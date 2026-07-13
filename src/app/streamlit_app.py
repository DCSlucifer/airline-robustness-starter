# ruff: noqa: E402
"""
Streamlit application for interactive airline network robustness analysis.

Provides a web UI for exploring network metrics, running attack simulations,
and testing defense strategies with visual hierarchy and attack/defense replay.
"""

import inspect
import logging
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import pydeck as pdk
import streamlit as st

# Project root setup
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_DATA_DIR = (ROOT / "data").resolve()
logger = logging.getLogger(__name__)


def stretch_button(label: str, **kwargs: Any) -> bool:
    """Render a full-width button across supported Streamlit versions."""
    if "width" in inspect.signature(st.button).parameters:
        kwargs["width"] = "stretch"
    else:  # Streamlit < 1.50
        kwargs["use_container_width"] = True
    return st.button(label, **kwargs)


def stretch_pydeck_chart(deck: pdk.Deck) -> None:
    """Render a full-width PyDeck chart across supported Streamlit versions."""
    width_parameter = inspect.signature(st.pydeck_chart).parameters.get("width")
    if width_parameter is not None and width_parameter.default == "stretch":
        st.pydeck_chart(deck, width="stretch")
    else:  # Streamlit < 1.50
        st.pydeck_chart(deck, use_container_width=True)


def apply_steps_to_graph(
    G: nx.DiGraph,
    attack_log: list[dict],
    attack_step: int,
    defense_log: list[dict],
    defense_step: int,
) -> nx.DiGraph:
    H = G.copy()

    # Apply removals up to attack_step
    removed_nodes: set[str] = set()
    removed_edges: set[tuple[str, str]] = set()

    for entry in attack_log[:attack_step]:
        if not isinstance(entry, dict):
            continue
        if "removed_node" in entry and entry["removed_node"] is not None:
            removed_nodes.add(entry["removed_node"])
        if "removed_nodes" in entry and entry["removed_nodes"]:
            removed_nodes.update(entry["removed_nodes"])
        if "removed_edge" in entry and entry["removed_edge"]:
            removed_edges.add(tuple(entry["removed_edge"]))
        if "removed_edges" in entry and entry["removed_edges"]:
            removed_edges.update(tuple(e) for e in entry["removed_edges"])

    for n in removed_nodes:
        if n in H:
            H.remove_node(n)

    for u, v in removed_edges:
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        elif H.has_edge(v, u):
            H.remove_edge(v, u)

    # Apply additions up to defense_step
    for entry in defense_log[:defense_step]:
        if not isinstance(entry, dict):
            continue
        if "added_edges" in entry and entry["added_edges"]:
            for u, v in entry["added_edges"]:
                if u in H and v in H:
                    H.add_edge(u, v)

    return H


def sanitize_path(filename: str) -> Path:
    """Validates user-provided filename is within allowed data directory."""
    return sanitize_data_path(filename, ALLOWED_DATA_DIR)


# Imports
from src.ai.factory import make_client
from src.ai.guardrails import GuardrailError
from src.ai.orchestrator import run_whatif
from src.ai.rag.advisor import answer as rag_answer
from src.ai.rag.embedder import OpenAIEmbedder
from src.ai.rag.index import INDEX_PATH
from src.ai.rag.store import VectorStore
from src.app.ui_state import (
    committed_scenario_state,
    fresh_load_state,
    graph_fingerprint,
    prefer_graph,
    provider_error_message,
    rag_index_readiness,
    safe_error_metadata,
    sanitize_data_path,
)
from src.attacks import (
    community_bridge_attack,
    edge_betweenness_attack,
    geographic_attack_radius,
    targeted_node_removal,
)
from src.clustering import (
    cluster_aggregates,
    community_clustering,
    geographic_clustering,
    get_unclustered_nodes,
)
from src.constants import ATTACK_NODE_COLOR, DEFAULT_TOP_N_HIGHLIGHTED, NODE_SIZE_ATTACKED
from src.data_io import load_airports, load_routes, merge_airports_routes
from src.defenses import greedy_edge_addition
from src.graph_build import build_digraph
from src.metrics import topological_report
from src.viz import build_cluster_layer, build_edge_layer, build_node_layer, compute_node_emphasis


# --- Caching ---
@st.cache_data(ttl=300)
def cached_community_clustering(fingerprint: str, _G: nx.DiGraph) -> dict[str, int]:
    """Cache community detection by a hashed graph fingerprint."""
    return community_clustering(_G)


@st.cache_data(ttl=300)
def cached_geographic_clustering(fingerprint: str, _G: nx.DiGraph) -> dict[str, int]:
    """Cache geographic clustering by a hashed graph fingerprint."""
    return geographic_clustering(_G)


# --- App Config ---
st.set_page_config(
    page_title="Airline Network Robustness", layout="wide", initial_sidebar_state="expanded"
)
st.title("Airline Network Robustness")
st.caption("Explore network topology, simulate disruptions, and compare resilience strategies.")

# Minimal CSS
st.markdown(
    """
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .metric-label { color: #888; font-size: 11px; text-transform: uppercase; }
    .metric-value { color: #fff; font-size: 20px; font-weight: 600; }
    .metric-delta-up { color: #4ade80; font-size: 11px; }
    .metric-delta-down { color: #f87171; font-size: 11px; }
    .stRadio > div { flex-direction: row; gap: 8px; }
    .stRadio label { font-size: 13px; }
</style>
""",
    unsafe_allow_html=True,
)


def metric_card(label: str, value: Any, delta: float | None = None) -> str:
    delta_html = ""
    if delta is not None and delta != 0:
        cls = "metric-delta-up" if delta > 0 else "metric-delta-down"
        sign = "+" if delta > 0 else ""
        delta_html = f'<div class="{cls}">{sign}{delta:.1%}</div>'

    if isinstance(value, float):
        value_str = (
            "∞" if value == float("inf") else f"{value:.2f}" if value > 1 else f"{value:.1%}"
        )
    else:
        value_str = str(value)

    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value_str}</div>{delta_html}</div>'


def extract_attack_data(log: list[dict], step: int) -> tuple[set[str], set[tuple[str, str]]]:
    nodes, edges = set(), set()
    for entry in log[:step]:
        if "removed_node" in entry:
            nodes.add(entry["removed_node"])
        if "removed_nodes" in entry:
            nodes.update(entry["removed_nodes"])
        if "removed_edge" in entry:
            edges.add(tuple(entry["removed_edge"]))
        if "removed_edges" in entry:
            edges.update(tuple(e) for e in entry["removed_edges"])
    return nodes, edges


def extract_defense_data(log: list[dict], step: int) -> set[tuple[str, str]]:
    edges = set()
    for entry in log[:step]:
        if "added_edges" in entry:
            for u, v in entry["added_edges"]:
                a, b = (u, v) if u < v else (v, u)  # canonical undirected for display
                edges.add((a, b))
    return edges


def build_removed_nodes_layer(G_ref: nx.DiGraph, removed_nodes: set[str]) -> pdk.Layer | None:
    rows = []
    for n in removed_nodes:
        if n not in G_ref:
            continue
        d = G_ref.nodes[n]
        lat, lon = d.get("lat"), d.get("lon")
        if lat is None or lon is None:
            continue
        rows.append(
            {
                "iata": n,
                "name": d.get("name", n),
                "lat": lat,
                "lon": lon,
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return pdk.Layer(
        "ScatterplotLayer",
        df,
        get_position=["lon", "lat"],
        get_color=ATTACK_NODE_COLOR,
        get_radius=NODE_SIZE_ATTACKED,
        pickable=True,
        radius_min_pixels=3,
        radius_max_pixels=18,
    )


def log_safe_failure(context: str, error: Exception) -> None:
    """Log operational metadata without exception text, prompts, or API keys."""
    error_type, status_code = safe_error_metadata(error)
    logger.warning("%s failed (type=%s, status=%s)", context, error_type, status_code)


# --- Session State ---
for key, default in [
    ("G", None),  # original loaded graph
    ("G_base", None),  # scenario baseline (replay base)
    ("attack_log", []),
    ("defense_log", []),
    ("baseline_report", None),
    ("hardened_nodes", set()),
    ("defense_base_attack_step", 0),
    ("H_attack", None),
    ("H_defense", None),
    ("ai_result", None),
    ("rag_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- Sidebar: Data Loading ---
with st.sidebar:
    st.subheader("Load Data")
    airports_file = st.text_input("Airports", "airports.csv")
    routes_file = st.text_input("Routes", "routes.csv")

    if st.button("Load", type="primary"):
        try:
            airports = load_airports(str(sanitize_path(airports_file)))
            routes = load_routes(str(sanitize_path(routes_file)))
            airports, routes = merge_airports_routes(airports, routes)
            G = build_digraph(airports, routes, add_distance=True)
            loaded_state = fresh_load_state(G)
            loaded_state["baseline_report"] = topological_report(G, fast_mode=True)
            st.session_state.update(loaded_state)

            st.success(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            st.error(str(e))

G = st.session_state.get("G")
if G is None:
    st.info("Open sidebar to load graph data.")
    st.stop()
G_base = prefer_graph(st.session_state.get("G_base"), G)


# --- Layout ---
left, center, right = st.columns([1.2, 4, 1.2])

# --- Left Panel ---
with left:
    st.caption("VISUALIZATION")
    if "fast_mode" not in st.session_state:
        st.session_state["fast_mode"] = G.number_of_nodes() > 800
    fast_mode = st.checkbox(
        "Fast mode (recommended for large graphs)",
        key="fast_mode",
    )

    top_n = st.slider("Top-N nodes", 5, 50, DEFAULT_TOP_N_HIGHLIGHTED, key="top_n")
    emphasis_metric = st.selectbox(
        "Rank by", ["degree", "betweenness", "pagerank"], key="emph_metric"
    )
    labels_emphasized = st.checkbox("Labels: emphasized only", True, key="labels_emph")
    cluster_mode = st.radio(
        "Cluster", ["Off", "Community", "Geographic"], key="cluster", horizontal=True
    )

    st.caption("ATTACK")
    attack_type = st.selectbox(
        "Type",
        ["targeted_nodes", "edge_betweenness", "geographic_radius", "community_bridge"],
        key="atk_type",
    )
    chain_attack = st.checkbox(
        "Chain attack from current replay state", value=False, key="chain_attack"
    )

    if attack_type == "targeted_nodes":
        c1, c2 = st.columns(2)
        atk_metric = c1.selectbox("Metric", ["degree", "betweenness", "pagerank"], key="atk_m")
        atk_k = c2.number_input("k", 1, 100, 10, key="atk_k")
    elif attack_type == "edge_betweenness":
        atk_m = st.number_input("Edges (m)", 1, 100, 10, key="atk_edges")
    elif attack_type == "geographic_radius":
        c1, c2, c3 = st.columns(3)
        atk_lat = c1.number_input("Lat", min_value=-90.0, max_value=90.0, value=40.0, key="atk_lat")
        atk_lon = c2.number_input(
            "Lon", min_value=-180.0, max_value=180.0, value=-74.0, key="atk_lon"
        )
        atk_rad = c3.number_input("Radius km", min_value=0.0, value=500.0, key="atk_rad")
    elif attack_type == "community_bridge":
        atk_m = st.number_input("Bridges (m)", 1, 50, 10, key="atk_br")

    if stretch_button("Run Attack", type="primary"):
        with st.spinner("Running attack simulation..."):
            try:
                # Base graph for this attack run
                if chain_attack:
                    # chain from what user is currently replaying
                    G_attack_base = apply_steps_to_graph(
                        G_base,
                        st.session_state.get("attack_log", []),
                        int(st.session_state.get("atk_step", 0)),
                        st.session_state.get("defense_log", []),
                        int(st.session_state.get("def_step", 0)),
                    )
                else:
                    # fresh attack on scenario baseline
                    G_attack_base = G_base

                if attack_type == "targeted_nodes":
                    H, log = targeted_node_removal(
                        G_attack_base,
                        k=atk_k,
                        metric=atk_metric,
                        adaptive=True,
                        fast_mode=fast_mode,
                        report_every_n=max(1, atk_k // 20),
                    )

                elif attack_type == "edge_betweenness":
                    H, log = edge_betweenness_attack(
                        G_attack_base,
                        m=atk_m,
                        adaptive=True,
                        fast_mode=fast_mode,
                        report_every_n=max(1, atk_m // 20),
                        recompute_every=1,
                    )

                elif attack_type == "geographic_radius":
                    H, info = geographic_attack_radius(
                        G_attack_base, (atk_lat, atk_lon), atk_rad, fast_mode=fast_mode
                    )
                    log = [info]

                elif attack_type == "community_bridge":
                    H, info = community_bridge_attack(G_attack_base, m=atk_m, fast_mode=fast_mode)
                    log = [info]

                attack_state = {
                    "attack_log": log,
                    "H_attack": H,
                    "H_defense": None,
                    "defense_log": [],
                    "defense_base_attack_step": 0,
                    "def_step": 0,
                    "atk_step": len(log),
                }
                if chain_attack:
                    # The new log is relative to the replayed graph, so promote that
                    # graph to the scenario base before rendering the new attack.
                    G_base = G_attack_base
                    attack_state.update(
                        {
                            "G_base": G_attack_base,
                            "baseline_report": topological_report(
                                G_attack_base, fast_mode=fast_mode
                            ),
                            "ai_result": None,
                        }
                    )
                st.session_state.update(attack_state)

                st.toast("Attack complete")
            except Exception as e:
                st.error(str(e))

    st.caption("DEFENSE")
    c1, c2 = st.columns(2)
    def_budget = c1.number_input("Budget", 1, 10, 3, key="def_b")
    def_dist = c2.number_input("Max km", 500, 5000, 3000, key="def_d")

    if stretch_button("Run Defense"):
        with st.spinner("Designing resilience improvements..."):
            try:
                atk_step_for_def = int(
                    st.session_state.get("atk_step", len(st.session_state.get("attack_log", [])))
                )
                attack_log_now = st.session_state.get("attack_log", [])

                G_base_now = prefer_graph(st.session_state.get("G_base"), G)

                # Defense MUST be computed on attacked graph at current attack step
                G_for_defense = apply_steps_to_graph(
                    G_base_now, attack_log_now, atk_step_for_def, [], 0
                )

                H, log = greedy_edge_addition(
                    G_for_defense,
                    budget=def_budget,
                    max_distance_km=float(def_dist),
                    fast_mode=fast_mode,
                )

                st.session_state.update(
                    {
                        "defense_log": log,
                        "H_defense": H,
                        "defense_base_attack_step": atk_step_for_def,
                        "def_step": len(log),
                    }
                )

                if log:
                    st.toast(f"Defense complete (based on attack step {atk_step_for_def})")
                else:
                    st.info("No feasible defense edges matched the current constraints.")

            except Exception as e:
                st.error(str(e))

    has_scenario_changes = bool(
        st.session_state.get("attack_log") or st.session_state.get("defense_log")
    )
    if stretch_button("Commit current state as new baseline", disabled=not has_scenario_changes):
        attack_log_now = st.session_state.get("attack_log", [])
        defense_log_now = st.session_state.get("defense_log", [])
        atk_step_now = int(st.session_state.get("atk_step", 0))
        def_step_now = int(st.session_state.get("def_step", 0))

        G_base_now = prefer_graph(st.session_state.get("G_base"), G)

        committed = apply_steps_to_graph(
            G_base_now, attack_log_now, atk_step_now, defense_log_now, def_step_now
        )

        committed_state = committed_scenario_state(committed)
        committed_state["baseline_report"] = topological_report(committed, fast_mode=fast_mode)
        st.session_state.update(committed_state)
        G_base = committed
        st.toast("Committed. Baseline updated.")

    st.caption("ASK AI")
    ai_provider = st.selectbox("Provider", ["openai", "anthropic"], key="ai_provider")
    provider_label = "OpenAI" if ai_provider == "openai" else "Anthropic"
    whatif_api_key = st.text_input(
        f"{provider_label} API key (What-If)",
        type="password",
        key=f"whatif_{ai_provider}_api_key",
        help="Your key is used only for this session (BYOK).",
    )
    ai_query = st.text_input(
        "Ask a what-if question",
        key="ai_query",
        placeholder="What if a storm hits the US East Coast?",
    )
    if stretch_button("Ask AI"):
        st.session_state["ai_result"] = None
        clean_whatif_key = whatif_api_key.strip()
        clean_ai_query = ai_query.strip()
        if not clean_whatif_key:
            st.warning(f"Enter an {provider_label} API key to use the assistant.")
        elif not clean_ai_query:
            st.warning("Type a question first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    G_for_ai = prefer_graph(st.session_state.get("G_base"), G)
                    client = make_client(ai_provider, api_key=clean_whatif_key)
                    result = run_whatif(clean_ai_query, G_for_ai, client)
                    st.session_state["ai_result"] = result.model_dump()
                except GuardrailError:
                    st.error("The request could not be validated. Rephrase it and try again.")
                except Exception as error:
                    log_safe_failure("What-If provider request", error)
                    st.error(provider_error_message(error, "What-If assistant"))

    ai_result = st.session_state.get("ai_result")
    if ai_result:
        st.markdown(f"**Tool:** `{ai_result['tool_name']}`  ")
        st.caption(f"args: {ai_result['arguments']}")
        st.write(ai_result["explanation"])

    st.caption("RESILIENCE ADVISOR (RAG)")
    st.caption("Uses a separate OpenAI key for retrieval and the grounded answer.")
    rag_ready, rag_readiness_error = rag_index_readiness(INDEX_PATH)
    if not rag_ready:
        st.session_state["rag_result"] = None
        st.warning(rag_readiness_error)
    rag_api_key = st.text_input(
        "OpenAI API key (Advisor)",
        type="password",
        key="rag_openai_api_key",
        help="This key is separate from the What-If provider key and stays in this session.",
        disabled=not rag_ready,
    )
    rag_q = st.text_input(
        "Ask about resilience / disruptions",
        key="rag_q",
        placeholder="What disrupted European air travel in 2010?",
        disabled=not rag_ready,
    )
    if stretch_button("Ask Advisor", disabled=not rag_ready):
        st.session_state["rag_result"] = None
        clean_rag_key = rag_api_key.strip()
        clean_rag_query = rag_q.strip()
        if not clean_rag_key:
            st.warning("Enter an OpenAI API key for the Advisor.")
        elif not clean_rag_query:
            st.warning("Type a question first.")
        else:
            with st.spinner("Retrieving..."):
                try:
                    store = VectorStore.load(
                        INDEX_PATH,
                        expected_model="text-embedding-3-small",
                        require_nonempty=True,
                    )
                except Exception as error:
                    log_safe_failure("Advisor index load", error)
                    st.error("Knowledge index could not be loaded. Rebuild it and try again.")
                else:
                    try:
                        embedder = OpenAIEmbedder(api_key=clean_rag_key)
                        res = rag_answer(
                            clean_rag_query,
                            make_client("openai", api_key=clean_rag_key),
                            embedder,
                            store,
                        )
                    except Exception as error:
                        log_safe_failure("Advisor provider request", error)
                        st.error(provider_error_message(error, "Resilience Advisor"))
                    else:
                        st.session_state["rag_result"] = res.model_dump()

    rag_result = st.session_state.get("rag_result")
    if rag_result:
        st.write(rag_result["answer"])
        if rag_result["sources"]:
            st.caption("Sources")
            for i, s in enumerate(rag_result["sources"], start=1):
                st.markdown(f"{i}. [{s['title']}]({s['url']})")


# --- Center: Map ---
with center:
    attack_log = st.session_state.get("attack_log", [])
    defense_log = st.session_state.get("defense_log", [])

    # Compact step controls
    if attack_log or defense_log:
        c1, c2 = st.columns(2)
        attack_step = (
            c1.slider("Attack step", 0, max(1, len(attack_log)), len(attack_log), key="atk_step")
            if attack_log
            else 0
        )

        base_step = int(st.session_state.get("defense_base_attack_step", 0))
        defense_reset_needed = False
        if defense_log and attack_step < base_step:
            st.session_state["def_step"] = 0
            defense_reset_needed = True

        defense_step = (
            c2.slider("Defense step", 0, max(1, len(defense_log)), len(defense_log), key="def_step")
            if defense_log
            else 0
        )

        if defense_reset_needed:
            st.info(
                f"Defense was computed at attack step {base_step}. Set Attack step ≥ {base_step} to replay defense."
            )
    else:
        attack_step, defense_step = 0, 0
    removed_nodes, removed_edges = extract_attack_data(attack_log, attack_step)
    added_edges = extract_defense_data(defense_log, defense_step)
    hardened = st.session_state.get("hardened_nodes", set())
    current_step_G = apply_steps_to_graph(
        G_base, attack_log, attack_step, defense_log, defense_step
    )

    # Clustering
    use_clusters = cluster_mode != "Off"
    clusters, cluster_aggs = {}, None
    if use_clusters:
        fingerprint = graph_fingerprint(current_step_G)
        clusters = (
            cached_community_clustering(fingerprint, current_step_G)
            if cluster_mode == "Community"
            else cached_geographic_clustering(fingerprint, current_step_G)
        )
        cluster_aggs = cluster_aggregates(current_step_G, clusters)

    # Node emphasis
    emphasis = compute_node_emphasis(
        current_step_G, top_n, emphasis_metric, removed_nodes, hardened
    )

    # Build layers
    layers = build_edge_layer(current_step_G, removed_edges, added_edges)

    if use_clusters and cluster_aggs:
        cl = build_cluster_layer(cluster_aggs)
        if cl:
            layers.append(cl)
        unclustered = get_unclustered_nodes(current_step_G, clusters)
        sub_G = current_step_G.subgraph(unclustered)
        sub_emph = {n: emphasis.get(n, {}) for n in unclustered}
        nl, _ = build_node_layer(sub_G, sub_emph, labels_emphasized)
        if nl:
            layers.append(nl)
    else:
        nl, _ = build_node_layer(current_step_G, emphasis, labels_emphasized)
        if nl:
            layers.append(nl)
    # Overlay removed nodes so targeted attacks are visible
    removed_layer = build_removed_nodes_layer(G_base, removed_nodes)
    if removed_layer:
        layers.append(removed_layer)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1.4, pitch=0),
        tooltip={"text": "{iata}: {name}"},
        map_style="mapbox://styles/mapbox/dark-v10",
    )
    stretch_pydeck_chart(deck)

    # Compact legend
    st.markdown(
        '<div style="text-align:center;font-size:11px;color:#888;margin-top:4px;">🟠 Top-N  🔴 Removed  🟢 Added  🔵 Hardened  🟣 Cluster</div>',
        unsafe_allow_html=True,
    )

# --- Right: Metrics ---
with right:
    report = topological_report(current_step_G, fast_mode=fast_mode)
    baseline = topological_report(G_base, fast_mode=fast_mode)

    def delta(k):
        if baseline and k in baseline and baseline[k] and baseline[k] != 0:
            return (report[k] - baseline[k]) / abs(baseline[k])
        return None

    st.caption("CONNECTIVITY")
    st.markdown(
        metric_card(
            "GWCC",
            f"{100 * report['gwcc_frac']:.3f}%<br><span style='font-size:13px;color:#9aa4b2'>{report['gwcc_n']}/{report['n_nodes']} nodes</span>",
            delta("gwcc_frac"),
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        metric_card(
            "GSCC",
            f"{100 * report['gscc_frac']:.3f}%<br><span style='font-size:13px;color:#9aa4b2'>{report['gscc_n']}/{report['n_nodes']} nodes</span>",
            delta("gscc_frac"),
        ),
        unsafe_allow_html=True,
    )

    st.markdown(metric_card("Components", report["n_components"]), unsafe_allow_html=True)

    st.caption("EFFICIENCY")
    st.markdown(metric_card("ASPL", report["aspl_gwcc"]), unsafe_allow_html=True)
    st.markdown(metric_card("Diameter", report["diameter_gwcc"]), unsafe_allow_html=True)
    st.markdown(metric_card("OD ≤4 hops", report["pct_od_within_H"]), unsafe_allow_html=True)

    st.caption("SIZE")
    st.markdown(metric_card("Nodes", report["n_nodes"]), unsafe_allow_html=True)
    st.markdown(metric_card("Edges", report["n_edges"]), unsafe_allow_html=True)

    # Compact log summary
    if attack_log:
        st.caption(f"Attack: {len(removed_nodes)} nodes, {len(removed_edges)} edges removed")
    if defense_log:
        st.caption(f"Defense: {len(added_edges)} edges added")
