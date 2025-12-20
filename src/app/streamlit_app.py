"""
Streamlit application for interactive airline network robustness analysis.

Provides a web UI for exploring network metrics, running attack simulations,
and testing defense strategies with visual hierarchy and attack/defense replay.
"""
import streamlit as st
import pandas as pd
import networkx as nx
import pydeck as pdk
from pathlib import Path
import sys
import os
from typing import Set, Tuple, List, Dict, Any, Optional

# Project root setup
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALLOWED_DATA_DIR = (ROOT / "data").resolve()

def apply_steps_to_graph(
    G: nx.DiGraph,
    attack_log: List[Dict],
    attack_step: int,
    defense_log: List[Dict],
    defense_step: int,
) -> nx.DiGraph:
    H = G.copy()

    # Apply removals up to attack_step
    removed_nodes: Set[str] = set()
    removed_edges: Set[Tuple[str, str]] = set()

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

    for (u, v) in removed_edges:
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
    clean_name = os.path.basename(filename)
    resolved = (ALLOWED_DATA_DIR / clean_name).resolve()
    if not str(resolved).startswith(str(ALLOWED_DATA_DIR)):
        raise ValueError(f"Access denied: '{filename}' is outside allowed directory")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {clean_name}")
    return resolved


# Imports
from src.data_io import load_airports, load_routes, merge_airports_routes
from src.graph_build import build_digraph
from src.metrics import topological_report
from src.attacks import (
    targeted_node_removal,
    edge_betweenness_attack,
    geographic_attack_radius,
    community_bridge_attack,
    random_node_failures,
)
from src.defenses import greedy_edge_addition, node_hardening_list
from src.constants import DEFAULT_TOP_N_HIGHLIGHTED, ATTACK_NODE_COLOR, NODE_SIZE_ATTACKED
from src.clustering import (
    community_clustering,
    geographic_clustering,
    cluster_aggregates,
    get_unclustered_nodes,
)
from src.viz import compute_node_emphasis, build_node_layer, build_edge_layer, build_cluster_layer


# --- Caching ---
@st.cache_data(ttl=300)
def cached_community_clustering(_hash: str, _G: nx.DiGraph) -> Dict[str, int]:
    """Cached community detection. _G prefixed to skip hashing."""
    return community_clustering(_G)


@st.cache_data(ttl=300)
def cached_geographic_clustering(_hash: str, _G: nx.DiGraph) -> Dict[str, int]:
    """Cached geographic clustering. _G prefixed to skip hashing."""
    return geographic_clustering(_G)


def graph_hash(G: nx.DiGraph) -> str:
    return f"{G.number_of_nodes()}_{G.number_of_edges()}"


# --- App Config ---
st.set_page_config(page_title="Airline Network Robustness", layout="wide", initial_sidebar_state="collapsed")

# Minimal CSS
st.markdown("""
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
""", unsafe_allow_html=True)


def metric_card(label: str, value: Any, delta: Optional[float] = None) -> str:
    delta_html = ""
    if delta is not None and delta != 0:
        cls = "metric-delta-up" if delta > 0 else "metric-delta-down"
        sign = "+" if delta > 0 else ""
        delta_html = f'<div class="{cls}">{sign}{delta:.1%}</div>'

    if isinstance(value, float):
        value_str = "∞" if value == float('inf') else f"{value:.2f}" if value > 1 else f"{value:.1%}"
    else:
        value_str = str(value)

    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value_str}</div>{delta_html}</div>'


def extract_attack_data(log: List[Dict], step: int) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    nodes, edges = set(), set()
    for i, entry in enumerate(log[:step]):
        if "removed_node" in entry:
            nodes.add(entry["removed_node"])
        if "removed_nodes" in entry:
            nodes.update(entry["removed_nodes"])
        if "removed_edge" in entry:
            edges.add(tuple(entry["removed_edge"]))
        if "removed_edges" in entry:
            edges.update(tuple(e) for e in entry["removed_edges"])
    return nodes, edges


def extract_defense_data(log: List[Dict], step: int) -> Set[Tuple[str, str]]:
    edges = set()
    for entry in log[:step]:
        if "added_edges" in entry:
            for u, v in entry["added_edges"]:
                a, b = (u, v) if u < v else (v, u)   # canonical undirected for display
                edges.add((a, b))
    return edges

def build_removed_nodes_layer(G_ref: nx.DiGraph, removed_nodes: Set[str]) -> Optional[pdk.Layer]:
    rows = []
    for n in removed_nodes:
        if n not in G_ref:
            continue
        d = G_ref.nodes[n]
        lat, lon = d.get("lat"), d.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({
            "iata": n,
            "name": d.get("name", n),
            "lat": lat,
            "lon": lon,
        })

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



# --- Session State ---
for key, default in [
    ("G", None),                 # original loaded graph
    ("G_base", None),            # scenario baseline (replay base)
    ("attack_log", []),
    ("defense_log", []),
    ("baseline_report", None),
    ("hardened_nodes", set()),
    ("defense_base_attack_step", 0),
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
            st.session_state.update({
        "G": G,
        "G_base": G,  # baseline của kịch bản hiện tại
        "baseline_report": topological_report(G, fast_mode=True),
        "attack_log": [],
        "defense_log": [],
        "hardened_nodes": set(),
        "defense_base_attack_step": 0,
        "atk_step": 0,
        "def_step": 0,
    })


            st.success(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            st.error(str(e))

G = st.session_state.get("G")
if G is None:
    st.info("Open sidebar to load graph data.")
    st.stop()
G_base = st.session_state.get("G_base") or G


# --- Layout ---
left, center, right = st.columns([1.2, 4, 1.2])

# --- Left Panel ---
with left:
    st.caption("VISUALIZATION")
    fast_mode = st.checkbox(
        "Fast mode (recommended for large graphs)",
        value=(G.number_of_nodes() > 800),
        key="fast_mode",
    )

    top_n = st.slider("Top-N nodes", 5, 50, DEFAULT_TOP_N_HIGHLIGHTED, key="top_n")
    emphasis_metric = st.selectbox("Rank by", ["degree", "betweenness", "pagerank"], key="emph_metric")
    labels_emphasized = st.checkbox("Labels: emphasized only", True, key="labels_emph")
    cluster_mode = st.radio("Cluster", ["Off", "Community", "Geographic"], key="cluster", horizontal=True)

    st.caption("ATTACK")
    attack_type = st.selectbox("Type", ["targeted_nodes", "edge_betweenness", "geographic_radius", "community_bridge"], key="atk_type", label_visibility="collapsed")
    chain_attack = st.checkbox("Chain attack from current replay state", value=False, key="chain_attack")


    if attack_type == "targeted_nodes":
        c1, c2 = st.columns(2)
        atk_metric = c1.selectbox("Metric", ["degree", "betweenness", "pagerank"], key="atk_m")
        atk_k = c2.number_input("k", 1, 100, 10, key="atk_k")
    elif attack_type == "edge_betweenness":
        atk_m = st.number_input("Edges (m)", 1, 100, 10, key="atk_edges")
    elif attack_type == "geographic_radius":
        c1, c2, c3 = st.columns(3)
        atk_lat = c1.number_input("Lat", value=40.0, key="atk_lat")
        atk_lon = c2.number_input("Lon", value=-74.0, key="atk_lon")
        atk_rad = c3.number_input("Radius km", value=500.0, key="atk_rad")
    elif attack_type == "community_bridge":
        atk_m = st.number_input("Bridges (m)", 1, 50, 10, key="atk_br")

    if st.button("Run Attack", type="primary", use_container_width=True):
        with st.spinner("..."):
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
                    H, log = targeted_node_removal(G_attack_base, k=atk_k, metric=atk_metric, adaptive=True, fast_mode=fast_mode, report_every_n=max(1, atk_k // 20))

                elif attack_type == "edge_betweenness":
                    H, log = edge_betweenness_attack(G_attack_base, m=atk_m, adaptive=True, fast_mode=fast_mode, report_every_n=max(1, atk_m // 20), recompute_every=1)

                elif attack_type == "geographic_radius":
                    H, info = geographic_attack_radius(G_attack_base, (atk_lat, atk_lon), atk_rad)
                    info["report"] = topological_report(H, fast_mode=fast_mode)
                    log = [info]

                elif attack_type == "community_bridge":
                    H, info = community_bridge_attack(G_attack_base, m=atk_m)
                    info["report"] = topological_report(H, fast_mode=fast_mode)
                    log = [info]

                st.session_state.update({
                "attack_log": log,
                "H_attack": H,
                "defense_log": [],                 # reset defense vì attack mới
                "defense_base_attack_step": 0,
                "def_step": 0,
                "atk_step": len(log),
            })

                st.toast("Attack complete")
            except Exception as e:
                st.error(str(e))

    st.caption("DEFENSE")
    c1, c2 = st.columns(2)
    def_budget = c1.number_input("Budget", 1, 10, 3, key="def_b")
    def_dist = c2.number_input("Max km", 500, 5000, 3000, key="def_d")

    if st.button("Run Defense", use_container_width=True):
        with st.spinner("..."):
            try:
                atk_step_for_def = int(st.session_state.get("atk_step", len(st.session_state.get("attack_log", []))))
                attack_log_now = st.session_state.get("attack_log", [])

                G_base_now = st.session_state.get("G_base") or st.session_state.get("G")

                # Defense MUST be computed on attacked graph at current attack step
                G_for_defense = apply_steps_to_graph(G_base_now, attack_log_now, atk_step_for_def, [], 0)

                H, log = greedy_edge_addition(
                    G_for_defense,
                    budget=def_budget,
                    max_distance_km=float(def_dist),
                    fast_mode=fast_mode
                )

                st.session_state.update({
                    "defense_log": log,
                    "H_defense": H,
                    "defense_base_attack_step": atk_step_for_def,
                    "def_step": len(log),
                })

                st.toast(f"Defense complete (based on attack step {atk_step_for_def})")

            except Exception as e:
                st.error(str(e))

    if st.button("Commit current state as new baseline", use_container_width=True):
        attack_log_now = st.session_state.get("attack_log", [])
        defense_log_now = st.session_state.get("defense_log", [])
        atk_step_now = int(st.session_state.get("atk_step", 0))
        def_step_now = int(st.session_state.get("def_step", 0))

        G_base_now = st.session_state.get("G_base") or st.session_state.get("G")

        committed = apply_steps_to_graph(G_base_now, attack_log_now, atk_step_now, defense_log_now, def_step_now)

        st.session_state.update({
            "G_base": committed,
            "attack_log": [],
            "defense_log": [],
            "defense_base_attack_step": 0,
            "atk_step": 0,
            "def_step": 0,
        })
        st.toast("Committed. Baseline updated.")


# --- Center: Map ---
with center:
    attack_log = st.session_state.get("attack_log", [])
    defense_log = st.session_state.get("defense_log", [])

    # Compact step controls
    if attack_log or defense_log:
        c1, c2 = st.columns(2)
        attack_step = c1.slider("Attack step", 0, max(1, len(attack_log)), len(attack_log), key="atk_step") if attack_log else 0

        base_step = int(st.session_state.get("defense_base_attack_step", 0))
        defense_reset_needed = False
        if defense_log and attack_step < base_step:
            st.session_state["def_step"] = 0
            defense_reset_needed = True

        defense_step = c2.slider("Defense step", 0, max(1, len(defense_log)), len(defense_log), key="def_step") if defense_log else 0

        if defense_reset_needed:
            st.info(f"Defense was computed at attack step {base_step}. Set Attack step ≥ {base_step} to replay defense.")
    else:
        attack_step, defense_step = 0, 0
    removed_nodes, removed_edges = extract_attack_data(attack_log, attack_step)
    added_edges = extract_defense_data(defense_log, defense_step)
    hardened = st.session_state.get("hardened_nodes", set())

    # Clustering
    use_clusters = cluster_mode != "Off"
    clusters, cluster_aggs = {}, None
    if use_clusters:
        h = graph_hash(G)
        clusters = cached_community_clustering(h, G) if cluster_mode == "Community" else cached_geographic_clustering(h, G)
        cluster_aggs = cluster_aggregates(G, clusters)

    # Node emphasis
    current_step_G = apply_steps_to_graph(G_base, attack_log, attack_step, defense_log, defense_step)
    emphasis = compute_node_emphasis(current_step_G, top_n, emphasis_metric, removed_nodes, hardened)


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
    st.pydeck_chart(deck, use_container_width=True)

    # Compact legend
    st.markdown('<div style="text-align:center;font-size:11px;color:#888;margin-top:4px;">🟠 Top-N  🔴 Removed  🟢 Added  🔵 Hardened  🟣 Cluster</div>', unsafe_allow_html=True)

# --- Right: Metrics ---
with right:
    # Build graph at the selected replay steps
    current_step_G = apply_steps_to_graph(G_base, attack_log, attack_step, defense_log, defense_step)
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
