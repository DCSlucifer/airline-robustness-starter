"""
Minimal Streamlit app to experiment with attacks/defenses.
"""
# --- IMPORTS (đừng sửa khác đi) ---
import streamlit as st
import pandas as pd
import networkx as nx
import json
import pydeck as pdk
from pathlib import Path

# Bảo đảm Python nhìn thấy thư mục gốc dự án (…/airline-robustness-starter)
import sys
ROOT = Path(__file__).resolve().parents[2]  # lên 2 cấp từ src/app/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Dùng import tuyệt đối (không còn dấu ..)
from src.data_io import load_airports, load_routes, merge_airports_routes
from src.graph_build import build_digraph
from src.centrality import node_centralities
from src.metrics import topological_report
from src.attacks import (
    targeted_node_removal,
    edge_betweenness_attack,
    geographic_attack_radius,
    community_bridge_attack,
)
from src.defenses import greedy_edge_addition
# --- STREAMLIT APP ---



st.set_page_config(page_title="Airline Network Robustness", layout="wide")

st.title("✈️ Airline Network Robustness — Interactive Demo")

# Sidebar for data loading
with st.sidebar:
    st.header("Data Loading")
    airports_path = st.text_input("Airports CSV", "data/airports.csv")
    routes_path = st.text_input("Routes CSV", "data/routes.csv")

    if st.button("Load graph"):
        try:
            airports = load_airports(airports_path)
            routes = load_routes(routes_path)
            airports, routes = merge_airports_routes(airports, routes)
            G = build_digraph(airports, routes, add_distance=True)
            st.session_state["G"] = G
            st.success(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

G = st.session_state.get("G")

if G is None:
    st.info("Please load the graph from the sidebar to start.")
    st.stop()

def render_map(G):
    # Extract node data
    nodes = []
    for n, data in G.nodes(data=True):
        if "lat" in data and "lon" in data:
            nodes.append({
                "iata": n,
                "lat": data["lat"],
                "lon": data["lon"],
                "name": data.get("name", n)
            })
    df_nodes = pd.DataFrame(nodes)

    # Extract edge data (arcs)
    arcs = []
    # Limit edges for performance if needed, but let's try full set first or a sample
    # For visual clarity, maybe just sample 2000 edges if it's too heavy
    edges_to_plot = list(G.edges())
    if len(edges_to_plot) > 5000:
        import random
        edges_to_plot = random.sample(edges_to_plot, 5000)

    for u, v in edges_to_plot:
        if u in G.nodes and v in G.nodes:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            if "lat" in u_data and "lon" in u_data and "lat" in v_data and "lon" in v_data:
                arcs.append({
                    "source": [u_data["lon"], u_data["lat"]],
                    "target": [v_data["lon"], v_data["lat"]],
                    "source_iata": u,
                    "dest_iata": v
                })
    df_arcs = pd.DataFrame(arcs)

    # Layers
    layer_nodes = pdk.Layer(
        "ScatterplotLayer",
        df_nodes,
        get_position=["lon", "lat"],
        get_color=[200, 30, 0, 160],
        get_radius=50000,
        pickable=True,
        radius_min_pixels=3,
        radius_max_pixels=10,
    )

    layer_arcs = pdk.Layer(
        "ArcLayer",
        df_arcs,
        get_source_position="source",
        get_target_position="target",
        get_source_color=[0, 128, 200, 80],
        get_target_color=[200, 0, 80, 80],
        get_width=1,
        width_min_pixels=1,
    )

    view_state = pdk.ViewState(latitude=20.0, longitude=0.0, zoom=1.5, pitch=0)

    r = pdk.Deck(
        layers=[layer_arcs, layer_nodes],
        initial_view_state=view_state,
        tooltip={"text": "{iata}: {name}"},
        map_style="mapbox://styles/mapbox/light-v9"
    )
    st.pydeck_chart(r)

# Tabs for different activities
tab1, tab2, tab3 = st.tabs(["Explore & Rankings", "Attacks", "Defenses"])

with tab1:
    st.header("Network Overview & Rankings")

    st.subheader("Global Connectivity Map")
    if st.checkbox("Show Interactive Map", value=True):
         render_map(G)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Topological Metrics")
        if st.button("Calculate Baseline Metrics"):
            rep = topological_report(G)
            st.json(rep)

    with col2:
        st.subheader("Node Centrality Rankings")
        if st.button("Calculate Rankings"):
            with st.spinner("Computing centralities (this may take a moment)..."):
                df_cent = node_centralities(G)
                st.session_state["df_cent"] = df_cent
            st.success("Done!")

    if "df_cent" in st.session_state:
        df = st.session_state["df_cent"]
        st.dataframe(df.head(50))

        st.download_button(
            "Download Rankings CSV",
            df.to_csv(index=False).encode("utf-8"),
            "node_rankings.csv",
            "text/csv",
            key='download-csv'
        )

with tab2:
    st.header("Run an Attack")
    attack_type = st.selectbox("Attack Type", ["targeted_nodes", "random_nodes", "edge_betweenness", "geographic_radius", "community_bridge"])

    colA, colB, colC = st.columns(3)

    # Dynamic controls based on attack type
    if attack_type == "targeted_nodes":
        with colA:
            metric = st.selectbox("Targeted metric", ["degree","betweenness","pagerank","CI"])
        with colB:
            k = st.number_input("k nodes to remove", 1, 500, 10)
        with colC:
            adaptive = st.checkbox("Adaptive recomputation", value=True)

    elif attack_type == "random_nodes":
        with colA:
            k = st.number_input("k nodes to remove", 1, 500, 10)
        with colB:
            R = st.number_input("Repetitions (R)", 1, 50, 5)
        with colC:
            st.write("") # Spacer

    elif attack_type == "edge_betweenness":
        with colA:
            m = st.number_input("m edges to remove", 1, 500, 10)
        with colB:
            adaptive = st.checkbox("Adaptive recomputation", value=True)
        with colC:
             st.write("")

    elif attack_type == "geographic_radius":
        with colA:
            lat = st.number_input("Center Lat", value=0.0, format="%.4f")
        with colB:
            lon = st.number_input("Center Lon", value=0.0, format="%.4f")
        with colC:
            radius = st.number_input("Radius (km)", value=1000.0, format="%.1f")

    elif attack_type == "community_bridge":
        with colA:
            m = st.number_input("m edges to remove", 1, 500, 10)
        with colB:
             st.write("")
        with colC:
             st.write("")

    if st.button("Run Attack"):
        with st.spinner("Simulating attack..."):
            if attack_type == "targeted_nodes":
                H, log = targeted_node_removal(G, k=int(k), metric=metric, adaptive=adaptive)
                st.session_state["H"] = H
                st.session_state["attack_log"] = log
            elif attack_type == "random_nodes":
                # Random returns a list of reports, not a single graph H
                from src.attacks import random_node_failures
                log = random_node_failures(G, k=int(k), R=int(R))
                st.session_state["H"] = None # No single resulting graph
                st.session_state["attack_log"] = log
            elif attack_type == "edge_betweenness":
                H, log = edge_betweenness_attack(G, m=int(m), adaptive=adaptive)
                st.session_state["H"] = H
                st.session_state["attack_log"] = log
            elif attack_type == "geographic_radius":
                H, info = geographic_attack_radius(G, (float(lat), float(lon)), float(radius))
                st.session_state["H"] = H
                st.session_state["attack_log"] = [info]
            elif attack_type == "community_bridge":
                H, info = community_bridge_attack(G, m=int(m))
                st.session_state["H"] = H
                st.session_state["attack_log"] = [info]

        st.success("Attack finished.")
        st.json(st.session_state["attack_log"])

with tab3:
    st.header("Run a Defense")
    budget = st.number_input("Budget (edges to add)", 1, 50, 3)
    dist_cap = st.number_input("Max distance km", 100.0, 20000.0, 3000.0, step=100.0)

    if st.button("Run Defense"):
        with st.spinner("Optimizing network..."):
            H, log = greedy_edge_addition(G, budget=int(budget), max_distance_km=float(dist_cap))
            st.session_state["H_def"] = H
            st.session_state["defense_log"] = log
        st.success("Defense finished.")
        st.json(log)

st.markdown("---")
st.markdown("Use the sidebar to load your own data.")
