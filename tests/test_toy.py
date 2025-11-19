from src.graph_build import build_digraph
from src.metrics import topological_report
import pandas as pd

def test_topological_report():
    airports = pd.DataFrame([
        {"airport_id":1, "name":"A","city":"X","country":"N","iata":"AAA","icao":"A","lat":0.0,"lon":0.0},
        {"airport_id":2, "name":"B","city":"X","country":"N","iata":"BBB","icao":"B","lat":0.0,"lon":1.0},
        {"airport_id":3, "name":"C","city":"X","country":"N","iata":"CCC","icao":"C","lat":1.0,"lon":0.0},
    ])
    routes = pd.DataFrame([{"source_iata":"AAA","dest_iata":"BBB"},{"source_iata":"BBB","dest_iata":"CCC"}])
    G = build_digraph(airports, routes, add_distance=True)
    rep = topological_report(G)
    assert rep["n_nodes"] == 3
    assert rep["n_edges"] == 2
