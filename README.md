# Airline Network Robustness Analysis Framework

A graph-theoretic toolkit for simulating disruptions to global aviation networks and evaluating defensive strategies through interactive visualization.

---

## Overview

Air transportation networks are critical infrastructure vulnerable to cascading failures from targeted attacks, random disruptions, or localized disasters. This framework models the global airline network as a directed graph where airports are nodes and routes are edges, enabling systematic analysis of network resilience.

The project provides:
- Simulation of multiple attack strategies (targeted, random, geographic, community-based)
- Defense mechanisms to reinforce network connectivity
- Quantitative metrics for network health assessment
- An interactive Streamlit dashboard with geographic visualization

This tool supports academic research, infrastructure planning exercises, and coursework in complex network analysis.

---

## Features

### Attack Models
- **Targeted node removal**: Removes airports ranked by centrality (degree, betweenness, PageRank, or Collective Influence). Supports adaptive mode where rankings are recomputed after each removal.
- **Random node failures**: Monte Carlo simulation of random airport outages across multiple repetitions.
- **Edge betweenness attack**: Removes high-betweenness edges that serve as critical bridges.
- **Geographic radius attack**: Disables all airports within a specified distance from a coordinate (simulating regional disasters).
- **Community bridge attack**: Targets edges connecting different network communities.

### Defense Models
- **Greedy edge addition**: Strategically adds routes to maximize connectivity, subject to geographic distance constraints.
- **Node hardening**: Identifies critical airports that should be prioritized for infrastructure reinforcement.

### Metrics Reported
- GWCC/GSCC fraction (connectivity)
- Number of weakly connected components
- Average Shortest Path Length and diameter (efficiency)
- Percentage of origin-destination pairs reachable within H hops (practical reachability)

### Interactive Visualization
The Streamlit UI implements:
- **Map visualization** of attack and defense vectors using PyDeck
- **Visual hierarchy**: Top-N nodes are emphasized (larger, brighter); low-priority nodes are dimmed
- **Node clustering**: Community-based or geographic grouping to reduce visual clutter
- **Step replay sliders** for progressive attack/defense visualization
- Real-time metric updates with delta indicators

---

## Demo

### Running the Dashboard

```bash
cd airline-robustness-starter
.venv\Scripts\python.exe -m streamlit run src/app/streamlit_app.py
```

On macOS/Linux:
```bash
.venv/bin/python -m streamlit run src/app/streamlit_app.py
```

The application opens at `http://localhost:8501`.

### Screenshot Placeholders

Place demo images in a `docs/images/` directory:

```
docs/images/
├── dashboard_overview.png    # Full UI with 3-column layout
├── attack_demo.gif           # Animated attack sequence
├── clustering_view.png       # Community or geographic clustering
└── defense_edges.png         # Green edges showing added routes
```

Embed in README:
```markdown
![Dashboard Overview](docs/images/dashboard_overview.png)
```

### Using the UI

1. **Load data**: Open sidebar, click "Load" with default files
2. **Visual hierarchy**: Adjust "Top-N nodes" slider; change "Rank by" metric
3. **Clustering**: Select "Community" or "Geographic" to aggregate minor nodes
4. **Run attack**: Choose attack type, set parameters, click "Run Attack"
5. **Replay**: Move the attack step slider to see progressive damage
6. **Run defense**: Set budget and max distance, click "Run Defense"
7. **Observe metrics**: Right panel updates with current network health

---

## Method Summary

### Graph Model
The airline network is represented as a directed graph G = (V, E), where V is the set of airports and E is the set of routes. Edge weights represent geographic distance (Haversine formula).

### Key Metrics

| Metric | Definition |
|--------|------------|
| **GWCC** | Giant Weakly Connected Component: the largest set of nodes connected when edge directions are ignored. Reported as fraction of total nodes. |
| **GSCC** | Giant Strongly Connected Component: the largest set where every node is reachable from every other following edge directions. |
| **ASPL** | Average Shortest Path Length: mean number of hops between all node pairs in the GWCC. |
| **Diameter** | Maximum shortest path length in the GWCC. |
| **OD within H hops** | Fraction of origin-destination pairs reachable within H transfers (default H=4). |

### Attack Strategies
Attacks remove nodes or edges iteratively. Adaptive attacks recompute centrality scores after each removal; static attacks use initial rankings throughout.

### Defense Strategy
Greedy edge addition evaluates candidate edges (between high-degree nodes across communities) and selects those that maximize GWCC while minimizing ASPL, subject to a maximum geographic distance constraint.

---

## Installation

**Requirements**: Python 3.10 or higher

```bash
# Clone repository
git clone https://github.com/DCSlucifer/airline-robustness-starter.git
cd airline-robustness-starter

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core packages: `pandas`, `numpy`, `networkx`, `matplotlib`, `plotly`, `streamlit`, `pydeck`, `pyyaml`

Optional GIS packages (for extended geographic analysis): `geopandas`, `folium`, `pyproj`

If GIS packages cause installation issues, comment them out in `requirements.txt`; the core functionality remains available.

---

## Quickstart

### Minimum Commands

```bash
# Activate environment
.venv\Scripts\activate    # Windows
source .venv/bin/activate # macOS/Linux

# Run interactive dashboard
python -m streamlit run src/app/streamlit_app.py
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from the repository root directory |
| Port 8501 in use | Use `--server.port 8502` flag |
| Clustering causes `UnhashableParamError` | Ensure you have the latest `streamlit_app.py` with `_G` parameter prefix |

---

## Data

### Source
The framework is designed for the [OpenFlights dataset](https://openflights.org/data.html), which provides global airport and route information under the Open Database License.

### Expected Format

**airports.csv** (required columns):
| Column | Description |
|--------|-------------|
| `iata` | 3-letter IATA airport code (node identifier) |
| `lat` | Latitude |
| `lon` | Longitude |
| `name` | Airport name |

Additional columns (`city`, `country`, `airport_id`, `icao`) are preserved but not required for core functionality.

**routes.csv** (required columns):
| Column | Description |
|--------|-------------|
| `source_iata` | Origin airport IATA code |
| `dest_iata` | Destination airport IATA code |

### Replacing Data
Place custom CSV files in `data/` and update filenames in the Streamlit sidebar or `config/default.yaml`.

---

## Project Structure

```
airline-robustness-starter/
├── config/
│   └── default.yaml          # Simulation parameters (k, m, H, budget, distances)
├── data/
│   ├── airports.csv          # Airport metadata
│   └── routes.csv            # Route definitions
├── outputs/                  # Simulation results (JSON logs, figures)
├── src/
│   ├── app/
│   │   └── streamlit_app.py  # Interactive dashboard
│   ├── attacks.py            # Attack simulation algorithms
│   ├── defenses.py           # Defense strategy implementations
│   ├── clustering.py         # Community and geographic clustering
│   ├── viz.py                # PyDeck visualization layers
│   ├── metrics.py            # Topological metric calculations
│   ├── centrality.py         # Node centrality computations
│   ├── geo.py                # Haversine distance, radius queries
│   ├── graph_build.py        # NetworkX graph construction
│   ├── data_io.py            # CSV loading and validation
│   ├── simulate.py           # CLI entry point
│   └── constants.py          # Configuration constants
├── tests/                    # Unit tests (pytest)
└── requirements.txt
```

---

## Configuration

Parameters are stored in `config/default.yaml`:

```yaml
random_seed: 42
hops_H: 4                    # Hop limit for OD reachability
repetitions_R: 10            # Monte Carlo repetitions
adaptive: true               # Recompute centrality after each removal
distance_km_max: 3000        # Maximum distance for new edges (km)
budget_b: 5                  # Number of edges to add in defense
k_nodes: 10                  # Nodes to remove in targeted attack
m_edges: 20                  # Edges to remove in edge attack
collective_influence_l: 2   # CI algorithm radius

airports_csv: data/airports.csv
routes_csv: data/routes.csv
output_dir: outputs
```

Command-line arguments override these values when using `simulate.py`.

---

## Reproducibility

### Running Batch Experiments

```bash
# Targeted degree attack (10 nodes, adaptive)
python -m src.simulate --attack targeted_nodes --metric degree --k 10 --adaptive

# Random failures (50 nodes, 20 repetitions)
python -m src.simulate --attack random_nodes --k 50 --R 20

# Geographic attack (500 km radius around JFK coordinates)
python -m src.simulate --attack geographic_radius --lat 40.64 --lon -73.78 --radius_km 500

# Defense simulation
python -m src.simulate --mode defense --budget 5 --distance_km_max 3000
```

### Outputs
Results are saved to `outputs/`:
- `baseline_report.json`: Initial network metrics
- `attack_log.json` or `defense_log.json`: Step-by-step simulation log

### Running Tests

```bash
python -m pytest tests/ -v
```

---

## UI Implementation Notes

### Visual Hierarchy
Nodes are ranked by the selected centrality metric. The top-N nodes receive:
- Larger marker radius (80,000 m vs 25,000 m)
- Higher opacity (220 vs 50 on 0-255 scale)
- Orange color for emphasis

Attacked nodes appear in red; hardened nodes in blue.

### Clustering
Two aggregation methods reduce visual clutter for large networks:
- **Community**: Groups nodes by label propagation algorithm
- **Geographic**: Groups nodes by 5° latitude/longitude grid cells

Clusters with fewer than 3 nodes remain as individual markers.

### Performance Considerations
- Edge rendering is sampled to 5,000 edges maximum
- Centrality and clustering computations are cached (5-minute TTL)
- Use clustering mode for networks exceeding 1,000 nodes

---

## Known Limitations

- **Topology-only model**: Edge weights represent distance, not traffic volume or capacity
- **Static snapshots**: Does not model temporal dynamics or schedule-based connectivity
- **Single-layer network**: Does not distinguish airline alliances or code-share relationships
- **Greedy defense**: Edge addition uses heuristic optimization, not globally optimal solutions

---

## Roadmap

Completed:
- Core attack and defense simulations
- Interactive Streamlit dashboard
- Visual hierarchy and clustering
- Map visualization with attack/defense vectors

Future work:
- Integration of passenger flow data for weighted resilience analysis
- Multi-layer network modeling (airline alliances, cargo vs passenger)
- Post-2020 resilience extensions: pandemic-induced network restructuring analysis
- Export of robustness curves and comparative attack profiles

---

## Credits and References

### Course Context
Developed as a Network Science coursework project exploring infrastructure resilience.

### Data Source
Airport and route data derived from [OpenFlights](https://openflights.org/data.html), used under the Open Database License.

### Relevant Literature
- Albert, R., Jeong, H., & Barabási, A.-L. (2000). Error and attack tolerance of complex networks. *Nature*, 406, 378-382.
- Lordan, O., Sallan, J. M., Simo, P., & Gonzalez-Prieto, D. (2014). Robustness of the air transport network. *Transportation Research Part E*, 68, 155-163.

---

## License

No license file is currently included in the repository. Contact the repository owner for licensing terms before redistribution.
