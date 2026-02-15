# Airline Network Robustness Analysis

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://dcslucifer-airline-robustness-starte-srcappstreamlit-app-vhymx8.streamlit.app/)
> **Try it now:** [https://dcslucifer-airline-robustness-starte-srcappstreamlit-app-vhymx8.streamlit.app/](https://dcslucifer-airline-robustness-starte-srcappstreamlit-app-vhymx8.streamlit.app/)
A graph-theoretic framework for simulating disruptions to global aviation networks and evaluating defensive strategies.

## Overview

Air transportation networks are critical infrastructure vulnerable to cascading failures from targeted attacks, random disruptions, or localized disasters. This framework models the global airline network as a directed graph where airports are nodes and routes are edges, enabling systematic analysis of network resilience.

**Key capabilities:**

- Simulation of multiple attack strategies (targeted, random, geographic, community-based)
- Defense mechanisms to reinforce network connectivity
- Quantitative metrics for network health assessment
- Interactive Streamlit dashboard with geographic visualization

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Attack Models](#attack-models)
- [Defense Models](#defense-models)
- [Metrics](#metrics)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [CLI Usage](#cli-usage)
- [Testing](#testing)
- [References](#references)

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/DCSlucifer/airline-robustness-starter.git
cd airline-robustness-starter

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `networkx` | Graph algorithms and data structures |
| `pandas` | Data manipulation |
| `numpy` | Numerical computations |
| `streamlit` | Interactive web dashboard |
| `pydeck` | Geographic map visualization |
| `matplotlib` | Static plotting |
| `plotly` | Interactive charts |

Optional GIS packages (`geopandas`, `folium`, `pyproj`) can be commented out if installation issues occur.

## Quick Start

```bash
# Activate environment
.venv\Scripts\activate    # Windows
source .venv/bin/activate # macOS/Linux

# Launch dashboard
python -m streamlit run src/app/streamlit_app.py
```

The application opens at `http://localhost:8501`.

### Dashboard Usage

1. **Load data** - Open sidebar, click "Load" with default files
2. **Visual hierarchy** - Adjust "Top-N nodes" slider; change "Rank by" metric
3. **Clustering** - Select "Community" or "Geographic" to aggregate minor nodes
4. **Run attack** - Choose attack type, set parameters, click "Run Attack"
5. **Replay** - Move the attack step slider to see progressive damage
6. **Run defense** - Set budget and max distance, click "Run Defense"
7. **Observe metrics** - Right panel updates with current network health

## Attack Models

### Targeted Node Removal

Removes airports ranked by centrality metrics. Supports adaptive mode where rankings are recomputed after each removal.

**Metrics supported:** `degree`, `betweenness`, `pagerank`, `CI` (Collective Influence)

```python
from src.attacks import targeted_node_removal

H, log = targeted_node_removal(G, k=10, metric="betweenness", adaptive=True)
```

### Random Node Failures

Monte Carlo simulation of random airport outages across multiple repetitions.

```python
from src.attacks import random_node_failures

reports = random_node_failures(G, k=50, R=20, seed=42)
```

### Edge Betweenness Attack

Removes high-betweenness edges that serve as critical bridges.

```python
from src.attacks import edge_betweenness_attack

H, log = edge_betweenness_attack(G, m=10, adaptive=True)
```

### Geographic Radius Attack

Disables all airports within a specified distance from a coordinate, simulating regional disasters.

```python
from src.attacks import geographic_attack_radius

H, info = geographic_attack_radius(G, center=(40.64, -73.78), radius_km=500)
```

### Community Bridge Attack

Targets edges connecting different network communities.

```python
from src.attacks import community_bridge_attack

H, info = community_bridge_attack(G, m=15)
```

## Defense Models

### Greedy Edge Addition

Strategically adds routes to maximize connectivity, subject to geographic distance constraints.

```python
from src.defenses import greedy_edge_addition

H, log = greedy_edge_addition(G, budget=5, max_distance_km=3000)
```

### Node Hardening

Identifies critical airports that should be prioritized for infrastructure reinforcement.

```python
from src.defenses import node_hardening_list

critical_nodes = node_hardening_list(G, top_n=10, metric="betweenness")
```

## Metrics

| Metric | Definition |
|--------|------------|
| **GWCC** | Giant Weakly Connected Component - largest set of nodes connected ignoring edge directions |
| **GSCC** | Giant Strongly Connected Component - largest set where every node is mutually reachable |
| **ASPL** | Average Shortest Path Length - mean hops between all node pairs in GWCC |
| **Diameter** | Maximum shortest path length in GWCC |
| **OD within H hops** | Fraction of origin-destination pairs reachable within H transfers (default H=4) |

```python
from src.metrics import topological_report

report = topological_report(G, H=4, fast_mode=False)
```

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

## Project Structure

```
airline-robustness-starter/
├── config/
│   └── default.yaml          # Simulation parameters
├── data/
│   ├── airports.csv          # Airport metadata
│   └── routes.csv            # Route definitions
├── outputs/                  # Simulation results
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
├── pyproject.toml            # Project metadata and tool config
└── requirements.txt          # Python dependencies
```

## Data Format

The framework uses the [OpenFlights dataset](https://openflights.org/data.html) format.

### airports.csv

| Column | Description |
|--------|-------------|
| `iata` | 3-letter IATA airport code (node identifier) |
| `lat` | Latitude |
| `lon` | Longitude |
| `name` | Airport name |

Additional columns (`city`, `country`, `airport_id`, `icao`) are preserved but not required.

### routes.csv

| Column | Description |
|--------|-------------|
| `source_iata` | Origin airport IATA code |
| `dest_iata` | Destination airport IATA code |

Custom datasets can be placed in `data/` and referenced via the Streamlit sidebar or `config/default.yaml`.

## CLI Usage

```bash
# Targeted degree attack (10 nodes, adaptive)
python -m src.simulate --attack targeted_nodes --metric degree --k 10 --adaptive

# Random failures (50 nodes, 20 repetitions)
python -m src.simulate --attack random_nodes --k 50 --R 20

# Geographic attack (500 km radius around JFK)
python -m src.simulate --attack geographic_radius --lat 40.64 --lon -73.78 --radius_km 500

# Defense simulation
python -m src.simulate --mode defense --budget 5 --distance_km_max 3000
```

Results are saved to `outputs/` as JSON logs.

## Testing

```bash
python -m pytest tests/ -v
```

All 69 tests should pass. Test coverage includes:

- Attack functions (targeted, random, edge betweenness, geographic, community bridge)
- Defense functions (greedy edge addition, node hardening)
- Metric calculations (GWCC, GSCC, ASPL, diameter, OD reachability)
- Clustering algorithms (community, geographic)
- Visualization components
- Edge cases (empty graphs, single nodes, disconnected components)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from the repository root directory |
| Port 8501 in use | Use `--server.port 8502` flag |
| `UnhashableParamError` with clustering | Ensure latest `streamlit_app.py` with `_G` parameter prefix |

## Known Limitations

- **Topology-only model** - Edge weights represent distance, not traffic volume or capacity
- **Static snapshots** - Does not model temporal dynamics or schedule-based connectivity
- **Single-layer network** - Does not distinguish airline alliances or code-share relationships
- **Greedy defense** - Edge addition uses heuristic optimization, not globally optimal solutions

## References

- Albert, R., Jeong, H., & Barabási, A.-L. (2000). Error and attack tolerance of complex networks. *Nature*, 406, 378-382.
- Lordan, O., Sallan, J. M., Simo, P., & Gonzalez-Prieto, D. (2014). Robustness of the air transport network. *Transportation Research Part E*, 68, 155-163.

## Data Source

Airport and route data derived from [OpenFlights](https://openflights.org/data.html), used under the Open Database License.

## License

MIT License. See [LICENSE](LICENSE) for details.
