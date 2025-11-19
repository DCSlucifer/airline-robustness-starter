# Airline Network Robustness Analysis Framework

## Overview

The **Airline Network Robustness Analysis Framework** is a comprehensive tool designed to stress-test global air connectivity. By modeling the global aviation network as a complex graph, this project enables researchers and analysts to simulate various disruption scenarios—ranging from targeted attacks to random failures—and evaluate the effectiveness of strategic defense mechanisms.

This framework integrates advanced graph theory metrics, geographic spatial analysis, and interactive visualizations to provide deep insights into network resilience, critical infrastructure identification, and topological efficiency.

## Key Features

### 🛡️ Attack Simulations
Simulate diverse disruption scenarios to test network resilience:
*   **Targeted Node Removal:** Attacks based on centrality metrics (Degree, Betweenness, PageRank, Collective Influence). Supports **adaptive** (recomputed after each step) and **static** modes.
*   **Random Failures:** Monte Carlo simulations to model random equipment failures or disruptions.
*   **Edge-Based Attacks:** Targeted removal of high-betweenness edges and "community bridges" that connect distinct network clusters.
*   **Geographic Disruptions:** Spatially localized failures affecting all airports within a specific radius of a coordinate.

### 🛡️ Defense Strategies
Evaluate mitigation strategies to improve network robustness:
*   **Greedy Edge Addition:** Heuristic algorithms to strategically add routes that maximize the Giant Weakly Connected Component (GWCC) and minimize Average Shortest Path Length (ASPL), subject to geographic distance constraints.
*   **Node Hardening:** Identification of critical nodes that require reinforced infrastructure or operational redundancy.

### 📊 Comprehensive Metrics
Quantify network health using rigorous topological metrics:
*   **Connectivity:** Giant Weakly/Strongly Connected Components (GWCC/GSCC) and component counts.
*   **Efficiency:** Average Shortest Path Length (ASPL) and Network Diameter.
*   **Reachability:** Percentage of Origin-Destination (OD) pairs reachable within a specified number of hops (*H*).

### 📈 Visualization
*   **Interactive Dashboard:** A Streamlit-based application for real-time data exploration, attack/defense simulation, and result visualization.
*   **Geospatial Mapping:** 3D interactive maps using PyDeck to visualize network structure and node importance.
*   **Robustness Curves:** Plot network degradation profiles under various attack strategies.

## Project Structure

```text
airline-robustness-starter/
├── config/                 # Configuration files
│   └── default.yaml        # Default simulation parameters
├── data/                   # Data directory (place OpenFlights CSVs here)
│   ├── airports.csv        # Airport metadata
│   └── routes.csv          # Route information
├── outputs/                # Simulation results, logs, and figures
├── src/                    # Source code
│   ├── app/                # Streamlit application
│   │   └── streamlit_app.py
│   ├── attacks.py          # Attack simulation logic
│   ├── centrality.py       # Centrality metric calculations
│   ├── data_io.py          # Data loading and processing
│   ├── defenses.py         # Defense strategy implementation
│   ├── geo.py              # Geographic utility functions
│   ├── graph_build.py      # Graph construction
│   ├── metrics.py          # Topological metric calculations
│   ├── simulate.py         # CLI entry point for simulations
│   └── viz.py              # Static plotting utilities
├── tests/                  # Unit tests
└── requirements.txt        # Python dependencies
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd airline-robustness-starter
    ```

2.  **Create a virtual environment:**
    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Interactive Dashboard (Recommended)
Launch the Streamlit application to explore the data and run simulations interactively:

```bash
streamlit run src/app/streamlit_app.py
```
The dashboard allows you to:
*   Load and visualize the network on an interactive 3D map.
*   Calculate and view node rankings.
*   Run targeted attacks and visualize the impact.
*   Simulate defense strategies and see the improvement in metrics.

### 2. Command Line Interface (CLI)
Run batch simulations using the `simulate.py` script.

**Example: Targeted Degree Attack**
```bash
python src/simulate.py --attack targeted_nodes --metric degree --k 10 --adaptive
```

**Example: Random Failure Simulation**
```bash
python src/simulate.py --attack random_nodes --k 50 --repetitions 20
```

**Example: Defense Simulation**
```bash
python src/simulate.py --defense greedy_edge_addition --budget 5
```

### 3. Configuration
Default parameters are stored in `config/default.yaml`. You can modify this file to adjust default paths, simulation parameters, and algorithm settings.

## Data Sources

This project is designed to work with airline data such as the [OpenFlights dataset](https://openflights.org/data.html).
*   **Airports:** Requires columns `iata`, `lat`, `lon`, `name`, `city`, `country`.
*   **Routes:** Requires columns `source_iata`, `dest_iata`.

Place your CSV files in the `data/` directory and update the paths in the Streamlit app or `config/default.yaml`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
