# Airline Network Robustness — Starter Kit

This starter kit matches your proposal for **“Airline Network Robustness: Stress‑Testing Global Air Connectivity.”**  
It gives you a reproducible skeleton, sample data, and ready-to-run scripts.

## Quick start

```bash
# 1) Create a virtual environment (any method you prefer is fine)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Put the OpenFlights CSVs in ./data : airports.dat and routes.dat
#    Or keep using the small sample CSVs included for smoke testing.

# 4) Run a quick simulation on the sample data
python src/simulate.py --attack targeted_nodes --metric degree --k 3

# 5) Launch the interactive demo
streamlit run src/app/streamlit_app.py
```

## Expected folder layout

```
airline-robustness-starter/
├─ config/
│  └─ default.yaml
├─ data/
│  ├─ sample_airports.csv
│  └─ sample_routes.csv
├─ outputs/                       # metrics, figures and logs get saved here
├─ src/
│  ├─ attacks.py
│  ├─ centrality.py
│  ├─ data_io.py
│  ├─ defenses.py
│  ├─ geo.py
│  ├─ graph_build.py
│  ├─ metrics.py
│  ├─ simulate.py                 # CLI to run experiments
│  ├─ viz.py
│  └─ app/streamlit_app.py        # interactive demo
├─ tests/
│  └─ test_toy.py
├─ notebooks/
│  └─ 00_quickstart.md
└─ requirements.txt
```

## What’s included

- **Attack models:** random failures, targeted (degree / betweenness / PageRank / Collective Influence), adaptive and static, edge-betweenness, geographic radius, community-bridge removal.
- **Defense models:** greedy edge addition under a max-distance constraint, cross-community linking, basic node-hardening list.
- **Metrics:** giant weakly/strongly connected components (GWCC/GSCC), components count, ASPL and diameter on GWCC, % OD pairs within *H* hops.
- **Visualization:** quick Matplotlib plots and a Streamlit app.

> ⚠️ The sample data are tiny and only for smoke tests. Use OpenFlights for real results. See `src/data_io.py` for loading helpers and column expectations.

## Tips

- For large networks, prefer **`--adaptive false`** first, then turn it on for the final runs.
- Re-run centralities when changing the underlying graph (e.g., after removing nodes in adaptive mode).
- Save run configurations to `outputs/run_*.json` to keep a full audit trail.

