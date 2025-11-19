# Quickstart (notebook placeholder)

1. Install dependencies and activate your environment.
2. Inspect `data/sample_*` to learn expected columns.
3. Run a small targeted attack:

```bash
python src/simulate.py --attack targeted_nodes --metric degree --k 3
```

4. Try a defense pass (edge addition):

```bash
python src/simulate.py --mode defense --budget 3 --distance_km_max 2000
```

5. Launch the demo:

```bash
streamlit run src/app/streamlit_app.py
```
