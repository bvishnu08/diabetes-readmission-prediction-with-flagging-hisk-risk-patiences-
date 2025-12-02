## How I Run the Diabetes Readmission Project

I put this run book together so my professor can follow the exact sequence I use—no guessing about what comes first or what files to look at. Think of it as the “connect-the-dots” guide that maps every artifact to the command that produces it.

---

### Project map (which file does what)

| Order | File / Folder | Purpose | What it feeds |
|-------|---------------|---------|---------------|
| 1 | `requirements.txt` | Locks the Python stack so joblib models load correctly | Everything |
| 2 | `data/raw/diabetic_data.csv` | UCI source data | `scripts/run_train.py`, notebooks |
| 3 | `src/*.py` | Library code (config, preprocess, model, train, evaluate, clinical utils) | Scripts, notebooks, dashboard |
| 4 | `scripts/run_train.py` | Builds processed datasets + models + thresholds | Evaluation + dashboard |
| 5 | `scripts/run_eval.py` | Reads processed data + models + thresholds, prints results | Slides/report + dashboard sanity |
| 6 | `dashboard.py` / `scripts/run_dashboard.py` | Streamlit UI tied to the same artifacts | Live demo |
| 7 | `notebooks/03_implementation_details.ipynb` (+ HTML) | Narrative version of the workflow | Submission artifact |
| 8 | Docs (`README.md`, `RUN_BOOK.md`, `COMPLETE_PROJECT_CODE.md`) | Explain the system front-to-back | Submission packet |

Follow the steps below in order and every file will line up with the outputs your professor expects.

---

### 0. Environment setup (what I do before touching the code)
1. Install Python 3.10+ (I use Anaconda).
2. Move into the repo folder: `cd /path/to/265_final`
3. Spin up a fresh virtual environment and install everything:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### 1. Data readiness check
- Confirm `data/raw/diabetic_data.csv` and `data/raw/IDS_mapping.csv` exist.  
- No command yet—the training step below will read them and write the processed splits used everywhere else.

---

### 2. Train the models (my first command)
Once the data is in place I immediately train, because that step does *everything*: cleaning, feature selection, fitting Logistic Regression + XGBoost, and tuning thresholds. This run is what connects the raw data to every downstream artifact.
```bash
python scripts/run_train.py
```
**Outputs:**
- `data/processed/train_processed.csv`
- `data/processed/test_processed.csv`
- `models/logreg_selected.joblib`
- `models/xgb_selected.joblib`
- `models/thresholds.json`

---

After it finishes I skim the console log for “✅ Saved models to models/…”. That tells me the dashboard + evaluation now have something to read.

---

### 3. Evaluate the trained models
Next I run the evaluation script to generate the CLI metrics, confusion matrices, and the clinical “safe discharge vs risk” summary. If I’m on a constrained system (like a VM), I set `OMP_NUM_THREADS=1` to avoid shared-memory issues.
```bash
python scripts/run_eval.py
```
If you need deterministic jobs on restricted systems, set `OMP_NUM_THREADS=1` before running to avoid shared-memory errors:
```bash
OMP_NUM_THREADS=1 python scripts/run_eval.py
```

---

### 4. Launch the dashboard
With artifacts in place, I fire up Streamlit so I can walk through the tuned metrics, plots, and prediction playground using the CURSOR_THEME styling. The dashboard only works after Steps 2–3, because it reads the same joblib pipelines and `thresholds.json`.
```bash
streamlit run dashboard.py
# or
python scripts/run_dashboard.py
```

---

### 5. Walk through the implementation notebook
Finally, I open `notebooks/03_implementation_details.ipynb` in Jupyter (Lab or classic), hit **Run → Run All**, and let it recreate the visualizations and experiments that mirror the production pipeline. Because the underlying functions live in `src/`, the notebook stays consistent with the scripts.
```bash
jupyter lab notebooks/03_implementation_details.ipynb
# Run all cells from top to bottom
```
`nbconvert` export (already generated): `notebooks/03_implementation_details.html`.

---

### 6. What I upload to Canvas (covers every file type)
To satisfy all the requested file types, I gather these:
- `P2 Final_submission report.pdf`
- `notebooks/03_implementation_details.ipynb`
- `notebooks/03_implementation_details.html`
- `dashboard.py`
- `scripts/run_train.py`, `scripts/run_eval.py`, `scripts/run_dashboard.py`
- Entire `src/` directory (zip if needed)
- `requirements.txt`
- Optional supporting docs: `COMPLETE_PROJECT_CODE.md`, `CODE_EXPLANATION.md`, `PIPELINE_DIFFERENCES.md`, `RUN_BOOK.md`

---

### 7. Troubleshooting tips I keep handy
- **OMP shared-memory error** (`Can't open SHM2`): I set `OMP_NUM_THREADS=1` or run outside containerized environments.
- **scikit-learn version mismatch warning**: reinstall from `requirements.txt` so the joblib artifacts load cleanly.
- **Streamlit can’t find artifacts**: rerun Step 2 so `models/` and `data/processed/` exist.
- **Notebook import errors**: make sure you launched Jupyter from the repo root so `src/` is on `PYTHONPATH`.

Following this order is exactly how I demo the project: set up → train → evaluate → dashboard → notebook → submission. Every file listed above connects directly to one of those steps, so nothing gets left out.

