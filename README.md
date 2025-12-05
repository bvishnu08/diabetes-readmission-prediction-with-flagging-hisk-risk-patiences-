# Diabetes 30-Day Readmission Prediction

> **🚀 EASIEST WAY: Download & Run Everything in One Command!**
> - **Mac/Linux:** `./download_and_run.sh` or `python download_and_run.py`
> - **Windows:** `download_and_run.bat` or `python download_and_run.py`
> 
> These scripts automatically download the repository from GitHub, set up the environment, install packages, train models, and evaluate them - **all in one command!**
> 
> **Alternative:** Manual setup → **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete step-by-step instructions.

This MSBA-265 project predicts whether a diabetic inpatient will be
readmitted within 30 days of discharge. The repo is organized so it can be
explained live (P2) and submitted as a polished artifact (P3):
clean preprocessing, two complementary models, threshold tuning focused on high
recall, and a Streamlit dashboard that mirrors the CLI outputs.

## Project goal

- **Clinical motivation:** flag high-risk discharges early so care managers can
  intervene, improve outcomes, and avoid CMS HRRP penalties.
- **Dataset:** UCI *Diabetes 130-US Hospitals* (~100k encounters).
- **Target:** `readmitted_binary = 1` if the encounter is followed by another
  admission within 30 days.
- **Metric focus:** maximize recall for the positive class (do not miss true
  readmissions), then choose the operating point with the highest F1-score.

## Modeling approach

| Model | Role | Feature subset | Notes |
|-------|------|----------------|-------|
| Logistic Regression | Interpretable baseline for clinicians | Top 20 features (SelectKBest) | ColumnTransformer preprocessing, `class_weight="balanced"` |
| XGBoost | Deployment candidate | Top 25 features (SelectKBest) | 300 estimators, depth 4, learning-rate 0.05 |

Why two models?
1. **Logistic Regression** anchors the story around transparency—coefficients
   and odds ratios can be discussed with stakeholders.
2. **XGBoost** achieves stronger F1/ROC-AUC at the same high-recall target, so
   it is the recommended deployment model.

### Feature pool & selection

`Config.candidate_features` contains a curated set of 41 demographic, utilization,
and diabetes-therapy signals (age bucket, admission source, prior utilization,
insulin changes, diabetes medications, etc.). During training we run SelectKBest on this pool:
- LR keeps the top 20 features (easier to interpret while still capturing key signals).
- XGBoost keeps the top 25 (more expressive while still limited).

### Threshold tuning

After fitting each model we score the held-out test split, sweep thresholds from
0.05 → 0.95, retain candidates with recall ≥ 0.65, and choose the threshold
with the highest F1-score for the readmission class. The final thresholds and
feature lists are persisted to `models/thresholds.json`, which keeps evaluation
and the dashboard perfectly in sync.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── dashboard.py
├── docs/
│   ├── RUN_BOOK.md
│   ├── QUICK_START.md
│   ├── PROJECT_STRUCTURE.md
│   ├── COMPLETE_PROJECT_CODE.md
│   ├── CODE_EXPLANATION.md
│   ├── PIPELINE_DIFFERENCES.md
│   ├── P3_SUBMISSION_CHECKLIST.md
│   ├── P3_SUBMISSION_SUMMARY.md
│   └── archive/ (older presentation files)
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv
│   │   └── IDS_mapping.csv
│   └── processed/{train,test}_processed.csv
├── models/
│   ├── logreg_selected.joblib
│   ├── xgb_selected.joblib
│   └── thresholds.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_implementation_details.ipynb
├── reports/
│   └── P2 Final_submission report.pdf
├── scripts/
│   ├── run_train.py
│   ├── run_eval.py
│   └── run_dashboard.py
└── src/
    ├── config.py
    ├── preprocess.py
    ├── feature_selection.py
    ├── model.py
    ├── train.py
    └── evaluate.py
```

## How to run

### 🚀 **EASIEST WAY: One Command (Recommended)**

After cloning/downloading the repository:

```bash
python run_all.py
```

This single command will:
- ✅ Create virtual environment
- ✅ Install all packages
- ✅ Train both models
- ✅ Evaluate models
- ✅ Show results

**Total time: 5-10 minutes**

---

### 📥 **Step 1: Get the Repository**

**Option A: Git Clone**
```bash
git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
```

**Option B: Download ZIP** (No Git needed)
1. Go to: https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
2. Click "Code" → "Download ZIP"
3. Extract to a **SHORT path** (e.g., `C:\Projects\` on Windows)
4. Rename folder to something short (e.g., `diabetes-project`)
5. Open terminal in that folder

---

### 🎯 **Step 2: Run Everything**

**Method 1: Automated Script (Easiest)**
```bash
python run_all.py
```

**Method 2: Step-by-Step (Manual)**
```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate          # Mac/Linux
# OR
.venv\Scripts\activate             # Windows

# 3. Install packages
pip install -r requirements.txt

# 4. Train models
python scripts/run_train.py

# 5. Evaluate models
python scripts/run_eval.py

# 6. Launch dashboard (optional)
streamlit run dashboard.py
```

---

### 📊 **Step 3: View Results**

**In Terminal:**
- Model metrics will be displayed after evaluation

**Interactive Dashboard:**
```bash
streamlit run dashboard.py
# Then open: http://localhost:8501
```

**Jupyter Notebooks:**
```bash
jupyter lab notebooks/03_implementation_details.ipynb
```

---

### ⚠️ **Windows Users - Important**

If you get "Filename too long" error:
- Use **ZIP download** instead of Git clone
- Extract to a **SHORT path** (e.g., `C:\Projects\`)
- See `docs/WINDOWS_PATH_LENGTH_FIX.md` for details

---

### 📖 **Detailed Guides**

- **Complete Setup:** `SETUP_GUIDE.md` - Full step-by-step instructions
- **Quick Start:** `docs/QUICK_START.md` - Quick reference
- **Clone Guide:** `HOW_TO_CLONE_AND_RUN.md` - Simple clone instructions
- **Windows Fixes:** `docs/WINDOWS_PATH_LENGTH_FIX.md` - Windows troubleshooting

---

Artifacts land in `models/` (joblib pipelines + thresholds) and
`data/processed/` (train/test CSVs) so notebooks, evaluation, and Streamlit all
consume the same outputs.

## Pipeline overview

1. **Preprocessing (`src/preprocess.py`)**
   - Strip whitespace, drop empty columns, replace `"?"` with `NaN`.
   - Map age buckets to ordinals and create `readmitted_binary`.
   - Restrict to the curated feature pool + target, stratified 80/20 split.
2. **Modeling (`src/model.py`, `src/train.py`)**
   - ColumnTransformer preprocessing, SelectKBest per model, fit LR + XGB.
3. **Threshold tuning (`src/train.py`)**
   - Enforce recall ≥ target (default 0.65) and pick the best F1 threshold.
4. **Evaluation (`src/evaluate.py`)**
   - Load processed test split and saved thresholds, print a side-by-side
     comparison with clinical interpretation (safe discharge vs. high-risk view),
     and recommend the stronger model.
5. **Dashboard (`dashboard.py`)**
   - Mirrors the CLI outputs: LR tab, XGB tab, ROC/confusion plots, and a
     prediction playground that uses the tuned thresholds.

## Interpretation & recommendation

Typical tuned metrics (vary slightly with the random split):

| Model | Threshold | Recall | Precision | F1 | ROC-AUC |
|-------|-----------|--------|-----------|----|---------|
| Logistic Regression (20 feats) | 0.45 | ≈0.70 | ≈0.15 | ≈0.24 | ≈0.64 |
| XGBoost (25 feats)             | 0.10 | ≈0.71 | ≈0.17 | ≈0.27 | ≈0.68 |

- **Logistic Regression** stays in the project as the interpretable, high-recall
  baseline that clinicians can interrogate.
- **XGBoost** is the recommended deployment model because it delivers the
  strongest F1 while maintaining recall at the target threshold.

## Future enhancements

- Add SHAP/feature-attribution notebooks for both models.
- Calibrate probabilities + quantify intervention cost/benefit.
- Extend the dashboard with fairness monitoring and cohort drill-downs.

## Key Metrics Explained

- **Recall (Sensitivity)**: Proportion of actual readmissions correctly identified
  - **High recall** = Fewer missed high-risk patients (our primary goal)
- **Precision**: Proportion of predicted readmissions that actually occurred
  - Lower precision is acceptable if it means catching more true positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Overall model discrimination ability
- **Accuracy**: Overall correctness (less important for imbalanced data)

## Reproducibility

- Fixed random seed (42) used throughout:
  - Train/test split
  - Model training (Logistic Regression, XGBoost)
  - Feature selection
- All paths and configurations centralized in `src/config.py`
- Processed datasets saved to disk for consistency

## Notes

- **Curated features**: Reduces noise and keeps the narrative simple.
- **Class imbalance**: LR uses class weights, XGBoost benefits from the tuned threshold.
- **Threshold tuning**: Done on the held-out split for clarity. In production, move this to a validation set or nested CV.
- **Interpretability**: Logistic Regression coefficients, plus XGBoost feature importances/SHAP (future enhancement).

## License

This project is for educational purposes (MSBA-265 course project).
