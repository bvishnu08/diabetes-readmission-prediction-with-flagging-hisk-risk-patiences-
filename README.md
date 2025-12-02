# Diabetes 30-Day Readmission Prediction

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
│   ├── COMPLETE_PROJECT_CODE.md
│   ├── CODE_EXPLANATION.md
│   └── PIPELINE_DIFFERENCES.md
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

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train Logistic Regression + XGBoost and store thresholds
python scripts/run_train.py

# Evaluate on the held-out test split
python scripts/run_eval.py

# Optional dashboard (Streamlit)
streamlit run dashboard.py
```

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
# Diabetes 30-Day Readmission Prediction

Predict readmission risk for diabetic patients at discharge. The project keeps
the codebase classroom-ready (MSBA-265 P2/P3) while prioritizing **recall for
the positive class** so we catch as many high-risk patients as possible.

## Project Goal

**Clinical + business framing**
- Dataset: UCI *Diabetes 130-US Hospitals* (`data/raw/diabetic_data.csv`)
- Target: `readmitted_binary = 1` if the patient returns within 30 days
- Primary metric: Recall for class 1 (don’t miss true readmissions)
- Secondary metrics: F1 (class 1) and ROC-AUC
- Use case: discharge screening → false positives trigger extra outreach, but
  false negatives become penalties + readmissions.

## Feature pool & selection

We use `Config.candidate_features`, a curated pool of 41 features covering:

- **Demographics**: `age` (ordinal bucket), `gender`, `race`
- **Hospital visit context**: `admission_type_id`, `admission_source_id`, `discharge_disposition_id`
- **Utilization / severity**: `time_in_hospital`, `num_lab_procedures`,
  `num_procedures`, `num_medications`, `number_outpatient`,
  `number_emergency`, `number_inpatient`, `number_diagnoses`
- **Diabetes control & therapy**: `max_glu_serum`, `A1Cresult`, `change`, `insulin`,
  `diabetesMed`, and 23 diabetes medication indicators

During training, `SelectKBest` selects the most informative features from this pool:
- Logistic Regression uses the top 20 features (better interpretability)
- XGBoost uses the top 25 features (higher predictive power)

This feature selection strategy ensures both models focus on the most relevant signals
while maintaining interpretability for the logistic regression baseline.

## Models

| Model | Positioning | Details |
|-------|-------------|---------|
| Logistic Regression | **Baseline / interpretable** | ColumnTransformer preprocessing, balanced class weights, fast to explain to clinicians |
| XGBoost | **Advanced / higher recall** | 300 estimators, depth 4, learning-rate 0.05, subsample 0.8 |

Both pipelines use the same preprocessing (median imputation + scaling for
numerics, most-frequent + one-hot for categoricals).

### Threshold tuning strategy

1. Generate probabilities on the held-out test split.
2. Sweep thresholds from 0.05 → 0.95.
3. Keep only thresholds where recall (class 1) ≥ **0.65**.
4. Among those, select the threshold with the highest **F1-score** for class 1.
5. Persist the chosen threshold + feature list to `models/thresholds.json`.

That gives us a clinically meaningful operating point and keeps evaluation,
plots, and the dashboard aligned.

## Pipeline Overview

1. **Preprocessing (`src/preprocess.py`)**
   - Strip whitespace, drop empty columns, replace `"?"` with `NaN`.
   - Convert `readmitted` → `readmitted_binary` and map age buckets to integers.
   - Restrict columns to `candidate_features + target`.
   - Stratified 80/20 split saved to `data/processed/train_processed.csv`
     and `test_processed.csv`.

2. **Modeling (`src/model.py`, `src/train.py`)**
   - Shared preprocessing via ColumnTransformer.
   - Build + fit Logistic Regression and XGBoost pipelines.

3. **Threshold tuning (`src/train.py`)**
   - Apply the recall ≥ 0.65 rule, pick highest F1.
   - Persist thresholds + features to `models/thresholds.json`.

4. **Evaluation (`src/evaluate.py`)**
   - Load processed test split + saved models/thresholds.
   - Print a side-by-side comparison and recommend the stronger model.

5. **Dashboard (`dashboard.py`)**
   - Consumes the same artifacts to display metrics + interactive scoring.

## How to run

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Train both models + save artifacts
python scripts/run_train.py

# 3. Evaluate on held-out test split
python scripts/run_eval.py
```

Artifacts:
- `models/logreg_selected.joblib`
- `models/xgb_selected.joblib`
- `models/thresholds.json` (thresholds + feature list)
- `data/processed/train_processed.csv`, `test_processed.csv`

Optional dashboard:
```bash
streamlit run dashboard.py
# or python scripts/run_dashboard.py
```

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── dashboard.py                 # Streamlit dashboard
│
├── docs/                        # Documentation
│   ├── RUN_BOOK.md             # Step-by-step execution guide
│   ├── COMPLETE_PROJECT_CODE.md # All project code in one file
│   ├── CODE_EXPLANATION.md     # Deep dive into codebase
│   └── PIPELINE_DIFFERENCES.md # Notebook vs. production notes
│
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv   # Raw UCI dataset
│   │   └── IDS_mapping.csv     # Feature ID mappings
│   └── processed/
│       ├── train_processed.csv # Processed training data
│       └── test_processed.csv # Processed test data
│
├── models/
│   ├── logreg_selected.joblib  # Logistic Regression model
│   ├── xgb_selected.joblib     # XGBoost model
│   └── thresholds.json         # Stored thresholds + feature list
│
├── notebooks/                  # EDA and experiments
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_implementation_details.ipynb
│
├── reports/                    # Project reports and submissions
│   └── P2 Final_submission report.pdf
│
├── scripts/                    # CLI entrypoints
│   ├── run_train.py           # Train both models
│   ├── run_eval.py            # Evaluate both models
│   └── run_dashboard.py       # Launch dashboard
│
└── src/                        # Library code
    ├── __init__.py
    ├── config.py              # Paths, candidate features, thresholds
    ├── preprocess.py          # One source of truth for cleaning/splits
    ├── feature_selection.py   # SelectKBest feature selection
    ├── model.py               # Build LR + XGBoost pipelines
    ├── train.py               # Train + threshold tuning
    ├── evaluate.py            # Side-by-side comparison + clinical view
    └── clinical_utils.py      # Clinical interpretation helpers
```

## Key Metrics Explained

- **Recall (Sensitivity)**: Proportion of actual readmissions correctly identified
  - **High recall** = Fewer missed high-risk patients (our primary goal)
- **Precision**: Proportion of predicted readmissions that actually occurred
  - Lower precision is acceptable if it means catching more true positives
- **F1-Score**: Harmonic mean of precision and recall
- **F2-Score**: Emphasizes recall over precision (beta=2)
  - Used for threshold tuning to prioritize catching readmissions
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
- **Class imbalance**: LR uses class weights, XGBoost uses scale_pos_weight=1
  but benefits from the tuned threshold.
- **Threshold tuning**: Done on the held-out split for clarity. In production,
  move this to a validation set or nested CV.
- **Interpretability**: Logistic Regression coefficients, plus XGBoost feature
  importances/SHAP (future enhancement).

## Future Enhancements

- SHAP/feature importance plots for both models.
- Calibrated probabilities + intervention cost modeling.
- Real-time scoring API + monitoring.
- Fairness analysis across demographics.

## License

This project is for educational purposes (MSBA-265 course project).
