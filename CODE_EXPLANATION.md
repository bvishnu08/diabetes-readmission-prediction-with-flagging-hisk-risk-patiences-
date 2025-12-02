# Code Explanation - Diabetes Readmission ML Pipeline

This document provides a comprehensive explanation of all code files in this project, their purpose, and how they work together.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Core Modules (`src/`)](#core-modules-src)
4. [Scripts (`scripts/`)](#scripts-scripts)
5. [Data Flow](#data-flow)
6. [How to Use](#how-to-use)

---

## Project Overview

This project implements a ML pipeline to predict 30-day readmission risk for
diabetic patients. The upgraded architecture now follows this flow:

- **Preprocessing** (`src/preprocess.py`): Clean the raw CSV, map age buckets to
  ordinals, restrict to a curated feature set, and persist the train/test split.
- **Modeling** (`src/model.py`, `src/train.py`): Build two pipelines—
  Logistic Regression (baseline) and XGBoost (advanced)—both sharing the same
  ColumnTransformer preprocessing.
- **Threshold tuning** (`src/train.py`): Sweep probability thresholds, enforce
  recall ≥ target (default 0.65), and pick the one with the best F1-score.
- **Evaluation** (`src/evaluate.py`): Load saved models + thresholds and print a
  side-by-side comparison suitable for P2/P3 write-ups.

---

## Directory Structure

```
.
├── data/
│   ├── raw/              # Original CSV files (diabetic_data.csv, IDS_mapping.csv)
│   └── processed/        # Cleaned/processed data (auto-generated, gitignored)
├── models/               # Saved trained models (.joblib files)
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── reports/              # Generated reports and visualizations
├── scripts/              # CLI entry points for training and evaluation
├── src/                  # Core library code (importable package)
└── tests/                # Unit tests
```

---

## Core Modules (`src/`)

### 1. `src/config.py` – Configuration + feature pool

**Purpose**: Single source of truth for paths, candidate feature lists, and all
training/tuning hyperparameters.

**Highlights**
- `candidate_features`: curated set of 41 features covering demographics, admission context,
  utilization, and diabetes therapy signals. SelectKBest runs on this pool.
- `lr_top_k` / `xgb_top_k`: how many features to keep for each model (20 / 25).
- Paths: raw data, processed splits, model directory, reports.
- Model artifacts: `logreg_selected.joblib`, `xgb_selected.joblib`,
  `thresholds.json`.
- Threshold knobs: `target_recall` (default 0.65) and `threshold_grid`
  (0.05 → 0.95 in 19 steps).

Change settings here and the rest of the pipeline follows suit.

---

### 2. `src/preprocess.py` - Data Preprocessing

**Purpose**: Clean the raw CSV once, enforce the candidate feature pool, and
write the exact train/test split consumed everywhere else.

**Highlights**
1. Strip whitespace, drop empty columns, replace `"?"` with `NaN`.
2. Map `readmitted` → `readmitted_binary` (`"<30"` → 1, otherwise 0).
3. Map age buckets to ordinal integers (`"[60-70)"` → 6).
4. Keep only `Config.candidate_features + target`.
5. Stratified 80/20 split (random_state=42) saved to
   `data/processed/train_processed.csv` and `test_processed.csv`.

Because the processed CSVs are persisted, training, evaluation, notebooks, and
Streamlit all see the exact same data—a key requirement for P2/P3 storytelling.

---

### 3. `src/model.py` - Pipelines

**Purpose**: Build reusable scikit-learn pipelines for both models.

**Key functions**
- `infer_feature_types(df)`: Splits a given feature subset into numeric vs
  categorical columns based on dtypes.
- `build_logreg_pipeline(numeric_cols, categorical_cols, random_state)`: Creates
  a ColumnTransformer (median imputer + scaler for numeric, most-frequent
  imputer + one-hot for categorical) followed by Logistic Regression with
  `class_weight="balanced"` and `max_iter=1000`.
- `build_xgb_pipeline(numeric_cols, categorical_cols)`: Same preprocessing, but
  the classifier is `xgboost.XGBClassifier` (300 trees, depth 4, lr 0.05).
- `build_pipeline(X_train, random_state)`: backward-compatible helper for the
  notebooks; internally delegates to `build_logreg_pipeline`.

Having both pipelines share the **exact** preprocessing logic prevents data
leakage and keeps the LR vs XGB comparison fair.

---

### 4. `src/train.py` - Model Training + Threshold Tuning

**Purpose**: Train both pipelines, run SelectKBest per model, tune thresholds,
and persist everything needed for evaluation and the dashboard.

**Workflow**
1. Load the cleaned split from `preprocess.train_test_split_clean`.
2. Run `select_top_k` twice:
   ```python
   X_train_lr, X_test_lr, lr_features = select_top_k(..., k=Config.lr_top_k)
   X_train_xgb, X_test_xgb, xgb_features = select_top_k(..., k=Config.xgb_top_k)
   ```
3. Fit pipelines on the reduced feature sets (LogReg + XGBoost).
4. Threshold sweep (`np.linspace(0.05, 0.95, 19)`):
   - Keep thresholds where recall ≥ `Config.target_recall` (default 0.65).
   - Among those, choose the one with the highest F1-score.
5. Save artifacts:
   - `models/logreg_selected.joblib`
   - `models/xgb_selected.joblib`
   - `models/thresholds.json` containing `{model: {threshold, recall, precision,
     f1, features}}` plus the recall target used.

This keeps the P2/P3 story reproducible: anyone who runs `scripts/run_train.py`
will get the exact same thresholds and models described in the report.

---

### 5. `src/evaluate.py` - Model Evaluation

**Purpose**: Load the persisted test split + trained models/thresholds and print
a P2-ready comparison.

**Core steps**
1. Read `data/processed/test_processed.csv` and `models/thresholds.json`.
2. Ensure the test frame contains the stored feature list; subset accordingly.
3. Load `logreg_selected.joblib` and `xgb_selected.joblib`.
4. For each model:
   ```python
   proba = model.predict_proba(X_test)[:, 1]
   preds = (proba >= saved_threshold).astype(int)
   ```
   Compute ROC-AUC, accuracy, precision/recall/F1 for class 1, confusion matrix,
   and a classification report.
5. Generate clinical interpretation using `clinical_utils.summarize_risk_view()`:
   - Observed readmission rate among LOW-RISK patients (safe discharge view)
   - Average predicted readmission risk across all patients
   - Percentage of patients flagged as high-risk vs. low-risk
6. Print two blocks with standard metrics + clinical interpretation, plus a recommendation (pick the higher F1 for class 1).

Because it reuses the saved thresholds and feature list, the metrics here match
the dashboard + report exactly—no silent re-splitting or re-selection happens.

---

## Scripts (`scripts/`)

### 1. `scripts/run_train.py`

Adds the project root to `sys.path`, imports `train_all_models`, and runs it.
No CLI flags—just:

```bash
python scripts/run_train.py
```

### 2. `scripts/run_eval.py`

Same minimal wrapper for `evaluate_all`:

```bash
python scripts/run_eval.py
```

### 3. `scripts/run_dashboard.py`

Convenience helper to launch Streamlit without typing the full command:

```bash
python scripts/run_dashboard.py   # Internally calls `streamlit run dashboard.py`
```

### 6. `src/feature_selection.py` - Feature Selection

**Purpose**: Select the top-K most informative features for each model using mutual information.

**Key functions**
- `select_top_k(X, y, model_name, config)`: Uses `SelectKBest` with `mutual_info_classif` to score and select the top-K features for a given model type (logreg or xgb).
- Handles mixed data types by temporarily label-encoding categorical features before scoring.
- Returns the reduced feature matrix and a list of selected feature names.

**Why it's important**: Feature selection reduces overfitting, improves interpretability (especially for Logistic Regression), and ensures each model focuses on the most relevant signals from our 41-feature candidate pool.

---

### 7. `src/clinical_utils.py` - Clinical Interpretation

**Purpose**: Translate technical ML metrics into clinically meaningful insights for healthcare professionals.

**Key functions**
- `summarize_risk_view(y_true, y_pred, p_readmit)`: Generates a clinical-style summary that translates binary predictions into "safe to discharge" vs. "high risk" language.

**What it calculates**:
- Number and percentage of patients flagged as HIGH RISK vs. LOW RISK
- Observed readmission rate among LOW-RISK patients (most important for discharge safety)
- Average predicted readmission risk across all patients
- Average predicted "safe discharge" probability

**Why it's important**: Makes the model's output actionable for clinicians by focusing on discharge safety rather than abstract metrics like "precision" or "F1-score".

---

## Data Flow

Here's how data flows through the pipeline:

```
1. Raw Data (CSV: data/raw/diabetic_data.csv)
   ↓
2. preprocess.load_raw() + basic_clean()
   → Cleaned pandas DataFrame (candidate_features + target)
   ↓
3. preprocess.train_test_split_clean()
   → X_train, X_test, y_train, y_test (saved to data/processed/)
   ↓
4. feature_selection.select_top_k()
   → X_train_lr (20 features), X_train_xgb (25 features)
   ↓
5. model.build_logreg_pipeline() / build_xgb_pipeline()
   → sklearn Pipeline (preprocessing + classifier)
   ↓
6. train.train_all_models()
   → Trained models saved to models/ + thresholds.json
   ↓
7. evaluate.evaluate_all()
   → Metrics (ROC-AUC, precision, recall, F1) + clinical interpretation
```

**Key Points**:
- Data is loaded once, then split into train/test
- Same preprocessing is applied to train and test
- Model is saved after training for reuse
- Evaluation uses the same preprocessing pipeline

---

## How to Use

### Option 1: Using Scripts (Recommended)

```bash
# Train both models and save artifacts
python scripts/run_train.py

# Evaluate on held-out test set
python scripts/run_eval.py

# Launch Streamlit dashboard
python scripts/run_dashboard.py
```

### Option 2: Running Modules Directly

```bash
# Train
python src/train.py

# Evaluate
python src/evaluate.py
```

### Option 3: Using as Python Modules

```bash
# Train
python -m src.train

# Evaluate
python -m src.evaluate
```

### Option 4: Importing in Code

```python
from src.train import train_all_models
from src.evaluate import evaluate_all
from src.config import Config

# Train both models
train_all_models()

# Evaluate both models
evaluate_all()
```

---

## Notebooks

- `01_eda.ipynb`: Quick exploratory data analysis of the raw dataset.
- `02_modeling.ipynb`: Sandbox for prototyping ideas before promoting them into `src/`.
- `03_implementation_details.ipynb`: Mirrors the production pipeline by importing functions from `src/`. Contains experiments with Logistic Regression and XGBoost, showcasing the production pipeline's components.

---

## Key Design Decisions

1. **Modular Architecture**: Each module has a single responsibility
2. **Configuration Object**: Centralized config makes it easy to change defaults
3. **Error Handling**: Graceful handling of missing files, incompatible models
4. **Flexible Execution**: Can be used as scripts, modules, or imported functions
5. **Reproducibility**: Fixed random seeds ensure consistent results
6. **Type Hints**: Help catch errors and improve IDE support

---

## Future Improvements

Potential enhancements you could add:

1. **More Models**: Add Random Forest, XGBoost, etc.
2. **Feature Engineering**: More sophisticated preprocessing
3. **Hyperparameter Tuning**: Grid search or random search
4. **Cross-Validation**: More robust evaluation
5. **Logging**: Better tracking of experiments
6. **Visualizations**: Plot ROC curves, feature importance
7. **Model Registry**: Track multiple model versions

---

## Dashboard (`dashboard.py`)

### Purpose
Interactive web dashboard built with Streamlit for visualizing model performance, data analysis, and exploration.

### Features

1. **Model Performance Visualization**:
   - ROC curves with AUC scores
   - Precision-Recall curves
   - Confusion matrices
   - Detailed classification metrics

2. **Data Analysis**:
   - Target variable distribution
   - Numeric feature distributions by readmission status
   - Categorical feature analysis
   - Correlation heatmaps

3. **Interactive Explorer**:
   - Raw data preview
   - Data statistics
   - Missing value analysis
   - Column information

4. **Integration**:
   - Loads processed data from `data/processed/train_processed.csv` and `test_processed.csv`
   - Loads trained models from `models/logreg_selected.joblib` and `models/xgb_selected.joblib`
   - Loads thresholds and feature lists from `models/thresholds.json`
   - Uses the same preprocessing pipeline as training/evaluation for consistency

### Usage

```bash
# Run directly
streamlit run dashboard.py

# Or use the script
python scripts/run_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Dashboard Tabs

1. **Overview**: Key metrics, classification report, dataset info
2. **Model Performance**: ROC/PR curves, confusion matrix, feature importance
3. **Data Analysis**: Target distribution, feature distributions
4. **Visualizations**: Correlation heatmaps, interactive feature plots
5. **Data Explorer**: Raw data, statistics, missing values, column info

---

## Summary

This codebase implements a clean, maintainable ML pipeline with:
- **Clear separation of concerns**: Each file has one job
- **Reusable components**: Functions can be imported or run directly
- **Error handling**: Graceful failures with helpful messages
- **Flexibility**: Multiple ways to run the code
- **Testability**: Unit tests ensure correctness
- **Visualization**: Interactive dashboard for model and data exploration

The pipeline follows best practices for ML projects and is easy to extend with new features or models.

