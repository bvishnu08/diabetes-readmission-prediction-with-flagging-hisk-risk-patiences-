# Pipeline Differences: Production vs. Notebook Experiments

## TL;DR

- The **production pipeline** (`src/`) now contains *two* models (LogReg + XGBoost) built on a curated pool of 41 candidate features. `SelectKBest` selects the top 20 features for Logistic Regression and top 25 for XGBoost. The corresponding notebook cells import the exact same preprocessing/model-building functions, so results match aside from random noise.
- The **optional notebook experiments** are free to add extra features, SMOTE, different hyperparameters, etc. Anything not in `src/` should be considered exploratory until promoted here.

## Current Scores (rerun as needed)

- Logistic Regression (production, 20 features) – ROC-AUC ≈ 0.63–0.64, Recall ≈ 0.70.
- XGBoost (production, 25 features) – typically ROC-AUC ≈ 0.67–0.68, Recall ≈ 0.71.
- Notebook experiments may climb higher if you open up feature count or tuning.

Always rerun `python scripts/run_train.py && python scripts/run_eval.py` (or the
notebook cells) to refresh numbers before reporting them.

## Key Differences

| Aspect | Production (scripts / src) | Notebook optional sections |
|--------|---------------------------|-----------------------------|
| **Features** | SelectKBest selects top 20 (LR) or top 25 (XGB) from `Config.candidate_features` (41-feature pool) | Free to add/remove columns, try PCA, etc. |
| **Missing values** | Median (numeric) / most-frequent (categorical) inside ColumnTransformer | Same or heavier (e.g., MICE, domain-specific fills) |
| **Age handling** | Bucket → ordinal (0–9) | Experiments may keep as string or try one-hot |
| **Models** | Logistic Regression + XGBoost | May include RF, LightGBM, extra XGBoost variations |
| **Class imbalance** | Managed via class weights + tuned thresholds | Can add SMOTE, different `scale_pos_weight`, cost-sensitive losses |
| **Thresholds** | Recall-first tuning (≥0.65) stored in `thresholds.json` | Often uses default 0.5 or custom heuristics |

## What Changed

- Production now bundles both models (LogReg + XGBoost) and saves their tuned
  thresholds, so the dashboard and evaluation use identical numbers.
- The notebooks import the same preprocessing/model builders for the baseline
  cells; optional cells remain a playground for new features or techniques.
- Any experiment promoted into `src/` should also be documented here to capture
  the rationale.

## When to Use What

- **Need P2/P3-ready numbers?** Run `scripts/run_train.py` + `scripts/run_eval.py`.
- **Need to explore new ideas?** Use the notebook optional sections, but document
  any divergences here before promoting them.

## Next Steps

- Promote any promising experimental settings into the production pipeline once you’re ready.
- Keep documenting divergences here so it’s always clear why scores differ.
