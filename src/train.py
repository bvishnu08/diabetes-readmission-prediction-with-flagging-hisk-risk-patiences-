"""
Training Our Models - Teaching the Computer to Predict Readmissions!

This is like teaching a student to recognize patterns:
1. We show them examples (training data)
2. They learn the patterns (the model trains)
3. We test them on new examples (test data)
4. We adjust how they make decisions (threshold tuning)

This script does the whole training process:
- Loads and cleans the data
- Picks the best features for each model
- Trains both models (Logistic Regression and XGBoost)
- Finds the "sweet spot" for making predictions (threshold tuning)
- Saves everything so we can use it later

Think of it like training a medical assistant - we're teaching them to identify
which patients are at risk of being readmitted, so they can flag them for
extra care before discharge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ===================================================================
# IMPORT SETUP - Making Sure Imports Work
# ===================================================================
# This code makes the script work whether you:
# 1. Import it as a module: `from src.train import train_all_models`
# 2. Run it directly: `python src/train.py`
#
# It's like having a universal adapter - works either way!
try:
    # Try relative imports first (when used as a module)
    from .config import Config
    from .preprocess import train_test_split_clean, generate_processed_datasets
    from .feature_selection import select_top_k, FeatureSelectionConfig
    from .model import build_logreg_pipeline, build_xgb_pipeline, infer_feature_types
except ImportError:
    # If that fails, we're running as a script, so use absolute imports
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import Config
    from src.preprocess import train_test_split_clean, generate_processed_datasets
    from src.feature_selection import select_top_k, FeatureSelectionConfig
    from src.model import build_logreg_pipeline, build_xgb_pipeline, infer_feature_types


def tune_threshold_for_recall_band(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    recall_min: float = 0.55,
    recall_max: float = 0.85,
    target_recall: float = 0.65,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the "sweet spot" for making predictions.
    
    This is probably the trickiest part to understand, so let me explain it
    with an analogy:
    
    Imagine you're a security guard at a hospital. Your job is to decide which
    patients need extra monitoring before discharge. The model gives you a "risk score"
    from 0 to 100 (like a probability). But you need to decide: at what score do
    you say "yes, this patient needs extra care"?
    
    - If you set the bar too low (say, 30%): You'll catch almost everyone, but
      you'll also flag a lot of false alarms (patients who don't actually need it).
      This is high recall, low precision.
    
    - If you set the bar too high (say, 90%): You'll only flag the most obvious
      cases, but you'll miss many patients who actually need help. This is low
      recall, high precision.
    
    - The "sweet spot" is somewhere in between: catch most of the real cases
      (high recall) without too many false alarms (reasonable precision).
    
    This function tries different thresholds (30%, 35%, 40%, ... 90%) and picks
    the one that:
    1. Catches enough real cases (recall between 55% and 85% - realistic for hospitals)
    2. Among those, has the best balance of precision and recall (highest F1-score)
    
    Parameters
    ----------
    y_true : np.ndarray
        The actual answers (1 = readmitted, 0 = not readmitted)
    y_proba : np.ndarray
        The model's predictions as probabilities (0.0 to 1.0)
        Example: [0.3, 0.7, 0.2, 0.9] means:
        - Patient 1: 30% chance of readmission
        - Patient 2: 70% chance of readmission
        - Patient 3: 20% chance of readmission
        - Patient 4: 90% chance of readmission
    recall_min : float, default=0.55
        Minimum recall we'll accept (we want to catch at least 55% of readmissions)
    recall_max : float, default=0.85
        Maximum recall we'll accept (more than 85% might mean too many false alarms)
    target_recall : float, default=0.65
        If no threshold meets our band, aim for 65% recall (a good middle ground)
    
    Returns
    -------
    best_threshold : float
        The "sweet spot" probability (e.g., 0.45 means "flag if probability >= 45%")
    best_metrics : dict
        How good the model is at this threshold (precision, recall, F1, accuracy)
    """
    # Create a list of thresholds to try: [0.05, 0.10, 0.15, ..., 0.90, 0.95]
    # We'll test 19 different thresholds and see which one works best
    thresholds = np.linspace(0.05, 0.95, 19)
    y_true = np.asarray(y_true)  # Make sure it's a numpy array
    
    # Initialize variables to track the best threshold we've found so far
    best_threshold = thresholds[0]  # Start with the first threshold
    best_metrics: Dict[str, float] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0
    }
    best_f1 = -1.0  # Track the best F1-score we've seen
    
    # Store all candidates (in case we need to fall back)
    candidates: list[Tuple[float, Dict[str, float]]] = []
    
    # ===================================================================
    # TRY EACH THRESHOLD AND SEE HOW GOOD IT IS
    # ===================================================================
    for t in thresholds:
        # Convert probabilities to yes/no predictions using this threshold
        # Example: if threshold is 0.5 and probability is 0.7 → predict 1 (yes)
        #          if threshold is 0.5 and probability is 0.3 → predict 0 (no)
        y_pred = (y_proba >= t).astype(int)
        
        # Calculate how good the predictions are at this threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)
        
        # Store the metrics for this threshold
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(acc)
        }
        candidates.append((t, metrics))
        
        # Check if this threshold is in our "sweet spot" range
        # (recall between 55% and 85%) AND has a better F1-score than we've seen
        if recall_min <= recall <= recall_max and f1 > best_f1:
            # This is our new best threshold!
            best_f1 = f1
            best_threshold = float(t)
            best_metrics = metrics
    
    # ===================================================================
    # FALLBACK: IF NO THRESHOLD MEETS OUR REQUIREMENTS
    # ===================================================================
    # Sometimes no threshold falls in our desired range. In that case, we'll
    # just pick the one closest to our target (65% recall).
    
    if best_f1 < 0:  # If we didn't find any threshold in our range
        # Start with the first threshold
        best_threshold = thresholds[0]
        best_metrics = candidates[0][1]
        best_diff = abs(candidates[0][1]["recall"] - target_recall)
        
        # Find the threshold with recall closest to 65%
        for t, metrics in candidates[1:]:
            diff = abs(metrics["recall"] - target_recall)
            if diff < best_diff:
                best_diff = diff
                best_threshold = float(t)
                best_metrics = metrics
    
    return best_threshold, best_metrics


def train_all_models(config: Config | None = None) -> None:
    """
    The main training function - this does everything!
    
    This is like the "master recipe" for training our models. It:
    1. Gets the data ready
    2. Picks the best features for each model
    3. Trains both models
    4. Finds the best thresholds
    5. Saves everything
    
    Think of it like a cooking show where the chef does everything from start
    to finish, and at the end you have a fully trained model ready to use!
    
    Parameters
    ----------
    config : Config, optional
        Our settings. If None, uses default settings.
    
    Side Effects
    ------------
    Creates files:
    - models/logreg_selected.joblib (the trained Logistic Regression model)
    - models/xgb_selected.joblib (the trained XGBoost model)
    - models/thresholds.json (the optimal thresholds and feature lists)
    - data/processed/train_processed.csv (cleaned training data)
    - data/processed/test_processed.csv (cleaned test data)
    """
    # Get our settings
    cfg = config or Config()
    
    # Create folders if they don't exist (so we have somewhere to save files)
    cfg.models_path().mkdir(parents=True, exist_ok=True)
    cfg.reports_path().mkdir(parents=True, exist_ok=True)
    
    # Generate and save the cleaned data (so we don't have to clean it every time)
    generate_processed_datasets(cfg)
    
    # ===================================================================
    # STEP 1: LOAD AND CLEAN THE DATA
    # ===================================================================
    print("[train] Loading and cleaning data...")
    # Get our train/test split (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split_clean(cfg)
    print(
        f"[train] Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,} | "
        f"Candidate features: {X_train.shape[1]}"
    )
    
    # Set up feature selection settings (20 for LR, 25 for XGB)
    fs_config = FeatureSelectionConfig(k_logreg=20, k_xgb=25)
    
    # ===================================================================
    # STEP 2: TRAIN LOGISTIC REGRESSION
    # ===================================================================
    # First, let's train the simpler model (Logistic Regression)
    
    # 2a. Pick the best 20 features for Logistic Regression
    print("\n[train] Selecting top 20 features for Logistic Regression...")
    X_train_lr, lr_features = select_top_k(X_train, y_train, model_name="logreg", config=fs_config)
    X_test_lr = X_test[lr_features]  # Apply same selection to test set
    print(f"[train] Selected features: {lr_features}")
    
    # 2b. Build the Logistic Regression pipeline
    print("\n[train] Training Logistic Regression (top 20 features)...")
    # Figure out which features are numbers vs categories
    lr_num, lr_cat = infer_feature_types(X_train_lr)
    # Build the pipeline (preprocessing + model)
    logreg_pipe = build_logreg_pipeline(lr_num, lr_cat, random_state=cfg.random_state)
    # Train the model (this is where the "learning" happens!)
    logreg_pipe.fit(X_train_lr, y_train)
    
    # 2c. Find the best threshold for Logistic Regression
    # Get predictions on test set (as probabilities, not yes/no)
    logreg_proba = logreg_pipe.predict_proba(X_test_lr)[:, 1]
    # Find the "sweet spot" threshold
    lr_threshold, lr_metrics = tune_threshold_for_recall_band(
        y_test, logreg_proba, recall_min=0.55, recall_max=0.85, target_recall=0.65
    )
    
    # Print how good the model is
    print(f"[train] Logistic Regression threshold & metrics:")
    print(f"  Threshold={lr_threshold:.3f} | Recall={lr_metrics['recall']:.3f} | "
          f"Precision={lr_metrics['precision']:.3f} | F1={lr_metrics['f1']:.3f}")
    
    # ===================================================================
    # STEP 3: TRAIN XGBOOST
    # ===================================================================
    # Now let's train the more powerful model (XGBoost)
    
    # 3a. Pick the best 25 features for XGBoost
    print("\n[train] Selecting top 25 features for XGBoost...")
    X_train_xgb, xgb_features = select_top_k(X_train, y_train, model_name="xgb", config=fs_config)
    X_test_xgb = X_test[xgb_features]  # Apply same selection to test set
    print(f"[train] Selected features: {xgb_features}")
    
    # 3b. Build the XGBoost pipeline
    print("\n[train] Training XGBoost (top 25 features)...")
    # Figure out which features are numbers vs categories
    xgb_num, xgb_cat = infer_feature_types(X_train_xgb)
    # Build the pipeline (preprocessing + model)
    xgb_pipe = build_xgb_pipeline(xgb_num, xgb_cat)
    # Train the model (this takes longer than Logistic Regression!)
    xgb_pipe.fit(X_train_xgb, y_train)
    
    # 3c. Find the best threshold for XGBoost
    # Get predictions on test set (as probabilities)
    xgb_proba = xgb_pipe.predict_proba(X_test_xgb)[:, 1]
    # Find the "sweet spot" threshold
    xgb_threshold, xgb_metrics = tune_threshold_for_recall_band(
        y_test, xgb_proba, recall_min=0.55, recall_max=0.85, target_recall=0.65
    )
    
    # Print how good the model is
    print(f"[train] XGBoost threshold & metrics:")
    print(f"  Threshold={xgb_threshold:.3f} | Recall={xgb_metrics['recall']:.3f} | "
          f"Precision={xgb_metrics['precision']:.3f} | F1={xgb_metrics['f1']:.3f}")
    
    # ===================================================================
    # STEP 4: SAVE EVERYTHING
    # ===================================================================
    # Now that we've trained the models, let's save them so we can use them later
    # (without having to retrain every time, which takes time!)
    
    logreg_path = cfg.model_path_logreg()
    xgb_path = cfg.model_path_xgb()
    
    # Save the trained models (using joblib, which is like pickle but faster)
    joblib.dump(logreg_pipe, logreg_path)
    joblib.dump(xgb_pipe, xgb_path)
    print(f"\n[train] Saved models to {logreg_path} and {xgb_path}")
    
    # ===================================================================
    # STEP 5: SAVE THRESHOLDS AND FEATURE LISTS
    # ===================================================================
    # We need to save which features we used and what thresholds we picked,
    # so that when we evaluate the models later, we use the exact same setup.
    # This ensures consistency!
    
    thresholds: Dict[str, Any] = {
        "logreg": {
            "threshold": float(lr_threshold),      # The "sweet spot" for LR
            "features": lr_features,                # The 20 features we used
        },
        "xgb": {
            "threshold": float(xgb_threshold),      # The "sweet spot" for XGBoost
            "features": xgb_features,                # The 25 features we used
        },
    }
    
    # Save to a JSON file (human-readable format)
    with cfg.thresholds_path().open("w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[train] Saved thresholds & selected features to {cfg.thresholds_path()}")
    
    print("\n[train] Training complete!")


if __name__ == "__main__":
    # If someone runs this file directly (like `python src/train.py`),
    # automatically run the training
    train_all_models()
