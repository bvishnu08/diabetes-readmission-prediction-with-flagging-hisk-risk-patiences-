"""
Evaluating Our Models - How Good Are They Really?

This is like giving our trained models a final exam. We've trained them, now
we need to see how well they actually perform on new data they've never seen.

This script:
1. Loads the trained models (the "students" we trained)
2. Tests them on the test set (the "final exam")
3. Compares them side-by-side (which one is better?)
4. Gives a recommendation (which one should we actually use?)
5. Shows clinical interpretation (safe to discharge vs high risk)

Think of it like a science fair judging:
- We test both models on the same problems
- We measure how well they do (accuracy, precision, recall, etc.)
- We compare them and pick a winner
- We explain why we picked that one
- We translate the results into clinical language (safe vs high risk)

This helps us decide: should we use the simpler Logistic Regression model
(which is easier to explain) or the more complex XGBoost model (which might
be more accurate)? And it helps clinicians understand what the model means
for their patients!
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ===================================================================
# IMPORT SETUP - Making Sure Imports Work
# ===================================================================
# Same as train.py - makes the script work whether imported or run directly
try:
    from .config import Config
    from .preprocess import train_test_split_clean
    from .clinical_utils import summarize_risk_view  # NEW: Clinical interpretation!
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import Config
    from src.preprocess import train_test_split_clean
    from src.clinical_utils import summarize_risk_view  # NEW: Clinical interpretation!


def _load_thresholds(cfg: Config) -> Dict[str, Any]:
    """
    Load the thresholds and feature lists we saved during training.
    
    This is like opening a notebook where we wrote down:
    - "For Logistic Regression, use these 20 features and threshold 0.45"
    - "For XGBoost, use these 25 features and threshold 0.10"
    
    We need this because we want to evaluate the models using the EXACT same
    setup we used during training. If we use different features or thresholds,
    the results won't be fair!
    
    Parameters
    ----------
    cfg : Config
        Our settings object (knows where the file is)
    
    Returns
    -------
    Dict[str, Any]
        A dictionary with the thresholds and features for each model
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist (you need to run training first!)
    """
    thresholds_path = cfg.thresholds_path()
    
    # Check if the file exists
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"thresholds.json not found at {thresholds_path}. "
            "Run scripts/run_train.py first to train and save thresholds."
        )
    
    # Read the JSON file (it's human-readable, you can open it in a text editor!)
    with thresholds_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _evaluate_one_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    threshold: float,
    target_col: str = "readmitted_binary",
) -> Dict[str, Any]:
    """
    Test one model and see how good it is - with clinical interpretation!
    
    This is like giving one student a test and grading it, then explaining
    the results to a doctor. We:
    1. Give them the test questions (the features)
    2. Get their answers (the model's predictions)
    3. Compare to the answer key (the true labels)
    4. Calculate their grade (the metrics)
    5. Translate to clinical language (safe vs high risk)
    
    Parameters
    ----------
    name : str
        The model's name (for display purposes, like "Logistic Regression")
    model
        The trained model (can make predictions)
    X_test : pd.DataFrame
        Test features (the "questions" - patient characteristics)
    y_test : pd.Series
        True labels (the "answer key" - did they actually get readmitted?)
    features : list[str]
        Which features to use (the ones we selected during training)
    threshold : float
        The probability threshold (when do we say "yes, readmission"?)
    target_col : str, default="readmitted_binary"
        The name of the target column (for consistency)
    
    Returns
    -------
    Dict[str, Any]
        A report card with all the metrics (precision, recall, F1, etc.)
        plus clinical interpretation (safe discharge vs high risk)
    """
    # ===================================================================
    # STEP 1: USE ONLY THE FEATURES WE TRAINED WITH
    # ===================================================================
    # This is important! We must use the EXACT same features we used during
    # training. If we use different features, it's like giving the student
    # different questions than they studied for - not fair!
    
    X_sel = X_test[features]
    
    # ===================================================================
    # STEP 2: GET THE MODEL'S PREDICTIONS
    # ===================================================================
    # The model gives us probabilities (like "70% chance of readmission")
    # We need to convert those to yes/no predictions using our threshold
    
    if hasattr(model, "predict_proba"):
        # Get probabilities for class 1 (readmission)
        # Example output: [0.3, 0.7, 0.2, 0.9] means:
        # - Patient 1: 30% chance
        # - Patient 2: 70% chance
        # - Patient 3: 20% chance
        # - Patient 4: 90% chance
        p_readmit = model.predict_proba(X_sel)[:, 1]
    else:
        # This shouldn't happen, but just in case...
        raise ValueError(f"Model {name} does not support predict_proba, adjust evaluation logic.")
    
    # ===================================================================
    # STEP 3: CONVERT PROBABILITIES TO YES/NO PREDICTIONS
    # ===================================================================
    # Using our threshold: if probability >= threshold, predict 1 (yes), else 0 (no)
    # Example: if threshold is 0.5 and probability is 0.7 → predict 1
    #          if threshold is 0.5 and probability is 0.3 → predict 0
    y_pred = (p_readmit >= threshold).astype(int)
    
    # ===================================================================
    # STEP 4: CALCULATE STANDARD METRICS
    # ===================================================================
    # Now we compare predictions to reality and calculate metrics
    
    # Precision, Recall, F1-score for the "readmission" class
    # - Precision: Of all the patients we flagged, how many actually got readmitted?
    #   (High precision = few false alarms)
    # - Recall: Of all the patients who actually got readmitted, how many did we catch?
    #   (High recall = we don't miss many real cases)
    # - F1-score: A balance between precision and recall
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    
    # Overall accuracy: What percentage did we get right?
    acc = accuracy_score(y_test, y_pred)
    
    # ROC-AUC: How well can the model distinguish between readmitted and not?
    # (Higher is better, ranges from 0.5 = random guessing to 1.0 = perfect)
    try:
        auc = roc_auc_score(y_test, p_readmit)
    except Exception:
        # Shouldn't happen, but just in case...
        auc = float("nan")
    
    # Confusion matrix: shows how many correct/incorrect predictions
    cm = confusion_matrix(y_test, y_pred)
    
    # Detailed classification report (per-class breakdown)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, digits=3, zero_division=0)
    
    # ===================================================================
    # STEP 5: CREATE CLINICAL INTERPRETATION
    # ===================================================================
    # Translate model predictions into clinical language
    # This helps doctors and nurses understand what the model means
    clinical_summary = summarize_risk_view(y_test, y_pred, p_readmit)
    
    # ===================================================================
    # STEP 6: PRINT RESULTS
    # ===================================================================
    # Print all the metrics in a nice, easy-to-read format
    
    print("=" * 70)
    print(f"{name.upper()} (features: {len(features)})")
    print("=" * 70)
    print(f"Threshold      : {threshold:.3f}")
    print(f"ROC-AUC        : {auc:.3f}")
    print(f"Accuracy       : {acc:.3f}")
    print(f"Recall (class1): {recall:.3f}")      # How many readmissions we caught
    print(f"Precision      : {precision:.3f}")   # How many flags were correct
    print(f"F1-score       : {f1:.3f}")          # Balance of precision and recall
    
    # Confusion matrix: [ [TN FP] ; [FN TP] ]
    # TN = True Negative (predicted no readmission, actually no readmission) ✓
    # FP = False Positive (predicted readmission, actually no readmission) ✗
    # FN = False Negative (predicted no readmission, actually readmission) ✗
    # TP = True Positive (predicted readmission, actually readmission) ✓
    print("\nConfusion matrix [ [TN FP] ; [FN TP] ]:")
    print(cm)
    
    print("\nClassification report:")
    print(report_str)
    
    # ===================================================================
    # STEP 7: PRINT CLINICAL INTERPRETATION
    # ===================================================================
    # This is the new part! We translate the results into clinical language
    # so doctors and nurses can understand what the model means for their patients
    
    print("-" * 70)
    print("CLINICAL INTERPRETATION – SAFE DISCHARGE VIEW")
    print("-" * 70)
    
    # How many patients are flagged as HIGH RISK vs LOW RISK?
    print(
        f"Patients flagged HIGH RISK : {clinical_summary['n_high']} "
        f"({clinical_summary['high_risk_percent']*100:.1f}% of test set)"
    )
    print(
        f"Patients flagged LOW RISK  : {clinical_summary['n_low']} "
        f"({clinical_summary['low_risk_percent']*100:.1f}% of test set)\n"
    )
    
    # This is the most important part for clinicians:
    # "If the model says 'safe to discharge', can I trust it?"
    print("Among LOW-RISK patients (model says 'safe to discharge'):")
    print(
        f"  Observed readmission rate    : "
        f"{clinical_summary['observed_readmit_rate_low']*100:.1f}%"
    )
    print(
        f"  Observed SAFE discharge rate : "
        f"{clinical_summary['observed_safe_rate_low']*100:.1f}%"
    )
    print(
        f"  Avg predicted SAFE chance    : "
        f"{clinical_summary['avg_p_safe_low']*100:.1f}%\n"
    )
    
    # Overall statistics across all patients
    print("Across ALL test patients:")
    print(
        f"  Avg predicted readmission risk      : "
        f"{clinical_summary['avg_p_readmit']*100:.1f}%"
    )
    print(
        f"  Avg predicted SAFE discharge chance : "
        f"{clinical_summary['avg_p_safe']*100:.1f}%"
    )
    print("\n")
    
    # ===================================================================
    # STEP 8: RETURN THE REPORT CARD
    # ===================================================================
    # Return everything so the caller can use it for comparison or reports
    return {
        "name": name,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "auc": float(auc),
        "report": report,
        "report_str": report_str,
        "cm": cm,
        "proba": p_readmit,      # Keep probabilities for plotting ROC curves
        "y_pred": y_pred,        # Keep predictions for detailed reports
        "clinical_summary": clinical_summary,  # NEW: Clinical interpretation!
    }


def evaluate_all(config: Config | None = None) -> None:
    """
    The main evaluation function - tests both models and compares them!
    
    This is like a science fair where we test both projects and see which one
    is better. We:
    1. Load the test data (the "final exam")
    2. Load both trained models (our "students")
    3. Test them both on the same problems
    4. Compare their scores
    5. Pick a winner and explain why
    6. Show clinical interpretation for both
    
    Parameters
    ----------
    config : Config, optional
        Our settings. If None, uses default settings.
    
    Side Effects
    ------------
    Prints a detailed comparison report to the console, including clinical
    interpretation for both models.
    """
    cfg = config or Config()
    
    # ===================================================================
    # STEP 1: LOAD THE TEST DATA
    # ===================================================================
    # Get the same test set we used during training
    # (We use the same split function with the same random seed, so we get
    #  the exact same test set - this ensures fair comparison!)
    X_train, X_test, y_train, y_test = train_test_split_clean(cfg)
    
    # We need the full DataFrame (with target column) for evaluation
    # Let's combine X_test and y_test back into a DataFrame
    df_test = X_test.copy()
    df_test[cfg.target_col] = y_test
    
    # Print a nice header
    print("\n" + "=" * 70)
    print("MODEL EVALUATION – 30-Day Readmission Prediction")
    print("=" * 70)
    print(f"[eval] Test size: {len(X_test):,} samples\n")
    
    # ===================================================================
    # STEP 2: LOAD THE THRESHOLDS AND FEATURES
    # ===================================================================
    # Remember, we saved which features and thresholds to use during training
    # We need to load those now so we use the exact same setup
    thresholds = _load_thresholds(cfg)
    
    # ===================================================================
    # STEP 3: LOAD THE TRAINED MODELS
    # ===================================================================
    # Load the models we saved during training
    # These are like loading saved game files - we're picking up where we left off!
    logreg_model = joblib.load(cfg.model_path_logreg())
    xgb_model = joblib.load(cfg.model_path_xgb())
    
    # ===================================================================
    # STEP 4: EXTRACT THE SETTINGS FOR EACH MODEL
    # ===================================================================
    # Get the thresholds and feature lists from the JSON file
    
    # Make sure both models are in the file
    if "logreg" not in thresholds or "xgb" not in thresholds:
        raise ValueError(
            "thresholds.json is missing 'logreg' or 'xgb' keys. "
            "Retrain with the updated training script."
        )
    
    # Get the settings for each model
    lr_cfg = thresholds["logreg"]
    xgb_cfg = thresholds["xgb"]
    
    # Extract the feature lists and thresholds
    lr_features = lr_cfg.get("features", [])      # The 20 features for LR
    xgb_features = xgb_cfg.get("features", [])    # The 25 features for XGBoost
    lr_threshold = float(lr_cfg.get("threshold", 0.5))    # The threshold for LR
    xgb_threshold = float(xgb_cfg.get("threshold", 0.5))   # The threshold for XGBoost
    
    # ===================================================================
    # STEP 5: TEST BOTH MODELS
    # ===================================================================
    # Evaluate Logistic Regression (with clinical interpretation!)
    lr_results = _evaluate_one_model(
        name="Logistic Regression (top 20 features)",
        model=logreg_model,
        X_test=X_test,
        y_test=y_test,
        features=lr_features,
        threshold=lr_threshold,
        target_col=cfg.target_col,
    )
    
    # Evaluate XGBoost (with clinical interpretation!)
    xgb_results = _evaluate_one_model(
        name="XGBoost (top 25 features)",
        model=xgb_model,
        X_test=X_test,
        y_test=y_test,
        features=xgb_features,
        threshold=xgb_threshold,
        target_col=cfg.target_col,
    )
    
    # ===================================================================
    # STEP 6: GIVE A RECOMMENDATION
    # ===================================================================
    # Based on the results, which model should we actually use?
    
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    # Get the key metrics for comparison
    lr_f1, xgb_f1 = lr_results["f1"], xgb_results["f1"]      # F1-scores
    lr_auc, xgb_auc = lr_results["auc"], xgb_results["auc"]   # ROC-AUC scores
    
    # Simple decision rule:
    # - If XGBoost has better F1 AND similar/better AUC → recommend XGBoost
    # - Otherwise → recommend Logistic Regression (it's more interpretable)
    #
    # Why F1-score? Because it balances precision and recall, which is what
    # we care about for this problem (we want to catch readmissions without
    # too many false alarms).
    #
    # Why consider AUC? Because it measures overall model quality. If F1 is
    # similar, we might prefer the model with better AUC.
    #
    # Why prefer Logistic Regression if similar? Because it's easier to explain
    # to doctors and administrators. Interpretability matters in healthcare!
    
    if xgb_f1 > lr_f1 and (np.isnan(lr_auc) or xgb_auc >= lr_auc - 0.01):
        # XGBoost wins! It's better at predicting readmissions.
        print(
            "Recommended deployment model: XGBoost (top 25 features)\n"
            f"- Higher F1-score ({xgb_f1:.3f} vs {lr_f1:.3f})\n"
            f"- ROC-AUC: {xgb_auc:.3f}\n"
            "Use Logistic Regression as the interpretable baseline in the report."
        )
    else:
        # Logistic Regression wins (or they're similar)!
        # We prefer it because it's easier to explain, even if performance is similar.
        print(
            "Recommended deployment model: Logistic Regression (top 20 features)\n"
            f"- F1-score: {lr_f1:.3f} vs {xgb_f1:.3f}\n"
            f"- ROC-AUC: {lr_auc:.3f} vs {xgb_auc:.3f}\n"
            "Still keep XGBoost as an experimental benchmark."
        )
    print()


if __name__ == "__main__":
    # If someone runs this file directly, automatically run the evaluation
    evaluate_all()
