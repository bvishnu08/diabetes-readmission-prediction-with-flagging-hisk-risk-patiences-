"""
Clinical Utilities - Translating Model Predictions into Clinical Language!

This module helps us interpret model predictions in a way that makes sense
to doctors and nurses. Instead of technical terms like "positive class" or
"precision", we use clinical terms like "HIGH RISK" and "safe to discharge".

Think of it like a translator:
- The model speaks "computer language" (probabilities, thresholds)
- Clinicians speak "clinical language" (high risk, safe to discharge)
- This module translates between them!

This makes it easier for doctors and nurses to understand what the model
is telling them, so they can make better decisions about patient care.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np


def summarize_risk_view(
    y_true,
    y_pred,
    p_readmit,
) -> Dict[str, Any]:
    """
    Create a clinical-style summary of model predictions.
    
    This function takes the model's predictions and translates them into
    language that clinicians can understand and use. Instead of saying
    "the model predicted class 1", we say "the model flagged this patient
    as HIGH RISK for readmission".
    
    The key insight: We want to know:
    - How many patients are flagged as "safe to discharge" (LOW RISK)?
    - Of those "safe" patients, how many actually get readmitted?
    - How confident is the model that they're safe?
    
    This helps clinicians understand:
    - "If the model says 'safe to discharge', can I trust it?"
    - "What percentage of patients are flagged as high risk?"
    - "What's the average risk level for patients we flag as safe?"
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The actual outcomes - what really happened:
        1 = patient WAS readmitted within 30 days (we missed them!)
        0 = patient was NOT readmitted within 30 days (good!)
    y_pred : array-like of shape (n_samples,)
        The model's predictions AFTER applying the threshold:
        1 = model flags patient as HIGH RISK (likely to be readmitted)
        0 = model flags patient as LOW RISK (safe to discharge)
    p_readmit : array-like of shape (n_samples,)
        The model's predicted probability of readmission (0.0 to 1.0)
        Example: 0.7 means "70% chance of readmission"
    
    Returns
    -------
    summary : dict
        A dictionary with all the clinical metrics:
        - n_total: Total number of patients
        - n_high: Number flagged as HIGH RISK
        - n_low: Number flagged as LOW RISK
        - high_risk_percent: Percentage flagged as HIGH RISK
        - low_risk_percent: Percentage flagged as LOW RISK
        - observed_readmit_rate_low: Of LOW RISK patients, how many actually got readmitted?
        - observed_safe_rate_low: Of LOW RISK patients, how many were actually safe?
        - avg_p_readmit: Average predicted readmission risk (across all patients)
        - avg_p_safe: Average predicted safe discharge chance (across all patients)
        - avg_p_safe_low: Average predicted safe chance for LOW RISK patients only
    """
    # Convert everything to numpy arrays so we can do math with them
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p_readmit = np.asarray(p_readmit)
    
    # Calculate probability of "safe discharge" (opposite of readmission)
    # If readmission probability is 0.7, then safe probability is 0.3
    p_safe = 1.0 - p_readmit
    
    # ===================================================================
    # COUNT PATIENTS BY RISK LEVEL
    # ===================================================================
    # How many patients total?
    n_total = len(y_true)
    
    # Create masks (True/False arrays) to filter patients by risk level
    high_mask = (y_pred == 1)  # Patients flagged as HIGH RISK
    low_mask = (y_pred == 0)   # Patients flagged as LOW RISK
    
    # Count how many in each category
    n_high = int(high_mask.sum())  # Number flagged as HIGH RISK
    n_low = int(low_mask.sum())    # Number flagged as LOW RISK
    
    # Calculate percentages
    high_risk_percent = n_high / n_total if n_total > 0 else 0.0  # % flagged as HIGH RISK
    low_risk_percent = n_low / n_total if n_total > 0 else 0.0    # % flagged as LOW RISK
    
    # ===================================================================
    # ANALYZE LOW-RISK PATIENTS (The "Safe to Discharge" Group)
    # ===================================================================
    # This is the most important metric for clinicians:
    # "If the model says 'safe to discharge', can I trust it?"
    
    if n_low > 0:  # Make sure we have some LOW RISK patients
        # Of the patients we said were "safe to discharge" (LOW RISK),
        # how many actually got readmitted? This tells us if we can trust the model.
        observed_readmit_rate_low = float((y_true[low_mask] == 1).mean())
        # observed_readmit_rate_low = 0.10 means "10% of our 'safe' patients actually got readmitted"
        
        # The opposite: how many were actually safe?
        observed_safe_rate_low = 1.0 - observed_readmit_rate_low
        # observed_safe_rate_low = 0.90 means "90% of our 'safe' patients were actually safe"
        
        # Average predicted "safe" probability for LOW RISK patients
        # This shows how confident the model was that these patients were safe
        avg_p_safe_low = float(p_safe[low_mask].mean())
        # avg_p_safe_low = 0.85 means "on average, we predicted 85% chance of safe discharge"
    else:
        # Edge case: if model flagged everyone as HIGH RISK
        observed_readmit_rate_low = 0.0
        observed_safe_rate_low = 0.0
        avg_p_safe_low = 0.0
    
    # ===================================================================
    # OVERALL STATISTICS
    # ===================================================================
    # Average predicted readmission risk across ALL patients (high and low risk)
    avg_p_readmit = float(p_readmit.mean())
    # avg_p_readmit = 0.15 means "on average, we predict 15% readmission risk for all patients"
    
    # Average predicted safe discharge chance across ALL patients
    avg_p_safe = float(p_safe.mean())
    # avg_p_safe = 0.85 means "on average, we predict 85% chance of safe discharge for all patients"
    
    # ===================================================================
    # RETURN THE SUMMARY
    # ===================================================================
    # Return everything in a dictionary so the caller can use it however they want
    return {
        "n_total": n_total,                          # Total number of patients
        "n_high": n_high,                            # Number flagged as HIGH RISK
        "n_low": n_low,                              # Number flagged as LOW RISK
        "high_risk_percent": high_risk_percent,      # % flagged as HIGH RISK
        "low_risk_percent": low_risk_percent,        # % flagged as LOW RISK
        "observed_readmit_rate_low": observed_readmit_rate_low,  # % of LOW RISK who actually got readmitted
        "observed_safe_rate_low": observed_safe_rate_low,        # % of LOW RISK who were actually safe
        "avg_p_readmit": avg_p_readmit,              # Average readmission risk (all patients)
        "avg_p_safe": avg_p_safe,                    # Average safe chance (all patients)
        "avg_p_safe_low": avg_p_safe_low,            # Average safe chance (LOW RISK patients only)
    }

