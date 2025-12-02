"""
Feature Selection - Picking the Best "Clues" for Our Models!

Imagine you're a detective trying to solve a case. You have 41 different clues,
but you can't use all of them - that would be too confusing! Instead, you want
to pick the 20-25 BEST clues that will help you solve the case.

That's what feature selection does - it picks the most useful features (patient
characteristics) that help us predict readmission.

We use "mutual information" to score each feature. Think of it like this:
- High score = "This feature tells me a lot about whether someone will be readmitted"
- Low score = "This feature doesn't really help me predict readmission"

Then we just pick the top-scoring features! Simple, right?

Why do we do this?
1. Simpler models are easier to understand and explain
2. Fewer features = faster training and prediction
3. Less chance of overfitting (memorizing the training data instead of learning patterns)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder


@dataclass
class FeatureSelectionConfig:
    """
    A simple settings box for feature selection.
    
    This is like a menu where you choose:
    - "How many features should Logistic Regression use?" → 20
    - "How many features should XGBoost use?" → 25
    
    We give Logistic Regression fewer features because it's simpler and works
    better with fewer features. XGBoost is more powerful and can handle more
    features without getting confused.
    """
    k_logreg: int = 20   # Logistic Regression gets top 20 features (simpler model)
    k_xgb: int = 25      # XGBoost gets top 25 features (can handle more complexity)

    def get_k_for_model(self, model_name: str) -> int:
        """
        Figure out how many features a model should use.
        
        This is like a translator - you say "logreg" or "xgb" and it tells you
        how many features that model should get.
        
        Parameters
        ----------
        model_name : str
            The name of the model. Can be:
            - "logreg", "lr", or "logistic" → returns 20
            - "xgb" or "xgboost" → returns 25
        
        Returns
        -------
        int
            How many features this model should use
        """
        # Convert to lowercase so "LogReg" and "logreg" both work
        name = model_name.lower()
        
        # If it's a Logistic Regression variant, return 20
        if name in ("logreg", "lr", "logistic"):
            return self.k_logreg
        # If it's an XGBoost variant, return 25
        elif name in ("xgb", "xgboost"):
            return self.k_xgb
        else:
            # If we don't recognize the name, raise an error
            raise ValueError(f"Unknown model name for feature selection: {model_name}")


def select_top_k(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    config: FeatureSelectionConfig | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pick the best K features using mutual information.
    
    This function is like a talent show judge - it looks at all the features,
    scores them based on how well they predict readmission, and picks the top K.
    
    How it works:
    1. Look at each feature and see how much it "knows" about readmission
    2. Score each feature (higher score = more useful)
    3. Pick the top K features (the highest-scoring ones)
    4. Return only those features
    
    The tricky part: we have both numbers (like age) and categories (like gender).
    We need to handle them differently, but the scoring method (mutual information)
    can handle both once we convert categories to numbers.
    
    Parameters
    ----------
    X : pd.DataFrame
        All our features (the "clues" we have)
    y : pd.Series
        The target (whether each patient was readmitted - 1 for yes, 0 for no)
    model_name : str
        Which model we're selecting features for ("logreg" or "xgb")
    config : FeatureSelectionConfig, optional
        Settings (how many features to pick). If None, uses defaults.
    
    Returns
    -------
    X_selected : pd.DataFrame
        The same data, but only with the top K features
    selected_features : List[str]
        The names of the features we picked (in order of importance)
    """
    # Use default settings if none provided
    if config is None:
        config = FeatureSelectionConfig()
    
    # Figure out how many features to pick (K)
    k = config.get_k_for_model(model_name)
    
    # Safety check: can't pick more features than we have!
    # If someone asks for 100 features but we only have 41, just use 41
    k = min(k, X.shape[1])
    
    # ===================================================================
    # HANDLE MIXED DATA TYPES (The Tricky Part!)
    # ===================================================================
    # Our data has both numbers (like age: 6) and categories (like gender: "Male")
    # The scoring method needs everything to be numbers, so we need to convert
    # categories to numbers first.
    
    # Make a copy so we don't mess up the original data
    X_encoded = X.copy()
    encoders = {}  # Store the encoders (in case we need to reverse it later)
    
    # Go through each column and check if it's categorical
    for col in X.columns:
        # Check if this column is text/categories (not numbers)
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            # Convert categories to numbers using LabelEncoder
            # Example: "Male", "Female" → 0, 1
            # Example: "Caucasian", "AfricanAmerican", "Hispanic" → 0, 1, 2
            le = LabelEncoder()
            
            # Convert to string, fill missing values with 'missing', then encode
            # This handles cases where data is missing
            X_encoded[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
            encoders[col] = le  # Save the encoder (not used later, but good practice)
        
        # Fallback for other weird data types
        elif not pd.api.types.is_numeric_dtype(X[col]):
            # Same process for any other non-numeric types
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
            encoders[col] = le
        
        # If it's already a number, we don't need to do anything!
        # Just leave it as-is.
    
    # ===================================================================
    # SCORE ALL FEATURES USING MUTUAL INFORMATION
    # ===================================================================
    # Now that everything is a number, we can score each feature
    
    # SelectKBest is like a judge that scores each feature
    # mutual_info_classif is the scoring method - it measures how much
    # information each feature gives us about the target (readmission)
    
    # Higher score = "This feature tells me a lot about readmission"
    # Lower score = "This feature doesn't help much"
    
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_encoded, y)  # Score all features
    
    # ===================================================================
    # PICK THE TOP K FEATURES
    # ===================================================================
    # Get the scores for all features
    # selector.scores_ is an array like [0.05, 0.12, 0.08, ...]
    # Each number is how "useful" that feature is
    
    # Create a dictionary: feature name → score
    # Example: {"age": 0.12, "gender": 0.05, "num_medications": 0.15, ...}
    feature_scores = dict(zip(X.columns, selector.scores_))
    
    # Sort features by score (highest first)
    # This is like ranking contestants: best score first, worst score last
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take the top K feature names
    # If K=20, we take the 20 highest-scoring features
    selected_features = [f[0] for f in sorted_features[:k]]
    
    # ===================================================================
    # RETURN THE FILTERED DATA
    # ===================================================================
    # Return the original data (not the encoded version) with only the
    # selected features. This preserves the original data types.
    
    X_selected = X[selected_features].copy()
    
    return X_selected, selected_features
