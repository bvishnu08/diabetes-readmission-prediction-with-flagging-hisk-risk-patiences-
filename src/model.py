"""
Building Our Machine Learning Models - The "Recipe" for Predictions!

This module is like a recipe book for building our prediction models. Just like
how a recipe tells you what ingredients to use and in what order, these functions
tell Python how to build our models step-by-step.

We have two "recipes":
1. Logistic Regression - A simpler, easier-to-understand model (like a basic
   chocolate chip cookie recipe)
2. XGBoost - A more complex, powerful model (like a fancy French pastry recipe)

Both recipes include:
- Preprocessing steps (cleaning and preparing the data)
- The actual model (the "brain" that makes predictions)

Think of it like this: before you can bake cookies, you need to:
1. Measure ingredients (preprocessing)
2. Mix them together (the model)
3. Bake (training)

That's what these functions do - they set up the whole "baking process"!
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Figure out which features are numbers and which are categories.
    
    This is like sorting your ingredients into two piles:
    - Numbers: things you can measure (like "5 cups of flour")
    - Categories: things that are just names (like "chocolate chips" or "vanilla")
    
    Why do we care? Because we need to treat them differently:
    - Numbers: we can do math with them (add, subtract, average)
    - Categories: we need to convert them to numbers first (like turning "Male"
      and "Female" into 0 and 1)
    
    Parameters
    ----------
    df : pd.DataFrame
        Our data with all the features (patient characteristics)
    
    Returns
    -------
    numeric_cols : List[str]
        List of column names that are numbers (like age, number of medications)
    categorical_cols : List[str]
        List of column names that are categories (like gender, race)
    """
    # Find all columns that are numbers (integers or decimals)
    # Examples: age (6), num_medications (5), time_in_hospital (3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Find all columns that are NOT numbers (text/categories)
    # Examples: gender ("Male", "Female"), race ("Caucasian", "AfricanAmerican")
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    return numeric_cols, categorical_cols


def build_logreg_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int,
) -> Pipeline:
    """
    Build a Logistic Regression model pipeline.
    
    Think of this like building a simple, reliable car:
    - It's not the fanciest, but it's easy to understand and explain
    - It gets the job done
    - You can explain to someone how it works
    
    Logistic Regression is great because:
    - It's interpretable (we can see which features matter most)
    - It's fast to train
    - It's less likely to overfit (memorize the training data)
    
    The pipeline does two things:
    1. Preprocessing: clean and prepare the data
    2. Classification: make predictions
    
    Parameters
    ----------
    numeric_cols : List[str]
        Which columns are numbers (like age, medications)
    categorical_cols : List[str]
        Which columns are categories (like gender, race)
    random_state : int
        A "magic number" to make results reproducible (same results every time)
    
    Returns
    -------
    Pipeline
        A complete, ready-to-use model that can be trained and make predictions
    """
    # ===================================================================
    # STEP 1: PREPARE NUMERIC FEATURES
    # ===================================================================
    # For numbers, we need to:
    # 1. Fill in missing values (if someone's age is missing, use the median age)
    # 2. Scale them (make sure all numbers are on the same scale, like converting
    #    feet to inches so everything is in the same units)
    
    # Why median? Because it's robust to outliers. If you have ages like
    # [20, 25, 30, 35, 100], the median (30) is better than the mean (42)
    # because that 100-year-old doesn't throw everything off.
    
    # Why scaling? Because Logistic Regression is sensitive to the scale of numbers.
    # If one feature is "number of medications" (0-20) and another is "age" (0-9),
    # the model might think medications are more important just because the numbers
    # are bigger. Scaling fixes this.
    
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Fill missing with median
            ("scaler", StandardScaler(with_mean=False)),    # Scale to same range
        ]
    )
    
    # ===================================================================
    # STEP 2: PREPARE CATEGORICAL FEATURES
    # ===================================================================
    # For categories (like gender: "Male" or "Female"), we need to:
    # 1. Fill in missing values (if gender is missing, use the most common gender)
    # 2. Convert to numbers (turn "Male"/"Female" into 0/1)
    
    # One-hot encoding means: if you have gender with values "Male" and "Female",
    # you create two columns:
    # - gender_Male: 1 if male, 0 if not
    # - gender_Female: 1 if female, 0 if not
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing with most common
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            # handle_unknown="ignore" means: if we see a new category we haven't
            # seen before, don't crash - just ignore it
        ]
    )
    
    # ===================================================================
    # STEP 3: COMBINE THE TRANSFORMERS
    # ===================================================================
    # We have two different "preparation methods" (one for numbers, one for categories)
    # Now we combine them so they work together
    
    transformers = []
    if numeric_cols:
        # If we have numeric columns, add the numeric transformer
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        # If we have categorical columns, add the categorical transformer
        transformers.append(("cat", categorical_transformer, categorical_cols))
    
    # Safety check: we need at least one type of feature!
    if not transformers:
        raise ValueError("No numeric or categorical columns found for preprocessing.")
    
    # ColumnTransformer applies different transformations to different columns
    # It's like having two chefs in the kitchen - one handles numbers, one handles categories
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    
    # ===================================================================
    # STEP 4: ADD THE ACTUAL MODEL (THE "BRAIN")
    # ===================================================================
    # Logistic Regression is like a simple decision-making process:
    # - It looks at all the features
    # - It gives each feature a "weight" (importance)
    # - It adds them up and says "yes" or "no"
    
    # class_weight="balanced" means: since we have more "no readmission" cases
    # than "readmission" cases, give more weight to the rare "readmission" cases
    # (so we don't just predict "no readmission" all the time)
    
    clf = LogisticRegression(
        max_iter=1000,              # Try up to 1000 times to find the best solution
        class_weight="balanced",     # Handle imbalanced data (more "no" than "yes")
        random_state=random_state,   # Same random seed = same results every time
        solver="lbfgs",              # The algorithm to use (efficient for our data size)
    )
    
    # ===================================================================
    # STEP 5: PUT IT ALL TOGETHER
    # ===================================================================
    # A Pipeline chains everything together:
    # 1. First, preprocess the data (clean it up)
    # 2. Then, feed it to the model (make predictions)
    
    # This is like an assembly line: data goes in → preprocessing → model → predictions come out
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),  # Step 1: Clean the data
            ("clf", clf),                  # Step 2: Make predictions
        ]
    )
    
    return pipe


def build_xgb_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Pipeline:
    """
    Build an XGBoost model pipeline.
    
    Think of this like building a high-performance sports car:
    - It's more complex and powerful
    - It can handle more complicated patterns
    - It might be harder to explain how it works, but it's really good at predictions
    
    XGBoost is great because:
    - It's very accurate (often beats simpler models)
    - It can find complex patterns in the data
    - It's less likely to overfit (with the right settings)
    
    The main difference from Logistic Regression:
    - XGBoost doesn't need scaling (trees don't care about the scale of numbers)
    - It can handle more features and more complexity
    
    Parameters
    ----------
    numeric_cols : List[str]
        Which columns are numbers
    categorical_cols : List[str]
        Which columns are categories
    
    Returns
    -------
    Pipeline
        A complete, ready-to-use XGBoost model
    """
    # ===================================================================
    # STEP 1: PREPARE NUMERIC FEATURES
    # ===================================================================
    # For XGBoost, we only need to fill missing values
    # We DON'T need scaling because tree-based models (like XGBoost) don't care
    # about the scale. They just ask "is this number bigger than that number?"
    # They don't care if it's 5 vs 10 or 500 vs 1000.
    
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # Just fill missing values
            # No scaler needed! Trees don't need it.
        ]
    )
    
    # ===================================================================
    # STEP 2: PREPARE CATEGORICAL FEATURES
    # ===================================================================
    # Same as Logistic Regression - fill missing and convert to numbers
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    
    # ===================================================================
    # STEP 3: COMBINE THE TRANSFORMERS
    # ===================================================================
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))
    
    if not transformers:
        raise ValueError("No numeric or categorical columns found for preprocessing.")
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    
    # ===================================================================
    # STEP 4: ADD THE XGBOOST MODEL
    # ===================================================================
    # XGBoost works by building many small "decision trees" and combining them.
    # Think of it like asking many people for advice and taking a vote.
    # Each tree is like one person's opinion, and the final prediction is the
    # majority vote.
    
    # Here's what each setting does:
    xgb = XGBClassifier(
        n_estimators=300,           # Build 300 trees (more = better but slower)
        max_depth=4,                # Each tree can be 4 levels deep (prevents overfitting)
        learning_rate=0.05,         # How fast to learn (smaller = slower but more careful)
        subsample=0.8,               # Use 80% of data for each tree (reduces overfitting)
        colsample_bytree=0.8,        # Use 80% of features for each tree (reduces overfitting)
        objective="binary:logistic", # We're doing binary classification (yes/no)
        eval_metric="logloss",       # How to measure "goodness" during training
        n_jobs=-1,                   # Use all CPU cores (faster training)
        scale_pos_weight=1.0,        # Balance between classes (1.0 = balanced)
        reg_lambda=1.0,              # Regularization (prevents overfitting)
        reg_alpha=0.0,               # More regularization (not used here)
    )
    
    # ===================================================================
    # STEP 5: PUT IT ALL TOGETHER
    # ===================================================================
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),  # Clean the data
            ("clf", xgb),                 # Make predictions with XGBoost
        ]
    )
    
    return pipe


def build_pipeline(
    X_train: pd.DataFrame,
    random_state: int = 42,
) -> Pipeline:
    """
    A helper function for backwards compatibility with old notebooks.
    
    Some of our notebooks were written before we had separate functions for
    Logistic Regression and XGBoost. This function lets those old notebooks
    still work by automatically building a Logistic Regression pipeline.
    
    Think of it like a "legacy mode" - it keeps old code working while we
    move to the new, better way of doing things.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training data (used to figure out which columns are numeric vs categorical)
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    Pipeline
        A Logistic Regression pipeline (the simpler model)
    """
    # Automatically figure out which columns are numbers and which are categories
    numeric_cols, categorical_cols = infer_feature_types(X_train)
    
    # Build and return a Logistic Regression pipeline
    return build_logreg_pipeline(numeric_cols, categorical_cols, random_state)
