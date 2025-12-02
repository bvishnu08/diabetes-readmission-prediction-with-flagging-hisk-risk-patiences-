"""
Data Preprocessing - Cleaning Up Our Messy Data!

Raw data is like a messy room - it has stuff everywhere, some things are missing,
and it's not organized the way we need it. This module is like a cleaning service
that takes the messy raw data and makes it nice and tidy for our models to use.

What we do here:
1. Load the raw CSV file (like opening a box of files)
2. Clean it up (fix missing values, convert weird formats)
3. Create our target variable (the thing we're trying to predict)
4. Split it into training and testing sets (like separating practice problems
   from the final exam)

Think of this as the "preparation" step before cooking - you gotta wash the
vegetables and chop them before you can make the meal!
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config

# ===================================================================
# AGE CONVERSION TABLE
# ===================================================================
# The raw data has ages as text like "[60-70)" which means "60 to 70 years old"
# We need to convert these to numbers so the computer can understand them.
# We'll use 0-9 where 0 = youngest, 9 = oldest.

# Think of it like converting letter grades to numbers:
# "[0-10)" → 0 (very young)
# "[10-20)" → 1 (teenager)
# ...
# "[90-100)" → 9 (very old)

AGE_BUCKET_MAP = {
    "[0-10)": 0,    # Babies and young kids
    "[10-20)": 1,   # Teenagers
    "[20-30)": 2,   # Young adults
    "[30-40)": 3,   # Adults
    "[40-50)": 4,   # Middle-aged
    "[50-60)": 5,   # Getting older
    "[60-70)": 6,   # Seniors
    "[70-80)": 7,   # Elderly
    "[80-90)": 8,   # Very elderly
    "[90-100)": 9,  # Centenarians!
}


def load_raw(cfg: Config) -> pd.DataFrame:
    """
    Load the raw data file from disk.
    
    This is like opening a book - we're just reading the CSV file and
    putting it into a pandas DataFrame (which is like a fancy Excel spreadsheet
    that Python can work with).
    
    Parameters
    ----------
    cfg : Config
        Our settings object that knows where the data file is
    
    Returns
    -------
    pd.DataFrame
        The raw data, just as it came from the UCI website
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist (maybe you forgot to download it?)
    """
    # Get the full path to the data file
    raw_path = cfg.resolved_raw_path()
    
    # Check if the file actually exists
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
    
    # Read the CSV file into a pandas DataFrame
    # Think of this as opening an Excel file, but in Python
    df = pd.read_csv(raw_path)
    
    return df


def basic_clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Clean up the messy raw data and get it ready for modeling.
    
    This function does a lot of "housekeeping":
    - Removes extra spaces from column names
    - Handles missing values (the dataset uses "?" for missing data)
    - Creates our target variable (did they get readmitted?)
    - Converts age from text to numbers
    - Keeps only the features we care about
    
    Think of this like cleaning your room:
    - Throw away stuff you don't need
    - Organize what you're keeping
    - Fix things that are broken or incomplete
    
    Parameters
    ----------
    df : pd.DataFrame
        The raw, messy data straight from the CSV file
    cfg : Config
        Our settings object (knows which features to keep, etc.)
    
    Returns
    -------
    pd.DataFrame
        Clean, organized data ready for machine learning
    """
    # Make a copy so we don't mess up the original data
    # It's like making a photocopy before you start editing
    df = df.copy()
    
    # Clean up column names - sometimes they have extra spaces
    # "  age  " becomes "age"
    df.columns = df.columns.str.strip()
    
    # Remove columns that are completely empty (all NaN values)
    # Like throwing away a blank notebook
    df = df.dropna(axis=1, how="all")
    
    # The UCI dataset uses "?" to mean "we don't know this value"
    # We'll convert "?" to NaN (which is Python's way of saying "missing")
    # This is like replacing "???" with a blank space
    df.replace("?", np.nan, inplace=True)
    
    # Make sure the target column exists (we need it to train the model!)
    if cfg.raw_target_col not in df.columns:
        raise KeyError(f"Target column '{cfg.raw_target_col}' not found in raw data.")
    
    # ===================================================================
    # CREATE OUR TARGET VARIABLE (What We're Trying to Predict)
    # ===================================================================
    # The original data has three categories:
    # - "<30" = readmitted within 30 days (BAD - we want to catch this!)
    # - ">30" = readmitted after 30 days (not as urgent)
    # - "NO" = not readmitted (GOOD!)
    #
    # For our problem, we only care about "<30" readmissions (the urgent ones).
    # So we'll create a binary variable:
    # - 1 = readmitted within 30 days (this is what we're trying to predict!)
    # - 0 = everything else (readmitted later, or not readmitted)
    
    df[cfg.target_col] = (df[cfg.raw_target_col] == "<30").astype(int)
    # This line says: "If readmitted == '<30', set to 1, otherwise set to 0"
    
    # ===================================================================
    # CONVERT AGE FROM TEXT TO NUMBERS
    # ===================================================================
    # Age is stored as text like "[60-70)" but we need it as a number
    # We'll use our conversion table to turn "[60-70)" into 6
    if "age" in df.columns:
        df["age"] = df["age"].map(AGE_BUCKET_MAP)
        # If age is missing or not in our table, it becomes NaN (that's okay)
    
    # ===================================================================
    # KEEP ONLY THE FEATURES WE CARE ABOUT
    # ===================================================================
    # We have a curated list of 41 features we want to use
    # Let's make sure they all exist in the data
    missing_features = [col for col in cfg.candidate_features if col not in df.columns]
    if missing_features:
        raise KeyError(
            "Configured core features missing from raw data: "
            + ", ".join(missing_features)
        )
    
    # Now keep ONLY the features we want, plus our target variable
    # It's like packing a suitcase - only take what you need!
    keep_cols = list(cfg.candidate_features) + [cfg.target_col]
    df = df[keep_cols].copy()
    
    # Remove any rows where we don't know the target (can't train on those!)
    # Like removing test papers where the student didn't write their name
    df = df.dropna(subset=[cfg.target_col])
    
    return df


def train_test_split_clean(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data, clean it, and split it into training and testing sets.
    
    This is like preparing for a test:
    - Training set = practice problems (80% of data)
    - Test set = the actual test (20% of data)
    
    We keep them separate because we want to see how well our model does on
    NEW data it hasn't seen before (the test set). If we test on the same
    data we trained on, that's like giving students the answers before the test!
    
    Parameters
    ----------
    cfg : Config
        Our settings object (knows how to split, random seed, etc.)
    
    Returns
    -------
    X_train : pd.DataFrame
        Features for training (the "practice problems")
    X_test : pd.DataFrame
        Features for testing (the "actual test")
    y_train : pd.Series
        Answers for training (so the model knows if it's right)
    y_test : pd.Series
        Answers for testing (so we can check how good the model is)
    """
    # Step 1: Load the raw CSV file
    # Like opening a book
    df_raw = load_raw(cfg)
    
    # Step 2: Clean it up
    # Like organizing and fixing the data
    df_clean = basic_clean(df_raw, cfg)
    
    # Step 3: Separate features (X) from target (y)
    # Features = the clues (patient characteristics)
    # Target = what we're trying to guess (will they be readmitted?)
    
    # Get all columns except the target
    feature_cols = [col for col in df_clean.columns if col != cfg.target_col]
    X = df_clean[feature_cols]  # All the features (the clues)
    y = df_clean[cfg.target_col]  # The target (the answer we're trying to predict)
    
    # Step 4: Split into training and testing
    # test_size=0.2 means 20% for testing, 80% for training
    # stratify=y means both sets have similar proportions of readmissions
    #   (like making sure both practice and test have similar difficulty)
    # random_state ensures we get the same split every time (reproducibility!)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,        # 20% for testing
        random_state=cfg.random_state,   # Same "magic number" = same split every time
        stratify=y,                      # Keep class balance in both sets
    )
    
    return X_train, X_test, y_train, y_test


def generate_processed_datasets(cfg: Config) -> None:
    """
    Clean the data and save it to CSV files for later use.
    
    This function does the full cleaning process and saves the results.
    Why save them? So we don't have to clean the data every single time we
    want to use it. It's like meal prep - do the work once, use it many times!
    
    The saved files are used by:
    - Training scripts (to train models)
    - Evaluation scripts (to test models)
    - Dashboard (to show data to users)
    - Notebooks (for exploration and analysis)
    
    Parameters
    ----------
    cfg : Config
        Our settings object (knows where to save files, etc.)
    
    Side Effects
    ------------
    Creates two CSV files:
    - data/processed/train_processed.csv (cleaned training data)
    - data/processed/test_processed.csv (cleaned test data)
    """
    # Create the "processed" folder if it doesn't exist
    # Like making sure you have a folder before putting files in it
    processed_dir = (cfg.project_root / cfg.data_processed_dir).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the cleaned, split data
    # This calls train_test_split_clean() which does all the work
    X_train, X_test, y_train, y_test = train_test_split_clean(cfg)
    
    # Combine features and target back together for saving
    # We separated them for the split, but for saving we want them together
    
    # Training set: features + target in one DataFrame
    train_df = X_train.copy()
    train_df[cfg.target_col] = y_train
    
    # Test set: features + target in one DataFrame
    test_df = X_test.copy()
    test_df[cfg.target_col] = y_test
    
    # Get the full paths where we'll save the files
    train_path = cfg.processed_train_path()
    test_path = cfg.processed_test_path()
    
    # Save to CSV files (index=False means don't save row numbers)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Print confirmation so the user knows it worked
    print(f"[preprocess] Saved train to {train_path}")
    print(f"[preprocess] Saved test  to {test_path}")
