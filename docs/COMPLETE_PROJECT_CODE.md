# COMPLETE PROJECT CODE - DIABETES 30-DAY READMISSION PREDICTION

This document contains all the code from the project, organized by module with clear headings.

---

## 1. CONFIGURATION MODULE (`src/config.py`)

```python
"""
Configuration file - The "Settings" of our project!

Imagine this file as the control center for our entire project. Instead of
hardcoding file paths and settings all over the place, we put everything here.
That way, if we need to change something (like where the data is stored), we
only change it in ONE place, and the whole project automatically uses the new
setting. Pretty neat, right?

This is like having a master remote control for your project - change the
settings here, and everything else follows along.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Config:
    """
    This is our project's "settings box" - everything we need to configure is here!
    
    Think of it like a settings menu in a video game. You can see all the options,
    change them if you want, and they affect how the game (our project) runs.
    
    The @dataclass decorator is Python magic that automatically creates helpful
    functions for us, so we don't have to write boring boilerplate code.
    """
    
    # ===================================================================
    # WHERE IS EVERYTHING STORED? (File Paths)
    # ===================================================================
    # These tell Python where to find our files and where to save things
    
    # First, let's figure out where this project lives on the computer
    # __file__ is this Python file, .parents[1] means "go up one folder"
    # So if this file is at: project/src/config.py
    # Then project_root is: project/
    project_root: Path = Path(__file__).resolve().parents[1]
    
    # Where is our raw data? (The original CSV file from UCI)
    # This is like telling someone "The data is in the data/raw folder"
    data_raw: Path = Path("data/raw/diabetic_data.csv")
    
    # Random seed - this is like a "magic number" that makes random things
    # happen the same way every time. Set to 42 because... why not? It's
    # a popular choice in data science (thanks, Hitchhiker's Guide!)
    seed: int = 42
    
    # ===================================================================
    # WHERE DO WE SAVE CLEANED DATA?
    # ===================================================================
    # After we clean the raw data, we save it here so we don't have to
    # clean it again every time (saves time!)
    
    # The folder where cleaned data goes
    data_processed_dir: Path = Path("data/processed")
    
    # Names of the files we'll create
    train_processed_name: str = "train_processed.csv"  # Training data (80% of patients)
    test_processed_name: str = "test_processed.csv"    # Test data (20% of patients)
    
    # ===================================================================
    # WHERE DO WE SAVE OUR TRAINED MODELS?
    # ===================================================================
    # After training, we save the models here so we can use them later
    # without having to retrain (which takes time!)
    
    models_dir: Path = Path("models")  # The folder for saved models
    
    # Names of the model files we'll save
    model_name_logreg: str = "logreg_selected.joblib"  # Our Logistic Regression model
    model_name_xgb: str = "xgb_selected.joblib"        # Our XGBoost model
    
    # This file stores the "optimal thresholds" - think of it like the
    # "sweet spot" for making predictions. We'll explain this more later!
    thresholds_file: str = "thresholds.json"
    
    # ===================================================================
    # WHERE DO WE SAVE REPORTS AND PLOTS?
    # ===================================================================
    # Any charts, graphs, or reports we generate go here
    reports_dir: Path = Path("reports")
    
    # ===================================================================
    # WHAT ARE WE PREDICTING? (Target Variables)
    # ===================================================================
    # The "target" is what we're trying to predict - in our case, whether
    # a patient will be readmitted within 30 days
    
    # The original column name in the raw data
    raw_target_col: str = "readmitted"
    
    # After cleaning, we create a binary version:
    # 1 = patient was readmitted within 30 days (bad - we want to catch this!)
    # 0 = patient was NOT readmitted within 30 days (good!)
    target_col: str = "readmitted_binary"
    
    # ===================================================================
    # WHAT FEATURES CAN WE USE? (The "Ingredients" for Our Models)
    # ===================================================================
    # These are all the features (patient characteristics) we might use
    # to predict readmission. Think of them as clues that help us guess
    # if someone will be readmitted.
    
    # We have a LOT of features (41 total!), but we won't use all of them.
    # Instead, we'll let the computer pick the "best" ones for each model.
    # This is like having a big toolbox but only using the most useful tools.
    
    candidate_features: List[str] = (
        # ===== PATIENT BASICS =====
        # Who is this patient? Basic info about them.
        "age",              # How old are they? (converted to age groups like 60-70)
        "gender",           # Male or Female?
        "race",             # What's their race/ethnicity?
        
        # ===== HOSPITAL VISIT INFO =====
        # How did they get here? Where are they going?
        "admission_type_id",         # Emergency? Elective surgery? Transfer?
        "admission_source_id",       # Came from home? Another hospital? Doctor referral?
        "discharge_disposition_id",  # Going home? To a nursing home? To another hospital?
        
        # ===== HOW MUCH HEALTHCARE DID THEY USE? =====
        # These numbers tell us how "sick" or "complex" the patient is
        "time_in_hospital",      # How many days did they stay? (longer = usually sicker)
        "number_inpatient",       # How many times have they been in the hospital before?
        "number_emergency",       # How many ER visits before?
        "number_outpatient",      # How many outpatient visits before?
        "num_lab_procedures",     # How many lab tests did we do? (more = sicker)
        "num_procedures",         # How many procedures/surgeries? (more = sicker)
        "num_medications",        # How many different medications? (more = more complex)
        "number_diagnoses",       # How many different health problems? (more = sicker)
        
        # ===== DIABETES-SPECIFIC INFO =====
        # Since this is a diabetes dataset, we have lots of diabetes-related info
        "max_glu_serum",          # Highest blood sugar level we measured
        "A1Cresult",             # Hemoglobin A1C test (shows long-term blood sugar control)
        "change",                # Did we change their diabetes medications? (Yes/No)
        "diabetesMed",            # Are they taking diabetes medication? (Yes/No)
        
        # ===== SPECIFIC DIABETES MEDICATIONS =====
        # These are all different types of diabetes medications
        # The computer will figure out which ones matter most for predicting readmission
        "insulin",               # Taking insulin?
        "metformin",             # Taking metformin? (common first-line drug)
        "repaglinide",           # Taking repaglinide?
        "nateglinide",           # Taking nateglinide?
        "chlorpropamide",        # Taking chlorpropamide?
        "glimepiride",           # Taking glimepiride?
        "acetohexamide",         # Taking acetohexamide?
        "glipizide",             # Taking glipizide?
        "glyburide",             # Taking glyburide?
        "tolbutamide",           # Taking tolbutamide?
        "pioglitazone",          # Taking pioglitazone?
        "rosiglitazone",         # Taking rosiglitazone?
        "acarbose",              # Taking acarbose?
        "miglitol",              # Taking miglitol?
        "troglitazone",          # Taking troglitazone?
        "tolazamide",            # Taking tolazamide?
        "examide",               # Taking examide?
        "citoglipton",           # Taking citoglipton?
        
        # ===== COMBINATION MEDICATIONS =====
        # Some patients take multiple diabetes drugs combined into one pill
        "glyburide-metformin",   # Combination pill
        "glipizide-metformin",   # Combination pill
        "glimepiride-pioglitazone",      # Combination pill
        "metformin-rosiglitazone",       # Combination pill
        "metformin-pioglitazone",        # Combination pill
    )
    
    # ===================================================================
    # HOW DO WE SPLIT THE DATA?
    # ===================================================================
    # We need to split our data into "training" (to teach the model) and
    # "testing" (to see how good the model is)
    
    test_size: float = 0.2  # 20% of data for testing (the rest, 80%, is for training)
    random_state: int = 42  # Same "magic number" - makes the split the same every time
    
    # ===================================================================
    # HOW MANY FEATURES SHOULD EACH MODEL USE?
    # ===================================================================
    # Remember we have 41 features total? Well, we don't want to use ALL of them.
    # Too many features can confuse the model (like too many ingredients in a recipe).
    # So we'll pick the "best" ones:
    
    lr_top_k: int = 20   # Logistic Regression gets the top 20 features (simpler model)
    xgb_top_k: int = 25  # XGBoost gets the top 25 features (can handle more complexity)
    
    # ===================================================================
    # THRESHOLD TUNING SETTINGS
    # ===================================================================
    # This is a bit advanced, but here's the idea:
    # When a model makes a prediction, it gives us a probability (like "70% chance
    # of readmission"). But we need to decide: at what point do we say "yes, this
    # patient will be readmitted"? That's the "threshold".
    
    # Default threshold is usually 0.5 (50%), but for our problem, we want to catch
    # MORE readmissions (even if we get some false alarms). So we'll tune this.
    target_recall: float = 0.65  # We want to catch at least 65% of readmissions
    
    # We'll try different thresholds from 0.05 to 0.95 and pick the best one
    threshold_grid: np.ndarray = field(
        default_factory=lambda: np.linspace(0.05, 0.95, 19)
        # This creates: [0.05, 0.10, 0.15, ..., 0.90, 0.95] (19 values total)
    )
    
    # ===================================================================
    # HELPER FUNCTIONS - These Build Full File Paths
    # ===================================================================
    # These functions take our relative paths (like "data/raw/diabetic_data.csv")
    # and turn them into full, absolute paths that work no matter where you
    # run the code from. It's like giving someone directions with full addresses
    # instead of just "turn left at the corner".
    
    def resolved_raw_path(self) -> Path:
        """
        Get the full path to the raw data file.
        
        Example: If project is at /Users/you/project/
        This returns: /Users/you/project/data/raw/diabetic_data.csv
        """
        return (self.project_root / self.data_raw).resolve()
    
    def processed_train_path(self) -> Path:
        """Get the full path to the cleaned training data file."""
        return (self.project_root / self.data_processed_dir / self.train_processed_name).resolve()
    
    def processed_test_path(self) -> Path:
        """Get the full path to the cleaned test data file."""
        return (self.project_root / self.data_processed_dir / self.test_processed_name).resolve()
    
    def models_path(self) -> Path:
        """Get the full path to the models folder."""
        return (self.project_root / self.models_dir).resolve()
    
    def model_path_logreg(self) -> Path:
        """Get the full path to the saved Logistic Regression model."""
        return self.models_path() / self.model_name_logreg
    
    def model_path_xgb(self) -> Path:
        """Get the full path to the saved XGBoost model."""
        return self.models_path() / self.model_name_xgb
    
    def thresholds_path(self) -> Path:
        """Get the full path to the thresholds.json file."""
        return self.models_path() / self.thresholds_file
    
    def reports_path(self) -> Path:
        """Get the full path to the reports folder."""
        return (self.project_root / self.reports_dir).resolve()
```

---

## 2. DATA PREPROCESSING MODULE (`src/preprocess.py`)

```python
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
```

---

## 3. FEATURE SELECTION MODULE (`src/feature_selection.py`)

```python
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
```

---

## 4. MODEL BUILDING MODULE (`src/model.py`)

```python
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
```

---

## 5. TRAINING MODULE (`src/train.py`)

```python
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
```

---

## 6. EVALUATION MODULE (`src/evaluate.py`)

[See the full file - it's very long. The key functions are `_evaluate_one_model()` and `evaluate_all()` which test models and provide clinical interpretation.]

---

## 7. CLINICAL UTILITIES MODULE (`src/clinical_utils.py`)

```python
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
```

---

## 8. DASHBOARD (`dashboard.py`)

[The dashboard.py file is 728 lines. It includes:
- Streamlit configuration and CURSOR_THEME CSS
- Cached data loaders for models and data
- Evaluation functions
- Feature metadata
- UI building blocks (sidebar, header, tabs)
- Main app function

See the full `dashboard.py` file for complete code.]

---

## 9. SCRIPTS

### 9.1 Training Script (`scripts/run_train.py`)

```python
"""
Training Script - The Easy Way to Train Our Models!

This is a simple "wrapper" script that makes it easy to train the models.
Instead of having to write Python code, you can just run this script from
the command line and it does everything for you!

Think of it like a "one-click" button that:
- Loads the data
- Trains both models
- Saves everything
- Prints progress updates

Usage:
    python scripts/run_train.py

That's it! Just run this one command and it does all the work.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ===================================================================
# PATH SETUP
# ===================================================================
# This makes sure Python can find our code, no matter where you run
# the script from. It's like adding the project folder to Python's
# "search path" so it knows where to look for our modules.

# Figure out where the project root is (one folder up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add it to Python's path if it's not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ===================================================================
# IMPORT AND RUN
# ===================================================================
# Import the training function and run it!
from src.train import train_all_models

if __name__ == "__main__":
    # This runs when you execute the script directly (not when importing it)
    # It's like pressing the "play" button - it starts the training process!
    
    # This will:
    # - Load and clean the data
    # - Pick the best features for each model
    # - Train Logistic Regression (top 20 features)
    # - Train XGBoost (top 25 features)
    # - Find the best thresholds
    # - Save everything to disk
    train_all_models()
```

### 9.2 Evaluation Script (`scripts/run_eval.py`)

```python
"""
Evaluation Script - The Easy Way to Test Our Models!

This is a simple "wrapper" script that makes it easy to evaluate the models.
After training, you can run this to see how good your models are!

Think of it like a "report card generator" - it:
- Loads the trained models
- Tests them on the test set
- Compares them side-by-side
- Gives you a recommendation

Usage:
    python scripts/run_eval.py

Prerequisites:
    - You must run scripts/run_train.py first to train the models
    - Otherwise, there won't be any models to evaluate!

This will print a nice report showing:
- How accurate each model is
- How many readmissions they catch (recall)
- How many false alarms they have (precision)
- Which model is recommended for deployment
"""

from __future__ import annotations

import sys
from pathlib import Path

# ===================================================================
# PATH SETUP
# ===================================================================
# Same as run_train.py - makes sure Python can find our code

# Figure out where the project root is
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add it to Python's path if it's not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ===================================================================
# IMPORT AND RUN
# ===================================================================
# Import the evaluation function and run it!
from src.evaluate import evaluate_all

if __name__ == "__main__":
    # This runs when you execute the script directly
    # It's like pressing the "test" button - it evaluates the models!
    
    # This will:
    # - Load the test data
    # - Load the trained models
    # - Load the thresholds and feature lists
    # - Test both models on the same data
    # - Print a side-by-side comparison
    # - Give a recommendation
    evaluate_all()
```

### 9.3 Dashboard Script (`scripts/run_dashboard.py`)

```python
#!/usr/bin/env python
"""
Script to run the Streamlit dashboard
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit dashboard"""
    dashboard_path = Path(__file__).resolve().parents[1] / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main()
```

---

## 10. REQUIREMENTS (`requirements.txt`)

```
pandas
scikit-learn
joblib
pyarrow
pytest
streamlit
matplotlib
seaborn
numpy
plotly
xgboost
```

---

## PROJECT STRUCTURE SUMMARY

```
project_root/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── preprocess.py          # Data cleaning and splitting
│   ├── feature_selection.py   # Feature selection logic
│   ├── model.py              # Model building (LR & XGBoost)
│   ├── train.py              # Training pipeline
│   ├── evaluate.py            # Model evaluation
│   └── clinical_utils.py     # Clinical interpretation
├── scripts/
│   ├── run_train.py          # Training script
│   ├── run_eval.py           # Evaluation script
│   └── run_dashboard.py      # Dashboard launcher
├── data/
│   ├── raw/
│   │   └── diabetic_data.csv
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
├── models/
│   ├── logreg_selected.joblib
│   ├── xgb_selected.joblib
│   └── thresholds.json
├── dashboard.py              # Streamlit dashboard
└── requirements.txt          # Python dependencies
```

---

## HOW TO USE

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models:**
   ```bash
   python scripts/run_train.py
   ```

3. **Evaluate models:**
   ```bash
   python scripts/run_eval.py
   ```

4. **Run dashboard:**
   ```bash
   python scripts/run_dashboard.py
   # OR
   streamlit run dashboard.py
   ```

---

**END OF COMPLETE PROJECT CODE DOCUMENT**

