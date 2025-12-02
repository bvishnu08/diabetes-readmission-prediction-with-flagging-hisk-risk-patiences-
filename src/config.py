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
