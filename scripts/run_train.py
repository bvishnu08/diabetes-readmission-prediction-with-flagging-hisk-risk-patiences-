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
