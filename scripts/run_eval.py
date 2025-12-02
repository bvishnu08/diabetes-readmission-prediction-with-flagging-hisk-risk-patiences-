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
