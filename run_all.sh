#!/bin/bash
# Complete Setup and Run Script for Mac/Linux
# This script does everything: setup, install, train, evaluate

set -e  # Exit on any error

echo "=========================================="
echo "Diabetes Readmission Prediction - Setup & Run"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Project directory: $SCRIPT_DIR"
echo ""

# Step 1: Check Python
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ ERROR: Python not found! Please install Python 3.8+ first."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"
echo ""

# Step 2: Create virtual environment if it doesn't exist
echo "🔧 Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    echo "   Creating new virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Step 3: Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Step 4: Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ pip upgraded"
echo ""

# Step 5: Install requirements
echo "📦 Installing required packages..."
echo "   (This may take a few minutes...)"
pip install -r requirements.txt --quiet
echo "✅ All packages installed"
echo ""

# Step 6: Verify data file exists
echo "🔍 Checking data files..."
if [ ! -f "data/raw/diabetic_data.csv" ]; then
    echo "❌ ERROR: data/raw/diabetic_data.csv not found!"
    echo "   Please make sure the data file is in the correct location."
    exit 1
fi
echo "✅ Data file found"
echo ""

# Step 7: Run training
echo "=========================================="
echo "🚀 Starting Training Process"
echo "=========================================="
echo ""
echo "This will:"
echo "  - Clean and preprocess the data"
echo "  - Train Logistic Regression model"
echo "  - Train XGBoost model"
echo "  - Save all models and processed data"
echo ""
echo "⏳ Training in progress... (This takes 2-5 minutes)"
echo ""

# Set OMP_NUM_THREADS to avoid potential issues
export OMP_NUM_THREADS=1

$PYTHON_CMD scripts/run_train.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
else
    echo ""
    echo "❌ Training failed! Please check the error messages above."
    exit 1
fi
echo ""

# Step 8: Run evaluation
echo "=========================================="
echo "📊 Running Model Evaluation"
echo "=========================================="
echo ""
echo "⏳ Evaluating models... (This takes 1-2 minutes)"
echo ""

$PYTHON_CMD scripts/run_eval.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
else
    echo ""
    echo "❌ Evaluation failed! Please check the error messages above."
    exit 1
fi
echo ""

# Step 9: Summary
echo "=========================================="
echo "✅ ALL STEPS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "📁 Generated files:"
echo "  - data/processed/train_processed.csv"
echo "  - data/processed/test_processed.csv"
echo "  - models/logreg_selected.joblib"
echo "  - models/xgb_selected.joblib"
echo "  - models/thresholds.json"
echo ""
echo "🎯 Next steps:"
echo "  1. View results in the terminal output above"
echo "  2. Launch dashboard: streamlit run dashboard.py"
echo "  3. Open notebooks: jupyter lab notebooks/03_implementation_details.ipynb"
echo ""
echo "💡 To run dashboard, use:"
echo "   source .venv/bin/activate"
echo "   streamlit run dashboard.py"
echo ""
echo "=========================================="

