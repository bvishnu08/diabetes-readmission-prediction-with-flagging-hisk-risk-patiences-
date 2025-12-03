#!/bin/bash
# Complete Download and Run Script for Mac/Linux
# This script downloads the repository from GitHub and runs everything automatically

set -e  # Exit on any error

echo "=========================================="
echo "Diabetes Readmission Prediction"
echo "Complete Download & Run Script"
echo "=========================================="
echo ""

# Repository URL
REPO_URL="https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git"
REPO_NAME="diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-"

# Check if repository is already cloned
if [ -d "$REPO_NAME" ]; then
    echo "📁 Repository folder already exists: $REPO_NAME"
    echo "   Using existing folder..."
    cd "$REPO_NAME"
else
    echo "📥 Downloading repository from GitHub..."
    echo "   URL: $REPO_URL"
    echo ""
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        echo "❌ ERROR: Git is not installed!"
        echo "   Please install Git first:"
        echo "   - Mac: brew install git"
        echo "   - Linux: sudo apt-get install git"
        echo "   Or download ZIP from: $REPO_URL"
        exit 1
    fi
    
    # Clone the repository
    git clone "$REPO_URL" "$REPO_NAME"
    
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to clone repository!"
        echo "   Please check your internet connection and try again."
        exit 1
    fi
    
    echo "✅ Repository downloaded successfully!"
    cd "$REPO_NAME"
fi

echo ""
echo "📁 Working directory: $(pwd)"
echo ""

# Now run the setup and training script
echo "=========================================="
echo "🚀 Starting Setup and Training"
echo "=========================================="
echo ""

# Check if run_all.sh exists
if [ -f "run_all.sh" ]; then
    echo "✅ Found run_all.sh script"
    echo "   Running automated setup..."
    echo ""
    chmod +x run_all.sh
    ./run_all.sh
else
    echo "⚠️  run_all.sh not found, running manual setup..."
    echo ""
    
    # Manual setup steps
    # Step 1: Check Python
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
    
    # Step 2: Create virtual environment
    echo "🔧 Setting up virtual environment..."
    if [ ! -d ".venv" ]; then
        $PYTHON_CMD -m venv .venv
        echo "✅ Virtual environment created"
    else
        echo "✅ Virtual environment already exists"
    fi
    echo ""
    
    # Step 3: Activate and install
    echo "🔌 Activating virtual environment..."
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo ""
    
    echo "📦 Installing required packages..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    echo "✅ All packages installed"
    echo ""
    
    # Step 4: Verify data
    if [ ! -f "data/raw/diabetic_data.csv" ]; then
        echo "❌ ERROR: data/raw/diabetic_data.csv not found!"
        exit 1
    fi
    echo "✅ Data file found"
    echo ""
    
    # Step 5: Run training
    echo "🚀 Starting training..."
    export OMP_NUM_THREADS=1
    python scripts/run_train.py
    echo ""
    
    # Step 6: Run evaluation
    echo "📊 Running evaluation..."
    python scripts/run_eval.py
    echo ""
fi

echo "=========================================="
echo "✅ COMPLETE! Everything is ready!"
echo "=========================================="
echo ""
echo "📁 Project location: $(pwd)"
echo ""
echo "🎯 Next steps:"
echo "  1. View results in the terminal output above"
echo "  2. Launch dashboard:"
echo "     cd $(pwd)"
echo "     source .venv/bin/activate"
echo "     streamlit run dashboard.py"
echo ""
echo "=========================================="

