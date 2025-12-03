#!/usr/bin/env python3
"""
Complete Setup and Run Script (Cross-Platform)
This script does everything: setup, install, train, evaluate
Works on Mac, Linux, and Windows
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a command and return success status"""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python():
    """Check if Python is available"""
    print("🔍 Checking Python installation...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ ERROR: Python 3.8+ required. Found Python {version.major}.{version.minor}")
        return False
    print(f"✅ Found Python {version.major}.{version.minor}.{version.micro}")
    return True

def setup_venv():
    """Create and activate virtual environment"""
    venv_path = Path(".venv")
    venv_python = venv_path / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    print("🔧 Setting up virtual environment...")
    if not venv_path.exists():
        print("   Creating new virtual environment...")
        success, output = run_command([sys.executable, "-m", "venv", ".venv"])
        if not success:
            print(f"❌ ERROR: Failed to create virtual environment: {output}")
            return False
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
    
    return True

def install_requirements():
    """Install all required packages"""
    venv_python = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    print("⬆️  Upgrading pip...")
    success, _ = run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    if not success:
        print("⚠️  Warning: pip upgrade failed, continuing anyway...")
    
    print("📦 Installing required packages...")
    print("   (This may take a few minutes...)")
    success, output = run_command([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
    if not success:
        print(f"❌ ERROR: Failed to install packages: {output}")
        return False
    print("✅ All packages installed")
    return True

def check_data():
    """Verify data file exists"""
    print("🔍 Checking data files...")
    data_path = Path("data/raw/diabetic_data.csv")
    if not data_path.exists():
        print(f"❌ ERROR: {data_path} not found!")
        print("   Please make sure the data file is in the correct location.")
        return False
    print("✅ Data file found")
    return True

def run_training():
    """Run the training script"""
    venv_python = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    print("==========================================")
    print("🚀 Starting Training Process")
    print("==========================================")
    print()
    print("This will:")
    print("  - Clean and preprocess the data")
    print("  - Train Logistic Regression model")
    print("  - Train XGBoost model")
    print("  - Save all models and processed data")
    print()
    print("⏳ Training in progress... (This takes 2-5 minutes)")
    print()
    
    # Set OMP_NUM_THREADS to avoid potential issues
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    success, output = run_command([str(venv_python), "scripts/run_train.py"], shell=False)
    if not success:
        print()
        print(f"❌ Training failed! Error: {output}")
        return False
    
    print()
    print("✅ Training completed successfully!")
    return True

def run_evaluation():
    """Run the evaluation script"""
    venv_python = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    print("==========================================")
    print("📊 Running Model Evaluation")
    print("==========================================")
    print()
    print("⏳ Evaluating models... (This takes 1-2 minutes)")
    print()
    
    # Set OMP_NUM_THREADS to avoid potential issues
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    success, output = run_command([str(venv_python), "scripts/run_eval.py"], shell=False)
    if not success:
        print()
        print(f"❌ Evaluation failed! Error: {output}")
        return False
    
    print()
    print("✅ Evaluation completed successfully!")
    return True

def main():
    """Main execution function"""
    print("==========================================")
    print("Diabetes Readmission Prediction - Setup & Run")
    print("==========================================")
    print()
    
    # Change to script directory
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    print(f"📁 Project directory: {script_dir}")
    print()
    
    # Step 1: Check Python
    if not check_python():
        sys.exit(1)
    print()
    
    # Step 2: Setup virtual environment
    if not setup_venv():
        sys.exit(1)
    print()
    
    # Step 3: Install requirements
    if not install_requirements():
        sys.exit(1)
    print()
    
    # Step 4: Check data
    if not check_data():
        sys.exit(1)
    print()
    
    # Step 5: Run training
    if not run_training():
        sys.exit(1)
    print()
    
    # Step 6: Run evaluation
    if not run_evaluation():
        sys.exit(1)
    print()
    
    # Step 7: Summary
    print("==========================================")
    print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("==========================================")
    print()
    print("📁 Generated files:")
    print("  - data/processed/train_processed.csv")
    print("  - data/processed/test_processed.csv")
    print("  - models/logreg_selected.joblib")
    print("  - models/xgb_selected.joblib")
    print("  - models/thresholds.json")
    print()
    print("🎯 Next steps:")
    print("  1. View results in the terminal output above")
    print("  2. Launch dashboard: streamlit run dashboard.py")
    print("  3. Open notebooks: jupyter lab notebooks/03_implementation_details.ipynb")
    print()
    
    if os.name == "nt":
        print("💡 To run dashboard (Windows):")
        print("   .venv\\Scripts\\activate")
        print("   streamlit run dashboard.py")
    else:
        print("💡 To run dashboard (Mac/Linux):")
        print("   source .venv/bin/activate")
        print("   streamlit run dashboard.py")
    print()
    print("==========================================")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)

