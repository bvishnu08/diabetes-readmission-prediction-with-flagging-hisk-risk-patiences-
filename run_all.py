#!/usr/bin/env python3
"""
================================================================================
COMPLETE SETUP AND RUN SCRIPT - THE MASTER SCRIPT THAT DOES EVERYTHING!
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This is the main script you run to set up and run the entire project.
It automates everything so you don't have to run multiple commands manually.

STEPS IT PERFORMS:
1. Checks that Python is installed (needs Python 3.8 or higher)
2. Creates a virtual environment (.venv folder) - this keeps packages isolated
3. Installs all required packages from requirements.txt
4. Verifies that the data file exists
5. Trains both machine learning models (Logistic Regression + XGBoost)
6. Evaluates the models and shows results
7. Prints a summary of what was created

HOW TO USE:
-----------
Just run this one command from the project root directory:
    python run_all.py

That's it! Everything happens automatically.

WORKS ON:
---------
- Mac (macOS)
- Linux
- Windows (Command Prompt, PowerShell, Git Bash)

TIME TO COMPLETE:
-----------------
About 5-10 minutes total (most time is downloading/installing packages)
"""

# ============================================================================
# IMPORTS - These are Python libraries we need to use
# ============================================================================
import sys          # For checking Python version and exiting
import subprocess    # For running terminal commands from Python
import os           # For checking operating system and environment variables
from pathlib import Path  # For working with file paths (works on all OS)

# ============================================================================
# HELPER FUNCTION: Run a Terminal Command
# ============================================================================
def run_command(cmd, shell=False, env=None):
    """
    WHAT THIS DOES:
    Runs a terminal command (like "python --version" or "pip install pandas")
    and returns whether it succeeded or failed.
    
    PARAMETERS:
    - cmd: The command to run (can be a string or list of strings)
    - shell: Whether to run in shell mode (usually False)
    - env: Environment variables to pass (like OMP_NUM_THREADS)
    
    RETURNS:
    - (True, output) if command succeeded
    - (False, error_message) if command failed
    
    EXAMPLE:
    success, output = run_command(["python", "--version"])
    if success:
        print(f"Python version: {output}")
    """
    try:
        # Convert string commands to list format (needed for subprocess)
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        # Run the command and capture the output
        # check=True means it will raise an error if command fails
        # capture_output=True means we get the output text
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True, env=env)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        # Command failed - return the error message
        return False, e.stderr

def check_python():
    """
    WHAT THIS DOES:
    Checks if Python is installed and if it's the right version (3.8 or higher).
    We need Python 3.8+ because that's when some features we use were added.
    
    WHY WE NEED THIS:
    If Python isn't installed or is too old, nothing else will work.
    Better to check early and give a clear error message.
    
    RETURNS:
    - True if Python 3.8+ is found
    - False if Python is missing or too old
    """
    print("🔍 Checking Python installation...")
    
    # Get the Python version that's currently running this script
    version = sys.version_info  # This gives us (major, minor, micro) like (3, 9, 7)
    
    # Check if version is too old (need at least 3.8)
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ ERROR: Python 3.8+ required. Found Python {version.major}.{version.minor}")
        print("   Please install Python 3.8 or higher from: https://www.python.org/downloads/")
        return False
    
    # Python version is good!
    print(f"✅ Found Python {version.major}.{version.minor}.{version.micro}")
    return True

def setup_venv():
    """
    WHAT THIS DOES:
    Creates a virtual environment (a folder called .venv with its own Python installation).
    
    WHY WE NEED THIS:
    A virtual environment keeps this project's packages separate from other Python projects.
    This prevents conflicts - for example, if Project A needs pandas 1.0 and Project B needs
    pandas 2.0, they can both exist without breaking each other.
    
    HOW IT WORKS:
    - Creates a .venv folder in the project root
    - Copies Python into that folder
    - When activated, Python uses packages from .venv instead of system Python
    
    RETURNS:
    - True if virtual environment was created or already exists
    - False if creation failed
    """
    # Path to the virtual environment folder
    venv_path = Path(".venv")
    
    # Path to Python inside the virtual environment
    # Windows uses "Scripts", Mac/Linux use "bin"
    venv_python = venv_path / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    print("🔧 Setting up virtual environment...")
    print("   (This creates an isolated Python environment for this project)")
    
    # Check if virtual environment already exists
    if not venv_path.exists():
        print("   Creating new virtual environment...")
        # Run: python -m venv .venv
        # This creates the .venv folder with a fresh Python installation
        success, output = run_command([sys.executable, "-m", "venv", ".venv"])
        if not success:
            print(f"❌ ERROR: Failed to create virtual environment: {output}")
            print("   Make sure you have write permissions in this directory.")
            return False
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
        print("   (Skipping creation - using existing .venv folder)")
    
    return True

def install_requirements():
    """
    WHAT THIS DOES:
    Installs all the Python packages needed for this project.
    Reads requirements.txt and installs everything listed there.
    
    PACKAGES INSTALLED:
    - pandas: For working with data tables (CSV files)
    - scikit-learn: For machine learning models (Logistic Regression)
    - xgboost: For the XGBoost model
    - streamlit: For the interactive web dashboard
    - matplotlib, seaborn: For making charts and graphs
    - joblib: For saving/loading trained models
    - And more...
    
    WHY WE UPGRADE PIP FIRST:
    pip is the package installer. Upgrading it ensures we have the latest version,
    which can install packages faster and handle dependencies better.
    
    RETURNS:
    - True if all packages installed successfully
    - False if installation failed
    """
    # Path to Python inside the virtual environment
    venv_python = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    # Step 1: Upgrade pip to the latest version
    print("⬆️  Upgrading pip (package installer)...")
    print("   (This ensures we have the latest version for better package installation)")
    success, _ = run_command([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    if not success:
        # If pip upgrade fails, it's usually okay - we can still try to install packages
        print("⚠️  Warning: pip upgrade failed, continuing anyway...")
        print("   (This is usually fine - we'll try installing packages with current pip)")
    
    # Step 2: Install all packages from requirements.txt
    print("📦 Installing required packages...")
    print("   (This reads requirements.txt and installs: pandas, scikit-learn, xgboost, streamlit, etc.)")
    print("   (This may take 2-5 minutes depending on your internet speed...)")
    
    # Run: .venv/bin/python -m pip install -r requirements.txt
    # This installs all packages listed in requirements.txt into the virtual environment
    success, output = run_command([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
    if not success:
        print(f"❌ ERROR: Failed to install packages: {output}")
        print("   Common causes:")
        print("   - No internet connection")
        print("   - requirements.txt file is missing")
        print("   - Disk space is full")
        return False
    
    print("✅ All packages installed successfully!")
    return True

def check_data():
    """
    WHAT THIS DOES:
    Checks if the raw data file exists before we try to train models.
    
    WHY WE NEED THIS:
    If the data file is missing, training will fail with a confusing error.
    Better to check early and give a clear message about where to get the data.
    
    THE DATA FILE:
    - Location: data/raw/diabetic_data.csv
    - Contains: ~100,000 patient records from UCI ML Repository
    - Source: Diabetes 130-US hospitals dataset
    
    RETURNS:
    - True if data file exists
    - False if data file is missing
    """
    print("🔍 Checking data files...")
    print("   (Looking for the dataset we'll use to train the models)")
    
    data_path = Path("data/raw/diabetic_data.csv")
    
    if not data_path.exists():
        print(f"❌ ERROR: {data_path} not found!")
        print("   The data file is required to train the models.")
        print("   Please make sure the data file is in the correct location.")
        print()
        print("   If the file is missing, download it from:")
        print("   https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008")
        print("   Then place it in: data/raw/diabetic_data.csv")
        return False
    
    print("✅ Data file found")
    print(f"   Location: {data_path}")
    return True

def run_training():
    """
    WHAT THIS DOES:
    Runs the training script that trains both machine learning models.
    
    WHAT HAPPENS DURING TRAINING:
    1. Loads raw data from data/raw/diabetic_data.csv
    2. Cleans the data (removes missing values, fixes formatting issues)
    3. Splits data into training (80%) and testing (20%) sets
    4. Selects the best features for each model:
       - Logistic Regression: Top 20 features
       - XGBoost: Top 25 features
    5. Trains Logistic Regression model
    6. Trains XGBoost model
    7. Finds the best threshold for each model (sweeps from 0.05 to 0.95)
    8. Saves everything to disk:
       - models/logreg_selected.joblib (Logistic Regression model)
       - models/xgb_selected.joblib (XGBoost model)
       - models/thresholds.json (thresholds and feature lists)
       - data/processed/train_processed.csv (cleaned training data)
       - data/processed/test_processed.csv (cleaned test data)
    
    WHY OMP_NUM_THREADS=1:
    Some libraries (like XGBoost) can have issues with multi-threading in certain
    environments. Setting this to 1 prevents those errors. It might be slightly
    slower but more reliable.
    
    TIME:
    Usually takes 2-5 minutes depending on your computer speed.
    
    RETURNS:
    - True if training completed successfully
    - False if training failed
    """
    # Path to Python inside the virtual environment
    venv_python = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    print("==========================================")
    print("🚀 Starting Training Process")
    print("==========================================")
    print()
    print("This will:")
    print("  1. Load and clean the raw data")
    print("  2. Split data into training (80%) and testing (20%)")
    print("  3. Select best features for each model")
    print("  4. Train Logistic Regression model (top 20 features)")
    print("  5. Train XGBoost model (top 25 features)")
    print("  6. Find best thresholds for each model")
    print("  7. Save models and processed data to disk")
    print()
    print("⏳ Training in progress... (This takes 2-5 minutes)")
    print("   (You'll see progress messages from the training script)")
    print()
    
    # Set OMP_NUM_THREADS to avoid potential multi-threading issues
    # This is a workaround for some environments that have issues with
    # OpenMP (a library used for parallel processing)
    env = os.environ.copy()  # Copy current environment variables
    env["OMP_NUM_THREADS"] = "1"  # Force single-threaded mode
    
    # Run the training script using Python from the virtual environment
    # This calls scripts/run_train.py which does all the actual training
    success, output = run_command([str(venv_python), "scripts/run_train.py"], shell=False, env=env)
    
    if not success:
        print()
        print(f"❌ Training failed! Error: {output}")
        print("   Common causes:")
        print("   - Data file is missing or corrupted")
        print("   - Not enough disk space to save models")
        print("   - Memory error (dataset is large)")
        return False
    
    print()
    print("✅ Training completed successfully!")
    print("   Models saved to: models/")
    print("   Processed data saved to: data/processed/")
    return True

def run_evaluation():
    """
    WHAT THIS DOES:
    Tests the trained models on the test data and shows performance metrics.
    
    WHAT HAPPENS DURING EVALUATION:
    1. Loads the test data from data/processed/test_processed.csv
    2. Loads both trained models from models/ folder
    3. Loads the thresholds from models/thresholds.json
    4. Makes predictions on the test data using both models
    5. Calculates performance metrics:
       - Accuracy: Overall correctness
       - Recall: How many readmissions we catch (we want this HIGH)
       - Precision: How many predictions are correct
       - F1-Score: Balance between precision and recall
       - ROC-AUC: Overall model quality
    6. Prints a side-by-side comparison
    7. Recommends which model to use for deployment
    
    WHY WE EVALUATE:
    We need to know how good our models are before using them in real life.
    Evaluation tells us if the models are catching enough high-risk patients.
    
    TIME:
    Usually takes 1-2 minutes.
    
    RETURNS:
    - True if evaluation completed successfully
    - False if evaluation failed
    """
    # Path to Python inside the virtual environment
    venv_python = Path(".venv") / ("Scripts" if os.name == "nt" else "bin") / "python"
    
    print("==========================================")
    print("📊 Running Model Evaluation")
    print("==========================================")
    print()
    print("This will:")
    print("  1. Load the test data and trained models")
    print("  2. Make predictions on the test set")
    print("  3. Calculate performance metrics (accuracy, recall, precision, etc.)")
    print("  4. Compare both models side-by-side")
    print("  5. Recommend which model to use")
    print()
    print("⏳ Evaluating models... (This takes 1-2 minutes)")
    print()
    
    # Set OMP_NUM_THREADS to avoid potential multi-threading issues
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    # Run the evaluation script using Python from the virtual environment
    # This calls scripts/run_eval.py which does all the actual evaluation
    success, output = run_command([str(venv_python), "scripts/run_eval.py"], shell=False, env=env)
    
    if not success:
        print()
        print(f"❌ Evaluation failed! Error: {output}")
        print("   Common causes:")
        print("   - Models weren't trained yet (run training first)")
        print("   - Model files are missing or corrupted")
        print("   - Test data file is missing")
        return False
    
    print()
    print("✅ Evaluation completed successfully!")
    print("   Check the output above to see model performance metrics.")
    return True

def main():
    """
    ============================================================================
    MAIN EXECUTION FUNCTION - THIS IS WHERE EVERYTHING STARTS
    ============================================================================
    
    WHAT THIS DOES:
    This is the main function that orchestrates the entire setup and run process.
    It calls each step in order and stops if any step fails.
    
    EXECUTION FLOW:
    1. Change to project directory (so all paths work correctly)
    2. Check Python version
    3. Create virtual environment
    4. Install packages
    5. Check data file exists
    6. Train models
    7. Evaluate models
    8. Print summary
    
    ERROR HANDLING:
    If any step fails, the script exits with an error code (1).
    This makes it clear that something went wrong.
    """
    print("==========================================")
    print("Diabetes Readmission Prediction - Setup & Run")
    print("==========================================")
    print("This script will set up everything and train the models automatically.")
    print()
    
    # Change to the directory where this script is located
    # This ensures all relative paths (like "data/raw/") work correctly
    # no matter where you run the script from
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    print(f"📁 Project directory: {script_dir}")
    print("   (All file paths will be relative to this directory)")
    print()
    
    # ========================================================================
    # STEP 1: Check Python Version
    # ========================================================================
    # We need Python 3.8+ for this project. Check early and fail fast if wrong version.
    print("STEP 1/6: Checking Python installation...")
    if not check_python():
        print("\n❌ Setup failed at Step 1: Python version check")
        sys.exit(1)  # Exit with error code 1
    print()
    
    # ========================================================================
    # STEP 2: Setup Virtual Environment
    # ========================================================================
    # Create an isolated Python environment so packages don't conflict with other projects.
    print("STEP 2/6: Setting up virtual environment...")
    if not setup_venv():
        print("\n❌ Setup failed at Step 2: Virtual environment creation")
        sys.exit(1)
    print()
    
    # ========================================================================
    # STEP 3: Install Required Packages
    # ========================================================================
    # Install all Python packages needed for this project (pandas, scikit-learn, etc.)
    print("STEP 3/6: Installing required packages...")
    if not install_requirements():
        print("\n❌ Setup failed at Step 3: Package installation")
        sys.exit(1)
    print()
    
    # ========================================================================
    # STEP 4: Check Data File Exists
    # ========================================================================
    # Verify the raw data file is present before trying to train models.
    print("STEP 4/6: Checking data files...")
    if not check_data():
        print("\n❌ Setup failed at Step 4: Data file check")
        sys.exit(1)
    print()
    
    # ========================================================================
    # STEP 5: Train Models
    # ========================================================================
    # This is the main work: train both Logistic Regression and XGBoost models.
    print("STEP 5/6: Training models...")
    if not run_training():
        print("\n❌ Setup failed at Step 5: Model training")
        sys.exit(1)
    print()
    
    # ========================================================================
    # STEP 6: Evaluate Models
    # ========================================================================
    # Test the trained models and show performance metrics.
    print("STEP 6/6: Evaluating models...")
    if not run_evaluation():
        print("\n❌ Setup failed at Step 6: Model evaluation")
        sys.exit(1)
    print()
    
    # ========================================================================
    # SUCCESS SUMMARY
    # ========================================================================
    print("==========================================")
    print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
    print("==========================================")
    print()
    print("📁 Generated files (you can find these in your project folder):")
    print("  ✅ data/processed/train_processed.csv  (Cleaned training data)")
    print("  ✅ data/processed/test_processed.csv   (Cleaned test data)")
    print("  ✅ models/logreg_selected.joblib       (Trained Logistic Regression)")
    print("  ✅ models/xgb_selected.joblib          (Trained XGBoost)")
    print("  ✅ models/thresholds.json              (Best thresholds and features)")
    print()
    print("=" * 70)
    print("📋 WHAT TO DO NEXT - STEP BY STEP (FOR BEGINNERS)")
    print("=" * 70)
    print()
    print("👋 Don't worry if you're new to this! Everything is explained simply below.")
    print()
    print("STEP 1: ✅ YOU ALREADY SAW THE RESULTS!")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   Scroll up in this terminal window and look for 'MODEL EVALUATION'.")
    print("   You'll see numbers showing how good your models are!")
    print("   ✅ This step is already done - you don't need to do anything!")
    print()
    print("STEP 2: VERIFY EVERYTHING WORKED (Takes 10 seconds)")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   Type this command (then press Enter):")
    print("   → python test_models.py")
    print()
    print("   What this does: Checks that all files were created correctly")
    print("   What you'll see: A list with ✅ checkmarks next to each file")
    print("   If you see '✅ ALL CHECKS PASSED!' - you're good to go!")
    print()
    print("STEP 3: LAUNCH THE INTERACTIVE DASHBOARD (The Fun Part! 🎉)")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   This opens a website in your browser with beautiful charts!")
    print()
    if os.name == "nt":
        # Windows instructions
        print("   For Windows - Type these commands ONE AT A TIME (press Enter after each):")
        print()
        print("   Command 1: Activate the virtual environment")
        print("   → .venv\\Scripts\\activate")
        print("   (You'll see (.venv) appear at the start of your prompt - that means it worked!)")
        print()
        print("   Command 2: Run the dashboard")
        print("   → streamlit run dashboard.py")
        print("   (Your browser will open automatically at http://localhost:8501)")
        print()
        print("   To stop the dashboard: Press Ctrl+C in this terminal")
    else:
        # Mac/Linux instructions
        print("   For Mac/Linux - Type these commands ONE AT A TIME (press Enter after each):")
        print()
        print("   Command 1: Activate the virtual environment")
        print("   → source .venv/bin/activate")
        print("   (You'll see (.venv) appear at the start of your prompt - that means it worked!)")
        print()
        print("   Command 2: Run the dashboard")
        print("   → streamlit run dashboard.py")
        print("   (Your browser will open automatically at http://localhost:8501)")
        print()
        print("   To stop the dashboard: Press Ctrl+C in this terminal")
    print()
    print("STEP 4: VIEW FULL RESULTS AND METRICS ⭐ IMPORTANT!")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   To see the complete evaluation results with all metrics, confusion matrices,")
    print("   and clinical interpretation, follow these steps:")
    print()
    if os.name == "nt":
        # Windows instructions
        print("   Command 1: Activate the virtual environment")
        print("   → .venv\\Scripts\\activate")
        print()
        print("   Command 2: Run evaluation to see full results")
        print("   → python scripts/run_eval.py")
        print()
        print("   This will show you:")
        print("   ✅ All metrics (Threshold, ROC-AUC, Accuracy, Recall, Precision, F1-Score)")
        print("   ✅ Confusion matrices for both models")
        print("   ✅ Classification reports")
        print("   ✅ Clinical interpretation (HIGH RISK vs LOW RISK patients)")
        print("   ✅ Model recommendation")
    else:
        # Mac/Linux instructions
        print("   Command 1: Activate the virtual environment")
        print("   → source .venv/bin/activate")
        print()
        print("   Command 2: Run evaluation to see full results")
        print("   → python scripts/run_eval.py")
        print()
        print("   This will show you:")
        print("   ✅ All metrics (Threshold, ROC-AUC, Accuracy, Recall, Precision, F1-Score)")
        print("   ✅ Confusion matrices for both models")
        print("   ✅ Classification reports")
        print("   ✅ Clinical interpretation (HIGH RISK vs LOW RISK patients)")
        print("   ✅ Model recommendation")
    print()
    print("STEP 5: EXPLORE THE CODE (Optional - Only if you want to learn)")
    print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   This opens an interactive notebook to see how the code works.")
    print("   First activate the virtual environment (see Step 3, Command 1), then:")
    print("   → pip install jupyter")
    print("   → jupyter lab notebooks/03_implementation_details.ipynb")
    print()
    print("=" * 70)
    print("💡 TIP: For detailed beginner-friendly instructions, see README.md")
    print("   (It explains everything in simple terms, like teaching a friend!)")
    print("=" * 70)
    print()
    print("🎉 Project setup and training complete!")
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

