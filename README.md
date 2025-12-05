# 🏥 Diabetes 30-Day Readmission Prediction Project

> **MSBA-265 Course Project**  
> Predicts whether diabetic patients will be readmitted to the hospital within 30 days

---

## 📋 Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Quick Start - Run Everything in One Command](#quick-start)
3. [Getting the Repository](#getting-the-repository)
4. [How to Run Everything](#how-to-run-everything)
5. [How to Check Results](#how-to-check-results)
6. [Understanding the Terminal Files](#understanding-the-terminal-files)
7. [Project Structure](#project-structure)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 What This Project Does

This project builds machine learning models to predict if a diabetic patient will be readmitted to the hospital within 30 days of discharge. This helps hospitals:

- **Flag high-risk patients early** so care managers can intervene
- **Improve patient outcomes** by providing extra care to those who need it
- **Avoid penalties** from CMS (Centers for Medicare & Medicaid Services)

### Models We Built

| Model | Why We Use It | Features Used |
|-------|---------------|----------------|
| **Logistic Regression** | Easy to understand - doctors can see which factors matter most | Top 20 features |
| **XGBoost** | More accurate - catches more readmissions correctly | Top 25 features |

**Result:** XGBoost is recommended for deployment because it's more accurate while still catching most high-risk patients.

---

## 🚀 Quick Start

### **EASIEST WAY: One Command Does Everything!**

After you get the repository (see next section), just run:

```bash
python run_all.py
```

**That's it!** This single command will:
- ✅ Create a virtual environment (isolated Python environment)
- ✅ Install all required packages
- ✅ Clean and preprocess the data
- ✅ Train both models (Logistic Regression + XGBoost)
- ✅ Evaluate the models
- ✅ Show you the results

**Time:** 5-10 minutes total

---

## 📥 Getting the Repository

You have two options to get the code:

### **Option 1: Download ZIP (Easiest - No Git Needed)**

**Best for:** Windows users, beginners, or if you don't have Git installed

**Steps:**

1. **Go to GitHub:**
   - Open: https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
   - Click the green **"Code"** button
   - Click **"Download ZIP"**

2. **Extract the ZIP file:**
   - **Windows:** Right-click → "Extract All"
   - **Mac:** Double-click the ZIP file
   - **Extract to a SHORT path** (important for Windows!):
     - ✅ Good: `C:\Projects\diabetes-project`
     - ❌ Bad: `C:\Users\YourName\Documents\Very\Long\Path\Name\...`

3. **Rename the folder** to something short (e.g., `diabetes-project`)

4. **Open Terminal/Command Prompt** in that folder:
   - **Windows:** Right-click folder → "Open in Terminal" or "Open PowerShell here"
   - **Mac/Linux:** Right-click folder → "New Terminal at Folder"

5. **Run the project:**
   ```bash
   python run_all.py
   ```

### **Option 2: Git Clone (For Advanced Users)**

**Best for:** Users familiar with Git, or if you want to update the code easily

**Steps:**

1. **Open Terminal/Command Prompt**

2. **Navigate to a short path:**
   ```bash
   # Windows
   cd C:\Projects
   
   # Mac/Linux
   cd ~/Projects
   ```

3. **Clone the repository:**
   ```bash
   git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
   ```

4. **Enter the project folder:**
   ```bash
   cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
   ```

5. **Run the project:**
   ```bash
   python run_all.py
   ```

---

## 🎯 How to Run Everything

### **Method 1: Automated Script (Recommended - One Command)**

This is the easiest way! Just run:

```bash
python run_all.py
```

**What this does:**
- Creates a virtual environment (`.venv` folder) - this keeps your project's packages separate from other Python projects
- Installs all required packages from `requirements.txt` (pandas, scikit-learn, xgboost, streamlit, etc.)
- Runs the training script (`scripts/run_train.py`) - this trains both models
- Runs the evaluation script (`scripts/run_eval.py`) - this tests the models and shows results
- Prints a summary of what was created

**Expected Output:**
```
==========================================
Diabetes Readmission Prediction - Setup & Run
==========================================

🔍 Checking Python installation...
✅ Found Python 3.9.7

🔧 Setting up virtual environment...
✅ Virtual environment created

📦 Installing required packages...
✅ All packages installed

🚀 Starting Training Process
...
✅ Training completed successfully!

📊 Running Model Evaluation
...
✅ Evaluation completed successfully!

✅ ALL STEPS COMPLETED SUCCESSFULLY!
```

### **Method 2: Step-by-Step (Manual - If You Want More Control)**

If you want to understand each step, or if the automated script has issues, you can run each step manually:

#### **Step 1: Create Virtual Environment**

This creates an isolated Python environment so your project's packages don't conflict with other projects.

```bash
# Create the virtual environment
python -m venv .venv
```

**What this does:** Creates a folder called `.venv` with a fresh Python installation just for this project.

#### **Step 2: Activate Virtual Environment**

This "turns on" the virtual environment so Python uses the packages in `.venv` instead of your system Python.

```bash
# Mac/Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**What this does:** Changes your terminal prompt to show `(.venv)` at the beginning, meaning the virtual environment is active.

**How to know it worked:** You should see `(.venv)` at the start of your terminal prompt.

#### **Step 3: Install Required Packages**

This downloads and installs all the Python libraries needed for this project.

```bash
pip install -r requirements.txt
```

**What this does:** Reads `requirements.txt` (which lists all needed packages) and installs them:
- `pandas` - for working with data tables
- `scikit-learn` - for machine learning models
- `xgboost` - for the XGBoost model
- `streamlit` - for the interactive dashboard
- `matplotlib`, `seaborn` - for making charts
- And more...

**Time:** 2-5 minutes depending on your internet speed.

#### **Step 4: Train the Models**

This is where the magic happens! This script:
- Loads the raw data from `data/raw/diabetic_data.csv`
- Cleans the data (removes missing values, fixes formatting)
- Splits data into training (80%) and testing (20%)
- Selects the best features for each model
- Trains Logistic Regression (top 20 features)
- Trains XGBoost (top 25 features)
- Finds the best threshold for each model
- Saves everything to disk

```bash
python scripts/run_train.py
```

**What this creates:**
- `data/processed/train_processed.csv` - cleaned training data
- `data/processed/test_processed.csv` - cleaned test data
- `models/logreg_selected.joblib` - trained Logistic Regression model
- `models/xgb_selected.joblib` - trained XGBoost model
- `models/thresholds.json` - best thresholds and feature lists for each model

**Time:** 2-5 minutes.

**Expected Output:**
```
[preprocess] Loading data from data/raw/diabetic_data.csv
[preprocess] Cleaning data...
[preprocess] Splitting into train/test (80/20)...
[preprocess] Saved train to data/processed/train_processed.csv
[preprocess] Saved test to data/processed/test_processed.csv
[train] Training Logistic Regression...
[train] Training XGBoost...
[train] Saved models and thresholds
✅ Training completed!
```

#### **Step 5: Evaluate the Models**

This tests the trained models on the test data and shows you how good they are.

```bash
python scripts/run_eval.py
```

**What this does:**
- Loads the test data
- Loads both trained models
- Makes predictions on the test data
- Calculates metrics (accuracy, recall, precision, F1-score, ROC-AUC)
- Prints a side-by-side comparison
- Recommends which model to use

**Time:** 1-2 minutes.

**Expected Output:**
```
MODEL EVALUATION – 30-Day Readmission Prediction
================================================

Logistic Regression (20 features):
  Threshold: 0.45
  Accuracy: 0.65
  Recall: 0.70
  Precision: 0.15
  F1-Score: 0.24
  ROC-AUC: 0.64

XGBoost (25 features):
  Threshold: 0.10
  Accuracy: 0.68
  Recall: 0.71
  Precision: 0.17
  F1-Score: 0.27
  ROC-AUC: 0.68

RECOMMENDATION: Use XGBoost for deployment
```

#### **Step 6: (Optional) Launch Interactive Dashboard**

This opens a web browser with an interactive dashboard where you can:
- See model performance charts
- Make predictions for new patients
- Explore feature importance

```bash
streamlit run dashboard.py
```

**What this does:**
- Starts a local web server
- Opens your browser to `http://localhost:8501`
- Shows an interactive dashboard with charts and predictions

**To stop:** Press `Ctrl+C` in the terminal.

---

## ✅ What to Do After Running `python run_all.py`

After you run `python run_all.py` and see "✅ ALL STEPS COMPLETED SUCCESSFULLY!", here's exactly what to do next:

### **Step 1: Check the Results in Terminal (Already Done!)**

✅ **You already saw the results!** When `run_all.py` finished, it automatically ran the evaluation and printed the results in your terminal. Look for output like this:

```
MODEL EVALUATION – 30-Day Readmission Prediction
================================================

Logistic Regression (20 features):
  Threshold: 0.45
  Accuracy: 0.65
  Recall: 0.70
  Precision: 0.15
  F1-Score: 0.24
  ROC-AUC: 0.64

XGBoost (25 features):
  Threshold: 0.10
  Accuracy: 0.68
  Recall: 0.71
  Precision: 0.17
  F1-Score: 0.27
  ROC-AUC: 0.68

RECOMMENDATION: Use XGBoost for deployment
```

**What this means:**
- Both models are trained and working
- XGBoost is recommended (better performance)
- Models catch ~70% of readmissions (good recall)

---

### **Step 2: Verify Everything Was Created (Optional but Recommended)**

Make sure all files were created correctly:

```bash
python test_models.py
```

**What this does:**
- Checks that all model files exist
- Verifies models can be loaded (not corrupted)
- Shows file sizes and feature counts

**Expected output:**
```
✅ models/logreg_selected.joblib (Size: 45.2 KB)
✅ models/xgb_selected.joblib (Size: 123.5 KB)
✅ models/thresholds.json (Size: 2.1 KB)
✅ data/processed/train_processed.csv (Size: 1250.3 KB)
✅ data/processed/test_processed.csv (Size: 312.8 KB)
✅ ALL CHECKS PASSED!
```

**If you see errors:** The models might not have trained correctly. Re-run `python run_all.py`.

---

### **Step 3: View Results Again (If You Want to See Them Again)**

If you want to see the evaluation results again (or if you missed them):

```bash
# First, activate the virtual environment (if not already activated)
# Windows:
.venv\Scripts\activate

# Mac/Linux:
source .venv/bin/activate

# Then run evaluation:
python scripts/run_eval.py
```

This will print the same results you saw when `run_all.py` finished.

---

### **Step 4: Launch the Interactive Dashboard (Recommended!)**

See your results visually with charts and graphs:

**For Windows:**
```bash
# Step 1: Activate virtual environment
.venv\Scripts\activate

# Step 2: Run the dashboard
streamlit run dashboard.py
```

**For Mac/Linux:**
```bash
# Step 1: Activate virtual environment
source .venv/bin/activate

# Step 2: Run the dashboard
streamlit run dashboard.py
```

**What happens:**
- A web browser will open automatically
- Go to: http://localhost:8501
- You'll see:
  - Model performance charts
  - ROC curves
  - Confusion matrices
  - Feature importance plots
  - Prediction playground (try making predictions!)

**To stop the dashboard:** Press `Ctrl+C` in the terminal.

---

### **Step 5: Explore the Code (Optional - For Learning)**

If you want to understand how everything works:

```bash
# Make sure virtual environment is activated
.venv\Scripts\activate      # Windows
# OR
source .venv/bin/activate    # Mac/Linux

# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter lab notebooks/03_implementation_details.ipynb
```

This opens an interactive notebook where you can:
- See the code step-by-step
- Run cells individually
- See data visualizations
- Understand the implementation

---

### **Quick Summary: What You Should Do**

**Minimum (to verify it worked):**
1. ✅ Check terminal output (already done - you saw the results!)
2. ✅ Run `python test_models.py` to verify files

**Recommended (to see everything):**
1. ✅ Check terminal output
2. ✅ Run `python test_models.py`
3. ✅ Launch dashboard: `streamlit run dashboard.py` (after activating venv)

**Optional (for deep dive):**
1. ✅ All of the above
2. ✅ Open Jupyter notebook to explore code

---

## 📊 How to Check Results

After running the project, here are three ways to see your results:

### **Method 1: Terminal Output (Automatic)**

When you run `python scripts/run_eval.py`, it automatically prints the results in your terminal:

```
MODEL EVALUATION – 30-Day Readmission Prediction
================================================

Logistic Regression Results:
  - Accuracy: 65%
  - Recall: 70% (catches 70% of readmissions)
  - Precision: 15% (15% of predictions are correct)
  - F1-Score: 0.24
  - ROC-AUC: 0.64

XGBoost Results:
  - Accuracy: 68%
  - Recall: 71% (catches 71% of readmissions)
  - Precision: 17% (17% of predictions are correct)
  - F1-Score: 0.27
  - ROC-AUC: 0.68

RECOMMENDATION: XGBoost is recommended for deployment
```

**What the metrics mean:**
- **Recall (Sensitivity):** How many actual readmissions we catch. Higher is better. We want this to be high (≥70%) so we don't miss high-risk patients.
- **Precision:** How many of our predictions are correct. Lower precision is okay if it means catching more true positives.
- **F1-Score:** Balance between precision and recall. Higher is better.
- **ROC-AUC:** Overall model quality. 0.5 = random guessing, 1.0 = perfect. Our models get ~0.65-0.68.

### **Method 2: Interactive Dashboard (Visual)**

Launch the dashboard to see charts and make predictions:

```bash
# Make sure virtual environment is activated first!
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate      # Windows

# Then run the dashboard
streamlit run dashboard.py
```

**What you'll see:**
- **Model Performance Tab:** Side-by-side comparison with charts
- **ROC Curves:** Visual comparison of model performance
- **Confusion Matrices:** Shows true positives, false positives, etc.
- **Feature Importance:** Which factors matter most for predictions
- **Prediction Playground:** Enter patient info and get a prediction

**Dashboard URL:** http://localhost:8501 (opens automatically)

### **Method 3: Verify Files Were Created**

Check that all the important files were created:

```bash
# Run the verification script
python test_models.py
```

**What this checks:**
- ✅ Models exist (`models/logreg_selected.joblib`, `models/xgb_selected.joblib`)
- ✅ Thresholds file exists (`models/thresholds.json`)
- ✅ Processed data exists (`data/processed/train_processed.csv`, `data/processed/test_processed.csv`)
- ✅ Models can be loaded (tests that they're not corrupted)
- ✅ Shows file sizes and feature counts

**Expected Output:**
```
============================================================
MODEL FILES VERIFICATION
============================================================

1. Checking file existence:
   ✅ models/logreg_selected.joblib (Size: 45.2 KB)
   ✅ models/xgb_selected.joblib (Size: 123.5 KB)
   ✅ models/thresholds.json (Size: 2.1 KB)
   ✅ data/processed/train_processed.csv (Size: 1250.3 KB)
   ✅ data/processed/test_processed.csv (Size: 312.8 KB)

2. Loading thresholds:
   ✅ Thresholds loaded successfully
   - Logistic Regression threshold: 0.45
   - XGBoost threshold: 0.10
   - LR selected features: 20 features
   - XGB selected features: 25 features

3. Testing model loading:
   ✅ Logistic Regression model loads successfully
   ✅ XGBoost model loads successfully

✅ ALL CHECKS PASSED! Everything is working correctly.
```

### **Method 4: Jupyter Notebooks (For Deep Dive)**

If you want to explore the code and data in detail:

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate      # Windows

# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter lab notebooks/03_implementation_details.ipynb
```

**What you'll see:**
- Interactive code cells you can run
- Data visualizations
- Step-by-step explanations
- Experiments and analysis

---

## 💻 Understanding the Terminal Files

Here's what each file does and when to use it:

### **`run_all.py` - The Master Script**

**What it does:** Runs everything automatically in the correct order.

**When to use:** This is the main script you should run. It does everything for you.

**What it contains:**
```python
# Step 1: Check Python version (needs 3.8+)
# Step 2: Create virtual environment (.venv folder)
# Step 3: Install packages from requirements.txt
# Step 4: Check that data file exists
# Step 5: Run training (scripts/run_train.py)
# Step 6: Run evaluation (scripts/run_eval.py)
# Step 7: Print summary
```

**How to run:**
```bash
python run_all.py
```

**Comments in the code explain:**
- Why we create a virtual environment (to isolate packages)
- What each step does
- How to handle errors
- What files are created

---

### **`scripts/run_train.py` - Training Script**

**What it does:** Trains both machine learning models.

**When to use:** 
- Automatically called by `run_all.py`
- Or run manually if you want to retrain models

**What it does step-by-step:**
```python
# 1. Sets up Python path so it can find our code modules
# 2. Imports the training function from src/train.py
# 3. Calls train_all_models() which:
#    - Loads raw data from data/raw/diabetic_data.csv
#    - Cleans the data (removes missing values, fixes formatting)
#    - Splits into 80% training, 20% testing
#    - Selects top 20 features for Logistic Regression
#    - Selects top 25 features for XGBoost
#    - Trains Logistic Regression model
#    - Trains XGBoost model
#    - Finds best threshold for each model (sweeps 0.05 to 0.95)
#    - Saves models to models/ folder
#    - Saves thresholds to models/thresholds.json
```

**How to run:**
```bash
# Make sure virtual environment is activated first!
python scripts/run_train.py
```

**Comments in the code explain:**
- Why we set up the Python path (so imports work from any directory)
- What the training function does
- What files are created

---

### **`scripts/run_eval.py` - Evaluation Script**

**What it does:** Tests the trained models and shows performance metrics.

**When to use:**
- Automatically called by `run_all.py`
- Or run manually after training to see results

**What it does step-by-step:**
```python
# 1. Sets up Python path so it can find our code modules
# 2. Imports the evaluation function from src/evaluate.py
# 3. Calls evaluate_all() which:
#    - Loads test data from data/processed/test_processed.csv
#    - Loads trained models from models/ folder
#    - Loads thresholds from models/thresholds.json
#    - Makes predictions on test data
#    - Calculates metrics (accuracy, recall, precision, F1, ROC-AUC)
#    - Prints side-by-side comparison
#    - Recommends which model to use
```

**How to run:**
```bash
# Make sure virtual environment is activated first!
python scripts/run_eval.py
```

**Comments in the code explain:**
- Why we need to evaluate (to see how good the models are)
- What metrics mean
- How the recommendation is made

---

### **`scripts/run_dashboard.py` - Dashboard Launcher**

**What it does:** Starts the Streamlit web dashboard.

**When to use:** When you want to see visual results and make predictions interactively.

**What it does step-by-step:**
```python
# 1. Finds the dashboard.py file in the project root
# 2. Checks that it exists
# 3. Runs: streamlit run dashboard.py
# 4. Opens browser to http://localhost:8501
```

**How to run:**
```bash
# Make sure virtual environment is activated first!
python scripts/run_dashboard.py

# OR directly:
streamlit run dashboard.py
```

**Comments in the code explain:**
- How it finds the dashboard file
- What port it runs on (8501)
- How to stop it (Ctrl+C)

---

### **`test_models.py` - Verification Script**

**What it does:** Checks that all files were created correctly and models can be loaded.

**When to use:** After training, to verify everything worked.

**What it checks:**
```python
# 1. Checks if model files exist:
#    - models/logreg_selected.joblib
#    - models/xgb_selected.joblib
#    - models/thresholds.json
#    - data/processed/train_processed.csv
#    - data/processed/test_processed.csv

# 2. Loads thresholds.json and displays:
#    - Threshold values for each model
#    - Number of features selected
#    - Feature names

# 3. Tests loading the models:
#    - Tries to load Logistic Regression model
#    - Tries to load XGBoost model
#    - Verifies they're not corrupted

# 4. Checks processed data:
#    - Reads train/test CSV files
#    - Shows row and column counts
```

**How to run:**
```bash
# Make sure virtual environment is activated first!
python test_models.py
```

**Comments in the code explain:**
- What each file is for
- Why we verify (to catch errors early)
- What to do if files are missing

---

### **`download_and_run.py` - Complete Bootstrap Script**

**What it does:** Downloads the repository from GitHub AND runs everything automatically.

**When to use:** If you don't have the code yet and want to download and run in one step.

**What it does step-by-step:**
```python
# 1. Checks if Git is installed
# 2. Clones repository from GitHub (or uses existing folder)
# 3. Changes into the repository folder
# 4. Calls run_all.py to set up and run everything
```

**How to run:**
```bash
# Run from any directory (it will download the repo)
python download_and_run.py
```

**Comments in the code explain:**
- How to handle Git not being installed (suggests ZIP download)
- How to handle Windows path length errors
- What to do if cloning fails

---

## 📁 Project Structure

Here's what each folder and file does:

```
diabetes-readmission-prediction/
│
├── README.md                    # This file! Complete guide to everything
├── requirements.txt             # List of all Python packages needed
├── run_all.py                   # Master script: runs everything automatically
├── download_and_run.py          # Downloads repo from GitHub and runs everything
├── test_models.py               # Verifies that models were created correctly
│
├── dashboard.py                 # Streamlit web dashboard (interactive UI)
│
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv    # Original dataset (100k+ patient records)
│   │   └── IDS_mapping.csv      # Mapping file for IDs
│   └── processed/
│       ├── train_processed.csv  # Cleaned training data (80% of data)
│       └── test_processed.csv   # Cleaned test data (20% of data)
│
├── models/
│   ├── logreg_selected.joblib   # Trained Logistic Regression model
│   ├── xgb_selected.joblib      # Trained XGBoost model
│   └── thresholds.json          # Best thresholds and feature lists
│
├── scripts/
│   ├── run_train.py             # Training script (trains both models)
│   ├── run_eval.py              # Evaluation script (tests models)
│   └── run_dashboard.py         # Dashboard launcher
│
├── src/
│   ├── config.py                # Configuration (file paths, feature lists)
│   ├── preprocess.py            # Data cleaning and splitting functions
│   ├── feature_selection.py    # Feature selection logic
│   ├── model.py                 # Model pipeline definitions
│   ├── train.py                 # Training logic (trains models, finds thresholds)
│   └── evaluate.py              # Evaluation logic (calculates metrics)
│
└── notebooks/
    ├── 01_eda.ipynb             # Exploratory data analysis
    ├── 02_modeling.ipynb        # Modeling experiments
    └── 03_implementation_details.ipynb  # Final implementation with explanations
```

**Key Files Explained:**

- **`requirements.txt`:** Lists all Python packages. When you run `pip install -r requirements.txt`, it installs everything needed.
- **`run_all.py`:** The main script you run. It does everything automatically.
- **`dashboard.py`:** The web interface. Run with `streamlit run dashboard.py`.
- **`src/config.py`:** Central configuration. Defines file paths, feature lists, model parameters.
- **`src/preprocess.py`:** Data cleaning functions. Removes missing values, fixes formatting.
- **`src/train.py`:** Training logic. Trains models, selects features, finds thresholds.
- **`src/evaluate.py`:** Evaluation logic. Tests models, calculates metrics, makes recommendations.

---

## 🔧 Troubleshooting

### **Problem: "Python not found" or "python: command not found"**

**Solution:**
- Make sure Python is installed: https://www.python.org/downloads/
- On Mac/Linux, try `python3` instead of `python`
- On Windows, make sure Python is added to PATH during installation

---

### **Problem: "pip: command not found"**

**Solution:**
- Python 3.4+ includes pip. If missing, install it:
  ```bash
  python -m ensurepip --upgrade
  ```
- Or use `python3 -m pip` instead of `pip`

---

### **Problem: "ModuleNotFoundError" or "No module named 'pandas' (or other package)"**

**Solution:**
- Make sure virtual environment is activated (you should see `(.venv)` in your terminal)
- Install packages: `pip install -r requirements.txt`
- If still failing, try: `python -m pip install -r requirements.txt`

---

### **Problem: "Filename too long" (Windows)**

**Solution:**
- This happens because Windows has a 260-character path limit
- **Best fix:** Download ZIP instead of using Git clone
- Extract to a SHORT path like `C:\Projects\diabetes-project`
- Rename the folder to something short

**Alternative:** Enable long paths in Windows:
1. Open PowerShell as Administrator
2. Run: `New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force`
3. Restart your computer

---

### **Problem: "OMP: Error #179" or "SHM2 failed"**

**Solution:**
- This is a harmless warning that can happen in some environments
- The script already handles this by setting `OMP_NUM_THREADS=1`
- If you see this error, it's already being worked around automatically
- You can ignore it

---

### **Problem: "data/raw/diabetic_data.csv not found"**

**Solution:**
- Make sure you're in the project root directory (where `README.md` is)
- Check that `data/raw/diabetic_data.csv` exists
- If missing, you need to download the dataset from UCI ML Repository:
  - Go to: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008
  - Download `diabetic_data.csv`
  - Place it in `data/raw/` folder

---

### **Problem: "Permission denied" when running scripts**

**Solution:**
- **Mac/Linux:** Make scripts executable:
  ```bash
  chmod +x run_all.py
  chmod +x scripts/*.py
  ```
- **Windows:** Usually not an issue, but try running as Administrator if needed

---

### **Problem: Dashboard won't open or "Address already in use"**

**Solution:**
- The dashboard runs on port 8501. If something else is using it:
  - Close other Streamlit apps
  - Or run on a different port: `streamlit run dashboard.py --server.port 8502`
- Make sure virtual environment is activated
- Check that `dashboard.py` exists in the project root

---

### **Problem: Models train but evaluation shows poor results**

**Solution:**
- This is normal! The models are designed to catch as many readmissions as possible (high recall), even if it means more false alarms (lower precision)
- Check the recall metric - it should be ≥70% (meaning we catch 70% of actual readmissions)
- The models prioritize catching high-risk patients over being perfectly accurate

---

### **Problem: Virtual environment activation doesn't work**

**Solution:**
- **Windows Command Prompt:** Use `.venv\Scripts\activate.bat`
- **Windows PowerShell:** You may need to allow scripts:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **Mac/Linux:** Make sure you're using `source .venv/bin/activate` (not just `.venv/bin/activate`)

---

### **Still Having Issues?**

1. **Check Python version:** Should be 3.8 or higher
   ```bash
   python --version
   ```

2. **Verify you're in the right directory:**
   ```bash
   # Should show README.md, run_all.py, etc.
   ls  # Mac/Linux
   dir  # Windows
   ```

3. **Try running step-by-step manually** (see Method 2 in "How to Run Everything")

4. **Check that data file exists:**
   ```bash
   # Should show the file
   ls data/raw/diabetic_data.csv  # Mac/Linux
   dir data\raw\diabetic_data.csv  # Windows
   ```

---

## 📚 Additional Information

### **What Metrics Mean**

- **Recall (Sensitivity):** Out of all actual readmissions, how many did we catch?  
  - 70% recall = We catch 70% of patients who will be readmitted
  - **We want this HIGH** (≥70%) so we don't miss high-risk patients

- **Precision:** Out of all our predictions of "will be readmitted," how many were correct?  
  - 15% precision = 15% of our "high-risk" predictions are correct
  - **Lower is okay** if it means catching more true positives

- **F1-Score:** Balance between precision and recall  
  - Harmonic mean: `2 * (precision * recall) / (precision + recall)`
  - Higher is better

- **ROC-AUC:** Overall model quality  
  - 0.5 = random guessing (coin flip)
  - 1.0 = perfect predictions
  - Our models: ~0.65-0.68 (better than random, but room for improvement)

### **Why Two Models?**

1. **Logistic Regression:** Easy to interpret. Doctors can see which factors matter most (coefficients). Good for explaining to stakeholders.

2. **XGBoost:** More accurate. Catches more readmissions correctly. Recommended for actual deployment.

### **Reproducibility**

- Fixed random seed (42) used throughout
- Same seed for: train/test split, model training, feature selection
- This ensures you get the same results every time you run it

---

## 🎓 For Professors/Instructors

### **Quick Setup for Grading**

1. **Download the repository:**
   ```bash
   git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
   cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
   ```

2. **Run everything:**
   ```bash
   python run_all.py
   ```

3. **Verify results:**
   ```bash
   python test_models.py
   ```

4. **View dashboard (optional):**
   ```bash
   source .venv/bin/activate  # Mac/Linux
   streamlit run dashboard.py
   ```

**Expected time:** 5-10 minutes for full setup and training.

**Files to check:**
- `models/logreg_selected.joblib` - Logistic Regression model
- `models/xgb_selected.joblib` - XGBoost model
- `models/thresholds.json` - Thresholds and feature lists
- `data/processed/train_processed.csv` - Processed training data
- `data/processed/test_processed.csv` - Processed test data

---

## 📝 License

This project is for educational purposes (MSBA-265 course project).

---

## 🙏 Acknowledgments

- **Dataset:** UCI Machine Learning Repository - Diabetes 130-US hospitals for years 1999-2008
- **Course:** MSBA-265
- **Purpose:** Educational project for predicting 30-day hospital readmissions

---

**Questions?** Check the troubleshooting section above, or review the code comments in each script file - they explain what each part does!

**Ready to start?** Run `python run_all.py` and everything will be set up automatically! 🚀
