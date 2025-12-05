# Complete Setup Guide for Professor

This guide provides step-by-step instructions to set up and run the Diabetes Readmission Prediction project on any local system.

---

## 📧 **Quick Start Message for Professors**

**Copy and paste this message to share with your professor:**

---

Hi Professor,

Here is my MSBA-265 project repository:

**GitHub Link:** https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-

**TO RUN EVERYTHING (3 Simple Steps):**

1. **Download the repository:**
   - Go to the GitHub link above
   - Click green "Code" button → "Download ZIP"
   - **Extract ZIP to a SHORT path** (e.g., `C:\Projects\`)
   - **Rename folder to something SHORT** (e.g., `diabetes-project`)
   - ⚠️ **Windows users:** Avoid deep nested paths to prevent "Filename too long" errors

2. **Open terminal/command prompt in the extracted folder**

3. **Run this ONE command:**
   ```bash
   python run_all.py
   ```

**That's it!** The script will automatically:
- ✅ Create virtual environment
- ✅ Install all packages
- ✅ Train both models (Logistic Regression + XGBoost)
- ✅ Evaluate models
- ✅ Show results

**Total time: 5-10 minutes**

**All documentation is included in the repository:**
- `README.md` - Project overview
- `SETUP_GUIDE.md` - Detailed setup instructions (this file)
- `docs/QUICK_START.md` - Quick reference guide
- `docs/RUN_BOOK.md` - Step-by-step execution guide

Thank you!

---

# 📚 **COMPLETE STEP-BY-STEP GUIDE**

---

## **STEP 1: Understanding the Files** 📁

### **1.1 Project Structure Overview**

```
diabetes-project/
├── README.md                    # Main project overview
├── SETUP_GUIDE.md              # This file - setup instructions
├── COMPLETE_RUN_GUIDE.md       # Complete run instructions
├── HOW_TO_CLONE_AND_RUN.md    # Simple clone guide
├── requirements.txt            # Python package dependencies
├── dashboard.py                # Interactive Streamlit dashboard
│
├── run_all.py                  # ⭐ ONE-COMMAND RUNNER (Use this!)
├── download_and_run.py         # Download from GitHub + run
│
├── docs/                       # All documentation
│   ├── QUICK_START.md
│   ├── RUN_BOOK.md
│   ├── PROJECT_STRUCTURE.md
│   ├── WINDOWS_PATH_LENGTH_FIX.md
│   └── ... (more guides)
│
├── src/                        # Source code (core project)
│   ├── config.py              # Configuration settings
│   ├── preprocess.py          # Data cleaning & splitting
│   ├── feature_selection.py   # Feature selection
│   ├── model.py               # Model definitions
│   ├── train.py               # Training logic
│   └── evaluate.py            # Evaluation logic
│
├── scripts/                    # Easy-to-run scripts
│   ├── run_train.py           # Train models
│   ├── run_eval.py            # Evaluate models
│   └── run_dashboard.py      # Launch dashboard
│
├── data/
│   ├── raw/
│   │   ├── diabetic_data.csv  # ⭐ Original dataset (MUST EXIST)
│   │   └── IDS_mapping.csv    # Feature ID mappings
│   └── processed/              # Created after training
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/                     # Created after training
│   ├── logreg_selected.joblib # ⭐ Logistic Regression model
│   ├── xgb_selected.joblib    # ⭐ XGBoost model
│   └── thresholds.json        # ⭐ Thresholds & features
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_modeling.ipynb     # Modeling experiments
│   └── 03_implementation_details.ipynb  # Main implementation
│
└── reports/                    # Project reports
    └── P2 Final_submission report.pdf
```

### **1.2 Key Files Explained**

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_all.py` | ⭐ **EASIEST** - Runs everything automatically | Use this first! |
| `download_and_run.py` | Downloads from GitHub + runs everything | First time setup |
| `scripts/run_train.py` | Trains both models | After setup, to retrain |
| `scripts/run_eval.py` | Evaluates models and shows metrics | After training |
| `dashboard.py` | Interactive web dashboard | To visualize results |
| `test_models.py` | Verifies models work correctly | To check everything |
| `requirements.txt` | Lists all Python packages needed | Auto-installed by scripts |

### **1.3 What Gets Created After Running**

**After running `python run_all.py`, these files are created:**

- ✅ `models/logreg_selected.joblib` - Trained Logistic Regression
- ✅ `models/xgb_selected.joblib` - Trained XGBoost
- ✅ `models/thresholds.json` - Thresholds and selected features
- ✅ `data/processed/train_processed.csv` - Processed training data
- ✅ `data/processed/test_processed.csv` - Processed test data
- ✅ `.venv/` - Virtual environment (Python packages)

---

## **STEP 2: How to Run** 🚀

### **2.1 Method 1: One Command (EASIEST - Recommended)** ⭐

**If you already have the repository:**

```bash
python run_all.py
```

**That's it!** This single command:
- Creates virtual environment
- Installs all packages
- Trains both models
- Evaluates models
- Shows results

**Time: 5-10 minutes**

---

### **2.2 Method 2: Download from GitHub + Run**

**For first-time setup:**

**Windows:**
```cmd
python download_and_run.py
```

**Mac/Linux:**
```bash
python download_and_run.py
# OR
./download_and_run.sh
```

This will:
1. Download repository from GitHub
2. Set up everything
3. Run training and evaluation

---

### **2.3 Method 3: Manual Step-by-Step**

If you prefer more control:

#### **Step 2.3.1: Navigate to Project Folder**
```bash
cd /path/to/diabetes-project
```

#### **Step 2.3.2: Create Virtual Environment**
```bash
python -m venv .venv
```

#### **Step 2.3.3: Activate Virtual Environment**
```bash
# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**You'll see `(.venv)` in your prompt when activated.**

#### **Step 2.3.4: Install Packages**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### **Step 2.3.5: Train Models**
```bash
python scripts/run_train.py
```

#### **Step 2.3.6: Evaluate Models**
```bash
python scripts/run_eval.py
```

---

## **STEP 3: How to Execute (Run Specific Tasks)** ⚙️

### **3.1 Execute Training Only**

```bash
# Activate virtual environment first
source .venv/bin/activate    # Mac/Linux
# OR
.venv\Scripts\activate        # Windows

# Run training
python scripts/run_train.py
```

**What it does:**
- Cleans and preprocesses data
- Selects best features
- Trains Logistic Regression (20 features)
- Trains XGBoost (25 features)
- Tunes thresholds
- Saves models to `models/` folder

**Time: 2-5 minutes**

---

### **3.2 Execute Evaluation Only**

```bash
# Activate virtual environment first
source .venv/bin/activate    # Mac/Linux
# OR
.venv\Scripts\activate        # Windows

# Run evaluation
python scripts/run_eval.py
```

**What it does:**
- Loads trained models
- Tests on held-out test set
- Calculates metrics (Recall, Precision, F1, ROC-AUC)
- Shows confusion matrices
- Provides clinical interpretation
- Recommends best model

**Time: 1-2 minutes**

---

### **3.3 Execute Dashboard**

```bash
# Activate virtual environment first
source .venv/bin/activate    # Mac/Linux
# OR
.venv\Scripts\activate        # Windows

# Run dashboard
streamlit run dashboard.py
# OR
python scripts/run_dashboard.py
```

**What it does:**
- Starts local web server
- Opens browser automatically
- Shows interactive visualizations
- Displays model metrics
- Provides prediction playground

**Access:** http://localhost:8501

**To stop:** Press `Ctrl+C` in terminal

---

### **3.4 Execute Verification Script**

```bash
# Activate virtual environment first
source .venv/bin/activate    # Mac/Linux
# OR
.venv\Scripts\activate        # Windows

# Run verification
python test_models.py
```

**What it does:**
- Checks all files exist
- Verifies models can be loaded
- Displays thresholds
- Shows file sizes
- Confirms everything works

---

## **STEP 4: How to Study (Explore the Project)** 📖

### **4.1 Study the Code Structure**

**Start with these files in order:**

1. **`src/config.py`** - Configuration and settings
   - Feature lists
   - File paths
   - Model parameters

2. **`src/preprocess.py`** - Data preprocessing
   - Data cleaning
   - Train/test splitting
   - Feature engineering

3. **`src/model.py`** - Model definitions
   - Pipeline structure
   - Preprocessing steps
   - Model architectures

4. **`src/train.py`** - Training logic
   - Feature selection
   - Model training
   - Threshold tuning

5. **`src/evaluate.py`** - Evaluation logic
   - Metric calculations
   - Clinical interpretation
   - Model comparison

---

### **4.2 Study with Jupyter Notebooks**

**Open notebooks in order:**

#### **Notebook 1: Exploratory Data Analysis**
```bash
jupyter lab notebooks/01_eda.ipynb
```
**What to study:**
- Data overview
- Feature distributions
- Missing values
- Basic statistics

#### **Notebook 2: Modeling Experiments**
```bash
jupyter lab notebooks/02_modeling.ipynb
```
**What to study:**
- Model experiments
- Hyperparameter testing
- Initial results

#### **Notebook 3: Implementation Details** ⭐
```bash
jupyter lab notebooks/03_implementation_details.ipynb
```
**What to study:**
- Complete pipeline walkthrough
- Feature importance
- Model comparisons
- Visualizations
- Code explanations

**This is the main notebook - most comprehensive!**

---

### **4.3 Study the Documentation**

**Read in this order:**

1. **`README.md`** - Project overview and quick start
2. **`SETUP_GUIDE.md`** - This file - complete setup guide
3. **`docs/RUN_BOOK.md`** - Detailed execution guide
4. **`docs/CODE_EXPLANATION.md`** - Deep dive into code
5. **`docs/PROJECT_STRUCTURE.md`** - Repository structure map

---

### **4.4 Study the Results**

**After running evaluation, study:**

1. **Terminal Output:**
   - Model metrics
   - Confusion matrices
   - Clinical interpretations

2. **Dashboard:**
   - Interactive charts
   - ROC curves
   - Feature importance plots

3. **Generated Files:**
   - `models/thresholds.json` - See thresholds and features
   - `data/processed/*.csv` - See processed data

---

## **STEP 5: How to Check (Verify Everything Works)** ✅

### **5.1 Check Files Exist**

**Verify all required files are present:**

```bash
# Check data file (MUST EXIST)
ls data/raw/diabetic_data.csv    # Mac/Linux
dir data\raw\diabetic_data.csv   # Windows

# Check models (created after training)
ls models/*.joblib models/*.json    # Mac/Linux
dir models\*.joblib models\*.json   # Windows
```

**Required files:**
- ✅ `data/raw/diabetic_data.csv` - Must exist before running
- ✅ `models/logreg_selected.joblib` - Created after training
- ✅ `models/xgb_selected.joblib` - Created after training
- ✅ `models/thresholds.json` - Created after training

---

### **5.2 Check Models Work**

**Run verification script:**

```bash
python test_models.py
```

**Expected output:**
```
✅ All files exist
✅ Thresholds loaded successfully
✅ Logistic Regression model loads successfully
✅ XGBoost model loads successfully
✅ All checks passed!
```

---

### **5.3 Check Model Files Manually**

#### **View Thresholds (JSON - Readable):**

**Mac/Linux:**
```bash
cat models/thresholds.json
```

**Windows:**
```cmd
type models\thresholds.json
```

**Or open in Notepad:**
```cmd
notepad models\thresholds.json
```

**You'll see:**
- Threshold values for each model
- Selected features for each model
- Model configurations

---

#### **Load Models in Python:**

```bash
python
```

Then:
```python
import joblib
import json

# Load thresholds
with open('models/thresholds.json', 'r') as f:
    thresholds = json.load(f)
print("Thresholds:", thresholds)

# Load Logistic Regression
lr_model = joblib.load('models/logreg_selected.joblib')
print("LR model loaded:", type(lr_model))

# Load XGBoost
xgb_model = joblib.load('models/xgb_selected.joblib')
print("XGB model loaded:", type(xgb_model))
```

---

### **5.4 Check Evaluation Results**

**Run evaluation to see results:**

```bash
python scripts/run_eval.py
```

**Check for:**
- ✅ Model metrics displayed
- ✅ Confusion matrices shown
- ✅ Clinical interpretations provided
- ✅ Model recommendation given
- ✅ No errors in output

---

### **5.5 Check Dashboard Works**

**Launch dashboard:**

```bash
streamlit run dashboard.py
```

**Check:**
- ✅ Browser opens automatically
- ✅ Dashboard loads at http://localhost:8501
- ✅ Metrics displayed
- ✅ Charts visible
- ✅ No errors

---

### **5.6 Complete Verification Checklist**

**After running the project, verify:**

- [ ] Virtual environment created (`.venv` folder exists)
- [ ] Packages installed (no import errors)
- [ ] Data file exists (`data/raw/diabetic_data.csv`)
- [ ] Models trained (`models/*.joblib` files exist)
- [ ] Thresholds saved (`models/thresholds.json` exists)
- [ ] Processed data created (`data/processed/*.csv` files exist)
- [ ] Evaluation completed (metrics displayed)
- [ ] Models can be loaded (test_models.py passes)
- [ ] Dashboard runs (if tested)

**Quick test command:**
```bash
python test_models.py && python scripts/run_eval.py
```

If both succeed, everything works! ✅

---

## **STEP 6: Troubleshooting** 🆘

### **Problem: "Python not found"**
**Solution:**
- Install Python 3.8+ from python.org
- Make sure Python is in your PATH
- Try `python3` instead of `python` (Mac/Linux)

### **Problem: "Filename too long" (Windows)**
**Solution:**
- Use ZIP download instead of Git clone
- Extract to SHORT path (e.g., `C:\Projects\`)
- Rename folder to something SHORT
- See: `docs/WINDOWS_PATH_LENGTH_FIX.md`

### **Problem: "ModuleNotFoundError"**
**Solution:**
- Activate virtual environment: `source .venv/bin/activate`
- Install packages: `pip install -r requirements.txt`
- Check you're in the project folder

### **Problem: "OMP Error" or "SHM2 Error"**
**Solution:**
```bash
# Mac/Linux
OMP_NUM_THREADS=1 python scripts/run_train.py

# Windows
set OMP_NUM_THREADS=1
python scripts/run_train.py
```

### **Problem: "FileNotFoundError: data/raw/diabetic_data.csv"**
**Solution:**
- Make sure you're in the project root folder
- Check the file exists: `ls data/raw/diabetic_data.csv`
- Verify you downloaded the complete repository

---

## **Quick Reference: All Commands**

```bash
# 1. Navigate to project
cd /path/to/diabetes-project

# 2. ONE COMMAND TO RUN EVERYTHING
python run_all.py

# OR step by step:

# 2. Create virtual environment
python -m venv .venv

# 3. Activate it
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows

# 4. Install packages
pip install -r requirements.txt

# 5. Train models
python scripts/run_train.py

# 6. Verify models
python test_models.py

# 7. Evaluate models
python scripts/run_eval.py

# 8. Launch dashboard
streamlit run dashboard.py
```

---

## **What Each File Does**

| File | Purpose |
|------|---------|
| `run_all.py` | ⭐ Runs everything automatically |
| `scripts/run_train.py` | Trains both models |
| `scripts/run_eval.py` | Evaluates models and shows results |
| `scripts/run_dashboard.py` | Launches interactive dashboard |
| `test_models.py` | Verifies models work correctly |
| `dashboard.py` | Streamlit dashboard application |
| `src/config.py` | Configuration and settings |
| `src/preprocess.py` | Data cleaning and splitting |
| `src/train.py` | Model training logic |
| `src/evaluate.py` | Model evaluation logic |
| `requirements.txt` | Python package dependencies |

---

**Everything is ready to run. Just follow these steps in order!** 🎉
