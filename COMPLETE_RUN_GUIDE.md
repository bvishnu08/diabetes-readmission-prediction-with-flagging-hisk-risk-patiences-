# 🚀 Complete Guide: How to Run This Project

**Repository:** https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-

---

## 📋 **Table of Contents**

1. [Quick Start (One Command)](#quick-start)
2. [Method 1: Git Clone](#method-1-git-clone)
3. [Method 2: Download ZIP](#method-2-download-zip)
4. [Step-by-Step Manual Setup](#step-by-step-manual-setup)
5. [Viewing Results](#viewing-results)
6. [Troubleshooting](#troubleshooting)

---

## ⚡ **Quick Start**

**After getting the repository, run this ONE command:**

```bash
python run_all.py
```

**That's it!** Everything runs automatically.

---

## 📥 **Method 1: Git Clone**

### Prerequisites
- Git installed
- Python 3.8+ installed
- Internet connection

### Steps

#### Step 1: Open Terminal
- **Mac/Linux:** Terminal
- **Windows:** Command Prompt or PowerShell

#### Step 2: Navigate to Short Path
```bash
# Windows
cd C:\Projects

# Mac/Linux
cd ~/Projects
```

#### Step 3: Clone Repository
```bash
git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
```

#### Step 4: Enter Project Folder
```bash
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
```

#### Step 5: Run Everything
```bash
python run_all.py
```

**Expected Output:**
```
==========================================
🚀 Running Model Training
==========================================
[preprocess] Saved train to...
[train] Training Logistic Regression...
[train] Training XGBoost...
✅ Training completed successfully!

==========================================
📊 Running Model Evaluation
==========================================
MODEL EVALUATION – 30-Day Readmission Prediction
...
✅ Evaluation completed successfully!
```

---

## 📦 **Method 2: Download ZIP** (Recommended for Windows)

### Prerequisites
- Python 3.8+ installed
- Internet connection
- **No Git needed!**

### Steps

#### Step 1: Download ZIP
1. Go to: https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
2. Click the green **"Code"** button
3. Click **"Download ZIP"**

#### Step 2: Extract to Short Path
- **Windows:** Extract to `C:\Projects\` (create folder if needed)
- **Mac/Linux:** Extract to `~/Projects/`
- **Important:** Rename extracted folder to something SHORT (e.g., `diabetes-project`)

#### Step 3: Open Terminal in Folder
```bash
# Windows
cd C:\Projects\diabetes-project

# Mac/Linux
cd ~/Projects/diabetes-project
```

#### Step 4: Run Everything
```bash
python run_all.py
```

---

## 🔧 **Step-by-Step Manual Setup**

If you prefer to run each step manually:

### Step 1: Create Virtual Environment
```bash
python -m venv .venv
```

### Step 2: Activate Virtual Environment
```bash
# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**You'll see `(.venv)` in your prompt when activated.**

### Step 3: Upgrade pip
```bash
pip install --upgrade pip
```

### Step 4: Install Packages
```bash
pip install -r requirements.txt
```

**This takes 2-5 minutes. Wait for completion.**

### Step 5: Verify Data File
```bash
# Mac/Linux
ls data/raw/diabetic_data.csv

# Windows
dir data\raw\diabetic_data.csv
```

**Should show the file exists.**

### Step 6: Train Models
```bash
python scripts/run_train.py
```

**This takes 2-5 minutes. You'll see:**
- Data preprocessing
- Feature selection
- Model training
- Threshold tuning
- Models saved

### Step 7: Evaluate Models
```bash
python scripts/run_eval.py
```

**You'll see:**
- Model performance metrics
- Confusion matrices
- Clinical interpretations
- Model recommendations

### Step 8: Launch Dashboard (Optional)
```bash
streamlit run dashboard.py
```

**Then open:** http://localhost:8501

---

## 📊 **Viewing Results**

### 1. Terminal Output
After running evaluation, you'll see:
- Model metrics (Recall, Precision, F1, ROC-AUC)
- Confusion matrices
- Clinical interpretations
- Model recommendations

### 2. Interactive Dashboard
```bash
# Activate virtual environment first
source .venv/bin/activate    # Mac/Linux
# OR
.venv\Scripts\activate       # Windows

# Run dashboard
streamlit run dashboard.py
```

**Features:**
- Model comparison charts
- ROC curves
- Feature importance plots
- Interactive prediction tool
- Confusion matrices

### 3. Jupyter Notebooks
```bash
jupyter lab notebooks/03_implementation_details.ipynb
```

**Contains:**
- Code with explanations
- Visualizations
- Step-by-step analysis

### 4. Generated Files
- **Models:** `models/logreg_selected.joblib`, `models/xgb_selected.joblib`
- **Thresholds:** `models/thresholds.json`
- **Processed Data:** `data/processed/train_processed.csv`, `test_processed.csv`

---

## 🆘 **Troubleshooting**

### Problem: "Python not found"
**Solution:**
- Install Python 3.8+ from python.org
- Make sure Python is in your PATH
- Try `python3` instead of `python` (Mac/Linux)

### Problem: "Git is not installed"
**Solution:**
- Use Method 2 (Download ZIP) instead
- Or install Git: https://git-scm.com/download/win

### Problem: "Filename too long" (Windows)
**Solution:**
- Use ZIP download instead of Git clone
- Extract to a SHORT path (e.g., `C:\Projects\`)
- Rename folder to something short
- See: `docs/WINDOWS_PATH_LENGTH_FIX.md`

### Problem: "ModuleNotFoundError"
**Solution:**
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`
- Check you're in the project folder

### Problem: "OMP Error" or "SHM2 Error"
**Solution:**
- Run with: `OMP_NUM_THREADS=1 python scripts/run_train.py`
- Windows: `set OMP_NUM_THREADS=1` then run script

### Problem: "FileNotFoundError: data/raw/diabetic_data.csv"
**Solution:**
- Make sure you're in the project root folder
- Check the file exists: `ls data/raw/diabetic_data.csv`

---

## 📖 **Additional Resources**

- **SETUP_GUIDE.md** - Complete setup instructions
- **docs/QUICK_START.md** - Quick reference guide
- **HOW_TO_CLONE_AND_RUN.md** - Simple clone instructions
- **docs/RUN_BOOK.md** - Detailed execution guide
- **docs/WINDOWS_PATH_LENGTH_FIX.md** - Windows-specific fixes

---

## ✅ **Verification Checklist**

After running, verify:

- [ ] Virtual environment created (`.venv` folder exists)
- [ ] Packages installed (no import errors)
- [ ] Models trained (`models/*.joblib` files exist)
- [ ] Thresholds saved (`models/thresholds.json` exists)
- [ ] Processed data created (`data/processed/*.csv` files exist)
- [ ] Evaluation completed (metrics displayed in terminal)
- [ ] Dashboard runs (if tested)

---

## 🎯 **Quick Reference Commands**

```bash
# One command to rule them all
python run_all.py

# Or step by step
python -m venv .venv
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
python scripts/run_train.py
python scripts/run_eval.py
streamlit run dashboard.py
```

---

## 📞 **Need Help?**

1. Check `SETUP_GUIDE.md` for detailed instructions
2. Check `docs/WINDOWS_PATH_LENGTH_FIX.md` for Windows issues
3. Check `docs/QUICK_START.md` for quick reference
4. Review error messages carefully - they usually tell you what's wrong

---

**That's it! Your project should be running!** 🎉

