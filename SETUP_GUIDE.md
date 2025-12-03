# Complete Setup Guide for Professor

This guide provides step-by-step instructions to set up and run the Diabetes Readmission Prediction project on any local system. You can use either:
- **Terminal/Command Prompt** (recommended)
- **Spyder IDE** (alternative method)

Both methods are fully documented below.

---

## **Step 1: Navigate to Project Folder**

Open terminal/command prompt and go to where the project is saved:

```bash
cd /path/to/265_final
```

*(Replace `/path/to/265_final` with the actual location where the project folder is)*

---

## **Step 2: Create Virtual Environment**

This creates an isolated Python environment so the project doesn't interfere with other Python projects:

**On Mac/Linux:**
```bash
python3 -m venv .venv
```

**On Windows:**
```bash
python -m venv .venv
```

*(If `python3` doesn't work, try `python`)*

---

## **Step 3: Activate Virtual Environment**

This "turns on" the virtual environment:

**On Mac/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

**You'll know it worked when you see `(.venv)` at the start of your terminal prompt!**

---

## **Step 4: Upgrade pip (Package Installer)**

```bash
pip install --upgrade pip
```

---

## **Step 5: Install All Required Packages**

This installs all the libraries the project needs:

```bash
pip install -r requirements.txt
```

**This will take a few minutes. Wait until it finishes!**

---

## **Step 6: Verify Data Files Exist**

Check that the raw data file is present:

**On Mac/Linux:**
```bash
ls data/raw/diabetic_data.csv
```

**On Windows:**
```bash
dir data\raw\diabetic_data.csv
```

**If you see the file listed, you're good! If you get an error, the data file is missing.**

---

## **Step 7: Run Training (This Creates Everything)**

This is the main command that does everything:
- Cleans the data
- Trains both models (Logistic Regression + XGBoost)
- Saves all the models and processed data

```bash
python scripts/run_train.py
```

**This will take 2-5 minutes. You'll see progress messages like:**
- `[preprocess] Saved train to...`
- `[train] Training Logistic Regression...`
- `[train] Training XGBoost...`
- `✅ Saved models to models/...`

**If you get an error about "OMP" or "SHM2", run this instead:**
```bash
OMP_NUM_THREADS=1 python scripts/run_train.py
```

*(On Windows, set the environment variable first: `set OMP_NUM_THREADS=1` then run the script)*

---

## **Step 8: Run Evaluation (Test the Models)**

This tests how good the models are:

```bash
python scripts/run_eval.py
```

**You'll see a detailed report comparing both models with metrics like:**
- Accuracy
- Recall (how many readmissions we catch)
- Precision (how many are false alarms)
- F1-Score
- ROC-AUC

**If you get an "OMP" error, use:**
```bash
OMP_NUM_THREADS=1 python scripts/run_eval.py
```

---

## **Step 9: Launch Dashboard (Optional - Interactive Web App)**

This opens a web browser with an interactive dashboard:

```bash
streamlit run dashboard.py
```

**This will:**
- Start a local web server
- Open your browser automatically
- Show the dashboard at `http://localhost:8501`

**To stop the dashboard:** Press `Ctrl+C` in the terminal

---

## **Step 10: Run Jupyter Notebook (Optional - For Exploration)**

If you want to see the implementation notebook:

```bash
jupyter lab notebooks/03_implementation_details.ipynb
```

**Or if you prefer classic Jupyter:**
```bash
jupyter notebook notebooks/03_implementation_details.ipynb
```

---

## **Alternative: Running in Spyder IDE**

If you prefer using Spyder IDE instead of the terminal, follow these steps:

### **Step 1: Complete Steps 1-5 Above First**

You still need to create the virtual environment and install packages using the terminal (Steps 1-5 above).

### **Step 2: Configure Spyder to Use Your Virtual Environment**

1. Open Spyder
2. Go to: **Tools → Preferences → Python Interpreter**
3. Select: **Use the following Python interpreter**
4. Browse and select your virtual environment's Python:
   - **Mac/Linux:** `/path/to/265_final/.venv/bin/python`
   - **Windows:** `C:\path\to\265_final\.venv\Scripts\python.exe`
5. Click **Apply** and **OK**
6. **Restart Spyder** for changes to take effect

### **Step 3: Verify Spyder is Using Correct Environment**

In Spyder's IPython console, run:
```python
import sys
print(sys.executable)
```

You should see the path to `.venv/bin/python` (or `.venv\Scripts\python.exe` on Windows).

### **Step 4: Run Training Script in Spyder**

**Option A: Run from Editor**
1. Open `scripts/run_train.py` in Spyder editor
2. Click **Run** (green play button) or press `F5`
3. Make sure "Run in current Python or IPython console" is selected

**Option B: Run from IPython Console**
```python
%run scripts/run_train.py
```

### **Step 5: Run Evaluation Script in Spyder**

**Option A: Run from Editor**
1. Open `scripts/run_eval.py` in Spyder editor
2. Click **Run** (F5)

**Option B: Run from IPython Console**
```python
%run scripts/run_eval.py
```

### **Step 6: Run Dashboard (Must Use Terminal)**

The dashboard needs to run in terminal (Spyder's console may not work well for Streamlit):

1. Open terminal/command prompt
2. Activate virtual environment:
   ```bash
   source .venv/bin/activate    # Mac/Linux
   # OR
   .venv\Scripts\activate      # Windows
   ```
3. Run dashboard:
   ```bash
   streamlit run dashboard.py
   ```

### **Spyder Troubleshooting**

**Problem:** `ModuleNotFoundError` in Spyder  
**Solution:** Spyder is not using the virtual environment. Re-check Step 2 above and restart Spyder.

**Problem:** Script runs but can't find files  
**Solution:** In Spyder, set working directory: **Run → Configuration per file → Working directory → The directory of the file being run**

**Problem:** Dashboard won't start from Spyder console  
**Solution:** Run dashboard from terminal/command prompt instead (Step 6 above).

---

## **Quick Reference: All Commands in Order**

```bash
# 1. Navigate to project
cd /path/to/265_final

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate    # Mac/Linux
# OR
.venv\Scripts\activate       # Windows

# 4. Install packages
pip install --upgrade pip
pip install -r requirements.txt

# 5. Train models (REQUIRED - do this first!)
python scripts/run_train.py

# 6. Evaluate models
python scripts/run_eval.py

# 7. Launch dashboard (optional)
streamlit run dashboard.py
```

---

## **Troubleshooting**

### **Problem:** `python3: command not found`  
**Solution:** Try `python` instead of `python3`

### **Problem:** `pip: command not found`  
**Solution:** Make sure virtual environment is activated (you should see `(.venv)` in prompt)

### **Problem:** `OMP: Error #179: Can't open SHM2`  
**Solution:** Run with `OMP_NUM_THREADS=1` before the command:
```bash
OMP_NUM_THREADS=1 python scripts/run_train.py
```

### **Problem:** `FileNotFoundError: Raw data not found`  
**Solution:** Make sure `data/raw/diabetic_data.csv` exists in the project folder

### **Problem:** `ModuleNotFoundError`  
**Solution:** Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

---

## **What Each File Does**

- **`requirements.txt`** - Lists all Python packages needed
- **`scripts/run_train.py`** - Trains both models (run this first!)
- **`scripts/run_eval.py`** - Tests the models and shows results
- **`dashboard.py`** - Interactive web dashboard
- **`src/` folder** - All the actual code (preprocessing, models, etc.)
- **`data/raw/`** - Original data file
- **`data/processed/`** - Cleaned data (created after training)
- **`models/`** - Saved trained models (created after training)

---

## **Project Structure Overview**

```
265_final/
├── SETUP_GUIDE.md          ← You are here! (Start here)
├── README.md               ← Project overview
├── requirements.txt        ← Python dependencies
├── dashboard.py            ← Streamlit dashboard
├── scripts/                ← Easy-to-run scripts
│   ├── run_train.py       ← Train models (run first!)
│   ├── run_eval.py        ← Evaluate models
│   └── run_dashboard.py   ← Launch dashboard
├── src/                    ← Core project code
│   ├── config.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── data/
│   ├── raw/               ← Original data (must exist)
│   └── processed/         ← Cleaned data (created after training)
├── models/                 ← Trained models (created after training)
└── notebooks/              ← Jupyter notebooks for exploration
```

---

**Everything is ready to run. Just follow these steps in order!**

