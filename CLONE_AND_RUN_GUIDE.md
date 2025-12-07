# 🚀 Complete Guide: Cloning and Running This Project

> **For anyone who clones this repository fresh** - Step-by-step instructions that work on Windows, Mac, and Linux

---

## 📋 **PREREQUISITES (What You Need First)**

Before you start, make sure you have:

1. **Python 3.8 or higher** installed
   - Check: Open terminal/command prompt and type `python --version`
   - If you see `Python 3.8.x` or higher → ✅ You're good!
   - If not installed: Download from https://www.python.org/downloads/
   - **Important:** During installation, check "Add Python to PATH" (Windows)

2. **Git (Optional - Only if using git clone)**
   - Check: Type `git --version` in terminal
   - If you see a version number → ✅ You're good!
   - If not: Download from https://git-scm.com/downloads
   - **Alternative:** You can download ZIP instead (no Git needed)

3. **Internet connection** (for downloading packages)

---

## 🎯 **METHOD 1: Git Clone (Recommended for Developers)**

### **Step 1: Open Terminal/Command Prompt**

- **Windows:** Press `Win + R`, type `cmd`, press Enter
- **Mac:** Press `Cmd + Space`, type "Terminal", press Enter
- **Linux:** Press `Ctrl + Alt + T`

### **Step 2: Navigate to a Short Path**

**Why?** Windows has a 260-character path limit. Use a short path to avoid errors.

```bash
# Windows
cd C:\Projects

# Mac/Linux
cd ~/Projects
```

**If the folder doesn't exist, create it:**
```bash
# Windows
mkdir C:\Projects
cd C:\Projects

# Mac/Linux
mkdir -p ~/Projects
cd ~/Projects
```

### **Step 3: Clone the Repository**

```bash
git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
```

**What this does:** Downloads all the project files from GitHub to your computer.

**Time:** 1-2 minutes (depending on internet speed)

**If you get an error:**
- **"git: command not found"** → Install Git (see Prerequisites) or use Method 2 (ZIP download)
- **"Filename too long"** (Windows) → Use Method 2 (ZIP download) or enable long paths (see troubleshooting)

### **Step 4: Enter the Project Folder**

```bash
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
```

**Note:** The folder name is long. You can rename it after cloning:
```bash
# After cloning, rename to something shorter
# Windows
rename diabetes-readmission-prediction-with-flagging-hisk-risk-patiences- diabetes-project

# Mac/Linux
mv diabetes-readmission-prediction-with-flagging-hisk-risk-patiences- diabetes-project
cd diabetes-project
```

### **Step 5: Run the Project**

```bash
python run_all.py
```

**That's it!** This single command will:
- ✅ Create virtual environment
- ✅ Install all packages
- ✅ Train both models
- ✅ Evaluate models
- ✅ Show results

**Time:** 5-10 minutes total

---

## 📦 **METHOD 2: Download ZIP (Easiest - No Git Needed)**

### **Step 1: Download the ZIP File**

1. Go to: https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Wait for download to complete

### **Step 2: Extract the ZIP File**

**Windows:**
1. Right-click the downloaded ZIP file
2. Click **"Extract All..."**
3. **IMPORTANT:** Extract to a SHORT path:
   - ✅ Good: `C:\Projects\`
   - ❌ Bad: `C:\Users\YourName\Documents\Very\Long\Path\Name\...`
4. Click "Extract"

**Mac:**
1. Double-click the ZIP file
2. It will extract automatically to your Downloads folder

**Linux:**
```bash
unzip diabetes-readmission-prediction-with-flagging-hisk-risk-patiences--main.zip
```

### **Step 3: Rename the Folder (Make It Shorter)**

**Why?** The folder name is very long. Rename it to something shorter.

**Windows:**
1. Right-click the extracted folder
2. Click "Rename"
3. Type: `diabetes-project`
4. Press Enter

**Mac/Linux:**
```bash
mv diabetes-readmission-prediction-with-flagging-hisk-risk-patiences--main diabetes-project
```

### **Step 4: Open Terminal in the Project Folder**

**Windows:**
1. Navigate to the folder in File Explorer
2. Right-click in the folder (empty space)
3. Click "Open in Terminal" or "Open PowerShell here"

**Mac:**
1. Navigate to the folder in Finder
2. Right-click the folder
3. Click "New Terminal at Folder"

**Linux:**
```bash
cd ~/Downloads/diabetes-project  # or wherever you extracted it
```

### **Step 5: Run the Project**

```bash
python run_all.py
```

**That's it!** Everything happens automatically.

---

## ✅ **WHAT TO EXPECT WHEN RUNNING `python run_all.py`**

### **Step-by-Step Output:**

```
==========================================
Diabetes Readmission Prediction - Setup & Run
==========================================

STEP 1/6: Checking Python installation...
✅ Found Python 3.9.7

STEP 2/6: Setting up virtual environment...
✅ Virtual environment created

STEP 3/6: Installing required packages...
📦 Installing packages... (This takes 2-5 minutes)
✅ All packages installed successfully!

STEP 4/6: Checking data files...
✅ Data file found: data/raw/diabetic_data.csv

STEP 5/6: Training models...
🚀 Starting Training Process
⏳ Training in progress... (This takes 2-5 minutes)
✅ Training completed successfully!

STEP 6/6: Evaluating models...
📊 Running Model Evaluation
⏳ Evaluating models... (This takes 1-2 minutes)
✅ Evaluation completed successfully!

✅ ALL STEPS COMPLETED SUCCESSFULLY!
```

### **After Completion, You'll See:**

1. **Model Performance Metrics:**
   - Confusion matrices
   - Accuracy, Recall, Precision, F1-Score, ROC-AUC
   - Classification reports
   - Clinical interpretation

2. **Files Created:**
   - `models/logreg_selected.joblib` - Trained Logistic Regression
   - `models/xgb_selected.joblib` - Trained XGBoost
   - `models/thresholds.json` - Best thresholds
   - `data/processed/train_processed.csv` - Cleaned training data
   - `data/processed/test_processed.csv` - Cleaned test data

3. **Next Steps Instructions:**
   - How to verify files
   - How to launch dashboard
   - How to view results again

---

## 🔧 **TROUBLESHOOTING COMMON ISSUES**

### **Issue 1: "python: command not found"**

**Problem:** Python is not installed or not in PATH.

**Solution:**
1. Install Python from https://www.python.org/downloads/
2. **During installation (Windows):** Check "Add Python to PATH"
3. Restart your terminal/command prompt
4. Try again: `python --version`

**Alternative (Windows):**
- Try `py` instead of `python`:
  ```bash
  py run_all.py
  ```

### **Issue 2: "Filename too long" (Windows)**

**Problem:** Windows has a 260-character path limit.

**Solution 1: Use ZIP Download (Easiest)**
- Download ZIP instead of using git clone
- Extract to a SHORT path like `C:\Projects\`
- Rename folder to something short like `diabetes-project`

**Solution 2: Enable Long Paths (Advanced)**
1. Open PowerShell as Administrator (Right-click → "Run as Administrator")
2. Run this command:
   ```powershell
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```
3. Restart your computer
4. Try cloning again

### **Issue 3: "pip: command not found" or "pip install fails"**

**Problem:** pip (Python package installer) is not working.

**Solution:**
1. Make sure Python is installed correctly
2. Try upgrading pip:
   ```bash
   python -m pip install --upgrade pip
   ```
3. If that doesn't work, try:
   ```bash
   python -m ensurepip --upgrade
   ```

### **Issue 4: "ModuleNotFoundError" or "No module named 'pandas'"

**Problem:** Packages are not installed in the virtual environment.

**Solution:**
1. Make sure you ran `python run_all.py` (it installs packages automatically)
2. If you're running scripts manually, activate the virtual environment first:
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Mac/Linux
   source .venv/bin/activate
   ```
3. Then install packages:
   ```bash
   pip install -r requirements.txt
   ```

### **Issue 5: "FileNotFoundError: data/raw/diabetic_data.csv"**

**Problem:** The data file is missing.

**Solution:**
1. Check if the file exists:
   ```bash
   # Windows
   dir data\raw\diabetic_data.csv
   
   # Mac/Linux
   ls data/raw/diabetic_data.csv
   ```
2. If it doesn't exist, the file might not have been cloned/downloaded
3. Check the repository on GitHub - the file should be there
4. If it's missing, contact the repository owner

### **Issue 6: "Permission denied" or "Access denied"**

**Problem:** You don't have permission to create files/folders.

**Solution:**
1. **Windows:** Run Command Prompt/PowerShell as Administrator
2. **Mac/Linux:** Make sure you have write permissions in the folder
3. Try moving the project to a folder you own (like `C:\Projects\` or `~/Projects/`)

### **Issue 7: Virtual environment creation fails**

**Problem:** `python -m venv .venv` fails.

**Solution:**
1. Make sure Python 3.8+ is installed
2. Try creating venv manually:
   ```bash
   python -m venv .venv
   ```
3. If it still fails, try:
   ```bash
   python3 -m venv .venv
   ```

### **Issue 8: "Out of memory" or "Killed" during training**

**Problem:** Not enough RAM or the process was killed.

**Solution:**
1. Close other applications to free up memory
2. The dataset is large (~10MB), make sure you have at least 2GB free RAM
3. If on a shared system, ask for more resources

### **Issue 9: Training takes too long or hangs**

**Problem:** The training process might be stuck.

**Solution:**
1. Wait at least 10 minutes (first run takes longer)
2. Check if your computer is still working (move mouse, check CPU usage)
3. If truly stuck, press `Ctrl+C` to stop, then try again
4. Make sure you have internet connection (for downloading packages)

### **Issue 10: "git: command not found"**

**Problem:** Git is not installed.

**Solution:**
1. Install Git from https://git-scm.com/downloads
2. **OR** use Method 2 (ZIP download) - no Git needed!

---

## 📝 **VERIFICATION: Make Sure Everything Worked**

After running `python run_all.py`, verify everything worked:

### **Step 1: Check Files Were Created**

```bash
python test_models.py
```

**Expected output:**
```
✅ models/logreg_selected.joblib
✅ models/xgb_selected.joblib
✅ models/thresholds.json
✅ data/processed/train_processed.csv
✅ data/processed/test_processed.csv
✅ ALL CHECKS PASSED!
```

### **Step 2: View Full Results and Metrics ⭐ IMPORTANT!**

**To see the complete evaluation results with all metrics, confusion matrices, and clinical interpretation:**

```bash
# 1. Activate virtual environment first
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

# 2. Run evaluation to see full results
python scripts/run_eval.py
```

**What you'll see:**
- Complete metrics for both models (Threshold, ROC-AUC, Accuracy, Recall, Precision, F1-Score)
- Confusion matrices showing True Positives, False Positives, True Negatives, False Negatives
- Classification reports with detailed breakdown
- Clinical interpretation (HIGH RISK vs LOW RISK patients)
- Model recommendation

**This gives you the full picture of model performance!** 📊

---

## 🎨 **NEXT STEPS: Launch the Dashboard**

After everything is set up, you can launch the interactive dashboard:

```bash
# 1. Activate virtual environment
# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

# 2. Launch dashboard
streamlit run dashboard.py
```

Your browser will open automatically at `http://localhost:8501`

**To stop the dashboard:** Press `Ctrl+C` in the terminal

---

## 📞 **STILL HAVING ISSUES?**

If you're still having problems:

1. **Check the README.md** - It has more detailed instructions
2. **Check docs/WINDOWS_FIX.md** - Windows-specific troubleshooting
3. **Check docs/WINDOWS_PATH_LENGTH_FIX.md** - For path length issues
4. **Make sure you're using Python 3.8+** - Check with `python --version`
5. **Try the ZIP download method** - It's simpler and avoids Git issues

---

## ✅ **QUICK REFERENCE: Commands Summary**

```bash
# 1. Clone or download the repository
git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
# OR download ZIP from GitHub

# 2. Enter the project folder
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
# OR if you renamed it:
cd diabetes-project

# 3. Run everything (ONE COMMAND!)
python run_all.py

# 4. Verify everything worked
python test_models.py

# 5. View full results and metrics (IMPORTANT!)
# Windows:
.venv\Scripts\activate
python scripts/run_eval.py

# Mac/Linux:
source .venv/bin/activate
python scripts/run_eval.py

# 6. (Optional) Launch dashboard
# Windows:
.venv\Scripts\activate
streamlit run dashboard.py

# Mac/Linux:
source .venv/bin/activate
streamlit run dashboard.py
```

---

**That's it! You should now be able to clone and run the project successfully!** 🎉

