# 🚀 Quick Start Guide - One Command to Run Everything

**👋 For Complete Beginners:** This guide shows you the easiest way to get started. Don't worry - everything is explained step-by-step!

---

## **EASIEST WAY: Download & Run in One Command**

You can download everything from GitHub and run the entire project with **just one command**!

**What this does:** Downloads the code, sets everything up, trains the models, and shows you the results - all automatically!

---

## **For Mac/Linux Users:**

### **Option 1: Python Script (Recommended - Easiest!)**

**Step 1:** Open Terminal

**Step 2:** Navigate to where you want to download the project (e.g., your Downloads folder):
```bash
cd ~/Downloads
```

**Step 3:** Download and run everything:
```bash
python download_and_run.py
```

**That's it!** The script will:
- Download the repository from GitHub
- Set up everything automatically
- Train the models
- Show you the results

**If you get "python: command not found":** Try `python3` instead:
```bash
python3 download_and_run.py
```

### **Option 2: Bash Script (Alternative)**

If you prefer using a shell script:
```bash
chmod +x download_and_run.sh
./download_and_run.sh
```

---

## **For Windows Users:**

### **Option 1: Python Script (Recommended - Easiest!)**

**Step 1:** Open Command Prompt or PowerShell

**Step 2:** Navigate to where you want to download the project (e.g., C:\Users\YourName\Downloads):
```cmd
cd C:\Users\YourName\Downloads
```

**Step 3:** Download and run everything:
```cmd
python download_and_run.py
```

**That's it!** The script will:
- Download the repository from GitHub
- Set up everything automatically
- Train the models
- Show you the results

**If you get "python is not recognized":** 
- Make sure Python is installed and added to PATH
- Or try: `py download_and_run.py`

### **Option 2: Batch File (Alternative)**

If you prefer using a batch file:
```cmd
download_and_run.bat
```

**Note:** Double-click the file in Windows Explorer, or run it from Command Prompt.

---

## **What These Scripts Do Automatically:**

1. ✅ **Download** the repository from GitHub (if not already downloaded)
2. ✅ **Create** virtual environment
3. ✅ **Install** all required packages
4. ✅ **Train** both models (Logistic Regression + XGBoost)
5. ✅ **Evaluate** the models
6. ✅ **Show** results

**Total time: 5-10 minutes** (mostly waiting for training)

---

## **Prerequisites:**

- **Python 3.8+** installed
- **Git** installed (optional - ZIP download works without Git)
- **Internet connection** (for first-time download)

## **⚠️ Windows Users - Important:**

If you get a "Filename too long" error:
- **Use ZIP download** instead of Git clone
- **Extract to a SHORT path** (e.g., `C:\Projects\`)
- **Rename folder to something SHORT** (e.g., `diabetes-project`)
- See `docs/WINDOWS_PATH_LENGTH_FIX.md` for detailed instructions

---

---

## **If You Already Have the Repository:**

If you've already downloaded/cloned the repository (you have the project folder on your computer), use these scripts instead:

**For Mac/Linux:**
```bash
# Option 1: Python script (recommended)
python run_all.py

# Option 2: Shell script
chmod +x run_all.sh
./run_all.sh
```

**For Windows:**
```cmd
REM Option 1: Python script (recommended)
python run_all.py

REM Option 2: Batch file
run_all.bat
```

**What's the difference?**
- `download_and_run.py` - Downloads from GitHub first, then runs everything
- `run_all.py` - Assumes you already have the code, just sets up and runs

**These skip the download step and go straight to setup and training.**

---

## **Troubleshooting:**

### **"Git is not installed"**
- **Mac:** `brew install git`
- **Linux:** `sudo apt-get install git`
- **Windows:** Download from https://git-scm.com/download/win
- **Alternative:** Download ZIP from GitHub and extract, then use `run_all.sh`/`run_all.bat`

### **"Python not found"**
- Make sure Python 3.8+ is installed
- Try `python3` instead of `python`

### **"Permission denied" (Mac/Linux)**
- Make script executable: `chmod +x download_and_run.sh`
- Or use: `python download_and_run.py`

---

---

## **After Running - What to Do Next:**

Once the script completes and you see "✅ ALL STEPS COMPLETED SUCCESSFULLY!", here's what to do:

### **Step 1: Check the Results (Already Done!)**
✅ The results are already shown in your terminal! Look for the "MODEL EVALUATION" section.

### **Step 2: Launch the Interactive Dashboard (Recommended!)**

**For Mac/Linux:**
```bash
# Navigate to the project folder (if not already there)
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-

# Activate the virtual environment
source .venv/bin/activate

# Run the dashboard
streamlit run dashboard.py
```

**For Windows:**
```cmd
REM Navigate to the project folder (if not already there)
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-

REM Activate the virtual environment
.venv\Scripts\activate

REM Run the dashboard
streamlit run dashboard.py
```

**What happens:**
- Your web browser will open automatically
- You'll see a beautiful dashboard with charts and graphs
- You can explore model performance and make predictions

**To stop the dashboard:** Press `Ctrl+C` in the terminal

**For detailed step-by-step instructions (with explanations for beginners), see the main [README.md](../README.md) file!**

---

## **Repository URL:**

```
https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
```

---

**That's it! One command does everything!** 🎉

