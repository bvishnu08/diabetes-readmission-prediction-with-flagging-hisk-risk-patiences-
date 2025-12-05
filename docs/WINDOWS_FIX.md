# Windows Error Fix Guide

## Problem
When running `download_and_run.py` on Windows, you may see:
```
❌ Unexpected error: [WinError 2] The system cannot find the file specified: 'diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-'
```

## Solution 1: Download ZIP Instead (EASIEST)

If Git is not installed or you prefer not to use it:

1. **Go to the GitHub repository:**
   ```
   https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
   ```

2. **Click the green "Code" button**

3. **Click "Download ZIP"**

4. **Extract the ZIP file** to your desired location

5. **Rename the extracted folder** to:
   ```
   diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
   ```

6. **Open Command Prompt** in that folder

7. **Run directly:**
   ```cmd
   python run_all.py
   ```
   OR
   ```cmd
   run_all.bat
   ```

---

## Solution 2: Install Git and Try Again

1. **Download and install Git for Windows:**
   - Go to: https://git-scm.com/download/win
   - Download and install

2. **Restart Command Prompt** after installation

3. **Run the script again:**
   ```cmd
   python download_and_run.py
   ```

---

## Solution 3: Manual Setup (If Scripts Don't Work)

If both methods above don't work, set up manually:

1. **Download ZIP** (as in Solution 1)

2. **Extract and navigate to folder:**
   ```cmd
   cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
   ```

3. **Create virtual environment:**
   ```cmd
   python -m venv .venv
   ```

4. **Activate virtual environment:**
   ```cmd
   .venv\Scripts\activate
   ```

5. **Install packages:**
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. **Run training:**
   ```cmd
   python scripts\run_train.py
   ```

7. **Run evaluation:**
   ```cmd
   python scripts\run_eval.py
   ```

---

## What Was Fixed

The scripts have been updated to:
- ✅ Check if repository folder exists before trying to access it
- ✅ Provide clear error messages
- ✅ Give alternative instructions (ZIP download)
- ✅ Handle Windows-specific errors better

---

## Quick Test

After setup, verify everything works:

```cmd
python scripts\run_eval.py
```

You should see model evaluation results.

---

## Still Having Issues?

If you're still getting errors:

1. **Check Python is installed:**
   ```cmd
   python --version
   ```
   Should show Python 3.8 or higher

2. **Check you're in the right folder:**
   ```cmd
   dir
   ```
   Should show `scripts`, `src`, `data` folders

3. **Try running from project root:**
   ```cmd
   python run_all.py
   ```

---

**Updated scripts are now on GitHub with better Windows error handling!**

