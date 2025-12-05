# Quick Start Guide - One Command to Run Everything

## 🚀 **EASIEST WAY: Download & Run in One Command**

Your professor can download everything from GitHub and run the entire project with **just one command**!

---

## **For Mac/Linux Users:**

### **Option 1: Bash Script**
```bash
./download_and_run.sh
```

### **Option 2: Python Script (Recommended)**
```bash
python download_and_run.py
```

---

## **For Windows Users:**

### **Option 1: Batch File**
```cmd
download_and_run.bat
```

### **Option 2: Python Script (Recommended)**
```cmd
python download_and_run.py
```

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

## **If You Already Have the Repository:**

If you've already downloaded/cloned the repository, use these scripts instead:

- **Mac/Linux:** `./run_all.sh` or `python run_all.py`
- **Windows:** `run_all.bat` or `python run_all.py`

These skip the download step and go straight to setup and training.

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

## **After Running:**

Once the script completes, you can:

1. **View results** in the terminal output
2. **Launch dashboard:**
   ```bash
   cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
   source .venv/bin/activate    # Mac/Linux
   # OR
   .venv\Scripts\activate       # Windows
   streamlit run dashboard.py
   ```

---

## **Repository URL:**

```
https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
```

---

**That's it! One command does everything!** 🎉

