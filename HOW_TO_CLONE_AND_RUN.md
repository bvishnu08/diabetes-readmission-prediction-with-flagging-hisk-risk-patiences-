# 📥 How to Clone and Run This Project

**Repository:** [https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-](https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-)

---

## 🚀 **METHOD 1: Git Clone (If Git is Installed)**

### Step 1: Open Terminal/Command Prompt
- **Mac/Linux:** Open Terminal
- **Windows:** Open Command Prompt or PowerShell

### Step 2: Navigate to a Short Path
```bash
# Windows
cd C:\Projects

# Mac/Linux
cd ~/Projects
```

### Step 3: Clone the Repository
```bash
git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
```

### Step 4: Enter the Project Folder
```bash
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
```

### Step 5: Run Everything Automatically
```bash
python run_all.py
```

**That's it!** The script will:
- ✅ Create virtual environment
- ✅ Install all packages
- ✅ Train both models
- ✅ Evaluate models
- ✅ Show results

---

## 📦 **METHOD 2: Download ZIP (No Git Needed) - RECOMMENDED FOR WINDOWS**

### Step 1: Go to GitHub
Open: https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-

### Step 2: Download ZIP
1. Click the green **"Code"** button
2. Click **"Download ZIP"**

### Step 3: Extract to Short Path
- **Windows:** Extract to `C:\Projects\` (create folder if needed)
- **Mac/Linux:** Extract to `~/Projects/`
- **Important:** Rename the extracted folder to something SHORT (e.g., `diabetes-project`)

### Step 4: Open Terminal in That Folder
```bash
# Windows
cd C:\Projects\diabetes-project

# Mac/Linux
cd ~/Projects/diabetes-project
```

### Step 5: Run Everything
```bash
python run_all.py
```

---

## 🎯 **QUICK COPY-PASTE COMMANDS**

### For Windows:
```cmd
cd C:\Projects
git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
python run_all.py
```

### For Mac/Linux:
```bash
cd ~/Projects
git clone https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
cd diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-
python run_all.py
```

---

## ⚠️ **IMPORTANT NOTES**

### Windows Users:
- ⚠️ **Use short paths** to avoid "Filename too long" errors
- ✅ **Good:** `C:\Projects\diabetes-project`
- ❌ **Bad:** `C:\Users\YourName\Downloads\very-long-nested-path\...`
- 📖 See `docs/WINDOWS_PATH_LENGTH_FIX.md` for detailed help

### Requirements:
- **Python 3.8+** must be installed
- **Internet connection** for first-time download
- **5-10 minutes** for setup and training

---

## 🔍 **VERIFY IT WORKED**

After running `python run_all.py`, you should see:
- ✅ Virtual environment created
- ✅ Packages installed
- ✅ Models trained
- ✅ Evaluation results displayed

---

## 📚 **NEXT STEPS**

After setup is complete:

1. **View Results:** Check the terminal output for model metrics
2. **Launch Dashboard:**
   ```bash
   source .venv/bin/activate    # Mac/Linux
   # OR
   .venv\Scripts\activate       # Windows
   
   streamlit run dashboard.py
   ```
   Then open: `http://localhost:8501`

3. **Run Notebooks:**
   ```bash
   jupyter lab notebooks/03_implementation_details.ipynb
   ```

---

## 🆘 **TROUBLESHOOTING**

### "Git is not installed"
- **Solution:** Use Method 2 (Download ZIP) instead
- Or install Git: https://git-scm.com/download/win

### "Filename too long" (Windows)
- **Solution:** Extract ZIP to a short path like `C:\Projects\`
- See: `docs/WINDOWS_PATH_LENGTH_FIX.md`

### "Python not found"
- **Solution:** Install Python 3.8+ from python.org
- Make sure Python is in your PATH

### "ModuleNotFoundError"
- **Solution:** Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

---

## 📖 **MORE HELP**

- **Complete Setup Guide:** `SETUP_GUIDE.md`
- **Quick Start:** `docs/QUICK_START.md`
- **Windows Fixes:** `docs/WINDOWS_PATH_LENGTH_FIX.md`
- **Project Structure:** `docs/PROJECT_STRUCTURE.md`

---

**That's it! Your project is ready to run!** 🎉

