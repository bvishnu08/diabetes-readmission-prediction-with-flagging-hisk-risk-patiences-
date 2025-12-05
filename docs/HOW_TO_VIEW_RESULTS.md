# 📊 How to View Results - Complete Guide

After running `python run_all.py` and seeing "✅ ALL STEPS COMPLETED SUCCESSFULLY!", here's how to view your results in different ways.

---

## 🖥️ Method 1: View Results in Terminal (Windows & Mac)

### **What You Already Saw**

When `run_all.py` finished, it automatically printed the evaluation results in your terminal. **Scroll up** in your terminal window to see them!

**Look for this section:**
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

---

### **View Results Again (If You Want to See Them Again)**

#### **For Windows Users:**

**Step 1:** Open Command Prompt or PowerShell

**Step 2:** Navigate to your project folder:
```cmd
cd C:\Users\YourName\Downloads\diabetes-project
```
*(Replace with your actual folder path)*

**Step 3:** Activate the virtual environment:
```cmd
.venv\Scripts\activate
```

**What you'll see:** Your prompt will change to show `(.venv)` at the start:
```
(.venv) C:\Users\YourName\Downloads\diabetes-project>
```

**Step 4:** Run the evaluation script:
```cmd
python scripts\run_eval.py
```

**What you'll see:** The evaluation results will be printed in your terminal!

---

#### **For Mac/Linux Users:**

**Step 1:** Open Terminal

**Step 2:** Navigate to your project folder:
```bash
cd ~/Downloads/diabetes-project
```
*(Replace with your actual folder path)*

**Step 3:** Activate the virtual environment:
```bash
source .venv/bin/activate
```

**What you'll see:** Your prompt will change to show `(.venv)` at the start:
```
(.venv) username@computer:~/Downloads/diabetes-project$
```

**Step 4:** Run the evaluation script:
```bash
python scripts/run_eval.py
```

**What you'll see:** The evaluation results will be printed in your terminal!

---

### **What the Results Mean (Simple Explanation)**

- **Accuracy:** Overall correctness (65-68% = models are better than random guessing)
- **Recall:** How many readmissions we catch (70-71% = we catch most high-risk patients) ✅
- **Precision:** How many predictions are correct (15-17% = lower is okay, we prioritize catching all high-risk patients)
- **F1-Score:** Balance between precision and recall (higher is better)
- **ROC-AUC:** Overall model quality (0.64-0.68 = better than random, room for improvement)

**Bottom line:** Both models work! XGBoost is recommended because it's slightly better.

---

## 📓 Method 2: View Results in Jupyter Notebooks

Jupyter notebooks let you see results with code, charts, and explanations all in one place!

### **For Windows Users:**

#### **Step 1: Open Terminal/PowerShell**

Open Command Prompt or PowerShell

#### **Step 2: Navigate to Project Folder**

```cmd
cd C:\Users\YourName\Downloads\diabetes-project
```
*(Replace with your actual folder path)*

#### **Step 3: Activate Virtual Environment**

```cmd
.venv\Scripts\activate
```

**You should see:** `(.venv)` at the start of your prompt

#### **Step 4: Install Jupyter (If Not Already Installed)**

```cmd
pip install jupyter jupyterlab
```

**This downloads Jupyter** - it might take 1-2 minutes

#### **Step 5: Launch Jupyter Lab**

```cmd
jupyter lab
```

**What happens:**
- Your web browser will open automatically
- You'll see the Jupyter Lab interface
- Navigate to the `notebooks/` folder
- Open `03_implementation_details.ipynb`

**OR launch directly to the notebook:**

```cmd
jupyter lab notebooks\03_implementation_details.ipynb
```

**This opens the notebook directly!**

#### **Step 6: View Results in Notebook**

Once the notebook opens:
- **Scroll through the cells** to see all the results
- **Click on a cell** and press `Shift + Enter` to run it
- **See visualizations** (charts, graphs, tables)
- **Read explanations** between code cells

**To stop Jupyter:**
- Go back to your terminal
- Press `Ctrl + C`
- Type `y` when asked to shutdown

---

### **For Mac/Linux Users:**

#### **Step 1: Open Terminal**

Open Terminal application

#### **Step 2: Navigate to Project Folder**

```bash
cd ~/Downloads/diabetes-project
```
*(Replace with your actual folder path)*

#### **Step 3: Activate Virtual Environment**

```bash
source .venv/bin/activate
```

**You should see:** `(.venv)` at the start of your prompt

#### **Step 4: Install Jupyter (If Not Already Installed)**

```bash
pip install jupyter jupyterlab
```

**This downloads Jupyter** - it might take 1-2 minutes

#### **Step 5: Launch Jupyter Lab**

```bash
jupyter lab
```

**What happens:**
- Your web browser will open automatically (usually Safari or Chrome)
- You'll see the Jupyter Lab interface
- Navigate to the `notebooks/` folder
- Open `03_implementation_details.ipynb`

**OR launch directly to the notebook:**

```bash
jupyter lab notebooks/03_implementation_details.ipynb
```

**This opens the notebook directly!**

#### **Step 6: View Results in Notebook**

Once the notebook opens:
- **Scroll through the cells** to see all the results
- **Click on a cell** and press `Shift + Enter` to run it
- **See visualizations** (charts, graphs, tables)
- **Read explanations** between code cells

**To stop Jupyter:**
- Go back to your terminal
- Press `Ctrl + C`
- Type `y` when asked to shutdown

---

## 🎯 Quick Reference: All Methods to View Results

### **Method 1: Terminal Output (Easiest - Already Done!)**
- ✅ **Already shown** when you ran `python run_all.py`
- Scroll up in terminal to see results
- Or run: `python scripts/run_eval.py` (after activating venv)

### **Method 2: Jupyter Notebooks (Best for Learning)**
- **Windows:** `jupyter lab notebooks\03_implementation_details.ipynb`
- **Mac/Linux:** `jupyter lab notebooks/03_implementation_details.ipynb`
- See code, results, and visualizations together

### **Method 3: Interactive Dashboard (Most Visual)**
- **Windows:** `streamlit run dashboard.py` (after activating venv)
- **Mac/Linux:** `streamlit run dashboard.py` (after activating venv)
- See charts, graphs, and make predictions interactively

### **Method 4: Verify Files (Check Everything Worked)**
- Run: `python test_models.py`
- Verifies all files were created correctly

---

## 🔧 Troubleshooting

### **"jupyter: command not found"**
**Solution:** Make sure you activated the virtual environment first!
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

### **"ModuleNotFoundError: No module named 'jupyter'"**
**Solution:** Install Jupyter:
```bash
pip install jupyter jupyterlab
```

### **Browser didn't open automatically**
**Solution:** That's okay! Just open your browser manually and go to:
- The URL shown in the terminal (usually `http://localhost:8888`)

### **"Port already in use"**
**Solution:** Someone else is using Jupyter. Either:
- Close the other Jupyter instance
- Or wait a minute and try again

### **Notebook cells won't run**
**Solution:** Make sure you're in the project folder and virtual environment is activated

---

## 📝 Summary

**Easiest way to see results:**
1. ✅ **Already done!** - Results are in your terminal from `run_all.py`
2. Scroll up to see the "MODEL EVALUATION" section

**Best way to explore results:**
1. Open Jupyter Lab (instructions above)
2. Open `03_implementation_details.ipynb`
3. See all results with code and visualizations

**Most visual way:**
1. Launch dashboard: `streamlit run dashboard.py`
2. See interactive charts and graphs

---

**That's it! You now know all the ways to view your results!** 🎉

