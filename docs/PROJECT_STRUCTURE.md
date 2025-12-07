# Project Structure - Diabetes 30-Day Readmission Prediction

## 📁 Complete Repository Organization

```
diabetes-readmission-prediction/
│
├── 📄 README.md                    # Main project README (START HERE!)
├── 📄 requirements.txt             # Python dependencies
├── 📄 dashboard.py                 # Streamlit dashboard application
│
├── 📄 run_all.py                   # Master script: runs everything automatically
├── 📄 run_all.bat                  # Windows batch version
├── 📄 run_all.sh                   # Mac/Linux shell version
│
├── 📄 download_and_run.py          # Downloads repo and runs everything
├── 📄 download_and_run.bat         # Windows batch version
├── 📄 download_and_run.sh          # Mac/Linux shell version
│
├── 📄 test_models.py               # Verifies models were created correctly
│
├── 📄 CLONE_AND_RUN_GUIDE.md       # Complete guide for fresh clones
├── 📄 PROJECT_EXPLANATION_GUIDE.md # Technical explanation (what, why, where)
│
├── 📂 src/                         # Core source code
│   ├── __init__.py
│   ├── config.py                   # Configuration settings
│   ├── preprocess.py               # Data preprocessing
│   ├── feature_selection.py        # Feature selection
│   ├── model.py                    # Model definitions
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Model evaluation
│   └── clinical_utils.py           # Clinical risk interpretation
│
├── 📂 scripts/                     # Execution scripts
│   ├── run_train.py                # Train models
│   ├── run_eval.py                 # Evaluate models
│   └── run_dashboard.py            # Launch dashboard
│
├── 📂 data/                        # Data files
│   ├── raw/                        # Raw data (committed to git)
│   │   ├── diabetic_data.csv       # Main dataset (18 MB, 101,766 records)
│   │   └── IDS_mapping.csv         # ID mappings (2.5 KB)
│   └── processed/                  # Processed data (gitignored, auto-generated)
│       ├── train_processed.csv     # Cleaned training data (80%)
│       └── test_processed.csv      # Cleaned test data (20%)
│
├── 📂 models/                      # Trained models (gitignored temp files)
│   ├── logreg_selected.joblib      # Logistic Regression model
│   ├── xgb_selected.joblib         # XGBoost model
│   └── thresholds.json             # Optimized thresholds and features
│
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_modeling.ipynb           # Modeling experiments
│   └── 03_implementation_details.ipynb  # Implementation narrative
│
├── 📂 docs/                        # Documentation
│   ├── README.md                   # Documentation index
│   ├── HOW_TO_VIEW_RESULTS.md      # Detailed results viewing guide
│   ├── WINDOWS_FIX.md              # General Windows troubleshooting
│   ├── WINDOWS_PATH_LENGTH_FIX.md # Windows path length error fix
│   ├── WINDOWS_PIP_FIX.md          # Windows pip launcher error fix
│   ├── PROJECT_STRUCTURE.md       # This file - repository structure
│   ├── P3_SUBMISSION_CHECKLIST.md  # P3 submission checklist
│   ├── P3_SUBMISSION_SUMMARY.md    # Quick submission reference
│   ├── PRESENTATION_SLIDES_SHORT.Rmd  # Main presentation file
│   ├── CLEANUP_SUMMARY.md          # Repository cleanup summary
│   └── archive/                    # Archived presentations
│       ├── PRESENTATION_SLIDES.md
│       ├── PRESENTATION_SLIDES_SHORT.md
│       └── PRESENTATION_FILES_README.md
│
├── 📂 reports/                     # Reports and submissions
│   ├── P2 Final_submission report.pdf
│   ├── P2 Final_submission report.docx
│   └── P3_FINAL_REPORT.md          # P3 final report
│
└── 📂 tests/                       # Test files (empty, ready for tests)
```

## 🎯 File Organization Guide

### **Root Level Files:**
- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies
- `dashboard.py` - Streamlit dashboard

### **Source Code (`src/`):**
All core Python modules for data processing, modeling, and evaluation.

### **Scripts (`scripts/`):**
Executable scripts to run the pipeline end-to-end.

### **Data (`data/`):**
- `raw/` - Original data files (committed to git)
- `processed/` - Generated processed data (gitignored)

### **Models (`models/`):**
Trained model files and configuration (thresholds.json).

### **Notebooks (`notebooks/`):**
Jupyter notebooks for EDA, modeling, and implementation details.

### **Documentation (`docs/`):**
All project documentation, guides, and presentation files.

### **Reports (`reports/`):**
Submission reports (P2 and P3).

## ✅ Organization Complete!

All files are properly organized and in their correct locations.

