# Project Structure - Diabetes 30-Day Readmission Prediction

## 📁 Complete Repository Organization

```
265_final/
│
├── 📄 README.md                    # Main project README
├── 📄 requirements.txt              # Python dependencies
├── 📄 dashboard.py                 # Streamlit dashboard application
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
│   ├── raw/                        # Raw data (keep in git)
│   │   ├── diabetic_data.csv
│   │   └── IDS_mapping.csv
│   └── processed/                  # Processed data (gitignored)
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── 📂 models/                      # Trained models
│   ├── logreg_selected.joblib      # Logistic Regression model
│   ├── xgb_selected.joblib         # XGBoost model
│   ├── rf_selected.joblib          # Random Forest model
│   ├── logreg_10feat.joblib        # 10-feature LR model
│   └── thresholds.json             # Optimized thresholds
│
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_modeling.ipynb           # Modeling experiments
│   ├── 03_implementation_details.ipynb  # Implementation narrative
│   └── 03_implementation_details.html   # Exported HTML
│
├── 📂 docs/                        # Documentation
│   ├── README.md                   # Documentation guide
│   ├── RUN_BOOK.md                 # Step-by-step execution guide
│   ├── COMPLETE_PROJECT_CODE.md    # All code in one document
│   ├── CODE_EXPLANATION.md         # Detailed code explanations
│   ├── PIPELINE_DIFFERENCES.md     # Notebook vs production
│   ├── P3_SUBMISSION_CHECKLIST.md  # P3 submission checklist
│   ├── P3_SUBMISSION_SUMMARY.md    # Quick submission reference
│   ├── PRESENTATION_SLIDES_SHORT.Rmd  # Main presentation file
│   └── archive/                    # Old/redundant files
│       ├── PRESENTATION_SLIDES.md
│       ├── PRESENTATION_SLIDES_SHORT.md
│       └── PRESENTATION_FILES_README.md
│
├── 📂 reports/                     # Reports and submissions
│   ├── P2 Final_submission report.pdf
│   ├── P2 Final_submission report.docx
│   └── P3_FINAL_REPORT.md          # P3 final report
│
└── 📂 tests/                       # Test files (empty, for future use)
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

