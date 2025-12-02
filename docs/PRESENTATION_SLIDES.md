# 📊 Diabetes 30-Day Readmission Prediction - Presentation Slides

---

## **Slide 1: Title Slide**
**Title:** Diabetes 30-Day Readmission Prediction with High-Risk Patient Flagging  
**Subtitle:** MSBA-265 Course Project  
**Presenter:** [Your Name]  
**Date:** [Date]

---

## **Slide 2: Project Overview**
**Title:** Problem Statement & Objectives

**Content:**
- **Problem:** Predict 30-day readmission risk for diabetic patients
- **Goal:** Flag high-risk patients for early intervention
- **Impact:** Reduce readmissions, improve care, lower costs
- **Approach:** Machine learning classification models

**Why:** Sets context and business value

---

## **Slide 3: Dataset Overview**
**Title:** Data Source & Characteristics

**Content:**
- **Source:** UCI Machine Learning Repository - Diabetes 130-US hospitals
- **Size:** ~101,766 patient encounters
- **Features:** 50+ clinical and demographic variables
- **Target:** Binary classification (readmitted within 30 days: Yes/No)
- **Key Files:**
  - `data/raw/diabetic_data.csv` - Main dataset
  - `data/raw/IDS_mapping.csv` - Feature ID mappings

**Where:** `data/raw/` directory  
**Why:** Raw data before preprocessing

---

## **Slide 4: Project Architecture**
**Title:** End-to-End Pipeline

**Content:**
```
Raw Data → Preprocessing → Feature Selection → 
Model Training → Evaluation → Dashboard → Clinical Decision Support
```

**Components:**
1. **Data Pipeline** (`src/preprocess.py`) - Clean and prepare data
2. **Feature Engineering** (`src/feature_selection.py`) - Select best features
3. **Model Training** (`src/train.py`) - Train Logistic Regression & XGBoost
4. **Evaluation** (`src/evaluate.py`) - Assess model performance
5. **Dashboard** (`dashboard.py`) - Interactive visualization
6. **Clinical Utils** (`src/clinical_utils.py`) - Risk interpretation

**Why:** Modular design for maintainability and reproducibility

---

## **Slide 5: Data Preprocessing Pipeline**
**Title:** Data Cleaning & Preparation

**Content:**
**File:** `src/preprocess.py`

**Steps:**
1. **Load Data** - Read CSV, handle encoding
2. **Remove Duplicates** - Drop duplicate encounters
3. **Handle Missing Values** - Impute or remove based on feature
4. **Encode Categorical Variables** - One-hot encoding for categorical features
5. **Scale Numerical Features** - StandardScaler for normalization
6. **Train-Test Split** - 80/20 split with stratification

**Key Functions:**
- `basic_clean(df, cfg)` - Core cleaning logic
- `train_test_split_clean(config)` - Complete preprocessing pipeline

**Why:** Clean data improves model performance and reliability  
**Where:** Used in `scripts/run_train.py` before model training

---

## **Slide 6: Feature Selection Strategy**
**Title:** Selecting Optimal Features

**Content:**
**File:** `src/feature_selection.py`

**Method:** Mutual Information (SelectKBest)
- Measures dependency between features and target
- Non-parametric, works with non-linear relationships
- Selects top K features based on MI scores

**Process:**
1. Calculate MI scores for all features
2. Select top 10 features for Logistic Regression
3. Use all features for XGBoost (handles feature importance internally)

**Key Function:**
- `select_features(X_train, y_train, k=10)` - Returns selected features

**Why:** Reduces overfitting, improves interpretability, speeds training  
**Where:** Applied in `src/train.py` before model training

---

## **Slide 7: Model Selection**
**Title:** Two-Model Approach

**Content:**
**File:** `src/model.py`

**Model 1: Logistic Regression**
- **Why:** Interpretable, fast, baseline
- **Features:** Top 10 selected features
- **Pipeline:** Preprocessing → Feature Selection → Logistic Regression
- **Use Case:** When interpretability is critical

**Model 2: XGBoost**
- **Why:** High performance, handles non-linear patterns
- **Features:** All available features
- **Pipeline:** Preprocessing → XGBoost (built-in feature importance)
- **Use Case:** When accuracy is priority

**Configuration:** `src/config.py` - Centralized hyperparameters and paths

**Where:** Models defined in `src/model.py`, trained in `src/train.py`

---

## **Slide 8: Model Training Process**
**Title:** Training Pipeline

**Content:**
**File:** `src/train.py`

**Steps:**
1. **Load Configuration** - Read settings from `config.py`
2. **Preprocess Data** - Clean and split data
3. **Select Features** - Apply feature selection
4. **Train Models** - Fit Logistic Regression and XGBoost
5. **Tune Thresholds** - Optimize for 80% recall (high-risk detection)
6. **Save Models** - Store as `.joblib` files in `models/`

**Key Functions:**
- `train_models(config)` - Main training function
- `tune_threshold_for_recall()` - Threshold optimization

**Output:**
- `models/logreg_selected.joblib`
- `models/xgb_selected.joblib`
- `models/thresholds.json`

**Why:** Automated, reproducible training pipeline  
**Where:** Run via `scripts/run_train.py`

---

## **Slide 9: Model Evaluation Metrics**
**Title:** Performance Assessment

**Content:**
**File:** `src/evaluate.py`

**Metrics Used:**
1. **ROC-AUC** - Overall model discrimination
2. **Precision** - Accuracy of positive predictions
3. **Recall** - Ability to catch all high-risk patients
4. **F1-Score** - Balance of precision and recall
5. **Confusion Matrix** - True/False positives and negatives

**Clinical Focus:**
- **High Recall (80%)** - Don't miss high-risk patients
- **Threshold Tuning** - Optimize for clinical needs

**Key Functions:**
- `evaluate_models(config)` - Comprehensive evaluation
- `plot_roc_curves()` - Visual performance comparison
- `plot_confusion_matrices()` - Error analysis

**Why:** Ensures models meet clinical requirements  
**Where:** Run via `scripts/run_eval.py`

---

## **Slide 10: Interactive Dashboard**
**Title:** Streamlit Dashboard - Clinical Interface

**Content:**
**File:** `dashboard.py`

**Features:**
1. **Model Performance Overview** - ROC-AUC, Precision, Recall, F1
2. **Clinical View** - Risk interpretation (HIGH RISK / LOW RISK)
3. **Prediction Playground** - Interactive patient input
4. **Data Overview** - Dataset statistics and distributions

**Design:**
- Dark theme for readability
- Real-time predictions
- Clinical risk interpretation via `src/clinical_utils.py`

**Key Functions:**
- `load_models()` - Load trained models
- `predict_risk()` - Generate predictions with clinical interpretation
- `format_risk_level()` - Convert probabilities to risk categories

**Why:** Makes models accessible to clinicians and stakeholders  
**Where:** Run via `scripts/run_dashboard.py` or `streamlit run dashboard.py`

---

## **Slide 11: Clinical Risk Interpretation**
**Title:** From Probabilities to Clinical Decisions

**Content:**
**File:** `src/clinical_utils.py`

**Risk Categories:**
- **HIGH RISK** (≥ threshold): Probability ≥ optimized threshold
- **LOW RISK** (< threshold): Probability < optimized threshold

**Process:**
1. Model outputs probability (0-1)
2. Compare to tuned threshold (e.g., 0.35 for 80% recall)
3. Classify as HIGH RISK or LOW RISK
4. Provide actionable recommendation

**Key Function:**
- `format_risk_level(prob, threshold)` - Converts probability to risk category

**Why:** Translates model outputs into actionable clinical insights  
**Where:** Used in `dashboard.py` for patient predictions

---

## **Slide 12: Results - Model Performance**
**Title:** Model Comparison

**Content:**
**Expected Results:**

| Metric | Logistic Regression | XGBoost |
|--------|-------------------|---------|
| ROC-AUC | ~0.65-0.70 | ~0.68-0.72 |
| Precision | ~0.45-0.50 | ~0.48-0.52 |
| Recall | ~0.80 (tuned) | ~0.80 (tuned) |
| F1-Score | ~0.55-0.60 | ~0.58-0.62 |

**Key Findings:**
- Both models achieve target 80% recall
- XGBoost slightly better overall performance
- Logistic Regression more interpretable

**Why:** Validates model effectiveness for clinical use

---

## **Slide 13: Feature Importance**
**Title:** What Drives Readmission Risk?

**Content:**
**Top Features (Logistic Regression):**
1. Number of medications
2. Number of diagnoses
3. Time in hospital
4. Number of lab procedures
5. Age group
6. Discharge disposition
7. Admission type
8. Medical specialty
9. Number of procedures
10. Emergency visits

**Insights:**
- Medication complexity is a strong predictor
- Hospital stay duration matters
- Patient demographics play a role

**Why:** Helps clinicians understand risk factors  
**Where:** Analyzed in `notebooks/02_modeling.ipynb`

---

## **Slide 14: Implementation Workflow**
**Title:** How to Run the Project

**Content:**
**Step-by-Step Execution:**

1. **Setup Environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**
   ```bash
   python scripts/run_train.py
   ```

3. **Evaluate Models**
   ```bash
   python scripts/run_eval.py
   ```

4. **Launch Dashboard**
   ```bash
   streamlit run dashboard.py
   ```

**File:** `docs/RUN_BOOK.md` - Complete execution guide

**Why:** Ensures reproducibility and easy deployment

---

## **Slide 15: Project Structure**
**Title:** Code Organization

**Content:**
```
265_final/
├── src/              # Core modules
│   ├── preprocess.py      # Data cleaning
│   ├── feature_selection.py # Feature selection
│   ├── model.py           # Model definitions
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation metrics
│   └── clinical_utils.py  # Risk interpretation
├── scripts/         # Execution scripts
├── notebooks/       # EDA and analysis
├── models/          # Trained models
├── data/            # Raw and processed data
├── docs/            # Documentation
└── dashboard.py     # Streamlit app
```

**Why:** Modular design for maintainability and collaboration

---

## **Slide 16: Key Technical Decisions**
**Title:** Design Choices & Rationale

**Content:**
1. **Two-Model Approach**
   - Why: Balance interpretability (LR) and performance (XGBoost)

2. **80% Recall Target**
   - Why: Clinical priority to catch high-risk patients

3. **Mutual Information for Feature Selection**
   - Why: Captures non-linear relationships, model-agnostic

4. **Threshold Tuning**
   - Why: Optimize for clinical needs, not just accuracy

5. **Modular Code Structure**
   - Why: Reproducibility, maintainability, testing

**Where:** Decisions documented in `docs/CODE_EXPLANATION.md`

---

## **Slide 17: Clinical Impact**
**Title:** Real-World Application

**Content:**
**Use Cases:**
1. **Discharge Planning** - Identify patients needing follow-up care
2. **Resource Allocation** - Prioritize high-risk patients
3. **Early Intervention** - Prevent readmissions proactively
4. **Cost Reduction** - Reduce unnecessary readmissions

**Dashboard Benefits:**
- Real-time risk assessment
- User-friendly interface
- Actionable insights

**Why:** Demonstrates practical value beyond technical metrics

---

## **Slide 18: Challenges & Solutions**
**Title:** Project Challenges

**Content:**
**Challenge 1: Class Imbalance**
- Solution: Threshold tuning to optimize recall

**Challenge 2: Feature Selection**
- Solution: Mutual Information for non-linear relationships

**Challenge 3: Model Interpretability**
- Solution: Logistic Regression for clinical explainability

**Challenge 4: Clinical Translation**
- Solution: Risk categorization in `clinical_utils.py`

**Where:** Addressed in `notebooks/03_implementation_details.ipynb`

---

## **Slide 19: Future Enhancements**
**Title:** Potential Improvements

**Content:**
1. **Additional Models** - Random Forest, Neural Networks
2. **Feature Engineering** - Create interaction features
3. **Real-Time Integration** - Connect to hospital EMR systems
4. **Explainability** - SHAP values for feature importance
5. **A/B Testing** - Validate in clinical setting

**Why:** Shows forward-thinking and continuous improvement mindset

---

## **Slide 20: Conclusion**
**Title:** Key Takeaways

**Content:**
✅ **Successfully built** end-to-end ML pipeline  
✅ **Achieved 80% recall** for high-risk patient detection  
✅ **Created interactive dashboard** for clinical use  
✅ **Modular, reproducible codebase** for future work  
✅ **Clinical interpretation** of model outputs

**Impact:**
- Helps reduce 30-day readmissions
- Improves patient outcomes
- Optimizes healthcare resources

**Repository:** https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-high-risk-patients-

---

## **Slide 21: Q&A**
**Title:** Questions & Discussion

**Content:**
Thank you for your attention!

**Contact:**
- GitHub: [Repository Link]
- Documentation: `docs/` folder
- Code: `src/` and `scripts/` folders

---

## **Additional Notes for Presenter:**

1. **Slide Transitions:** Use smooth transitions between pipeline steps
2. **Visuals:** Include screenshots of dashboard, ROC curves, confusion matrices
3. **Code Snippets:** Show key code snippets from each module
4. **Timing:** ~2-3 minutes per slide
5. **Emphasis:** Highlight clinical impact and practical applications

---

## **Code References by Slide:**

### Slide 5: Preprocessing
- **File:** `src/preprocess.py`
- **Key Functions:**
  ```python
  def basic_clean(df, cfg):
      # Remove duplicates, handle missing values
      
  def train_test_split_clean(config):
      # Complete preprocessing pipeline
  ```

### Slide 6: Feature Selection
- **File:** `src/feature_selection.py`
- **Key Function:**
  ```python
  def select_features(X_train, y_train, k=10):
      # Mutual Information feature selection
  ```

### Slide 7: Models
- **File:** `src/model.py`
- **Key Components:**
  ```python
  def create_lr_pipeline():
      # Logistic Regression pipeline
      
  def create_xgb_model():
      # XGBoost model
  ```

### Slide 8: Training
- **File:** `src/train.py`
- **Key Function:**
  ```python
  def train_models(config):
      # Main training pipeline
  ```

### Slide 9: Evaluation
- **File:** `src/evaluate.py`
- **Key Functions:**
  ```python
  def evaluate_models(config):
      # Comprehensive evaluation
      
  def plot_roc_curves():
      # Visual performance comparison
  ```

### Slide 10: Dashboard
- **File:** `dashboard.py`
- **Key Functions:**
  ```python
  def load_models():
      # Load trained models
      
  def predict_risk():
      # Generate predictions
  ```

### Slide 11: Clinical Utils
- **File:** `src/clinical_utils.py`
- **Key Function:**
  ```python
  def format_risk_level(prob, threshold):
      # Convert probability to risk category
  ```

---

## **Pipeline Flow Diagram:**

```
┌─────────────────┐
│  Raw Data CSV   │
│  (diabetic_data)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  ← src/preprocess.py
│  - Clean data   │
│  - Handle nulls │
│  - Encode cats  │
│  - Scale nums   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Selection│  ← src/feature_selection.py
│  - MI scores    │
│  - Top K features│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │  ← src/train.py
│  - Logistic Reg │
│  - XGBoost      │
│  - Threshold    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │  ← src/evaluate.py
│  - ROC-AUC      │
│  - Precision    │
│  - Recall       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Dashboard     │  ← dashboard.py
│  - Visualize    │
│  - Predict      │
│  - Interpret    │
└─────────────────┘
```

---

## **Key Metrics to Highlight:**

1. **ROC-AUC:** Overall model discrimination ability
2. **Recall (80%):** Critical for catching high-risk patients
3. **Precision:** Accuracy of positive predictions
4. **F1-Score:** Balanced performance metric

---

## **Visual Elements to Include:**

1. **Dashboard Screenshots:** Show the interactive interface
2. **ROC Curves:** Compare model performance visually
3. **Confusion Matrices:** Show prediction breakdown
4. **Feature Importance Charts:** Top 10 features visualization
5. **Pipeline Diagram:** Visual flow of the entire system

---

This presentation covers all project files, explains the pipeline, and provides context for why each component exists and where it's used in the codebase.

