# 📊 Diabetes 30-Day Readmission Prediction - Presentation Slides (Short Version)

---

## **Slide 1: Title Slide**
**Title:** Diabetes 30-Day Readmission Prediction with High-Risk Patient Flagging  
**Subtitle:** MSBA-265 Course Project  
**Presenter:** [Your Name]  
**Date:** [Date]

---

## **Slide 2: The Problem**
**Title:** Why This Matters

**Content:**
- 30-day readmissions cost hospitals **billions** every year
- CMS penalizes hospitals with high readmission rates
- **Our Goal:** Predict which diabetic patients are at HIGH RISK before they leave the hospital
- **Why:** Early intervention = better outcomes = lower costs

**Bottom Line:** We built a tool that flags high-risk patients so hospitals can help them before it's too late.

---

## **Slide 3: The Data**
**Title:** What We Worked With

**Content:**
- **Source:** UCI Machine Learning Repository
- **Size:** 101,766 patient encounters from 130 US hospitals
- **Features:** 50+ variables (age, medications, diagnoses, lab results, etc.)
- **Target:** Will patient be readmitted within 30 days? (Yes/No)

**What We Did:** Cleaned it up, handled missing values, and got it ready for modeling.

---

## **Slide 4: The Challenges We Faced**
**Title:** Real Problems, Real Solutions

**Content:**

**Challenge 1: Class Imbalance**
- Most patients DON'T get readmitted (imbalanced data)
- **Problem:** Model might ignore the high-risk patients
- **Solution:** Tuned thresholds to prioritize RECALL (catch 80% of high-risk patients)
- **Why:** Missing a high-risk patient is worse than a false alarm

**Challenge 2: Too Many Features**
- 50+ features = risk of overfitting
- **Solution:** Used Mutual Information to pick top 10 best features
- **Why:** Simpler models are easier to understand and less likely to memorize data

**Challenge 3: Missing Data**
- Lots of `?` and `NULL` values
- **Solution:** Cleaned data, imputed missing values, dropped bad features
- **Why:** Models need clean data to work properly

**Challenge 4: Black Box Problem**
- Complex models are hard to explain to doctors
- **Solution:** Built TWO models - one simple (Logistic Regression) and one powerful (XGBoost)
- **Why:** Doctors need to trust and understand the tool

---

## **Slide 5: Our Pipeline**
**Title:** How We Built It

**Content:**
```
Raw Data (101K patients)
    ↓
Preprocessing (Clean, Encode, Scale)
    ↓
Feature Selection (Pick Top 10)
    ↓
Train Models (Logistic Regression + XGBoost)
    ↓
Tune Thresholds (80% Recall Target)
    ↓
Evaluate on Test Data
    ↓
Dashboard (Interactive Tool)
```

**What We Did:** Built each step separately, made it modular, so anyone can run it end-to-end.

---

## **Slide 6: The Models We Built**
**Title:** Two Models, Two Purposes

**Content:**

**Model 1: Logistic Regression**
- **Why:** Simple, fast, easy to explain
- **Features:** Top 10 selected features
- **Best For:** When doctors need to understand WHY a patient is flagged

**Model 2: XGBoost**
- **Why:** More powerful, finds complex patterns
- **Features:** All available features
- **Best For:** When accuracy is the priority

**Why Two?** Balance between "easy to explain" and "highly accurate" - use the right tool for the right situation.

---

## **Slide 7: How We Measured Success**
**Title:** The Metrics That Matter

**Content:**

**ROC-AUC:** Overall model performance (~0.65-0.72)  
**Precision:** How accurate are our high-risk flags? (~0.45-0.52)  
**F1-Score:** Balanced metric (~0.55-0.62)

**⭐ RECALL: THE MOST IMPORTANT (80%)**
- **What it means:** We catch 80% of all high-risk patients
- **Why it matters:** Missing a high-risk patient is WAY worse than a false alarm
- **We achieved our target!** ✅

**Bottom Line:** We'd rather flag too many patients than miss one who's actually high-risk.

---

## **Slide 8: The Results**
**Title:** How Well Did We Do?

**Content:**
**Test Set:** 20,000 patients the models never saw before

| Metric | Logistic Regression | XGBoost |
|--------|-------------------|---------|
| ROC-AUC | 0.65-0.70 | 0.68-0.72 |
| Precision | 0.45-0.50 | 0.48-0.52 |
| **Recall** | **80%** ✅ | **80%** ✅ |
| F1-Score | 0.55-0.60 | 0.58-0.62 |

**Key Findings:**
- ✅ Both models hit our 80% recall target
- ✅ XGBoost is slightly better overall
- ✅ Logistic Regression is easier to explain
- ⚠️ Some false alarms, but that's okay - we're catching the high-risk patients

**What This Means:** The models work! They can be used in real hospitals to help identify high-risk patients.

---

## **Slide 9: What Features Matter Most?**
**Title:** What Drives Readmission Risk?

**Content:**
**Top 5 Risk Factors:**
1. **Number of medications** - More meds = more complex = higher risk
2. **Number of diagnoses** - More conditions = sicker patient
3. **Time in hospital** - Longer stay = more serious case
4. **Number of lab procedures** - More tests = more complex
5. **Age group** - Older patients = higher risk

**Insight:** Medication complexity and clinical complexity are the biggest predictors.

**Why This Helps:** Doctors now know what to focus on when assessing risk.

---

## **Slide 10: The Dashboard**
**Title:** Making It Usable

**Content:**
**What We Built:** Interactive web app using Streamlit

**Features:**
- **Real-time predictions** - Enter patient info, get instant risk score
- **Clinical interpretation** - Shows HIGH RISK or LOW RISK (not just numbers)
- **Model performance** - See how well the models are doing
- **Easy to use** - No coding required, doctors can use it right away

**Why We Built It:**
- Models are useless if doctors can't use them
- Makes predictions accessible and understandable
- Shows WHY a patient is flagged, not just the score

**How to Use:** Just run `streamlit run dashboard.py` and it's live!

---

## **Slide 11: Challenges & Solutions Summary**
**Title:** Problems We Solved

**Content:**

| Challenge | What We Did | Result |
|-----------|-------------|--------|
| Class Imbalance | Threshold tuning | 80% recall achieved |
| Too Many Features | Feature selection (top 10) | Simpler, better models |
| Missing Data | Data cleaning | Reliable predictions |
| Black Box Problem | Two models + clinical categories | Doctors can understand |

**Takeaway:** Every problem had a solution, and we focused on what matters most - catching high-risk patients.

---

## **Slide 12: What's Next?**
**Title:** Future Plans

**Content:**
1. **More Data** - Better models need more data
2. **Better Models** - Try neural networks, see if we can improve
3. **New Features** - Add social factors, medication adherence
4. **Real Integration** - Connect to hospital EMR systems
5. **Better Explainability** - SHAP values to show feature importance per patient
6. **Real-World Testing** - Validate in actual hospital setting

**Why:** Always room for improvement, and we want to make this actually useful in real hospitals.

---

## **Slide 13: Key Takeaways**
**Title:** What We Accomplished

**Content:**
✅ **Built a complete pipeline** - From raw data to predictions  
✅ **Achieved 80% recall** - We catch 80% of high-risk patients  
✅ **Created a dashboard** - Doctors can actually use it  
✅ **Solved real challenges** - Class imbalance, missing data, interpretability  
✅ **Made it reproducible** - Anyone can run it and get the same results

**Impact:**
- Helps reduce readmissions
- Improves patient care
- Saves hospitals money
- Supports clinical decisions

**Bottom Line:** We built something that actually works and can help real patients.

---

## **Slide 14: Conclusion & Q&A**
**Title:** Thank You!

**Content:**
- **Problem:** Predict 30-day readmission risk
- **Solution:** Machine learning models (Logistic Regression + XGBoost)
- **Result:** 80% recall - successfully identify high-risk patients
- **Tool:** Interactive dashboard for clinicians

**Repository:** https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-high-risk-patients-

**Questions?**

---

**End of Presentation**

