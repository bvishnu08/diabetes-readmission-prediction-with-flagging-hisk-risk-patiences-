from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)


# -------------------------------------------------------------------
# Import project code (Config, preprocessing, clinical utils)
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.preprocess import generate_processed_datasets
from src.clinical_utils import summarize_risk_view

CFG = Config()


# -------------------------------------------------------------------
# Streamlit page config + light theme tweaks
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Safe Discharge Command Center",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CURSOR_THEME = """
<style>

:root {
    --bg-dark: #0d1117;
    --bg-panel: #161b22;
    --bg-card: #1f242d;
    --text-light: #e6edf3;
    --text-dim: #9ba1a6;
    --accent-blue: #2563eb;
    --accent-purple: #9333ea;
    --accent-green: #22c55e;
    --accent-orange: #f97316;
}

/* APP BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 10%, rgba(37,99,235,0.28), transparent 60%),
                radial-gradient(circle at 85% 0%, rgba(249,115,22,0.28), transparent 60%),
                linear-gradient(135deg, #0f172a, #1e293b);
    color: var(--text-light);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: var(--text-dim) !important;
}

/* TITLES */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: white;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    font-size: 1.05rem;
    color: var(--text-dim);
    margin-bottom: 1.2rem;
}

/* KPI CARDS */
.metric-card {
    background: var(--bg-card);
    padding: 1.1rem 1.4rem;
    border-radius: 1rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 28px rgba(0,0,0,0.35);
}
.metric-label {
    text-transform: uppercase;
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 800;
    color: white;
}
.metric-caption {
    font-size: 0.85rem;
    color: var(--text-dim);
}

/* PANELS */
.panel {
    background: var(--bg-card);
    padding: 1.4rem 1.6rem;
    border-radius: 1rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
}

/* SECTION TITLES */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.4rem;
}
.section-subtext {
    font-size: 1rem;
    color: var(--text-dim);
    margin-bottom: 0.8rem;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-radius: 999px;
    padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1.3rem;
    color: #cbd5e1;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(120deg, var(--accent-blue), var(--accent-purple));
    border-radius: 999px;
    color: white;
    font-weight: 600;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(120deg, var(--accent-blue), var(--accent-purple));
    padding: 0.5rem 1.3rem;
    border-radius: 999px;
    color: white;
    border: none;
    font-weight: 600;
    box-shadow: 0 14px 35px rgba(37,99,235,0.35);
}

/* DATAFRAME ROUNDED */
[data-testid="stDataFrame"] {
    border-radius: 1rem;
    overflow: hidden;
}

</style>
"""

st.markdown(CURSOR_THEME, unsafe_allow_html=True)



# -------------------------------------------------------------------
# Cached helpers: data + models + evaluation
# -------------------------------------------------------------------


@st.cache_data(show_spinner="Loading processed train/test data...")
def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = CFG.processed_train_path()
    test_path = CFG.processed_test_path()

    if not train_path.exists() or not test_path.exists():
        generate_processed_datasets(CFG)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


@st.cache_resource(show_spinner="Loading trained models...")
def load_models_and_thresholds() -> Tuple[Any, Any, Dict[str, Any]]:
    models_dir = CFG.models_path()
    models_dir.mkdir(parents=True, exist_ok=True)

    logreg = joblib.load(CFG.model_path_logreg())
    xgb = joblib.load(CFG.model_path_xgb())

    thresholds_path = CFG.thresholds_path()
    with thresholds_path.open("r", encoding="utf-8") as f:
        thresholds = json.load(f)

    return logreg, xgb, thresholds


def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    threshold: float,
) -> Dict[str, Any]:
    X_sel = X_test[features]

    p_readmit = model.predict_proba(X_sel)[:, 1]
    y_pred = (p_readmit >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, p_readmit)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)

    clinical = summarize_risk_view(y_test, y_pred, p_readmit)

    return {
        "name": name,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "auc": float(auc),
        "cm": cm,
        "clinical": clinical,
    }


@st.cache_data(show_spinner="Evaluating models on test set...")
def compute_evaluation() -> Dict[str, Any]:
    train_df, test_df = load_processed_data()
    logreg, xgb, thresholds = load_models_and_thresholds()

    X_test = test_df.drop(columns=[CFG.target_col])
    y_test = test_df[CFG.target_col]

    lr_cfg = thresholds["logreg"]
    xgb_cfg = thresholds["xgb"]

    lr_feats = lr_cfg["features"]
    xgb_feats = xgb_cfg["features"]

    lr_res = evaluate_model(
        "Logistic Regression (top 20)",
        logreg,
        X_test,
        y_test,
        lr_feats,
        lr_cfg["threshold"],
    )
    xgb_res = evaluate_model(
        "XGBoost (top 25)",
        xgb,
        X_test,
        y_test,
        xgb_feats,
        xgb_cfg["threshold"],
    )

    return {
        "lr": lr_res,
        "xgb": xgb_res,
        "lr_features": lr_feats,
        "xgb_features": xgb_feats,
        "y_test": y_test,
        "X_test": X_test,
    }


# -------------------------------------------------------------------
# Feature metadata for human-friendly labels + explanations
# -------------------------------------------------------------------

FEATURE_META: Dict[str, Dict[str, str]] = {
    "age": {
        "label": "Age band",
        "desc": "Age group at admission (0 = 0–10, …, 9 = 90–100). Higher values are older patients.",
    },
    "gender": {"label": "Gender", "desc": "Patient sex recorded at admission."},
    "race": {"label": "Race / ethnicity", "desc": "Race recorded in the hospital system."},
    "admission_type_id": {
        "label": "Admission type ID",
        "desc": "Emergency vs elective vs other admission types (numeric code).",
    },
    "admission_source_id": {
        "label": "Admission source ID",
        "desc": "Where the patient came from (ER, referral, transfer, etc.).",
    },
    "discharge_disposition_id": {
        "label": "Discharge disposition ID",
        "desc": "Where the patient is discharged to (home, SNF, rehab, etc.).",
    },
    "time_in_hospital": {
        "label": "Length of stay (days)",
        "desc": "Number of days in the hospital for this encounter.",
    },
    "number_inpatient": {
        "label": "# prior inpatient visits",
        "desc": "Number of previous inpatient hospitalizations in the last year.",
    },
    "number_emergency": {
        "label": "# prior ER visits",
        "desc": "Number of prior emergency room visits.",
    },
    "number_outpatient": {
        "label": "# prior outpatient visits",
        "desc": "Number of prior outpatient visits.",
    },
    "number_diagnoses": {
        "label": "# diagnoses",
        "desc": "Count of distinct diagnoses for this encounter.",
    },
    "num_lab_procedures": {"label": "# lab procedures", "desc": "Total number of lab tests ordered."},
    "num_procedures": {"label": "# procedures", "desc": "Number of procedures (imaging, surgery, etc.)."},
    "num_medications": {"label": "# medications", "desc": "Count of distinct medications prescribed."},
    "max_glu_serum": {"label": "Max glucose result", "desc": "Highest serum glucose category during the stay."},
    "A1Cresult": {"label": "A1C result", "desc": "Most recent HbA1c result category."},
    "change": {"label": "Medication change", "desc": "Whether diabetes meds were changed during this stay."},
    "diabetesMed": {"label": "On diabetes medication", "desc": "Whether the patient is on diabetes medication."},
    "insulin": {"label": "Insulin use", "desc": "Insulin dosage status (up, down, steady, or none)."},
    "metformin": {"label": "Metformin use", "desc": "Whether the patient is taking metformin."},
}


# -------------------------------------------------------------------
# UI building blocks
# -------------------------------------------------------------------


def sidebar_info(xgb_res: Dict[str, Any]) -> None:
    with st.sidebar:
        st.markdown("### 🩺 Safe Discharge Dashboard")
        st.markdown(
            "Goal: **support clinicians at discharge time** by flagging high-risk diabetes patients "
            "likely to be readmitted within 30 days."
        )
        st.markdown(
            "Built on your **MSBA-265 capstone** using logistic regression + XGBoost "
            "with threshold tuning."
        )

        st.markdown("---")
        st.markdown("#### Model focus for metrics")
        st.radio("Deployment model", ["XGBoost (top 25 features)"], index=0, key="model_choice")
        st.caption("Logistic Regression remains available as an interpretable baseline.")

        st.markdown("---")
        st.markdown("#### Focused model snapshot")
        st.markdown(
            f"- Model: XGBoost (25 features)\n"
            f"- Threshold: {xgb_res['threshold']:.3f}\n"
            f"- Recall (class 1): {xgb_res['recall']:.3f}\n"
            f"- F1-score: {xgb_res['f1']:.3f}"
        )


def header_section(train_df: pd.DataFrame, test_df: pd.DataFrame, xgb_res: Dict[str, Any]) -> None:
    st.title("Diabetes 30-Day Readmission – Safe Discharge Dashboard")
    st.write(
        "Predicting which inpatients with diabetes are at risk of being **readmitted within 30 days** "
        "so clinicians can make safer discharge decisions and schedule follow-up care."
    )

    train_rows = len(train_df)
    test_rows = len(test_df)
    positives_test = int(test_df[CFG.target_col].sum())

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">TRAIN ROWS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{train_rows:,}</div>', unsafe_allow_html=True)
        st.caption("80% of total patients (training set)")
        st.markdown("</div>", unsafe_allow_html=True)

    with kpi_cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">TEST ROWS</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{test_rows:,}</div>', unsafe_allow_html=True)
        st.caption("20% hold-out evaluation set")
        st.markdown("</div>", unsafe_allow_html=True)

    with kpi_cols[2]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">TARGET POSITIVES (TEST)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{positives_test:,}</div>', unsafe_allow_html=True)
        st.caption("Patients readmitted < 30 days (label = 1)")
        st.markdown("</div>", unsafe_allow_html=True)

    with kpi_cols[3]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">FOCUSED MODEL</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">XGBoost (top 25)</div>', unsafe_allow_html=True)
        st.caption(f"Threshold = {xgb_res['threshold']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)


def tab_data_overview(train_df: pd.DataFrame, test_df: pd.DataFrame, xgb_features: List[str]) -> None:
    st.subheader("📊 Data overview – what does our dataset look like?")
    st.markdown(
        "We use the **UCI Diabetes 130-US hospitals** dataset, cleaned and split via your preprocessing pipeline."
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Columns (incl. target)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{train_df.shape[1]}</div>', unsafe_allow_html=True)
        st.caption(f"{len(CFG.candidate_features)} candidate features + 1 binary target")
        st.markdown("</div>", unsafe_allow_html=True)

        numeric_cols = train_df.drop(columns=[CFG.target_col]).select_dtypes(include=[np.number]).columns
        cat_cols = [
            c for c in train_df.columns if c not in numeric_cols and c != CFG.target_col
        ]

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Feature types</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{len(numeric_cols)} numeric / {len(cat_cols)} categorical</div>',
            unsafe_allow_html=True,
        )
        st.caption("Based on cleaned training data.")
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Target distribution (test set)</div>', unsafe_allow_html=True)
        target_counts = (
            test_df[CFG.target_col].value_counts().rename({0: "No readmission", 1: "<30 days"})
        )
        st.bar_chart(target_counts)
        st.caption("We evaluate models on the same 20% hold-out test set.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Candidate feature pool</div>', unsafe_allow_html=True)
    st.write("Configured in `Config.candidate_features`. We highlight which are used by XGBoost.")
    feature_table = pd.DataFrame(
        {
            "Feature name": CFG.candidate_features,
            "Included in XGBoost 25?": ["✅" if f in xgb_features else "" for f in CFG.candidate_features],
        }
    )
    st.dataframe(feature_table, use_container_width=True, height=350)
    st.caption("SelectKBest trims this pool to the top 20 features for LR and 25 for XGBoost.")
    st.markdown("</div>", unsafe_allow_html=True)


def tab_model_performance(lr_res: Dict[str, Any], xgb_res: Dict[str, Any]) -> None:
    st.subheader("📈 Model performance – how well do we predict readmissions?")
    st.write(
        "Both models share the same 20% hold-out test set with metrics on the positive class "
        "(patients readmitted within 30 days)."
    )
    perf_df = pd.DataFrame(
        [
            {
                "Model": lr_res["name"],
                "Threshold": lr_res["threshold"],
                "ROC-AUC": lr_res["auc"],
                "Accuracy": lr_res["accuracy"],
                "Recall (class 1)": lr_res["recall"],
                "Precision (class 1)": lr_res["precision"],
                "F1-score": lr_res["f1"],
            },
            {
                "Model": xgb_res["name"],
                "Threshold": xgb_res["threshold"],
                "ROC-AUC": xgb_res["auc"],
                "Accuracy": xgb_res["accuracy"],
                "Recall (class 1)": xgb_res["recall"],
                "Precision (class 1)": xgb_res["precision"],
                "F1-score": xgb_res["f1"],
            },
        ]
    )
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.dataframe(perf_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Confusion matrices (test set)")
    st.caption("Rows = actual outcome, columns = model prediction.")
    cm_cols = st.columns(2)
    for col, res in zip(cm_cols, [lr_res, xgb_res]):
        with col:
            cm = res["cm"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: No readmit", "Actual: <30 days"],
                columns=["Pred: No readmit", "Pred: <30 days"],
            )
            fig = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig.update_layout(
                title=res["name"],
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### How to talk about this")
    st.markdown(
        "- **XGBoost** achieves the strongest F1 and ROC-AUC, hence it is the recommended deployment model.\n"
        "- **Logistic Regression** remains as an interpretable baseline for explaining which factors drive risk.\n"
        "- Since both models use the same hold-out set, their comparison is fair."
    )


def tab_clinical_view(xgb_res: Dict[str, Any]) -> None:
    st.subheader("🩺 Safe discharge view – how does this look clinically?")
    clinical = xgb_res["clinical"]
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "Patients flagged HIGH RISK",
            f"{clinical['n_high']:,}",
            f"{clinical['high_risk_percent']*100:.1f}% of test set",
        )
    with cols[1]:
        st.metric(
            "Patients flagged LOW RISK",
            f"{clinical['n_low']:,}",
            f"{clinical['low_risk_percent']*100:.1f}% of test set",
        )
    with cols[2]:
        st.metric(
            "Avg predicted safe chance",
            f"{clinical['avg_p_safe']*100:.1f}%",
            "Across all test patients",
        )
    st.markdown("---")
    st.markdown("##### Among LOW-RISK patients (model says “safe to discharge”):")
    st.write(
        f"- Observed readmission rate: **{clinical['observed_readmit_rate_low']*100:.1f}%**\n"
        f"- Observed safe discharge rate: **{clinical['observed_safe_rate_low']*100:.1f}%**\n"
        f"- Avg predicted safe chance: **{clinical['avg_p_safe_low']*100:.1f}%**"
    )
    st.markdown("---")
    st.markdown("##### Talking points for clinicians")
    st.markdown(
        "- LOW RISK does not mean zero chance – it indicates patients who historically had low readmission rates.\n"
        "- HIGH RISK patients may benefit from education, social work consults, med review, or delayed discharge.\n"
        "- Use this as another signal; clinical judgment still leads."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_feature_input(feature: str, train_df: pd.DataFrame) -> Any:
    meta = FEATURE_META.get(feature, {})
    label = meta.get("label", feature)
    desc = meta.get("desc", "")

    column = train_df[feature]
    if column.dtype == "object":
        options = sorted(column.dropna().unique().tolist())
        value = st.selectbox(
            f"{label} (`{feature}`)",
            options if options else ["Unknown"],
            key=f"input_{feature}",
        )
    else:
        median = float(column.median())
        value = st.number_input(
            f"{label} (`{feature}`)",
            value=median,
            key=f"input_{feature}",
        )
    if desc:
        st.caption(desc)
    return value


def build_patient_input_form(train_df: pd.DataFrame, xgb_features: List[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    groups: List[Tuple[str, List[str]]] = [
        (
            "Demographics & admission",
            ["age", "gender", "race", "admission_type_id", "admission_source_id", "discharge_disposition_id"],
        ),
        (
            "Utilization history",
            ["time_in_hospital", "number_inpatient", "number_emergency", "number_outpatient", "number_diagnoses"],
        ),
        (
            "Labs & medications",
            ["num_lab_procedures", "num_procedures", "num_medications", "max_glu_serum", "A1Cresult"],
        ),
        (
            "Core diabetes treatment",
            ["change", "diabetesMed", "insulin", "metformin"],
        ),
    ]

    for title, features in groups:
        used = [f for f in features if f in xgb_features]
        if not used:
            continue
        with st.expander(title, expanded=True):
            cols = st.columns(2)
            for idx, feat in enumerate(used):
                with cols[idx % 2]:
                    values[feat] = render_feature_input(feat, train_df)

    remaining = [f for f in xgb_features if f not in values]
    if remaining:
        with st.expander("Other features used by the model", expanded=False):
            cols = st.columns(2)
            for idx, feat in enumerate(remaining):
                with cols[idx % 2]:
                    values[feat] = render_feature_input(feat, train_df)

    return values


def tab_prediction_form(train_df: pd.DataFrame, xgb_model, xgb_threshold: float, xgb_features: List[str]) -> None:
    st.subheader("🎛 Prediction playground – estimate risk for a single patient")
    left, right = st.columns([1.6, 1.2])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Enter patient details</div>', unsafe_allow_html=True)
        st.caption(
            "Describe one inpatient with diabetes at the end of their stay. Each field matches a feature used by the XGBoost model."
        )
        input_values = build_patient_input_form(train_df, xgb_features)
        predict_clicked = st.button("🔮 Predict 30-day readmission risk", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        if predict_clicked:
            input_df = pd.DataFrame([input_values])
            proba = float(xgb_model.predict_proba(input_df[xgb_features])[:, 1][0])
            safe_prob = 1.0 - proba
            label = "HIGH RISK" if proba >= xgb_threshold else "LOW RISK"

            st.metric("Predicted 30-day readmission probability", f"{proba*100:.1f}%")
            st.metric("Estimated safe discharge probability", f"{safe_prob*100:.1f}%")
            st.metric("Risk label (using tuned threshold)", label)

            st.progress(proba)
            if label == "HIGH RISK":
                st.markdown(
                    "- The model estimates this patient has a **high chance of readmission**.\n"
                    "- Consider education, social work consults, med review, or delayed discharge.\n"
                    "- This uses the same XGBoost model validated on the 20% hold-out set."
                )
            else:
                st.markdown(
                    "- The model estimates this patient looks similar to safe discharges historically.\n"
                    "- Still requires clinical judgment, but signals they are lower priority for intervention."
                )
        else:
            st.info("Fill in the patient details on the left and click the prediction button.")
        st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------


def main():
    train_df, test_df = load_processed_data()
    logreg_model, xgb_model, thresholds = load_models_and_thresholds()
    eval_results = compute_evaluation()

    lr_res = eval_results["lr"]
    xgb_res = eval_results["xgb"]
    xgb_features = eval_results["xgb_features"]

    sidebar_info(xgb_res)
    header_section(train_df, test_df, xgb_res)

    tabs = st.tabs(
        [
            "📊 Data overview",
            "📈 Model performance",
            "🩺 Safe discharge view",
            "🎛 Prediction form",
        ]
    )

    with tabs[0]:
        tab_data_overview(train_df, test_df, xgb_features)

    with tabs[1]:
        tab_model_performance(lr_res, xgb_res)

    with tabs[2]:
        tab_clinical_view(xgb_res)

    with tabs[3]:
        tab_prediction_form(train_df, xgb_model, xgb_res["threshold"], xgb_features)


if __name__ == "__main__":
    main()

