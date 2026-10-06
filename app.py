
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="WDBC AI Diagnostic",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# WHITE / PREMIUM UI
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}

.stApp {
    background: #ffffff;
    color: #111827;
}

/* Header */

.hero {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #f8fafc 55%,
        #eef2ff 100%
    );

    border: 1px solid #e5e7eb;
    border-radius: 24px;

    padding: 35px 40px;

    margin-bottom: 25px;

    box-shadow:
        0 10px 35px rgba(15, 23, 42, 0.07);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #64748b;
    font-size: 17px;
}

.metric-card {

    background: #ffffff;

    border: 1px solid #e5e7eb;

    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.05);

    min-height: 120px;
}

.metric-title {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.metric-value {
    color: #111827;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-ci {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #111827;
    margin-top: 20px;
}

.info-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 18px;
    color: #475569;
}

.prediction-benign {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 18px;
    padding: 25px;
}

.prediction-malignant {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 18px;
    padding: 25px;
}

div[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}

button[kind="primary"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load("model.joblib")


@st.cache_data
def load_metadata():

    with open("feature_names.json") as f:
        features = json.load(f)

    with open("metrics.json") as f:
        metrics = json.load(f)

    return features, metrics


model = load_model()

FEATURES, METRICS = load_metadata()


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
🩺 WDBC AI Diagnostic
</div>

<div class="hero-subtitle">
Explainable Breast Cancer Classification using
XGBoost + SHAP + Stratified 5-Fold Cross-Validation
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧬 WDBC AI")

    st.markdown(
        """
        **Model**

        XGBoost Classifier

        **Dataset**

        Wisconsin Diagnostic Breast Cancer

        **Validation**

        Stratified 5-Fold CV

        **Explainability**

        SHAP
        """
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "🔬 Prediction",
            "🧠 SHAP Explainability",
            "📈 Validation"
        ]
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="section-title">Model Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metric_items = [
        ("Accuracy", "Accuracy"),
        ("Precision", "Precision"),
        ("Recall", "Recall"),
        ("ROC-AUC", "ROC-AUC")
    ]

    for col, (title, key) in zip(
        [c1, c2, c3, c4],
        metric_items
    ):

        m = METRICS["metrics"][key]

        with col:

            st.markdown(
                f"""
                <div class="metric-card">

                <div class="metric-title">
                {title}
                </div>

                <div class="metric-value">
                {m["mean"] * 100:.2f}%
                </div>

                <div class="metric-ci">
                95% CI:
                {m["ci_lower"] * 100:.2f}%
                –
                {m["ci_upper"] * 100:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown(
        '<div class="section-title">Dataset</div>',
        unsafe_allow_html=True
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Samples",
            METRICS["samples"]
        )

    with d2:
        st.metric(
            "Features",
            METRICS["features"]
        )

    with d3:
        st.metric(
            "Validation",
            "5-Fold CV"
        )


    st.markdown(
        '<div class="section-title">About the Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">

        This application uses an XGBoost classifier trained on
        the Wisconsin Diagnostic Breast Cancer dataset.

        Model performance is estimated using stratified 5-fold
        cross-validation. The dashboard reports both the
        mean performance and the corresponding 95% confidence
        interval across folds.

        SHAP is used to provide feature-level explanations
        for individual predictions.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔬 Prediction":

    st.markdown(
        '<div class="section-title">Patient Feature Input</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Enter the 30 WDBC numerical measurements below."
    )

    # Default values = dataset medians
    defaults = {
        f: 0.0 for f in FEATURES
    }

    # Use median values from sklearn dataset
    from sklearn.datasets import load_breast_cancer

    raw_data = load_breast_cancer()

    median_values = pd.DataFrame(
        raw_data.data,
        columns=raw_data.feature_names
    ).median()

    values = {}

    cols = st.columns(3)

    for i, feature in enumerate(FEATURES):

        with cols[i % 3]:

            values[feature] = st.number_input(
                feature,
                value=float(median_values[feature]),
                format="%.5f",
                key=f"input_{i}"
            )


    st.markdown("---")

    if st.button(
        "🔍 Predict Diagnosis",
        type="primary",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(
            input_df
        )[0]

        # sklearn WDBC:
        # 0 = malignant
        # 1 = benign

        if prediction == 0:

            confidence = probability[0] * 100

            st.markdown(
                f"""
                <div class="prediction-malignant">

                <h2>⚠️ Malignant Prediction</h2>

                <p>
                Model confidence:
                <strong>{confidence:.2f}%</strong>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            confidence = probability[1] * 100

            st.markdown(
                f"""
                <div class="prediction-benign">

                <h2>✓ Benign Prediction</h2>

                <p>
                Model confidence:
                <strong>{confidence:.2f}%</strong>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Prediction Probability")

        prob_df = pd.DataFrame({
            "Diagnosis": [
                "Malignant",
                "Benign"
            ],

            "Probability": [
                probability[0],
                probability[1]
            ]
        })

        st.bar_chart(
            prob_df.set_index("Diagnosis")
        )


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

elif page == "🧠 SHAP Explainability":

    st.markdown(
        '<div class="section-title">SHAP Explainability</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        SHAP explains how individual WDBC features contribute
        to the model's prediction.
        """
    )

    from sklearn.datasets import load_breast_cancer

    raw_data = load_breast_cancer()

    median_values = pd.DataFrame(
        raw_data.data,
        columns=raw_data.feature_names
    ).median()

    shap_values_input = pd.DataFrame(
        [median_values],
        columns=FEATURES
    )

    st.markdown(
        "### Example Patient Explanation"
    )

    # Extract trained XGBoost estimator
    xgb = model.named_steps["xgb"]

    explainer = shap.TreeExplainer(xgb)

    transformed = model.named_steps[
        "imputer"
    ].transform(shap_values_input)

    shap_values = explainer.shap_values(
        transformed
    )

    fig = plt.figure(
        figsize=(11, 7)
    )

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=transformed[0],
            feature_names=FEATURES
        ),
        show=False
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
        **Interpretation**

        Features pushing the prediction toward one class
        have positive or negative SHAP contributions depending
        on the model output direction.

        Larger absolute SHAP values indicate stronger
        contribution to the prediction.
        """
    )


# ============================================================
# VALIDATION
# ============================================================

elif page == "📈 Validation":

    st.markdown(
        '<div class="section-title">Cross-Validation Performance</div>',
        unsafe_allow_html=True
    )

    rows = []

    for metric_name, metric_data in METRICS["metrics"].items():

        rows.append({
            "Metric": metric_name,
            "Mean": metric_data["mean"],
            "Std": metric_data["std"],
            "95% CI Lower": metric_data["ci_lower"],
            "95% CI Upper": metric_data["ci_upper"]
        })

    validation_df = pd.DataFrame(rows)

    display_df = validation_df.copy()

    for col in [
        "Mean",
        "Std",
        "95% CI Lower",
        "95% CI Upper"
    ]:

        display_df[col] = (
            display_df[col] * 100
        ).round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Mean Performance"
    )

    chart_df = validation_df[
        ["Metric", "Mean"]
    ].copy()

    chart_df["Mean"] *= 100

    st.bar_chart(
        chart_df.set_index("Metric")
    )

    st.markdown(
        """
        <div class="info-box">

        <strong>95% Confidence Interval</strong><br><br>

        The confidence interval represents the uncertainty
        around the mean cross-validation performance estimated
        from the five validation folds.

        Formula:

        <br><br>

        <strong>
        CI = Mean ± t × (SD / √n)
        </strong>

        <br><br>

        where n = 5 folds and the confidence level is 95%.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "WDBC AI Diagnostic • XGBoost • SHAP • "
    "Stratified 5-Fold Cross-Validation • 95% CI"
)
