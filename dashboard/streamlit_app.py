
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ----- Make sure we can import from the src/ folder -----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocess import (
    load_telco_data,
    prepare_features_and_target,
    split_feature_types,
    build_preprocessor,
)

MODEL_PATH = os.path.join("models", "telco_churn_model.pkl")
DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Utilities 

@st.cache_resource
def load_model():
    """
    Load or build the sklearn Pipeline model.

    On Streamlit Cloud, we avoid loading a pickled model (version issues)
    and instead train the model on the fly from the Telco dataset.
    """
    # Try to load existing model (works locally if compatible)
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            # If unpickling fails (e.g., on Streamlit Cloud), fall back to retrain
            print(f"[WARN] Failed to load pickled model: {e}. Retraining instead.")

    # ---- Retrain model from raw data ----
    print(f"[INFO] Training model from data at: {DATA_PATH}")
    df = load_telco_data(DATA_PATH)
    X, y = prepare_features_and_target(df)

    cat_features, num_features = split_feature_types(X)
    preprocessor = build_preprocessor(cat_features, num_features)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    # Optional: try to save for local reuse (will work locally, may be read-only on cloud)
    try:
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
    except Exception as e:
        print(f"[WARN] Could not save model: {e}")

    return pipeline


def ensure_minimal_columns(df: pd.DataFrame):
    required_cols = {"customerID", "tenure", "MonthlyCharges", "TotalCharges"}
    missing = required_cols.difference(set(df.columns))
    if missing:
        st.warning(
            f"The following commonly used columns are missing: {sorted(missing)}. "
            "Model may still work if it was trained with different columns."
        )


def predict_for_dataframe(model, df: pd.DataFrame):
    df_features = df.copy()

    # Strip whitespace
    df_features = df_features.applymap(
        lambda x: x.strip() if isinstance(x, str) else x
    )

    # Force numeric on key numeric columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

    # Drop target if present
    if "Churn" in df_features.columns:
        df_features = df_features.drop(columns=["Churn"])

    # Predict
    proba = model.predict_proba(df_features)[:, 1]
    preds = (proba >= 0.5).astype(int)

    df_out = df.copy()
    df_out["Churn_Probability"] = proba
    df_out["Churn_Prediction"] = preds

    return df_out


# Hero & main charts

def hero_radial_chart(df_pred: pd.DataFrame):
    required = ["Churn_Prediction", "Churn_Probability", "tenure", "MonthlyCharges"]
    if not all(col in df_pred.columns for col in required):
        st.info("Not enough columns for hero radial chart.")
        return

    churn_rate = float((df_pred["Churn_Prediction"] == 1).mean() * 100)
    avg_tenure = float(df_pred["tenure"].mean())
    avg_monthly = float(df_pred["MonthlyCharges"].mean())
    active_share = float((df_pred["Churn_Prediction"] == 0).mean() * 100)

    metrics = ["Churn Rate (%)", "Avg Tenure", "Avg Monthly Charges", "Active Share (%)"]
    raw_values = [churn_rate, avg_tenure, avg_monthly, active_share]

    denom = [
        100.0,  # churn%
        max(df_pred["tenure"].max(), 1),
        max(df_pred["MonthlyCharges"].max(), 1),
        100.0,  # active%
    ]
    values_norm = [v / d * 100 for v, d in zip(raw_values, denom)]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_norm + [values_norm[0]],
            theta=metrics + [metrics[0]],
            fill="toself",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def donut_churn_distribution(df_pred: pd.DataFrame):
    if "Churn_Prediction" not in df_pred.columns:
        st.info("Prediction column missing for churn donut.")
        return

    counts = (
        df_pred["Churn_Prediction"]
        .value_counts()
        .rename(index={0: "No Churn", 1: "Churn"})
    )

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.6,
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


# Flow & time visuals 

def sankey_contract_to_churn(df_pred: pd.DataFrame):
    """Sankey: Contract segment → churn outcome."""
    if "Contract" not in df_pred.columns or "Churn_Prediction" not in df_pred.columns:
        st.info("Contract or prediction column missing for Sankey chart.")
        return

    df_tmp = df_pred.copy()
    df_tmp["ChurnLabel"] = df_tmp["Churn_Prediction"].map({0: "No Churn", 1: "Churn"})

    group = df_tmp.groupby(["Contract", "ChurnLabel"]).size().reset_index(name="count")

    contracts = group["Contract"].unique().tolist()
    churn_labels = ["No Churn", "Churn"]
    nodes = contracts + churn_labels
    node_index = {name: i for i, name in enumerate(nodes)}

    sources = [node_index[c] for c in group["Contract"]]
    targets = [node_index[cl] for cl in group["ChurnLabel"]]
    values = group["count"].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=nodes),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def animated_churn_by_tenure_gender(df_pred: pd.DataFrame):
    
    if "tenure" not in df_pred.columns or "gender" not in df_pred.columns:
        st.info("tenure or gender column missing for animated chart.")
        return

    df_tmp = df_pred.copy()
    grouped = (
        df_tmp.groupby(["tenure", "gender"])["Churn_Prediction"]
        .mean()
        .reset_index(name="churn_rate")
    )
    grouped["churn_rate"] *= 100.0

    fig = px.bar(
        grouped,
        x="gender",
        y="churn_rate",
        color="gender",
        animation_frame="tenure",
        range_y=[
            0,
            grouped["churn_rate"].max() * 1.1 if not grouped.empty else 100,
        ],
        labels={"churn_rate": "Churn Rate (%)"},
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


# Risk, CLTV, cohorts 

def cltv_bar_chart(df_pred: pd.DataFrame):
    """
    CLTV bar chart by Contract.
    CLTV ~ tenure * MonthlyCharges
    """
    if "tenure" not in df_pred.columns or "MonthlyCharges" not in df_pred.columns:
        st.info("Missing tenure or MonthlyCharges for CLTV chart.")
        return
    if "Contract" not in df_pred.columns:
        st.info("Contract column missing; cannot group CLTV by contract.")
        return

    df_tmp = df_pred.copy()
    df_tmp["CLTV"] = df_tmp["tenure"].fillna(0) * df_tmp["MonthlyCharges"].fillna(0)

    cltv_group = (
        df_tmp.groupby("Contract")["CLTV"]
        .sum()
        .reset_index()
        .sort_values("CLTV", ascending=False)
    )

    fig = px.bar(
        cltv_group,
        x="Contract",
        y="CLTV",
        labels={"CLTV": "Total CLTV"},
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def risk_heatmap(df_pred: pd.DataFrame):
    
    if "Contract" not in df_pred.columns or "InternetService" not in df_pred.columns:
        st.info("Need Contract and InternetService for risk heatmap.")
        return
    if "Churn_Probability" not in df_pred.columns:
        st.info("Churn_Probability column missing for risk heatmap.")
        return

    pivot = df_pred.pivot_table(
        index="Contract",
        columns="InternetService",
        values="Churn_Probability",
        aggfunc="mean",
    )

    fig = px.imshow(
        pivot,
        labels=dict(x="InternetService", y="Contract", color="Avg Churn Probability"),
        aspect="auto",
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


def cohort_churn_line(df_pred: pd.DataFrame):
    
    if "tenure" not in df_pred.columns or "Churn_Prediction" not in df_pred.columns:
        st.info("Need tenure and prediction for cohort line chart.")
        return

    bins = [0, 6, 12, 24, 36, 60, df_pred["tenure"].max() + 1]
    labels = ["0-6", "6-12", "12-24", "24-36", "36-60", "60+"]

    df_tmp = df_pred.copy()
    df_tmp["TenureCohort"] = pd.cut(
        df_tmp["tenure"], bins=bins, labels=labels, right=False
    )

    grouped = (
        df_tmp.groupby("TenureCohort")["Churn_Prediction"]
        .mean()
        .reset_index(name="churn_rate")
    )
    grouped["churn_rate"] *= 100.0

    fig = px.line(
        grouped,
        x="TenureCohort",
        y="churn_rate",
        markers=True,
        labels={
            "churn_rate": "Churn Rate (%)",
            "TenureCohort": "Tenure Cohort (months)",
        },
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)


# Streamlit layout 

st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    layout="wide",
)


st.title("Telco Customer Churn Prediction Dashboard")

st.write(
    "Upload a CSV file with customer records (Telco schema). "
    "The model will compute churn probabilities and show visual insights."
)

st.sidebar.header("Controls")
threshold = st.sidebar.slider(
    "High-risk churn probability threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.01,
)
# 📄 Sample CSV download
sample_data = {
    "customerID": ["0001-A", "0002-B", "0003-C"],
    "gender": ["Female", "Male", "Female"],
    "SeniorCitizen": [0, 1, 0],
    "Partner": ["Yes", "No", "No"],
    "Dependents": ["No", "No", "Yes"],
    "tenure": [12, 3, 24],
    "PhoneService": ["Yes", "No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["Fiber optic", "DSL", "Fiber optic"],
    "OnlineSecurity": ["No", "Yes", "No"],
    "OnlineBackup": ["Yes", "No", "Yes"],
    "DeviceProtection": ["No", "Yes", "No"],
    "TechSupport": ["Yes", "No", "No"],
    "StreamingTV": ["No", "No", "Yes"],
    "StreamingMovies": ["Yes", "No", "Yes"],
    "Contract": ["Month-to-month", "Two year", "One year"],
    "PaperlessBilling": ["Yes", "No", "Yes"],
    "PaymentMethod": ["Electronic check", "Credit card (automatic)", "Mailed check"],
    "MonthlyCharges": [70.9, 20.5, 85.3],
    "TotalCharges": [850.4, 61.5, 2000.0],
}
sample_df = pd.DataFrame(sample_data)

st.sidebar.download_button(
    label="📄 Download Sample Telco CSV",
    data=sample_df.to_csv(index=False).encode("utf-8"),
    file_name="sample_telco_customers.csv",
    mime="text/csv",
    help="Download a sample file and upload it below to see how the model works."
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


try:
    model = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    ensure_minimal_columns(df_raw)

    df_pred = predict_for_dataframe(model, df_raw)

    
    total_customers = len(df_pred)
    predicted_churn = int((df_pred["Churn_Prediction"] == 1).sum())
    predicted_churn_rate = (
        predicted_churn / total_customers * 100 if total_customers else 0.0
    )

    st.subheader("Data Preview (first rows with predictions)")
    st.dataframe(df_pred.head())

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total Customers", total_customers)
    with k2:
        st.metric("Predicted Churners", predicted_churn)
    with k3:
        st.metric("Predicted Churn Rate (%)", f"{predicted_churn_rate:.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Hero Overview", "Flows & Time", "Risk & CLTV", "Model Insights"]
    )

    with tab1:
        st.markdown("### Hero Radial Multi-Metric Chart")
        hero_radial_chart(df_pred)

        st.markdown("### Churn vs Non-Churn Donut")
        donut_churn_distribution(df_pred)

    with tab2:
        st.markdown("### Contract → Churn Sankey Flow")
        sankey_contract_to_churn(df_pred)

        st.markdown("### Animated Churn by Tenure and Gender")
        animated_churn_by_tenure_gender(df_pred)

    with tab3:
        st.markdown("### Customer Lifetime Value (CLTV) by Contract")
        cltv_bar_chart(df_pred)

        st.markdown("### Risk Heatmap (Contract × Internet Service)")
        risk_heatmap(df_pred)

    with tab4:
        st.markdown("### Cohort Churn Line Chart (by Tenure Cohort)")
        cohort_churn_line(df_pred)

        shap_bar_path = os.path.join("plots", "shap_summary_bar.png")
        if os.path.exists(shap_bar_path):
            st.markdown("### Feature Importance (RandomForest)")
            st.image(shap_bar_path,caption="Feature Importance (RandomForest)",
                    width=1000   
            )

        else:
            st.info(
                "Run src/explain_model.py to generate SHAP summary plots; "
                "they will appear here automatically."
            )

    
    st.subheader(f"High-risk customers (probability ≥ {threshold:.2f})")
    high_risk = df_pred[df_pred["Churn_Probability"] >= threshold]
    if not high_risk.empty:
        st.dataframe(high_risk.head(100))
    else:
        st.write("No high-risk customers at the current threshold.")

    
    st.subheader("Download predictions")
    csv_out = df_pred.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV with predictions",
        csv_out,
        file_name="telco_churn_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV file to start the analysis.")
