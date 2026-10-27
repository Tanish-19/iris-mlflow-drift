"""
Streamlit Web Dashboard for Iris MLflow Tracking and Data Drift Analysis.
Provides interactive exploration of datasets, MLflow experiment comparisons,
confusion matrices, and Kolmogorov-Smirnov data drift visual analysis.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_loader import load_iris_data, get_reference_dataframe, create_simulated_drift_data, get_dataset_summary
from drift_detection import run_ks_drift_detection

# Page config
st.set_page_config(
    page_title="Iris MLflow & Drift Monitor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-drift {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-nodrift {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_data():
    ref_df = get_reference_dataframe()
    curr_df = create_simulated_drift_data(ref_df, random_state=42)
    return ref_df, curr_df


ref_df, curr_df = get_cached_data()

# Sidebar Navigation
st.sidebar.title("🌸 Iris MLOps Suite")
st.sidebar.markdown("**Assessment:** ISE-2 MLflow & Drift Analysis")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "1. Overview & Architecture",
        "2. Dataset Explorer",
        "3. MLflow Experiments",
        "4. Data Drift Analysis (KS Test)",
        "5. Theory & Definitions (Viva Prep)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Local MLflow UI Command:**\n\n"
    "`mlflow ui --port 5000`\n\n"
    "Open `http://localhost:5000` to inspect artifact logs."
)

# ----------------- 1. OVERVIEW & ARCHITECTURE -----------------
if nav_choice == "1. Overview & Architecture":
    st.markdown("<div class='main-title'>Iris ML – MLflow Tracking & Data Drift Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>End-to-End MLOps Pipeline with Scikit-learn, MLflow, and Kolmogorov-Smirnov Statistical Testing</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", "150 Samples", "Scikit-learn Iris")
    with col2:
        st.metric("Features Tracked", "4 Continuous", "Length & Width")
    with col3:
        st.metric("MLflow Runs", "3 Logged", "C: 0.1, 1.0, 10.0")
    with col4:
        st.metric("Drift Method", "KS 2-Sample Test", "alpha = 0.05")
        
    st.markdown("### 🏛️ System Architecture Flow")
    st.markdown("""
    ```
    ┌──────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
    │ 1. Data Ingestion    │ ───► │ 2. Experiment Tracking  │ ───► │ 3. Evaluation & Reports │
    │ • Iris Reference CSV │      │ • MLflow Persistent Run │      │ • Confusion Matrix      │
    │ • Simulated Drift CSV│      │ • Params, Metrics & Art │      │ • Classification Report │
    └──────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                               │
                                               ▼
                                  ┌─────────────────────────┐
                                  │ 4. Drift Analysis       │
                                  │ • KS 2-Sample Test      │
                                  │ • P-Value vs Alpha 0.05 │
                                  │ • Drift/No-Drift Alert  │
                                  └─────────────────────────┘
    ```
    """)
    
    st.markdown("### 🎯 Project Objectives")
    st.markdown("""
    1. **MLflow Setup & Experiment Tracking:** Record hyperparameters (`C`, `max_iter`, `solver`), metrics (`Accuracy`, `Precision`, `Recall`, `F1`), and serialized pipeline artifacts.
    2. **Multiclass Classification:** Train regularized Logistic Regression models on Fisher's Iris dataset.
    3. **Controlled Covariate Shift Simulation:** Inject precise synthetic drift into `sepal_length` (+0.85 cm) and `petal_length` (+1.40 cm) while maintaining unshifted control features.
    4. **Non-parametric Statistical Drift Detection:** Implement two-sample Kolmogorov-Smirnov hypothesis tests to identify out-of-distribution features at 95% confidence level ($\alpha = 0.05$).
    5. **Educational Rigor:** Disentangle **Data Drift**, **Concept Drift**, and **Prediction Drift** for academic defense.
    """)

# ----------------- 2. DATASET EXPLORER -----------------
elif nav_choice == "2. Dataset Explorer":
    st.markdown("<div class='main-title'>Dataset Identification & Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Inspection of Fisher's 1936 Iris Dataset (150 Samples, 3 Classes, 4 Morphological Features)</div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Feature Distributions", "📋 Metadata Details"])
    
    with tab1:
        st.subheader("Reference vs. Current (Simulated Drift) Sample")
        col_ref, col_curr = st.columns(2)
        with col_ref:
            st.markdown("##### Reference Dataset (Original)")
            st.dataframe(ref_df.head(8), use_container_width=True)
        with col_curr:
            st.markdown("##### Current Dataset (Controlled Drift Applied)")
            st.dataframe(curr_df.head(8), use_container_width=True)
            
    with tab2:
        st.subheader("Class Distribution and Pairwise Scatter")
        feature_x = st.selectbox("X-axis feature:", ["sepal_length", "sepal_width", "petal_length", "petal_width"], index=2)
        feature_y = st.selectbox("Y-axis feature:", ["sepal_length", "sepal_width", "petal_length", "petal_width"], index=3)
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(
            data=ref_df,
            x=feature_x,
            y=feature_y,
            hue="species",
            palette="deep",
            s=70,
            ax=ax
        )
        ax.set_title(f"Pairwise Scatter: {feature_x} vs {feature_y} by Species", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)
        
    with tab3:
        summary = get_dataset_summary()
        st.json(summary)

# ----------------- 3. MLFLOW EXPERIMENTS -----------------
elif nav_choice == "3. MLflow Experiments":
    st.markdown("<div class='main-title'>MLflow Experiment Tracking & Run Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Comparison of Hyperparameter Runs recorded in MLflow Experiment: <code>Iris_Classification_Experiment</code></div>", unsafe_allow_html=True)
    
    summary_path = "reports/experiment_summary.csv"
    if os.path.exists(summary_path):
        exp_df = pd.read_csv(summary_path)
        st.dataframe(
            exp_df.style.highlight_max(subset=["accuracy", "precision", "recall", "f1_score"], color="#D1FAE5"),
            use_container_width=True
        )
        
        st.subheader("Model Performance Metric Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        metric_cols = ["accuracy", "precision", "recall", "f1_score"]
        exp_melted = exp_df.melt(id_vars=["run_name"], value_vars=metric_cols, var_name="Metric", value_name="Score")
        sns.barplot(data=exp_melted, x="run_name", y="Score", hue="Metric", ax=ax, palette="Blues")
        ax.set_ylim(0.85, 1.02)
        ax.set_title("Performance Metrics Across MLflow Runs", fontweight="bold")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Experiment summary report not found. Run `python src/train.py` first.")
        
    st.subheader("Logged Artifacts: Confusion Matrices")
    cols = st.columns(3)
    cm_files = [
        ("Run 1 (Baseline C=1.0)", "plots/confusion_matrix_run_1_baseline_c1.0.png"),
        ("Run 2 (High Reg C=0.1)", "plots/confusion_matrix_run_2_high_regularization_c0.1.png"),
        ("Run 3 (Low Reg C=10.0)", "plots/confusion_matrix_run_3_low_regularization_c10.0.png")
    ]
    for col, (title, path) in zip(cols, cm_files):
        with col:
            st.markdown(f"**{title}**")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.info(f"Artifact {path} will appear after training.")

# ----------------- 4. DATA DRIFT ANALYSIS -----------------
elif nav_choice == "4. Data Drift Analysis (KS Test)":
    st.markdown("<div class='main-title'>Multivariate Data Drift Detection (KS Test)</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Empirical Distribution Comparison between Reference and Current Data at significance level α = 0.05</div>", unsafe_allow_html=True)
    
    drift_df = run_ks_drift_detection(ref_df, curr_df, alpha=0.05)
    
    st.subheader("Feature-wise Drift Detection Summary Table")
    
    # Render table with styled badges
    disp_df = drift_df.copy()
    disp_df["Status"] = disp_df["drift_detected"].apply(
        lambda d: "🚨 DRIFT DETECTED" if d else "✅ NO SIGNIFICANT DRIFT"
    )
    st.dataframe(
        disp_df[["feature", "ks_statistic", "p_value", "Status", "ref_mean", "curr_mean", "mean_shift"]],
        use_container_width=True
    )
    
    st.markdown("### 📊 Distribution Comparison Visualizer")
    selected_feature = st.selectbox(
        "Select Feature to Inspect Empirical Cumulative Distributions:",
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    
    feat_row = drift_df[drift_df["feature"] == selected_feature].iloc[0]
    drift_status = feat_row["drift_status"]
    is_drifted = feat_row["drift_detected"]
    
    col_plot, col_info = st.columns([2, 1])
    
    with col_plot:
        fig, (ax_kde, ax_ecdf) = plt.subplots(1, 2, figsize=(12, 4.5))
        
        # KDE
        sns.kdeplot(ref_df[selected_feature], label="Reference", color="#1976D2", fill=True, alpha=0.3, ax=ax_kde)
        sns.kdeplot(curr_df[selected_feature], label="Current", color="#D32F2F" if is_drifted else "#2E7D32", fill=True, alpha=0.3, ax=ax_kde)
        ax_kde.set_title(f"Density (KDE): {selected_feature}", fontweight="bold")
        ax_kde.legend()
        
        # ECDF
        sns.ecdfplot(ref_df[selected_feature], label="Reference ECDF", color="#1976D2", ax=ax_ecdf, linewidth=2)
        sns.ecdfplot(curr_df[selected_feature], label="Current ECDF", color="#D32F2F" if is_drifted else "#2E7D32", ax=ax_ecdf, linewidth=2)
        ax_ecdf.set_title(f"ECDF (KS Distance D={feat_row['ks_statistic']})", fontweight="bold")
        ax_ecdf.legend()
        
        st.pyplot(fig)
        plt.close(fig)
        
    with col_info:
        st.markdown(f"#### Feature: `{selected_feature}`")
        if is_drifted:
            st.error(f"Status: **{drift_status}**")
        else:
            st.success(f"Status: **{drift_status}**")
        st.metric("KS Statistic (D)", f"{feat_row['ks_statistic']:.4f}")
        st.metric("p-value", f"{feat_row['p_value']}")
        st.metric("Mean Shift (Δμ)", f"{feat_row['mean_shift']:.3f} cm")
        st.caption("Significance threshold: α = 0.05. Rejection of H₀ occurs when p < α.")

# ----------------- 5. THEORY & DEFINITIONS -----------------
elif nav_choice == "5. Theory & Definitions (Viva Prep)":
    st.markdown("<div class='main-title'>Academic Foundations & Viva Preparation</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Formal definitions, mathematical formulations, and answers to common examiner questions.</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. The Three Forms of Drift in Production Machine Learning
    
    | Term | Formal Mathematical Definition | Real-World Intuitive Meaning |
    | :--- | :--- | :--- |
    | **Data Drift (Covariate Shift)** | $P(X)_{current} \neq P(X)_{reference}$ while $P(Y|X)$ remains constant. | The input feature distributions have changed, but the fundamental physical/biological relationship mapping inputs to labels hasn't. |
    | **Concept Drift** | $P(Y|X)_{current} \neq P(Y|X)_{reference}$ | The definition or relationship of the target variable changes over time (e.g., consumer habits changing after macroeconomic inflation). |
    | **Prediction Drift (Prior Shift)** | $P(\hat{Y})_{current} \neq P(\hat{Y})_{reference}$ | The model's output prediction frequency shifts, indicating either population changes or model bias. |
    
    ---
    
    ### 2. The Kolmogorov-Smirnov (KS) Two-Sample Test
    
    - **Why KS Test?** It is a non-parametric test, meaning it does **not** assume the underlying data follows a Normal (Gaussian) distribution.
    - **Test Statistic ($D$):**
      $$D = \sup_x |F_1(x) - F_2(x)|$$
      Where $F_1(x)$ and $F_2(x)$ are the empirical cumulative distribution functions of the reference and current datasets.
    - **Hypothesis Formulation:**
      - $H_0$ (Null Hypothesis): The two samples come from the same continuous distribution.
      - $H_1$ (Alternative Hypothesis): The two samples come from different continuous distributions.
    - **Decision Boundary:** With $\alpha = 0.05$, if $p < 0.05$, we reject $H_0$ at the 95% confidence level and alert that drift is detected.
    
    ---
    
    ### 3. Critical Examiner Question: "Does Data Drift mean model accuracy has dropped?"
    
    > **Answer:** **NO.** Detecting data drift tells us that the input features are out-of-distribution compared to the training set. However:
    > 1. The drift might be occurring along features that have negligible coefficients in the logistic regression equation.
    > 2. The shifted samples may still lie far from the decision boundary in a safe classification region.
    > 3. Therefore, data drift is an **early-warning telemetry indicator**, not proof of immediate model degradation.
    """)
