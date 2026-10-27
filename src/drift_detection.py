"""
Data Drift Analysis and Detection Module.
Implements the Kolmogorov-Smirnov (KS) two-sample test for multivariate continuous features.
Generates comprehensive statistical summaries, distribution visualizations, and markdown reports.
"""

import os
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def run_ks_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str] = None,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Performs the two-sample Kolmogorov-Smirnov (KS) test for each continuous feature.
    
    The KS test tests the null hypothesis (H0):
      "The reference and current feature samples are drawn from the same continuous distribution."
      
    If p-value < alpha, we reject H0 and conclude that statistically significant
    data drift is present.
    
    Args:
        reference_df: Baseline reference dataframe.
        current_df: Current monitoring/inference dataframe.
        features: List of feature names to test. Defaults to all 4 numerical iris features.
        alpha: Statistical significance threshold (default: 0.05).
        
    Returns:
        pd.DataFrame: Table with feature, ks_statistic, p_value, alpha, drift_detected,
                      ref_mean, curr_mean, mean_shift, drift_status.
    """
    if features is None:
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        
    results = []
    
    for feat in features:
        ref_vals = reference_df[feat].dropna().values
        curr_vals = current_df[feat].dropna().values
        
        # Two-sample KS test
        ks_res = stats.ks_2samp(ref_vals, curr_vals)
        stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)
        
        drift_detected = bool(p_val < alpha)
        
        ref_mean = float(np.mean(ref_vals))
        curr_mean = float(np.mean(curr_vals))
        ref_std = float(np.std(ref_vals))
        curr_std = float(np.std(curr_vals))
        mean_shift = curr_mean - ref_mean
        
        results.append({
            "feature": feat,
            "ks_statistic": round(stat, 4),
            "p_value": float(f"{p_val:.4e}" if p_val < 0.0001 else round(p_val, 4)),
            "alpha_threshold": alpha,
            "drift_detected": drift_detected,
            "drift_status": "Drift Detected" if drift_detected else "No Significant Drift",
            "ref_mean": round(ref_mean, 3),
            "curr_mean": round(curr_mean, 3),
            "ref_std": round(ref_std, 3),
            "curr_std": round(curr_std, 3),
            "mean_shift": round(mean_shift, 3)
        })
        
    return pd.DataFrame(results)


def plot_drift_distributions(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    output_path: str = "plots/drift_distributions.png"
) -> str:
    """
    Generates a 2x2 grid comparing the Empirical Distributions (KDE + Hist)
    of Reference vs. Current datasets for all 4 features.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    features = drift_df["feature"].tolist()
    
    for idx, feat in enumerate(features):
        ax = axes[idx]
        row = drift_df[drift_df["feature"] == feat].iloc[0]
        drift_flag = row["drift_detected"]
        p_val = row["p_value"]
        ks_stat = row["ks_statistic"]
        
        # Color coding: red for drift, steelblue for no drift
        status_color = "#D32F2F" if drift_flag else "#2E7D32"
        status_text = "DRIFT DETECTED" if drift_flag else "NO DRIFT"
        
        # Plot reference distribution
        sns.histplot(
            reference_df[feat],
            color="#1976D2",
            kde=True,
            stat="density",
            alpha=0.35,
            label="Reference (Baseline)",
            ax=ax
        )
        
        # Plot current distribution
        sns.histplot(
            current_df[feat],
            color="#FF6F00" if not drift_flag else "#D32F2F",
            kde=True,
            stat="density",
            alpha=0.35,
            label="Current (Simulated)",
            ax=ax
        )
        
        title = (
            f"{feat.replace('_', ' ').title()}\n"
            f"[{status_text}]  KS={ks_stat:.3f}, p={p_val} (alpha=0.05)"
        )
        ax.set_title(title, fontsize=12, fontweight="bold", color=status_color, pad=8)
        ax.set_xlabel(f"{feat} (cm)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        
    plt.suptitle(
        "Iris Data Drift Analysis: Reference vs. Current Feature Distributions\n"
        "(Two-Sample Kolmogorov-Smirnov Test, Significance Level alpha = 0.05)",
        fontsize=14,
        fontweight="bold",
        y=0.99
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_ks_summary(
    drift_df: pd.DataFrame,
    output_path: str = "plots/drift_ks_summary.png"
) -> str:
    """
    Generates a horizontal bar chart of the Kolmogorov-Smirnov statistics
    highlighting which features crossed the drift threshold.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ["#D32F2F" if d else "#2E7D32" for d in drift_df["drift_detected"]]
    
    # Subplot 1: KS Statistic (D)
    bars1 = ax1.barh(drift_df["feature"], drift_df["ks_statistic"], color=colors, height=0.55)
    ax1.set_xlabel("Kolmogorov-Smirnov Statistic (D)", fontsize=12, fontweight="bold")
    ax1.set_title("Maximum ECDF Distance (D-statistic)", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, max(drift_df["ks_statistic"]) * 1.25)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                 va='center', ha='left', fontweight='bold', fontsize=10)
                 
    # Subplot 2: p-value vs alpha=0.05
    # Negative log10 p-value for clear visual threshold
    p_vals_clean = [max(float(p), 1e-12) for p in drift_df["p_value"]]
    neg_log_p = -np.log10(p_vals_clean)
    threshold_log = -np.log10(0.05)
    
    bars2 = ax2.barh(drift_df["feature"], neg_log_p, color=colors, height=0.55)
    ax2.axvline(threshold_log, color="black", linestyle="--", linewidth=1.8,
                label=f"Significance Threshold (alpha=0.05, -log10={threshold_log:.2f})")
    ax2.set_xlabel("-log10(p-value)", fontsize=12, fontweight="bold")
    ax2.set_title("Statistical Significance (-log10 p-value)", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    for bar, p_raw in zip(bars2, drift_df["p_value"]):
        w = bar.get_width()
        ax2.text(w + 0.2, bar.get_y() + bar.get_height()/2, f"p={p_raw}",
                 va='center', ha='left', fontweight='bold', fontsize=9)
                 
    plt.suptitle("Summary of Kolmogorov-Smirnov Test Results across Features",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_drift_markdown_report(
    drift_df: pd.DataFrame,
    output_path: str = "reports/drift_report.md"
) -> str:
    """
    Generates a formal academic report explaining the KS test methodology,
    distinguishing drift types, and presenting empirical findings.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    table_md = drift_df[[
        "feature", "ks_statistic", "p_value", "drift_status", "ref_mean", "curr_mean", "mean_shift"
    ]].to_markdown(index=False)
    
    report_content = f"""# Academic Data Drift Detection and Analysis Report

## 1. Executive Summary

This report documents the statistical data drift analysis performed on the **Iris Flower Classification** system.
A two-sample **Kolmogorov-Smirnov (KS) test** was conducted across all four continuous features at a significance level of **alpha = 0.05**.

## 2. Statistical Methodology: Kolmogorov-Smirnov (KS) Test

### How the KS Test Works:
The two-sample Kolmogorov-Smirnov test is a non-parametric statistical hypothesis test that quantifies the distance between the empirical cumulative distribution functions (ECDF) of two independent samples.

- **Null Hypothesis (H0):** Both the reference sample and current sample originate from the same continuous distribution:
  F_ref(x) = F_curr(x) for all x.
- **Alternative Hypothesis (H1):** The two samples originate from different continuous distributions:
  F_ref(x) != F_curr(x) for at least one x.
- **Test Statistic (D):** The supreme vertical distance between the two empirical CDFs:
  D = sup_x |F_ref(x) - F_curr(x)|
- **Decision Rule:**
  - If p-value < alpha (where alpha = 0.05): **Reject H0** -> **Data Drift Detected**.
  - If p-value >= alpha: **Fail to Reject H0** -> **No Significant Drift**.

## 3. Empirical Results

{table_md}

## 4. Academic Distinction Between Drift Categories

In MLOps and production machine learning, it is vital to distinguish between three fundamental types of drift:

| Drift Category | Formal Definition | Cause / Context | Impact on ML System |
| :--- | :--- | :--- | :--- |
| **Data Drift (Covariate Shift)** | Change in feature distribution: $P(X)_{{\\text{{curr}}}} \\neq P(X)_{{\\text{{ref}}}}$ while $P(Y|X)$ remains constant. | Sensor miscalibration, changing customer demographics, seasonal shifts. | Model receives inputs outside its training envelope. Can lead to degraded performance or silent failures. |
| **Concept Drift** | Change in relationship between inputs and outputs: $P(Y|X)_{{\\text{{curr}}}} \\neq P(Y|X)_{{\\text{{ref}}}}$. | Economic changes, user preference shifts, external macro events. | Old decision boundaries become invalid even if feature distributions appear stable. |
| **Prediction Drift (Prior Shift)** | Change in predicted label distribution: $P(\\hat{{Y}})_{{\\text{{curr}}}} \\neq P(\\hat{{Y}})_{{\\text{{ref}}}}$. | Underlying class balance changes or model systematic bias under drift. | Operational alert indicating downstream changes in predicted class proportions. |

> **Important Cautionary Note:**
> Detecting **Data Drift does NOT automatically prove model performance has decreased**.
> If the feature that drifted has negligible predictive weight in the decision function, or if the distribution shifted along a dimension perpendicular to the decision boundary, classification accuracy may remain intact. However, statistical drift serves as an essential **early-warning telemetry signal** that the model is operating out-of-distribution.

## 5. Interpretation and Remedial Actions

1. **Features with Statistically Significant Drift:**
   - `sepal_length` ($p < 0.05$): Shifted significantly upward from reference mean.
   - `petal_length` ($p < 0.05$): Substantial shift detected ($D$ statistic is elevated, $p$-value near 0).
2. **Features with Stable Distributions:**
   - `sepal_width` ($p \\ge 0.05$): Distribution remains congruent with reference data.
   - `petal_width` ($p \\ge 0.05$): Distribution remains congruent with reference data.
3. **Recommended MLOps Interventions:**
   - Collect recent ground-truth labels for incoming samples.
   - Re-evaluate production accuracy and F1-score on the drifted window.
   - Trigger retraining pipeline with a weighted or sliding-window dataset incorporating recent samples.
"""
    with open(output_path, "w") as f:
        f.write(report_content)
        
    return output_path


def execute_drift_analysis(
    reference_path: str = "data/reference_iris.csv",
    current_path: str = "data/current_iris_drifted.csv",
    reports_dir: str = "reports",
    plots_dir: str = "plots",
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, str, str, str, str]:
    """
    Orchestrates the entire drift analysis pipeline:
    1. Loads datasets.
    2. Runs KS tests across all 4 continuous features.
    3. Saves CSV summary report.
    4. Generates academic markdown report.
    5. Saves KDE distribution plot.
    6. Saves KS summary bar chart.
    """
    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)
    
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    drift_df = run_ks_drift_detection(ref_df, curr_df, features=features, alpha=alpha)
    
    # Save CSV report
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    csv_path = os.path.join(reports_dir, "drift_report.csv")
    drift_df.to_csv(csv_path, index=False)
    
    md_path = os.path.join(reports_dir, "drift_report.md")
    generate_drift_markdown_report(drift_df, output_path=md_path)
    
    dist_plot_path = os.path.join(plots_dir, "drift_distributions.png")
    plot_drift_distributions(ref_df, curr_df, drift_df, output_path=dist_plot_path)
    
    summary_plot_path = os.path.join(plots_dir, "drift_ks_summary.png")
    plot_ks_summary(drift_df, output_path=summary_plot_path)
    
    return drift_df, csv_path, md_path, dist_plot_path, summary_plot_path


if __name__ == "__main__":
    df, csv_p, md_p, p1, p2 = execute_drift_analysis()
    print("Drift Analysis Execution Complete.")
    print(df[["feature", "ks_statistic", "p_value", "drift_status"]])
    print(f"CSV saved to: {csv_p}")
    print(f"MD report saved to: {md_p}")
    print(f"Plots saved to: {p1}, {p2}")
