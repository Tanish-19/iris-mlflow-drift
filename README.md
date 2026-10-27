# 🌸 Iris ML: MLflow Tracking & Data Drift Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.17%2B-brightgreen.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/Pytest-Passing-success.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, lightweight machine learning engineering project designed for the **ISE-2 Assessment: "Theory-ISE2-MLFlow Data Analysis"**. This project demonstrates local MLflow experiment tracking, multiclass classification on Fisher's Iris dataset, controlled data drift simulation, and rigorous non-parametric hypothesis testing using the two-sample Kolmogorov-Smirnov (KS) test.

Fully optimized for **macOS and Apple Silicon (M1/M2/M3, 8GB RAM)** without requiring cloud subscriptions, Docker, or external paid APIs.

---

## 📑 Table of Contents

- [1. Project Overview & Architecture](#1-project-overview--architecture)
- [2. Dataset Specification](#2-dataset-specification)
- [3. Project Directory Structure](#3-project-directory-structure)
- [4. Quickstart & Installation](#4-quickstart--installation)
- [5. Execution Workflow](#5-execution-workflow)
- [6. MLflow Experiment Tracking Results](#6-mlflow-experiment-tracking-results)
- [7. Data Drift Simulation & KS Statistical Testing](#7-data-drift-simulation--ks-statistical-testing)
- [8. Theoretical Distinctions: Data vs. Concept vs. Prediction Drift](#8-theoretical-distinctions-data-vs-concept-vs-prediction-drift)
- [9. Streamlit Interactive Dashboard](#9-streamlit-interactive-dashboard)
- [10. Automated Testing Suite](#10-automated-testing-suite)
- [11. Academic Viva & Assessment Q&A](#11-academic-viva--assessment-qa)

---

## 1. Project Overview & Architecture

Modern machine learning systems in production do not exist in a static vacuum. Over time, physical sensor degradation, environmental shifts, or changing demographic populations alter input data distributions—a phenomenon known as **Data Drift (Covariate Shift)**.

This project delivers a complete local MLOps pipeline:

```
┌─────────────────────────────────┐
│     Scikit-learn Iris Data      │
└────────────────┬────────────────┘
                 │
                 ├───► [data/reference_iris.csv] ───► Baseline Reference
                 │
                 └───► Controlled Perturbation   ───► [data/current_iris_drifted.csv]
                                                        (Simulated Covariate Shift)
                                                                 │
      ┌──────────────────────────────────────────────────────────┘
      ▼
┌───────────────────────────────┐        ┌───────────────────────────────┐
│  MLflow Experiment Tracking   │        │     Data Drift Analysis       │
│  - Logistic Regression        │        │  - Kolmogorov-Smirnov (KS)    │
│  - 3 Parametric Runs          │        │  - Empirical CDF Distance (D) │
│  - Accuracy, F1, Loss Metrics │        │  - p-value vs alpha = 0.05    │
│  - Plots, Reports & Models    │        │  - Feature Status Badges      │
└───────────────┬───────────────┘        └───────────────┬───────────────┘
                │                                        │
                ▼                                        ▼
    [Local MLflow UI :5001]                  [Streamlit Dashboard :8501]
```

### Key Deliverables
1. **Experiment Tracking**: Systematic logging of hyperparameters, evaluation metrics, confusion matrices, classification reports, and serialized pipelines using MLflow.
2. **Controlled Drift Simulation**: Intentional, reproducible distribution shifts applied to `sepal_length` (+0.85 cm) and `petal_length` (+1.40 cm) while maintaining unshifted control features.
3. **Statistical Hypothesis Testing**: Automated two-sample Kolmogorov-Smirnov tests comparing baseline vs. drifted distributions at significance level $\alpha = 0.05$.
4. **Academic Documentation**: Formal thesis report adhering to college ISE-2 formatting guidelines (`ISE2_MLFLOW_DOCUMENTATION.md`).
5. **Interactive UI**: Streamlit dashboard for real-time visualization and viva demonstration.

---

## 2. Dataset Specification

The project uses the canonical **Iris Flower Dataset** introduced by British statistician and biologist Ronald Fisher (1936).

| Attribute | Specification |
| :--- | :--- |
| **Dataset Source** | `sklearn.datasets.load_iris` |
| **Total Instances** | 150 instances (50 instances per class) |
| **Input Features** | 4 continuous numerical morphological features (in cm):<br>• `sepal_length`<br>• `sepal_width`<br>• `petal_length`<br>• `petal_width` |
| **Target Variable** | `species` (Multiclass: `0: setosa`, `1: versicolor`, `2: virginica`) |
| **Missing Values** | 0 missing / null values |
| **Class Balance** | Perfectly balanced (33.3% per class) |
| **Suitability** | Clean, benchmarked multiclass dataset with intuitive geometric feature boundaries, ideal for demonstrating controlled covariate shift and MLflow tracking without computational overhead. |

---

## 3. Project Directory Structure

```
iris-mlflow-drift/
├── data/
│   ├── reference_iris.csv             # Pristine baseline Iris dataset (150 rows)
│   └── current_iris_drifted.csv       # Simulated drifted dataset (controlled shift)
├── src/
│   ├── data_loader.py                 # Data ingestion, preparation, and drift simulation
│   ├── train.py                       # MLflow tracking, pipeline training, and 3 runs
│   └── drift_detection.py             # KS two-sample test, report generator, and plots
├── reports/
│   ├── experiment_summary.csv         # Comparison table across MLflow runs
│   ├── drift_report.csv               # Feature-wise KS statistics, p-values & drift flags
│   └── drift_report.md                # Comprehensive academic markdown drift report
├── plots/
│   ├── confusion_matrix_run_1_baseline_c1.0.png
│   ├── confusion_matrix_run_2_high_regularization_c0.1.png
│   ├── confusion_matrix_run_3_low_regularization_c10.0.png
│   ├── drift_distributions.png        # 4-panel KDE & histogram comparison
│   └── drift_ks_summary.png           # KS statistic bar chart & p-value thresholds
├── artifacts/
│   ├── classification_report_*.txt    # Per-run precision, recall, and F1 reports
│   └── dataset_info_*.json            # Serialized dataset metadata
├── tests/
│   └── test_project.py                # 9 automated unit and integration pytest tests
├── mlruns/                            # Persistent local MLflow tracking store
├── app.py                             # Streamlit interactive visual dashboard
├── requirements.txt                   # Pinned project dependencies
├── .gitignore                         # Excludes venv, pycache, and temporary files
├── README.md                          # Project overview and reproduction guide
└── ISE2_MLFLOW_DOCUMENTATION.md       # Exact 5-section college theory submission
```

---

## 4. Quickstart & Installation

### Step 1: Clone or Navigate to the Workspace
```bash
cd /Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift
```

### Step 2: Initialize Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Execution Workflow

Execute all components sequentially:

### 1. Ingest Data & Generate Controlled Drift
```bash
python src/data_loader.py
```
*Creates `data/reference_iris.csv` and `data/current_iris_drifted.csv`.*

### 2. Train Models & Track Experiments with MLflow
```bash
python src/train.py
```
*Executes 3 distinct hyperparameter configurations, logs metrics/artifacts to MLflow, and writes `reports/experiment_summary.csv`.*

### 3. Run Kolmogorov-Smirnov Data Drift Analysis
```bash
python src/drift_detection.py
```
*Performs two-sample KS hypothesis testing, generates distribution plots in `plots/`, and outputs `reports/drift_report.csv`.*

### 4. Run Automated Pytest Validation
```bash
pytest tests/test_project.py -v
```
*Validates 9 automated test cases covering data loading, training, prediction bounds, and drift detection assertions.*

### 5. Launch Local MLflow Tracking UI
```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --port 5001
```
*Open [http://localhost:5001](http://localhost:5001) in your browser to inspect experiments, parameters, and logged model artifacts.*

### 6. Launch Interactive Streamlit Dashboard (Optional)
```bash
streamlit run app.py --server.port 8501
```
*Open [http://localhost:8501](http://localhost:8501) for interactive visual presentation.*

---

## 6. MLflow Experiment Tracking Results

All runs are recorded under the experiment name: **`Iris_Classification_Experiment`**.

### Empirical Run Comparison Table

| Run Name | Model | Regularization $C$ | Max Iterations | Solver | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Train Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1: Baseline** | LogisticRegression | `1.0` | 200 | `lbfgs` | **0.9333** | 0.9333 | 0.9333 | 0.9333 | 0.9583 |
| **Run 2: High Regularization** | LogisticRegression | `0.1` | 100 | `lbfgs` | **0.8667** | 0.8750 | 0.8667 | 0.8653 | 0.9333 |
| **Run 3: Low Regularization** | LogisticRegression | `10.0` | 300 | `lbfgs` | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.9750 |

### Key MLflow Artifacts Logged Per Run
- **Pipeline Model:** `StandardScaler` + `LogisticRegression` packaged via `mlflow.sklearn.log_model()`.
- **Confusion Matrix:** High-resolution heatmap saved to `plots/` and logged to MLflow artifacts.
- **Classification Report:** Detailed per-class precision, recall, and support stored in `artifacts/`.
- **Dataset Metadata:** Schema and instance counts stored as JSON in `artifacts/`.

---

## 7. Data Drift Simulation & KS Statistical Testing

### Controlled Drift Setup
To simulate realistic real-world covariate shift in an academic setting:
- **`sepal_length`:** Injected synthetic upward shift ($\Delta\mu \approx +0.85\text{ cm}$, $+1.0\sigma$).
- **`petal_length`:** Injected synthetic upward shift ($\Delta\mu \approx +1.40\text{ cm}$, $+0.8\sigma$).
- **`sepal_width`:** Kept stable within original distribution ($\Delta\mu \approx +0.01\text{ cm}$).
- **`petal_width`:** Kept stable within original distribution ($\Delta\mu \approx -0.01\text{ cm}$).

### Empirical Kolmogorov-Smirnov Test Results ($\alpha = 0.05$)

| Feature | KS Statistic ($D$) | $p$-value | Significance Threshold ($\alpha$) | Ref Mean ($\mu_{ref}$) | Curr Mean ($\mu_{curr}$) | Mean Shift ($\Delta\mu$) | Drift Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sepal_length`** | **0.3733** | **$1.0722 \times 10^{-9}$** | 0.05 | 5.843 cm | 6.697 cm | +0.853 cm | 🚨 **Drift Detected** |
| **`sepal_width`** | 0.1000 | 0.4425 | 0.05 | 3.057 cm | 3.071 cm | +0.014 cm | ✅ **No Significant Drift** |
| **`petal_length`** | **0.4067** | **$1.7712 \times 10^{-11}$** | 0.05 | 3.758 cm | 5.163 cm | +1.405 cm | 🚨 **Drift Detected** |
| **`petal_width`** | 0.1333 | 0.1391 | 0.05 | 1.199 cm | 1.189 cm | -0.010 cm | ✅ **No Significant Drift** |

### Statistical Test Interpretation
- For `sepal_length` and `petal_length`, the test statistic $D$ represents a substantial maximum separation between the empirical CDFs ($D > 0.37$). The resulting $p$-values are orders of magnitude below $\alpha = 0.05$, conclusively rejecting $H_0$.
- For `sepal_width` ($p = 0.4425$) and `petal_width` ($p = 0.1391$), we fail to reject $H_0$, confirming that no statistically significant drift occurred.

---

## 8. Theoretical Distinctions: Data vs. Concept vs. Prediction Drift

| Dimension | **Data Drift (Covariate Shift)** | **Concept Drift** | **Prediction Drift (Prior Shift)** |
| :--- | :--- | :--- | :--- |
| **Mathematical Definition** | $P(X)_{curr} \neq P(X)_{ref}$ with $P(Y \mid X)$ constant. | $P(Y \mid X)_{curr} \neq P(Y \mid X)_{ref}$. | $P(\hat{Y})_{curr} \neq P(\hat{Y})_{ref}$. |
| **Physical Meaning** | Input feature distributions change, but the underlying physical laws or species traits remain identical. | The relationship between features and target labels changes over time. | The proportion of predicted classes shifts over time. |
| **Example Scenario** | Calipers measuring flowers are recalibrated in inches or biased by +0.8 cm. | Due to genetic hybridization, flowers with long petals now classify under a new species definition. | The model begins predicting 70% Setosa and only 10% Virginica. |
| **Performance Impact** | Model may perform fine if drift occurs far from decision boundary, but operates out-of-distribution. | Model performance **always degrades** because historical decision boundaries no longer map to truth. | Downstream automated business operations receive imbalanced class assignments. |

> ⚠️ **Critical Viva Principle:** Detecting Data Drift does **NOT** automatically prove that classification accuracy has declined. A feature may experience distribution shift, but if that feature has low weighting or the shift remains within an unambiguous classification margin, accuracy can remain stable. However, drift acts as a vital **early-warning telemetry indicator** to trigger human-in-the-loop review and model retraining.

---

## 9. Streamlit Interactive Dashboard

An optional interactive dashboard is included in `app.py`.

To launch:
```bash
streamlit run app.py --server.port 8501
```

### Features included:
1. **Interactive Dataset Explorer:** Pairwise bivariate scatterplots and raw data comparison.
2. **MLflow Run Comparison:** Dynamic metric bar charts and confusion matrix previews.
3. **Interactive KS Drift Inspector:** Select any of the 4 features to inspect overlaid Kernel Density Estimates (KDE) and Empirical Cumulative Distribution Functions (ECDF).
4. **Viva Preparation Tab:** Embedded mathematical formulas and conceptual cheat-sheets.

---

## 10. Automated Testing Suite

All project features are validated using `pytest`.

```bash
pytest tests/test_project.py -v
```

### Test Coverage Summary:
- `test_load_iris_data_shape_and_columns`: Verifies 150 rows, 4 columns, and class labels.
- `test_dataset_no_missing_values`: Verifies 0 null values.
- `test_dataset_summary_metadata`: Verifies structured metadata extraction.
- `test_model_training_and_predictions`: Validates pipeline execution and metric bounds $[0, 1]$.
- `test_confusion_matrix_artifact_generated`: Verifies plot serialization.
- `test_drift_simulation_creates_controlled_shift`: Validates synthetic shift injection.
- `test_ks_drift_detection_results_format`: Verifies output dataframe structure.
- `test_ks_drift_detection_identifies_shifted_features`: Asserts that shifted features trigger $p < 0.05$ and unshifted features maintain $p \ge 0.05$.
- `test_dataset_saving_and_drift_execution`: End-to-end integration test.

---

## 11. Academic Viva & Assessment Q&A

### Q1: Why did you use the Kolmogorov-Smirnov (KS) test instead of a Student's t-test?
**Answer:** The Student's t-test only checks whether the sample means differ and strictly assumes data is normally distributed. The two-sample Kolmogorov-Smirnov test is a non-parametric test that evaluates the entire shape of the empirical distribution function (including variance, skewness, and modality) without requiring Gaussian assumptions.

### Q2: What is the significance of the $D$-statistic in the KS test?
**Answer:** The $D$-statistic represents the maximum vertical distance between the empirical cumulative distribution functions of the reference and current datasets:
$$D = \sup_x |F_{ref}(x) - F_{curr}(x)|$$
A larger $D$ indicates greater divergence between the two distributions.

### Q3: How did you configure MLflow without Docker or Cloud setup?
**Answer:** MLflow was configured locally with a persistent filesystem backend directory (`file:./mlruns`). All runs were grouped under the experiment name `Iris_Classification_Experiment`, logging parameters, metrics, confusion matrix plots, and serialized scikit-learn models natively on macOS.

---

## 👤 Author & Academic Details
- **Student Name:** College ISE-2 Candidate
- **Course / Assessment:** ISE-2 Machine Learning & MLOps
- **Topic:** MLflow Experiment Tracking & Data Drift Analysis
