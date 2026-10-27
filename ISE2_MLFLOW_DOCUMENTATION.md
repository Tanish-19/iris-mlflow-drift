# ISE-2 ACADEMIC EVALUATION REPORT
## Course Assessment: Machine Learning & MLOps (Theory-ISE2-MLFlow Data Analysis)
## Title: Iris Flower Classification: MLflow Experiment Tracking and Kolmogorov-Smirnov Data Drift Analysis

### Student Details
- **Student Name:** Tanish Patgaonkar
- **Roll Number:** 42
- **PRN:** 2324000399
- **Academic Year:** 2026-27

---

## Section 1: MLflow Installation and Experiment Tracking

### 1.1 Environment Setup

The machine learning experiment and tracking environment was configured entirely on a local development workstation, maintaining complete compatibility with Apple Silicon architecture without cloud-dependent infrastructure or paid proprietary APIs.

- **Operating System:** macOS (Darwin 24.x, Apple Silicon M3, 8GB Unified Memory Architecture).
- **Python Version:** Python 3.13.5 (configured inside an isolated virtual environment via `venv`).
- **MLflow Version:** MLflow 3.17.2 (Skinny / Core Tracing distribution).
- **Key Supporting Libraries:**
  - `scikit-learn`: 1.9.1
  - `pandas`: 3.1.0
  - `numpy`: 2.5.4
  - `scipy`: 1.18.1
  - `matplotlib`: 3.11.2
  - `seaborn`: 0.13.2
  - `pytest`: 9.1.1
- **Installation and Verification Status:**
  - Virtual environment initialized at `.venv/`.
  - Packages installed cleanly via `.venv/bin/pip install -r requirements.txt`.
  - MLflow file store compatibility confirmed via `MLFLOW_ALLOW_FILE_STORE=true`.
  - Persistent tracking store validated at local URI: `file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/mlruns`.
  - Local MLflow tracking UI successfully started and verified on `http://127.0.0.1:5001` (HTTP status code 200 OK verified).

---

### 1.2 Experiment Tracking

- **Experiment Name:** `Iris_Classification_Experiment`
- **Experiment Objective:**
  The primary objective of this experiment is to systematically track, evaluate, and compare multiple hyperparameter configurations of a multiclass Logistic Regression classifier on the Iris dataset. By logging parameters, metrics, evaluation reports, and confusion matrix artifacts in a centralized tracking store, we ensure complete experiment reproducibility, traceability, and auditability throughout the machine learning development lifecycle.

- **Parameters Recorded:**
  1. `model_name`: Classifier architecture (`LogisticRegression`).
  2. `test_size`: Fraction of data allocated to the test split (`0.20`).
  3. `random_state`: Pseudo-random number generator seed (`42`).
  4. `max_iterations`: Maximum iterations for solver convergence (`100`, `200`, `300`).
  5. `regularization_parameter_C`: Inverse regularization strength parameter ($C = 0.1, 1.0, 10.0$).
  6. `solver`: Optimization algorithm (`lbfgs`).
  7. `preprocessing`: Standardization pipeline step (`StandardScaler`).

- **Metrics Recorded:**
  1. `accuracy`: Ratio of correctly classified flower instances over total test instances.
  2. `train_accuracy`: Training accuracy score to inspect potential bias and variance.
  3. `precision`: Macro-averaged precision across all three botanical classes.
  4. `recall`: Macro-averaged sensitivity across all three botanical classes.
  5. `f1_score`: Macro-averaged harmonic mean of precision and recall.
  6. `f1_score_weighted`: Support-weighted F1-score accounting for class distribution.

- **Artifacts Recorded:**
  1. `confusion_matrix_*.png`: High-resolution Seaborn heatmap depicting true vs. predicted species labels.
  2. `classification_report_*.txt`: Complete per-class precision, recall, F1, and support metrics.
  3. `dataset_info_*.json`: Serialized metadata regarding dataset shape, features, and split distributions.
  4. `iris_model/`: Serialized Scikit-learn Pipeline (preprocessing scaler + model weights) logged via `mlflow.sklearn.log_model()`.

- **Brief Description of Successful Execution:**
  Three distinct experimental runs were systematically executed using `src/train.py`. The runs tested varying degrees of regularization ($C \in \{0.1, 1.0, 10.0\}$). All runs successfully converged and were logged to the persistent tracking repository.

#### Table 1: MLflow Run Summary & Metric Comparison

| Run Name | Inverse Reg ($C$) | Max Iterations | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Train Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1: Baseline** | 1.0 | 200 | **0.9333** | 0.9333 | 0.9333 | 0.9333 | 0.9583 |
| **Run 2: High Regularization** | 0.1 | 100 | **0.8667** | 0.8750 | 0.8667 | 0.8653 | 0.9333 |
| **Run 3: Low Regularization** | 10.0 | 300 | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 0.9750 |

---

## Section 2: Dataset Identification

### 2.1 Problem Statement

Automated taxonomic classification of botanical specimens based on morphological dimensions is a classic supervised classification problem. The objective is to construct a predictive mathematical model that takes continuous physical dimensions of flower blossoms (sepal and petal lengths and widths) and accurately maps them into one of three distinct species: *Iris setosa*, *Iris versicolor*, or *Iris virginica*.

In production environments, physical measuring instruments may suffer calibration drift, regional climate changes may alter flower growth rates, or seasonal shifts may induce statistical distribution deviations. Therefore, this project combines model training with automated data drift monitoring.

---

### 2.2 Dataset Details

- **Dataset Name:** Fisher's Iris Flower Dataset.
- **Dataset Source:** Built-in standard dataset in Scikit-learn (`sklearn.datasets.load_iris`), originally collected by Edgar Anderson and published by Ronald A. Fisher in 1936.
- **Number of Records:** 150 total records (50 instances per class).
- **Important Features (4 Continuous Morphological Measurements in centimeters):**
  1. `sepal_length`: Sepal length in centimeters (Range: 4.3 to 7.9 cm; Mean: 5.84 cm; Std: 0.83 cm).
  2. `sepal_width`: Sepal width in centimeters (Range: 2.0 to 4.4 cm; Mean: 3.06 cm; Std: 0.44 cm).
  3. `petal_length`: Petal length in centimeters (Range: 1.0 to 6.9 cm; Mean: 3.76 cm; Std: 1.76 cm).
  4. `petal_width`: Petal width in centimeters (Range: 0.1 to 2.5 cm; Mean: 1.20 cm; Std: 0.76 cm).
- **Target Variable:** `species` (Categorical variable with 3 classes):
  - Class 0: *Iris setosa*
  - Class 1: *Iris versicolor*
  - Class 2: *Iris virginica*
- **Use Case Category:** Multiclass Supervised Machine Learning Classification.

---

### 2.3 Dataset Selection Rationale

The Iris dataset is exceptionally well-suited for this academic assessment for multiple reasons:
1. **Geometric Intuitiveness:** The features have clear physical interpretations and distinct clustering characteristics (e.g., *Setosa* is linearly separable, while *Versicolor* and *Virginica* exhibit slight non-linear overlap).
2. **Deterministic Baseline:** The absence of missing values, encoding complications, or extreme noise allows us to isolate the effects of hyperparameters in MLflow without confusing data quality artifacts.
3. **Controlled Drift Verification:** Because the feature distributions are well-understood continuous unimodal and bimodal densities, synthetic perturbations can be introduced with high statistical precision to validate hypothesis testing.
4. **Computational Efficiency:** Minimal CPU and RAM footprint enables rapid local iterations and instant execution on Apple Silicon laptops.

---

## Section 3: Drift Scenario and Detection Method

### 3.1 Drift Type

The specific drift simulated and evaluated in this project is **Data Drift (Covariate Shift)**.

Mathematically, Covariate Shift occurs when the marginal probability distribution of the input features shifts between the baseline training phase and the production inference phase, while the true conditional probability distribution of the target labels given the features remains stationary:

$$P(X)_{\text{current}} \neq P(X)_{\text{reference}} \quad \text{while} \quad P(Y \mid X)_{\text{current}} = P(Y \mid X)_{\text{reference}}$$

To prevent confusion during academic evaluation, we clearly distinguish Covariate Shift from related concepts:
- **Concept Drift:** Occurs when the functional relationship mapping inputs to outputs changes ($P(Y \mid X)_{\text{current}} \neq P(Y \mid X)_{\text{reference}}$), meaning old decision boundaries are no longer valid.
- **Prediction Drift:** Occurs when the distribution of the model's output predictions changes ($P(\hat{Y})_{\text{current}} \neq P(\hat{Y})_{\text{reference}}$).

---

### 3.2 Reference and Current Data

- **Reference Dataset ($D_{\text{ref}}$):** The pristine baseline 150-sample Iris dataset exported directly from Scikit-learn to `data/reference_iris.csv`. This represents the data distribution observed during model training and validation.
- **Current Dataset ($D_{\text{curr}}$):** A simulated production monitoring dataset created in `data/current_iris_drifted.csv` by applying controlled mathematical perturbations to specific numerical features:
  - `sepal_length`: Shifted upward by adding normal perturbation $\mathcal{N}(0.85, 0.15^2)$, producing an approximate mean shift of $+0.85$ cm ($\approx +1.0$ standard deviation).
  - `petal_length`: Shifted upward by adding normal perturbation $\mathcal{N}(1.40, 0.20^2)$, producing an approximate mean shift of $+1.40$ cm ($\approx +0.8$ standard deviations).
  - `sepal_width`: Maintained without intentional distribution shift ($\mu_{\text{shift}} \approx +0.01$ cm, representing natural zero-mean micro-noise).
  - `petal_width`: Maintained without intentional distribution shift ($\mu_{\text{shift}} \approx -0.01$ cm).

*Academic Disclosure:* This is an intentionally controlled synthetic perturbation designed to simulate sensor miscalibration or environmental growth variation, allowing us to evaluate whether our statistical detector correctly flags only the shifted features while avoiding false positives on the unshifted control features.

---

### 3.3 Detection Method

- **Statistical Test:** Two-Sample **Kolmogorov-Smirnov (KS) Test** (`scipy.stats.ks_2samp`).
- **How the KS Test Operates:**
  The two-sample KS test is a non-parametric hypothesis test that quantifies the difference between two empirical cumulative distribution functions (ECDF).
  Let $F_{\text{ref}}(x)$ be the empirical CDF of the reference feature, and $F_{\text{curr}}(x)$ be the empirical CDF of the current feature.
  The test statistic $D$ is the supremum of the absolute differences between the two CDF curves:
  $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$
- **Hypothesis Formulation:**
  - **Null Hypothesis ($H_0$):** Both the reference and current feature samples are drawn from identical continuous probability distributions ($F_{\text{ref}}(x) = F_{\text{curr}}(x)$ for all $x$).
  - **Alternative Hypothesis ($H_1$):** The two samples originate from different probability distributions ($F_{\text{ref}}(x) \neq F_{\text{curr}}(x)$ for at least one $x$).
- **Features Analyzed:** All 4 continuous features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`).
- **Detection Threshold ($\alpha$):** Significance level $\alpha = 0.05$ (corresponding to a 95% statistical confidence level).
- **Drift Detection Criteria:**
  - If $p\text{-value} < \alpha$ (i.e., $p < 0.05$): Reject $H_0$ $\implies$ **"Drift Detected"**.
  - If $p\text{-value} \ge \alpha$ (i.e., $p \ge 0.05$): Fail to reject $H_0$ $\implies$ **"No Significant Drift"**.

---

## Section 4: Results, Interpretation and Conclusion

### 4.1 Drift Analysis Results

The automated drift detection pipeline was executed using `src/drift_detection.py`. The empirical numerical results generated from the actual runs are summarized in Table 2 below.

#### Table 2: Kolmogorov-Smirnov Test Results Across Iris Features

| Feature Name | KS Statistic ($D$) | $p$-value | Alpha ($\alpha$) | Ref Mean ($\mu_{\text{ref}}$) | Curr Mean ($\mu_{\text{curr}}$) | Mean Shift ($\Delta\mu$) | Statistical Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`sepal_length`** | **0.3733** | **$1.0722 \times 10^{-9}$** | 0.05 | 5.843 cm | 6.681 cm | +0.837 cm | 🚨 **Drift Detected** |
| **`sepal_width`** | 0.1000 | 0.4425 | 0.05 | 3.057 cm | 3.069 cm | +0.012 cm | ✅ **No Significant Drift** |
| **`petal_length`** | **0.4067** | **$1.7712 \times 10^{-11}$** | 0.05 | 3.758 cm | 5.172 cm | +1.414 cm | 🚨 **Drift Detected** |
| **`petal_width`** | 0.1333 | 0.1391 | 0.05 | 1.199 cm | 1.188 cm | -0.012 cm | ✅ **No Significant Drift** |

---

### 4.2 Interpretation

1. **Shifted Feature Validation (`sepal_length` and `petal_length`):**
   - For `sepal_length`, the test yielded a KS statistic of $D = 0.3733$ with an extremely small $p$-value of $1.0722 \times 10^{-9}$. Since $p \ll 0.05$, the null hypothesis is conclusively rejected.
   - For `petal_length`, the KS statistic reached $D = 0.4067$ with $p = 1.7712 \times 10^{-11}$. The null hypothesis is decisively rejected.
   - Both features show significant empirical CDF divergence, successfully reflecting the controlled synthetic perturbation.

2. **Control Feature Validation (`sepal_width` and `petal_width`):**
   - For `sepal_width`, the KS statistic was small ($D = 0.1000$) with $p = 0.4425$. Because $p \ge 0.05$, we fail to reject $H_0$.
   - For `petal_width`, $D = 0.1333$ with $p = 0.1391 \ge 0.05$, failing to reject $H_0$.
   - This demonstrates that the statistical detector exhibits high specificity and does not generate false alarms when features undergo only non-significant stochastic variations.

3. **Academic Distinction on Performance Degradation:**
   - **Crucial Theoretical Point:** Detecting data drift does **not** equate to an automatic drop in model accuracy.
   - If the shifted samples still fall securely within their respective class decision boundaries (for example, if larger petal length simply makes an *Iris virginica* sample even more unmistakably distinct from *Setosa*), classification accuracy may remain 100%.
   - However, data drift indicates that the model is receiving out-of-distribution inputs that violate the statistical assumptions under which it was validated.

---

### 4.3 Conclusion and Recommended Action

- **Conclusion:**
  The local MLflow experiment tracking and Kolmogorov-Smirnov drift detection architecture has successfully demonstrated full end-to-end functionality. The classification models attained high predictive performance (up to 100% test accuracy under $C=10.0$ and 93.3% under baseline $C=1.0$), and the two-sample KS test detected covariate shift in precisely the target features with high confidence ($p < 10^{-8}$).

- **Recommended MLOps Interventions:**
  1. **Telemetry Alerting:** Configure automated alerts when any feature's KS test yields $p < 0.05$ over a rolling 24-hour window.
  2. **Active Label Acquisition:** Pull a random sample of drifted flower instances and send them for expert botanical re-verification to acquire ground-truth labels.
  3. **Performance Audit:** Compute real-time accuracy and F1-score on newly labeled drifted samples to verify whether decision boundary fidelity has degraded.
  4. **Retraining & Champion-Challenger Deployment:** Retrain the model pipeline on an augmented dataset combining baseline data and recently acquired drifted samples, log the candidate model to MLflow, and perform a shadow deployment before promoting to production.

---

## Section 5: GitHub Repository and Reproducibility

### 5.1 GitHub Repository URL
- **GitHub Repository URL:** `https://github.com/Tanish-19/iris-mlflow-drift`
- **Local Repository Root:** `/Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift`
- **Configured Branch:** `main` (Git repository initialized, clean local commits, synced with remote origin).

---

### 5.2 Source Code and Project Inventory

All executable source code is modularly structured:
- [`src/data_loader.py`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/src/data_loader.py): Iris dataset loader and controlled covariate shift generator.
- [`src/train.py`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/src/train.py): MLflow tracking suite executing 3 distinct hyperparameter configurations.
- [`src/drift_detection.py`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/src/drift_detection.py): Two-sample KS statistical test engine, distribution plotter, and report generator.
- [`app.py`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/app.py): Interactive Streamlit presentation dashboard.
- [`tests/test_project.py`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/tests/test_project.py): 9 comprehensive automated pytest validation test cases.

---

### 5.3 README and Execution Instructions

Complete execution commands to reproduce every result locally from scratch:

```bash
# 1. Clone repository and navigate to folder
cd iris-mlflow-drift

# 2. Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate datasets (Reference and Current Drifted)
python src/data_loader.py

# 4. Execute MLflow training suite (Runs 1, 2, and 3)
python src/train.py

# 5. Execute Kolmogorov-Smirnov drift analysis
python src/drift_detection.py

# 6. Execute automated pytest test suite
pytest tests/test_project.py -v

# 7. Launch local MLflow UI
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --port 5001

# 8. Launch Streamlit UI
streamlit run app.py --server.port 8501
```

---

### 5.4 Requirements / Dependencies

The complete dependency manifest is version-locked in `requirements.txt`:
```txt
Flask==3.1.3
Flask-CORS==6.0.5
matplotlib==3.11.2
mlflow==3.17.2
numpy==2.5.4
pandas==3.1.0
pytest==9.1.1
scikit-learn==1.9.1
scipy==1.18.1
seaborn==0.13.2
streamlit==1.66.1
tabulate==0.10.0
```

---

### 5.5 Screenshots and Drift Reports

The following visual artifacts were generated and saved in the repository:
1. **Drift Distribution Plot:** [`plots/drift_distributions.png`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/plots/drift_distributions.png) (Overlaid KDE and histograms for all four features).
2. **KS Test Summary Plot:** [`plots/drift_ks_summary.png`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/plots/drift_ks_summary.png) (Horizontal bar charts displaying $D$-statistics and $-\log_{10}(p\text{-values})$).
3. **Confusion Matrices:**
   - Run 1 (Baseline $C=1.0$): [`plots/confusion_matrix_run_1_baseline_c1.0.png`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/plots/confusion_matrix_run_1_baseline_c1.0.png)
   - Run 2 (High Reg $C=0.1$): [`plots/confusion_matrix_run_2_high_regularization_c0.1.png`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/plots/confusion_matrix_run_2_high_regularization_c0.1.png)
   - Run 3 (Low Reg $C=10.0$): [`plots/confusion_matrix_run_3_low_regularization_c10.0.png`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/plots/confusion_matrix_run_3_low_regularization_c10.0.png)
4. **Drift CSV Report:** [`reports/drift_report.csv`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/reports/drift_report.csv)
5. **Experiment Comparison CSV:** [`reports/experiment_summary.csv`](file:///Users/tanishpatgaonkar/.gemini/antigravity-ide/scratch/iris-mlflow-drift/reports/experiment_summary.csv)

---

### 5.6 MLflow Experiment and Drift-Analysis Evidence

#### MLflow Tracking Verification Evidence
- **Experiment ID:** `Iris_Classification_Experiment`
- **Active Runs Logged:**
  - `Run_1_Baseline_C1.0` (Run ID: `0c10d1d8776344acbdf54020bd3347aa`)
  - `Run_2_High_Regularization_C0.1` (Run ID: `4e4f60a28b30444dbf95637ecdd946b8`)
  - `Run_3_Low_Regularization_C10.0` (Run ID: `9e20c1aa03024d5ca849bb88a59c821f`)
- **Automated Pytest Pass Rate:** **9/9 tests passed (100%)** in 14.34 seconds.
- **Drift Test Verification:** $100\%$ precision in isolating shifted features (`sepal_length` and `petal_length`) while respecting non-rejection of the null hypothesis on control features (`sepal_width` and `petal_width`).
