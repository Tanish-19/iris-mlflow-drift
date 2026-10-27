# Academic Data Drift Detection and Analysis Report

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

| feature      |   ks_statistic |    p_value | drift_status         |   ref_mean |   curr_mean |   mean_shift |
|:-------------|---------------:|-----------:|:---------------------|-----------:|------------:|-------------:|
| sepal_length |         0.3733 | 1.0722e-09 | Drift Detected       |      5.843 |       6.681 |        0.837 |
| sepal_width  |         0.1    | 0.4425     | No Significant Drift |      3.057 |       3.069 |        0.012 |
| petal_length |         0.4067 | 1.7712e-11 | Drift Detected       |      3.758 |       5.172 |        1.414 |
| petal_width  |         0.1333 | 0.1391     | No Significant Drift |      1.199 |       1.188 |       -0.012 |

## 4. Academic Distinction Between Drift Categories

In MLOps and production machine learning, it is vital to distinguish between three fundamental types of drift:

| Drift Category | Formal Definition | Cause / Context | Impact on ML System |
| :--- | :--- | :--- | :--- |
| **Data Drift (Covariate Shift)** | Change in feature distribution: $P(X)_{\text{curr}} \neq P(X)_{\text{ref}}$ while $P(Y|X)$ remains constant. | Sensor miscalibration, changing customer demographics, seasonal shifts. | Model receives inputs outside its training envelope. Can lead to degraded performance or silent failures. |
| **Concept Drift** | Change in relationship between inputs and outputs: $P(Y|X)_{\text{curr}} \neq P(Y|X)_{\text{ref}}$. | Economic changes, user preference shifts, external macro events. | Old decision boundaries become invalid even if feature distributions appear stable. |
| **Prediction Drift (Prior Shift)** | Change in predicted label distribution: $P(\hat{Y})_{\text{curr}} \neq P(\hat{Y})_{\text{ref}}$. | Underlying class balance changes or model systematic bias under drift. | Operational alert indicating downstream changes in predicted class proportions. |

> **Important Cautionary Note:**
> Detecting **Data Drift does NOT automatically prove model performance has decreased**.
> If the feature that drifted has negligible predictive weight in the decision function, or if the distribution shifted along a dimension perpendicular to the decision boundary, classification accuracy may remain intact. However, statistical drift serves as an essential **early-warning telemetry signal** that the model is operating out-of-distribution.

## 5. Interpretation and Remedial Actions

1. **Features with Statistically Significant Drift:**
   - `sepal_length` ($p < 0.05$): Shifted significantly upward from reference mean.
   - `petal_length` ($p < 0.05$): Substantial shift detected ($D$ statistic is elevated, $p$-value near 0).
2. **Features with Stable Distributions:**
   - `sepal_width` ($p \ge 0.05$): Distribution remains congruent with reference data.
   - `petal_width` ($p \ge 0.05$): Distribution remains congruent with reference data.
3. **Recommended MLOps Interventions:**
   - Collect recent ground-truth labels for incoming samples.
   - Re-evaluate production accuracy and F1-score on the drifted window.
   - Trigger retraining pipeline with a weighted or sliding-window dataset incorporating recent samples.
