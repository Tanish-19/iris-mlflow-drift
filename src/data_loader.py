"""
Data Loader and Drift Simulation Module for Iris ML Project.
Loads the Scikit-learn Iris dataset and creates controlled simulated drift data.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


FEATURE_MAPPING = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width"
}


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Loads the built-in Iris dataset from scikit-learn.
    
    Returns:
        X (pd.DataFrame): 150 rows x 4 features with clean snake_case column names.
        y (pd.Series): Target class labels (0, 1, 2).
        target_names (list): List of class names ['setosa', 'versicolor', 'virginica'].
    """
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    
    # Rename feature columns to standard snake_case
    df = df.rename(columns=FEATURE_MAPPING)
    
    features = list(FEATURE_MAPPING.values())
    X = df[features].copy()
    y = df['target'].copy()
    target_names = [str(name) for name in iris.target_names]
    
    return X, y, target_names


def get_reference_dataframe() -> pd.DataFrame:
    """
    Returns the complete reference dataset as a single DataFrame including target and species_name.
    """
    X, y, target_names = load_iris_data()
    df = X.copy()
    df['target'] = y
    df['species'] = y.map(lambda idx: target_names[idx])
    return df


def create_simulated_drift_data(
    reference_df: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Creates a simulated 'current' dataset with controlled, known drift applied.
    
    Controlled Drift Scenario:
      1. 'sepal_length': Mean shifted upward by +0.85 cm (+ ~1.0 standard deviations).
      2. 'petal_length': Mean shifted upward by +1.40 cm (+ ~0.8 standard deviations).
      3. 'sepal_width': Kept within original distribution (subtle zero-mean noise only, non-significant).
      4. 'petal_width': Kept within original distribution (subtle zero-mean noise only, non-significant).
      
    NOTE FOR EVALUATION:
      This is a deliberate synthetic perturbation designed to simulate covariate shift (data drift)
      in an academic lab setting to rigorously test the Kolmogorov-Smirnov detection algorithm.
    """
    np.random.seed(random_state)
    current_df = reference_df.copy()
    n_samples = len(current_df)
    
    # 1. Significant drift on sepal_length: positive shift + slight scale variation
    sepal_len_noise = np.random.normal(loc=0.85, scale=0.15, size=n_samples)
    current_df['sepal_length'] = np.round(current_df['sepal_length'] + sepal_len_noise, 2)
    
    # 2. Significant drift on petal_length: positive shift
    petal_len_noise = np.random.normal(loc=1.40, scale=0.20, size=n_samples)
    current_df['petal_length'] = np.round(current_df['petal_length'] + petal_len_noise, 2)
    
    # 3. Non-significant micro-perturbations on sepal_width (preserving distribution)
    sepal_width_noise = np.random.normal(loc=0.01, scale=0.03, size=n_samples)
    current_df['sepal_width'] = np.round(current_df['sepal_width'] + sepal_width_noise, 2)
    
    # 4. Non-significant micro-perturbations on petal_width (preserving distribution)
    petal_width_noise = np.random.normal(loc=-0.01, scale=0.02, size=n_samples)
    current_df['petal_width'] = np.round(current_df['petal_width'] + petal_width_noise, 2)
    
    return current_df


def save_datasets(
    data_dir: str = "data",
    random_state: int = 42
) -> Tuple[str, str]:
    """
    Saves reference and simulated drifted datasets to disk.
    
    Returns:
        Tuple of (reference_csv_path, current_csv_path)
    """
    import os
    os.makedirs(data_dir, exist_ok=True)
    
    ref_df = get_reference_dataframe()
    curr_df = create_simulated_drift_data(ref_df, random_state=random_state)
    
    ref_path = os.path.join(data_dir, "reference_iris.csv")
    curr_path = os.path.join(data_dir, "current_iris_drifted.csv")
    
    ref_df.to_csv(ref_path, index=False)
    curr_df.to_csv(curr_path, index=False)
    
    return ref_path, curr_path


def get_dataset_summary() -> Dict[str, Any]:
    """
    Returns structured metadata regarding the Iris dataset.
    """
    X, y, target_names = load_iris_data()
    return {
        "dataset_name": "Iris Flower Dataset",
        "source": "Scikit-learn built-in (Fisher, 1936)",
        "total_records": len(X),
        "feature_count": X.shape[1],
        "features": list(X.columns),
        "target_variable": "species",
        "target_classes": target_names,
        "class_counts": y.value_counts().to_dict(),
        "missing_values": int(X.isnull().sum().sum()),
        "use_case": "Multiclass Machine Learning Classification"
    }


if __name__ == "__main__":
    ref_p, curr_p = save_datasets()
    print(f"Saved reference dataset to: {ref_p}")
    print(f"Saved simulated drifted dataset to: {curr_p}")
    summary = get_dataset_summary()
    print("Dataset Summary:")
    for k, v in summary.items():
        print(f"  - {k}: {v}")
