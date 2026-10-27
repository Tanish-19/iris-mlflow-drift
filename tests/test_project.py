"""
Automated Pytest Suite for Iris ML Project.
Validates dataset integrity, model training, prediction generation,
metric computation, drift simulation, and Kolmogorov-Smirnov detection.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_loader import (
    load_iris_data,
    get_reference_dataframe,
    create_simulated_drift_data,
    save_datasets,
    get_dataset_summary
)
from drift_detection import (
    run_ks_drift_detection,
    execute_drift_analysis
)
from train import (
    setup_mlflow,
    train_single_run,
    run_experiment_suite
)


class TestDatasetLoading:
    """Tests for dataset loading and structure."""
    
    def test_load_iris_data_shape_and_columns(self):
        X, y, target_names = load_iris_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 150, f"Expected 150 records, got {len(X)}"
        assert len(y) == 150, f"Expected 150 target records, got {len(y)}"
        assert X.shape[1] == 4, f"Expected 4 features, got {X.shape[1]}"
        expected_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        assert list(X.columns) == expected_features
        assert len(target_names) == 3
        assert set(target_names) == {"setosa", "versicolor", "virginica"}

    def test_dataset_no_missing_values(self):
        X, y, _ = load_iris_data()
        assert X.isnull().sum().sum() == 0, "Iris features contain unexpected NaN values"
        assert y.isnull().sum() == 0, "Iris target contains unexpected NaN values"

    def test_dataset_summary_metadata(self):
        summary = get_dataset_summary()
        assert summary["total_records"] == 150
        assert summary["feature_count"] == 4
        assert summary["missing_values"] == 0
        assert summary["use_case"] == "Multiclass Machine Learning Classification"


class TestModelTrainingAndEvaluation:
    """Tests for model training, predictions, and evaluation metrics."""
    
    @pytest.fixture
    def dataset_split(self):
        from sklearn.model_selection import train_test_split
        X, y, target_names = load_iris_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test, target_names

    def test_model_training_and_predictions(self, dataset_split, tmp_path):
        X_train, X_test, y_train, y_test, target_names = dataset_split
        setup_mlflow(str(tmp_path / "mlruns"))
        
        hyperparams = {
            "test_size": 0.20,
            "random_state": 42,
            "max_iter": 100,
            "C": 1.0,
            "solver": "lbfgs"
        }
        
        result = train_single_run(
            run_name="Test_Pytest_Run",
            hyperparams=hyperparams,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            target_names=target_names,
            plots_dir=str(tmp_path / "plots"),
            artifacts_dir=str(tmp_path / "artifacts")
        )
        
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        
        # Validate metric bounds [0.0, 1.0]
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0
        assert 0.0 <= result["f1_score"] <= 1.0
        assert result["accuracy"] >= 0.85, "Iris model should achieve at least 85% accuracy"

    def test_confusion_matrix_artifact_generated(self, dataset_split, tmp_path):
        X_train, X_test, y_train, y_test, target_names = dataset_split
        setup_mlflow(str(tmp_path / "mlruns"))
        
        plots_dir = str(tmp_path / "plots")
        train_single_run(
            run_name="Test_Artifact_Run",
            hyperparams={"C": 1.0, "max_iter": 100, "random_state": 42},
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            target_names=target_names,
            plots_dir=plots_dir,
            artifacts_dir=str(tmp_path / "artifacts")
        )
        
        expected_plot = os.path.join(plots_dir, "confusion_matrix_test_artifact_run.png")
        assert os.path.exists(expected_plot), f"Artifact {expected_plot} was not created"


class TestDriftDetection:
    """Tests for drift simulation and Kolmogorov-Smirnov detection."""
    
    def test_drift_simulation_creates_controlled_shift(self):
        ref_df = get_reference_dataframe()
        curr_df = create_simulated_drift_data(ref_df, random_state=42)
        
        assert len(curr_df) == len(ref_df)
        assert set(curr_df.columns) == set(ref_df.columns)
        
        # Verify Sepal Length and Petal Length mean increased by design
        assert curr_df["sepal_length"].mean() > ref_df["sepal_length"].mean() + 0.5
        assert curr_df["petal_length"].mean() > ref_df["petal_length"].mean() + 1.0

    def test_ks_drift_detection_results_format(self):
        ref_df = get_reference_dataframe()
        curr_df = create_simulated_drift_data(ref_df, random_state=42)
        
        drift_df = run_ks_drift_detection(ref_df, curr_df, alpha=0.05)
        
        assert isinstance(drift_df, pd.DataFrame)
        assert len(drift_df) == 4, "Expected drift results for all 4 features"
        
        expected_cols = [
            "feature", "ks_statistic", "p_value", "alpha_threshold",
            "drift_detected", "drift_status", "ref_mean", "curr_mean", "mean_shift"
        ]
        for col in expected_cols:
            assert col in drift_df.columns, f"Missing column: {col}"
            
        # P-values must be within [0, 1]
        for p in drift_df["p_value"]:
            assert 0.0 <= float(p) <= 1.0

    def test_ks_drift_detection_identifies_shifted_features(self):
        ref_df = get_reference_dataframe()
        curr_df = create_simulated_drift_data(ref_df, random_state=42)
        
        drift_df = run_ks_drift_detection(ref_df, curr_df, alpha=0.05)
        
        results_by_feature = dict(zip(drift_df["feature"], drift_df["drift_detected"]))
        
        # The two intentionally drifted features must be detected
        assert results_by_feature["sepal_length"] is True, "sepal_length drift was not detected"
        assert results_by_feature["petal_length"] is True, "petal_length drift was not detected"
        
        # The two unshifted features must NOT be detected as drifted
        assert results_by_feature["sepal_width"] is False, "sepal_width should not show significant drift"
        assert results_by_feature["petal_width"] is False, "petal_width should not show significant drift"


class TestFullPipelineIntegration:
    """End-to-end integration test for execution."""
    
    def test_dataset_saving_and_drift_execution(self, tmp_path):
        ref_p, curr_p = save_datasets(data_dir=str(tmp_path / "data"))
        assert os.path.exists(ref_p)
        assert os.path.exists(curr_p)
        
        drift_df, csv_p, md_p, p1, p2 = execute_drift_analysis(
            reference_path=ref_p,
            current_path=curr_p,
            reports_dir=str(tmp_path / "reports"),
            plots_dir=str(tmp_path / "plots")
        )
        assert os.path.exists(csv_p)
        assert os.path.exists(md_p)
        assert os.path.exists(p1)
        assert os.path.exists(p2)
