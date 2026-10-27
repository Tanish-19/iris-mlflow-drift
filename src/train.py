"""
Model Training and MLflow Experiment Tracking Module.
Trains Logistic Regression classifiers with varied hyperparameters, evaluates performance,
generates evaluation artifacts, logs everything to a local persistent MLflow tracking store,
and exports an experiment comparison summary.
"""

import os
import json

# Enable local file store and suppress tracing agent hint in MLflow 3.x
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import mlflow
import mlflow.sklearn

from data_loader import load_iris_data, get_dataset_summary


EXPERIMENT_NAME = "Iris_Classification_Experiment"


def setup_mlflow(tracking_dir: str = "mlruns") -> str:
    """
    Configures a persistent local MLflow tracking store and sets the active experiment.
    """
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    
    abs_tracking_dir = os.path.abspath(tracking_dir)
    os.makedirs(abs_tracking_dir, exist_ok=True)
    tracking_uri = f"file://{abs_tracking_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=tracking_uri
        )
    else:
        experiment_id = experiment.experiment_id
        
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri


def train_single_run(
    run_name: str,
    hyperparams: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_names: List[str],
    plots_dir: str = "plots",
    artifacts_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Trains a Logistic Regression pipeline and logs parameters, metrics, artifacts,
    and the serialized model to MLflow.
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        
        # 1. Parameter specification
        model_name = "LogisticRegression"
        test_size = hyperparams.get("test_size", 0.2)
        random_state = hyperparams.get("random_state", 42)
        max_iter = hyperparams.get("max_iter", 200)
        c_param = hyperparams.get("C", 1.0)
        solver = hyperparams.get("solver", "lbfgs")
        
        params_to_log = {
            "model_name": model_name,
            "test_size": test_size,
            "random_state": random_state,
            "max_iterations": max_iter,
            "regularization_parameter_C": c_param,
            "solver": solver,
            "preprocessing": "StandardScaler"
        }
        mlflow.log_params(params_to_log)
        
        # 2. Pipeline definition (StandardScaler + LogisticRegression)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=c_param,
                max_iter=max_iter,
                solver=solver,
                random_state=random_state
            ))
        ])
        
        # 3. Model training
        pipeline.fit(X_train, y_train)
        
        # 4. Model inference
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)
        
        # 5. Metric computation
        acc = float(accuracy_score(y_test, y_pred))
        train_acc = float(accuracy_score(y_train, y_train_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        
        metrics_to_log = {
            "accuracy": round(acc, 4),
            "train_accuracy": round(train_acc, 4),
            "precision": round(prec_macro, 4),
            "recall": round(rec_macro, 4),
            "f1_score": round(f1_macro, 4),
            "f1_score_weighted": round(f1_weighted, 4)
        }
        mlflow.log_metrics(metrics_to_log)
        
        # 6. Artifact: Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=target_names,
            yticklabels=target_names,
            cbar=False,
            ax=ax
        )
        ax.set_title(f"Confusion Matrix: {run_name}\nAccuracy: {acc*100:.1f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Species", fontsize=10)
        ax.set_ylabel("True Species", fontsize=10)
        plt.tight_layout()
        
        cm_filename = f"confusion_matrix_{run_name.lower().replace(' ', '_')}.png"
        cm_path = os.path.join(plots_dir, cm_filename)
        fig.savefig(cm_path, dpi=200)
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        
        # 7. Artifact: Classification Report
        report_text = classification_report(y_test, y_pred, target_names=target_names)
        report_filename = f"classification_report_{run_name.lower().replace(' ', '_')}.txt"
        report_path = os.path.join(artifacts_dir, report_filename)
        with open(report_path, "w") as f:
            f.write(f"=== CLASSIFICATION REPORT FOR RUN: {run_name} ===\n\n")
            f.write(report_text)
        mlflow.log_artifact(report_path, artifact_path="evaluation_reports")
        
        # 8. Artifact: Dataset Metadata Information
        dataset_meta = get_dataset_summary()
        dataset_meta["train_samples"] = len(X_train)
        dataset_meta["test_samples"] = len(X_test)
        meta_filename = f"dataset_info_{run_name.lower().replace(' ', '_')}.json"
        meta_path = os.path.join(artifacts_dir, meta_filename)
        with open(meta_path, "w") as f:
            json.dump(dataset_meta, f, indent=4)
        mlflow.log_artifact(meta_path, artifact_path="metadata")
        
        # 9. Model serialization & logging to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="iris_model",
            registered_model_name=None
        )
        
        summary_record = {
            "run_id": run_id,
            "run_name": run_name,
            "regularization_C": c_param,
            "max_iterations": max_iter,
            "solver": solver,
            "accuracy": metrics_to_log["accuracy"],
            "precision": metrics_to_log["precision"],
            "recall": metrics_to_log["recall"],
            "f1_score": metrics_to_log["f1_score"],
            "train_accuracy": metrics_to_log["train_accuracy"]
        }
        
        return summary_record


def run_experiment_suite(
    reports_dir: str = "reports",
    plots_dir: str = "plots",
    artifacts_dir: str = "artifacts"
) -> pd.DataFrame:
    """
    Executes at least 3 distinct experiments with varied regularization and solver settings,
    logs all runs to MLflow, and writes an experiment summary CSV.
    """
    setup_mlflow()
    X, y, target_names = load_iris_data()
    
    # Stratified split to ensure balanced representation across species
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3 Distinct configurations to evaluate bias-variance trade-offs
    configurations = [
        {
            "run_name": "Run_1_Baseline_C1.0",
            "hyperparams": {
                "test_size": 0.20,
                "random_state": 42,
                "max_iter": 200,
                "C": 1.0,
                "solver": "lbfgs"
            }
        },
        {
            "run_name": "Run_2_High_Regularization_C0.1",
            "hyperparams": {
                "test_size": 0.20,
                "random_state": 42,
                "max_iter": 100,
                "C": 0.1,
                "solver": "lbfgs"
            }
        },
        {
            "run_name": "Run_3_Low_Regularization_C10.0",
            "hyperparams": {
                "test_size": 0.20,
                "random_state": 42,
                "max_iter": 300,
                "C": 10.0,
                "solver": "lbfgs"
            }
        }
    ]
    
    results = []
    print(f"Starting MLflow Experiment Suite: '{EXPERIMENT_NAME}'")
    for cfg in configurations:
        print(f"  --> Executing: {cfg['run_name']} (C={cfg['hyperparams']['C']}, max_iter={cfg['hyperparams']['max_iter']})")
        res = train_single_run(
            run_name=cfg["run_name"],
            hyperparams=cfg["hyperparams"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            target_names=target_names,
            plots_dir=plots_dir,
            artifacts_dir=artifacts_dir
        )
        results.append(res)
        
    summary_df = pd.DataFrame(results)
    
    os.makedirs(reports_dir, exist_ok=True)
    summary_csv_path = os.path.join(reports_dir, "experiment_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nExperiment Summary successfully saved to: {summary_csv_path}")
    print(summary_df.to_string(index=False))
    
    return summary_df


if __name__ == "__main__":
    run_experiment_suite()
