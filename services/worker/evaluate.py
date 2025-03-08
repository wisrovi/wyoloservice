import mlflow
import os
from ultralytics import YOLO

def evaluate_model(model_path, test_data):
    """Evalúa un modelo con nuevos datos y registra métricas en MLflow."""
    model = YOLO(model_path)
    results = model.val(data=test_data)

    mlflow.log_metrics({
        "precision": results.results["precision"],
        "recall": results.results["recall"],
        "map50": results.results["map50"],
        "map50_95": results.results["map50_95"]
    })

    return results.results
