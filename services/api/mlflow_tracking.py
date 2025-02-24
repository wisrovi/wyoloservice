import mlflow

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def log_training_run(config, metrics):
    """Registra una ejecución en MLflow."""
    with mlflow.start_run():
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
