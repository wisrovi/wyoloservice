import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
artifacts = mlflow.artifacts.list_artifacts("s3://mlflow-artifacts/")
print(artifacts)