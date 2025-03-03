import torch
from ultralytics import YOLO

# Cargar el modelo PyTorch desde MLflow
logged_model = 'runs:/<run_id>/model'
loaded_pytorch_model = mlflow.pytorch.load_model(logged_model)

# Reconstruir el modelo YOLO
reconstructed_yolo_model = YOLO(loaded_pytorch_model)

# Usar el modelo reconstruido
results = reconstructed_yolo_model("imagen.jpg")