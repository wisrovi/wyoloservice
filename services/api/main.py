import json
import uuid
from fastapi import FastAPI, UploadFile, File
import os
import shutil
from api.database import db, TrainingHistory
from api.redis_queue import send_task_to_worker
from api.minio import upload_model, list_models, download_model

CONFIG_DIR = "/config_versions"
os.makedirs(CONFIG_DIR, exist_ok=True)

app = FastAPI()


@app.post("/train/")
def start_training(user_code: str, file: UploadFile = File(...)):
    """Registra un entrenamiento y lo encola en Redis."""

    task_id = str(uuid.uuid4()).replace("-", "").replace("_", "")

    config_path = os.path.join(CONFIG_DIR, f"{task_id}.yaml")
    with open(config_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        count = len(db.get_all())
    except Exception:
        count = 0

    # Insertar un nuevo usuario con el nuevo campo
    db.insert(
        TrainingHistory(
            id=count + 1,
            task_id=task_id,
            status="pending",
            user_code=user_code,
            config_path=config_path,
        )
    )

    send_task_to_worker(
        json.dumps(
            {
                "task_id": task_id,
                "config_path": config_path,
                "user_code": user_code,
            }
        )
    )
    return {"message": "Entrenamiento registrado", "task_id": task_id}


@app.get("/trainings/")
def get_trainings():
    """Consulta los entrenamientos registrados."""
    return db.get_all()


@app.get("/best_model/{experiment_name}")
def get_best_model(experiment_name: str):
    """Devuelve el mejor modelo del experimento."""

    versions = db.get_by_field(experiment_name=experiment_name)
    best_version = max(versions, key=lambda v: v.loss)
    if best_version:
        return {
            "experiment_name": experiment_name,
            "best_model": best_version.recommended_model,
            "metrics": {
                "loss": best_version.loss,
                "precision": best_version.precision,
                "recall": best_version.recall,
                "map50": best_version.map50,
                "map50_95": best_version.map50_95,
            },
        }
    return {"message": "No se encontró un mejor modelo"}


@app.get("/model_versions/{task_id}")
def get_model_versions(task_id: str):
    """Devuelve todas las versiones de un modelo almacenadas en MinIO."""

    versions = db.get_by_field(task_id=task_id)

    return [
        {"version": v.task_id.split("_v")[-1], "url": v.model_path} for v in versions
    ]


@app.get("/models/")
def list_all_models():
    """Lista todos los modelos almacenados en MinIO."""
    return list_models()


@app.get("/download_model/{model_name}")
def download_model_endpoint(model_name: str):
    """Descarga un modelo desde MinIO."""
    save_path = f"/tmp/{model_name}"
    download_model(model_name, save_path)
    return {"message": "Modelo descargado", "path": save_path}
