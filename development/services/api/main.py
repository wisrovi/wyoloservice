import os
import shutil
import uuid

import yaml
from app.database import TrainingHistory, db

# from api.minio import download_model, list_models

from fastapi import FastAPI, File, UploadFile
from wredis.queue import RedisQueueManager

CONFIG_DIR = "/config_versions"
os.makedirs(CONFIG_DIR, exist_ok=True)


config_path = "/code/app/config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)


redis_config = cfg.get("redis", {})
queue_manager = RedisQueueManager(
    host="redis",
    port=6379,
    db=redis_config.get("REDIS_DB"),
)


app = FastAPI()


@app.post("/recreate/")
def recreate(
    user_code: str,
    task_id: str,
    last_model: str,
    file: UploadFile = File(...),
    resume: bool = True,
    destinity: str = "recreate",
):
    config_path = os.path.join(CONFIG_DIR, f"{task_id}.yaml")
    with open(config_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    # validar si resume esta presente
    user_config["train"]["resume"] = True

    user_config["request_by"] = user_code
    user_config["task_id"] = task_id

    # esto fuerza el sistema a que use el modelo entregado y reportado como last_model
    user_config["sweeper"]["model"] = ["choice", last_model]
    user_config["model"] = last_model
    user_config["recreate"] = True

    # actualizar el config_path en el archivo de configuracion original
    with open(config_path, "w") as f:
        yaml.dump(user_config, f)

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

    try:
        count = len(db.get_all())
    except Exception:
        count = 0

    queue_manager.publish(
        queue_name=destinity,
        data={
            "task_id": task_id,
            "config_path": config_path,
            "user_code": user_code,
            "db_count": count + 1,
        },
    )

    return {"message": "Entrenamiento registrado", "task_id": task_id}


@app.post("/evaluate/")
def evaluate(user_code: str, data_yaml: str, model_path: str, topic: str = "evaluate"):
    """Ejecuta una evaluación de un modelo en un conjunto de datos.

    Args:
        user_code (str): Código del usuario que solicita la evaluación.
        data_yaml (str): Ruta al archivo YAML que contiene la configuración de los datos.
        model_path (str): Ruta al modelo a evaluar.
        topic (str): Nombre del tema en Redis donde se publicará la tarea. Por defecto es "evaluate".

    Returns:
        dict: Mensaje de confirmación y el ID de la tarea.
    """

    task_id = str(uuid.uuid4()).replace("-", "").replace("_", "")

    queue_manager.publish(
        queue_name=topic,
        data={
            "task_id": task_id,
            "user_code": user_code,
            "data_yaml": data_yaml,
            "model_path": model_path,
        },
    )
    return {"message": "Evaluación registrada", "task_id": task_id}


@app.post("/stop/")
def stop_training(user_code: str, file: UploadFile = File(...)):
    """Detiene un entrenamiento en curso."""

    # Detener el entrenamiento en curso
    start_training(user_code=user_code, file=file, resume=False)


@app.post("/train/")
def start_training(user_code: str, file: UploadFile = File(...), resume: bool = False):
    """Registra un entrenamiento y lo encola en Redis."""

    task_id = str(uuid.uuid4()).replace("-", "").replace("_", "")

    config_path = os.path.join(CONFIG_DIR, f"{task_id}.yaml")
    with open(config_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    # validar si resume esta presente
    if resume:
        user_config["train"]["resume"] = True

    user_config["request_by"] = user_code
    user_config["task_id"] = task_id

    # actualizar el config_path en el archivo de configuracion original
    with open(config_path, "w") as f:
        yaml.dump(user_config, f)

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

    queue_topic = user_config.get("debug", redis_config.get("TOPIC"))

    queue_manager.publish(
        queue_name=queue_topic,
        data={
            "task_id": task_id,
            "config_path": config_path,
            "user_code": user_code,
            "db_count": count + 1,
        },
    )

    return {"message": "Entrenamiento registrado", "task_id": task_id}


# @app.get("/trainings/")
# def get_trainings():
#     """Consulta los entrenamientos registrados."""
#     return db.get_all()


# @app.get("/best_model/{experiment_name}")
# def get_best_model(experiment_name: str):
#     """Devuelve el mejor modelo del experimento."""

#     versions = db.get_by_field(experiment_name=experiment_name)
#     best_version = max(versions, key=lambda v: v.loss)
#     if best_version:
#         return {
#             "experiment_name": experiment_name,
#             "best_model": best_version.recommended_model,
#             "metrics": {
#                 "loss": best_version.loss,
#                 "precision": best_version.precision,
#                 "recall": best_version.recall,
#                 "map50": best_version.map50,
#                 "map50_95": best_version.map50_95,
#             },
#         }
#     return {"message": "No se encontró un mejor modelo"}


# @app.get("/model_versions/{task_id}")
# def get_model_versions(task_id: str):
#     """Devuelve todas las versiones de un modelo almacenadas en MinIO."""

#     versions = db.get_by_field(task_id=task_id)

#     return [
#         {"version": v.task_id.split("_v")[-1], "url": v.model_path} for v in versions
#     ]


# @app.get("/models/")
# def list_all_models():
#     """Lista todos los modelos almacenados en MinIO."""
#     return list_models()


# @app.get("/download_model/{model_name}")
# def download_model_endpoint(model_name: str):
#     """Descarga un modelo desde MinIO."""
#     save_path = f"/tmp/{model_name}"
#     download_model(model_name, save_path)
#     return {"message": "Modelo descargado", "path": save_path}
