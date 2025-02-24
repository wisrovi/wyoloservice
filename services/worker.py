import os
import json
import yaml
import hydra
import mlflow
import optuna
import redis
import boto3
from loguru import logger
from glob import glob
from tqdm import tqdm
from ultralytics import YOLO, RTDETR
from api.database import db, TrainingHistory
from api.redis_queue import queue_manager, TOPIC

# Configuración de MinIO y Redis
MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MINIO_ENDPOINT = "http://minio:9000"
BUCKET_NAME = "models"

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
)
redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

# Configuración base por defecto
DEFAULT_CONFIG = {
    "experiment_name": "default_experiment",
    "model": "yolov8n.pt",
    "train": {"batch_size": 32, "epochs": 10},
    "sweeper": {"n_trials": 50, "sampler": "random"},
}


def upload_to_minio(task_id, model_path, version):
    """Guarda múltiples versiones del modelo en MinIO y devuelve la URL."""
    model_filename = f"{task_id}_v{version}.pt"
    s3.upload_file(model_path, BUCKET_NAME, model_filename)
    return f"{MINIO_ENDPOINT}/{BUCKET_NAME}/{model_filename}"


def save_best_model(task_id, model_path, version):
    """Registra el mejor modelo en la base de datos y lo sube a MinIO."""
    minio_url = upload_to_minio(task_id, model_path, version)

    # Verificar si el `task_id` existe en la base de datos
    existing_task = db.get_by_field(task_id=task_id)
    if existing_task is None:
        logger.warning(
            f"⚠️ Advertencia: No se encontró task_id {task_id} en la base de datos."
        )
        return

    db.update(
        1,
        TrainingHistory(
            id=existing_task,
            task_id=f"{task_id}_v{version}",
            model_path=minio_url,
            status="completed",
            recommended_model=minio_url,
        ),
    )
    logger.info(f"✅ Modelo recomendado guardado en MinIO: {minio_url}")


def obtener_carpeta(ruta):
    # Normalizar la ruta
    ruta_normalizada = os.path.normpath(ruta)

    # Si la ruta normalizada no es un directorio, extraer el directorio
    if not os.path.isdir(ruta_normalizada):
        return os.path.dirname(ruta_normalizada)
    return ruta_normalizada


def train_model(request_config, task_id):
    """Entrena YOLOv8 con búsqueda de hiperparámetros y guarda el mejor modelo."""

    def objective(trial):
        logger.info(f"🔍 Iniciando prueba {trial.number}...")

        request_config["train"]["epochs"] = trial.suggest_int(
            "train.epochs", 10, 100, step=10
        )
        request_config["train"]["batch"] = trial.suggest_categorical(
            "train.batch", [8, 16, 32, 64]
        )
        request_config["train"]["imgsz"] = trial.suggest_categorical(
            "train.imgsz", [320, 416, 512, 640]
        )

        experiment_name = request_config["experiment_name"]
        model_name = request_config["model"]

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"{experiment_name}_trial_{trial.number}"):
            mlflow.log_params(request_config["train"])
            mlflow.set_tag("mlflow.note.content", request_config.get("content", "NA"))

            logger.info(
                f"🚀 Iniciando entrenamiento con la configuración: {request_config}"
            )

            if request_config["type"] == "yolo":
                model = YOLO(model_name)
            elif request_config["type"] == "rtdetr":
                model = RTDETR(model_name)
            else:
                Exception("Is not a valid model")

            if "train" in request_config:
                results = model.train(**request_config["train"])

            if "val" in request_config:
                validation_results = model.val(**request_config["val"])

                mlflow.set_tag("experiment_type", str(results.task))

                val_results = {
                    # "task": str(results.task),
                    "precision_b": round(
                        results.results_dict["metrics/precision(B)"], 2
                    ),
                    "recall_b": round(results.results_dict["metrics/recall(B)"], 2),
                    "mAP50_b": round(results.results_dict["metrics/mAP50(B)"], 2),
                    "mAP95_b": round(results.results_dict["metrics/mAP50-95(B)"], 2),
                    "f1": round(results.box.f1[0], 2),
                    "precision": round(
                        validation_results.results_dict["metrics/precision(B)"], 2
                    ),
                    "recall": round(
                        validation_results.results_dict["metrics/recall(B)"], 2
                    ),
                    # "loss": loss_callback.losses[-1],
                }

                mlflow.log_metrics(val_results)

            if "test" in request_config:
                config_test = request_config["test"]

                path_dataset = obtener_carpeta(request_config["train"]["data"])
                PATH = f"{path_dataset}/test"

                for image_path in tqdm(
                    glob(f"{PATH}/*/*"), desc="Testing", unit="Images"
                ):
                    try:
                        model.predict(**config_test, source=image_path)
                    except:
                        pass

                config_test["source"] = PATH
                config_test["name"] = "test_metrics"
                model.val(**config_test)

            model_path = f"/models/{experiment_name}_trial_{trial.number}.pt"
            model.save(model_path)

            mlflow.log_artifact(model_path)
            mlflow.log_artifacts(PATH)

    logger.debug(
        f"Creando estudio de optimización con {request_config['sweeper']['n_trials']} pruebas..."
    )
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=request_config["sweeper"]["n_trials"])

    best_trial = study.best_trial
    best_model_path = (
        f"/models/{request_config['experiment_name']}_trial_{best_trial.number}.pt"
    )
    save_best_model(task_id, best_model_path, best_trial.number)

    logger.info(f"🏆 Mejor modelo guardado en MinIO: {best_model_path}")


@hydra.main(config_path="/config_versions", config_name="config", version_base=None)
def main(_):
    logger.info("✅ Worker iniciado. Esperando trabajos en la cola de Redis...")

    @queue_manager.on_message(TOPIC)
    def worker(task_data: dict):
        task_data = json.loads(task_data)
        logger.debug(f"📥 Nueva tarea recibida: {task_data}")

        if "task_id" not in task_data or "config_path" not in task_data:
            logger.error(
                f"⚠️ La tarea recibida no tiene la estructura esperada: {task_data}"
            )
            return

        # Leer el archivo YAML del usuario
        config_path = task_data["config_path"]
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)  # Convertir YAML a dict

                # 🚨 Eliminar `defaults` si existe
                user_config.pop("defaults", None)

                # 🚨 Fusionar con la configuración base
                final_config = DEFAULT_CONFIG.copy()
                final_config.update(user_config)  # 📌 Fusión con `update()`

        except Exception as e:
            logger.error(f"❌ Error al cargar YAML ({config_path}): {e}")

        train_model(final_config, task_data["task_id"])

    queue_manager.start()
    queue_manager.wait()


if __name__ == "__main__":
    main()
