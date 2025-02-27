import json
import math
import os
import signal
import sys
import tempfile
import traceback
import tracemalloc
from datetime import datetime
from functools import wraps

import hydra
import mlflow
import optuna
import torch
import yaml
from logbook import FileHandler, Logger
from loguru import logger

from api.database import TrainingHistory, db
from api.redis_queue import TOPIC, queue_manager
from worker_utils import (
    initialize_minio_client,
    train_yolo,
    catch_errors,
    clean_gpu,
    load_train_config,
    MinioS3Client,
)


# Configura el primer logger: solo errores en un archivo
logger.add("error_log.log", level="ERROR", rotation="10 MB", retention="7 days")

# Configura el segundo logger: mensajes normales en la consola
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
)

# Configura el tercer logger: todos los niveles en un archivo separado (completamente independiente)
FileHandler("full_log.log").push_application()
custom_logger = Logger("wyoloservice")


s3: MinioS3Client = initialize_minio_client()
# Configuración base por defecto
DEFAULT_CONFIG = {
    "experiment_name": "default_experiment",
    "model": "yolov8n.pt",
    "train": {"batch_size": 32, "epochs": 10},
    "sweeper": {"n_trials": 1, "sampler": "random"},
}


def save_best_model(
    task_id,
    model_path,
    version,
    results_model,
    project_name,
    data_path_for_test: str = None,
):
    """Registra el mejor modelo en la base de datos y lo sube a MinIO."""

    update_model = True

    # TODO: falta validar el modelo anterior y compararlo con el modelo actual, si es mejor, se reemplaza, de lo contrario solo se suben los resultados del modelo actual

    # if bucket_exists(bucket_name):
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         old_files = list_objects(bucket_name)

    #         temp_file_path = os.path.join(temp_dir, "old.pt")

    #         if data_path and old_files and f"{project_name}.pt" in old_files:
    #             old_model = download_file(
    #                 bucket_name, f"{project_name}.pt", temp_file_path
    #             )

    #             better_model = comparar_modelos_yolo(
    #                 modelo1_path=old_model,
    #                 modelo2_path=model_path,
    #                 data_path=data_path,
    #             )

    #             if better_model == "old.pt":
    #                 update_model = False

    if update_model:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(f"{temp_dir}/{project_name}-v{version}.txt", "w") as f:
                f.write(f"{project_name}/{task_id}/*")

            s3.upload_file(
                f"{temp_dir}/{project_name}-v{version}.txt",
                f"better-{BUCKET_NAME}",
                f"{project_name}-v{version}.txt",
            )
            
            minio_url = s3.upload_file(
                model_path,
                f"better-{BUCKET_NAME}",
                f"{project_name}-v{version}.pt",
            )
            
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
                    

        s3.upload_file(
            model_path,
            BUCKET_NAME,
            f"{project_name}/{task_id}/model.pt",
        )

    s3.upload_folder(
        
        folder_path=results_model,
        bucket_name=f"{BUCKET_NAME}",
        prefix=f"{project_name}/{task_id}/",
    )
    

    # Verificar si el `task_id` existe en la base de datos
    existing_task = db.get_by_field(task_id=task_id)
    if existing_task is None:
        logger.warning(
            f"⚠️ Advertencia: No se encontró task_id {task_id} en la base de datos."
        )
        return

    
    logger.info(f"✅ Modelo recomendado guardado en MinIO: {minio_url}")




def process_train(request_config_user, task_id):
    sweeper_config = request_config_user.get("sweeper", {})
    
    @clean_gpu
    @load_train_config(config_path=request_config_user.get("config_path"))
    def objective(trial, config=None):
        
        config = train_yolo(config, trial_number=trial.number)
        
        if math.isnan(config):
            raise optuna.TrialPruned("Rendimiento insuficiente.")

        metric = config.get("metric", float("nan"))
        
        return metric
    
    study = optuna.create_study(
        direction=sweeper_config.get("direction", "minimize"),
        study_name=sweeper_config.get("study_name", "default_study"),
        sampler=getattr(optuna.samplers, sweeper_config.get("sampler", "TPESampler"))(),
    )
    study.optimize(objective, n_trials=sweeper_config.get("n_trials", 10))
    
    done = True
    try:
        best_trial = study.best_trial
    except:
        done = False
        
    if done:
        RESULT_PATH = f'/models/{sweeper_config.get("study_name", "default_study")}/{request_config_user["type"]}/{request_config_user["task_id"]}'
        best_model_path = (
            f"{RESULT_PATH}/trail_history/trial_{best_trial.number}.pt"
        )

        save_best_model(
            task_id=task_id,
            project_name=sweeper_config.get("study_name", "default_study"),
            model_path=best_model_path,
            version=sweeper_config["version"],
            results_model=f"{RESULT_PATH}/{best_trial.number}/",
            data_path_for_test=request_config_user["train"]["data"],
            )
        
    


@queue_manager.on_message(TOPIC)    
def worker(task_data: dict):
    task_data = json.loads(task_data)
    
    logger.debug(f"📥 Nueva tarea recibida: {task_data}")
    if "task_id" not in task_data or "config_path" not in task_data:
        logger.error(
            f"⚠️ La tarea recibida no tiene la estructura esperada: {task_data}"
        )
        return
    
    config_path = task_data["config_path"]
    
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

    final_config["config_path"] = config_path
    final_config["task_id"] = task_data["task_id"]
    

    try:
        with open(config_path, "w") as archivo:
            yaml.dump(final_config, archivo, default_flow_style=False)
    except Exception as e:
        pass
    
    process_train(final_config, task_data["task_id"])




@hydra.main(config_path="/app", config_name="config", version_base=None)
def main(cfg):
    mlflow.set_tracking_uri(cfg.mlflow.MLFLOW_TRACKING_URI)
    logger.info(f"MLflow URI: {cfg.mlflow.MLFLOW_TRACKING_URI}")
    
    
    
    
    queue_manager.start()
    queue_manager.wait()



main()







exit()


# Configuración de MinIO y Redis
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MINIO_ENDPOINT = "http://minio:9000"
BUCKET_NAME = "models"
MINIO_ID = "minio"
MINIO_KEY = "minio123"
REDIS_HOST = "redis"
REDIS_PORT = 6379
threshold = 0.02






def obtener_carpeta(ruta):
    # Normalizar la ruta
    ruta_normalizada = os.path.normpath(ruta)

    # Si la ruta normalizada no es un directorio, extraer el directorio
    if not os.path.isdir(ruta_normalizada):
        return os.path.dirname(ruta_normalizada)
    return ruta_normalizada


def get_datetime():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")





def train_model(request_config_user, task_id):
    """Entrena YOLOv8 con búsqueda de hiperparámetros y guarda el mejor modelo."""

    # config use of GPU
    if False:
        if (
            torch.cuda.is_available()
            and "extras" in request_config_user
            and "gpu" in request_config_user["extras"]
            and "limit" in request_config_user["extras"]["gpu"]
            and 0.3 < float(request_config_user["extras"]["gpu"]["limit"]) < 1
        ):
            memory_fraction = int(request_config_user["extras"]["gpu"]["id"])
            gpu_id = float(request_config_user["extras"]["gpu"]["limit"])
            try:
                torch.cuda.set_per_process_memory_fraction(memory_fraction, gpu_id)
            except Exception as e:
                custom_logger.error("No se pudo limitar la GPU")

    sweeper_config = request_config_user.get("sweeper", {})

    @clean_gpu
    @load_train_config(config_path=request_config_user.get("config_path"))
    def objective(trial, config=None):
        
        config = train_yolo(config, trial_number=trial.number)
        
        metric = config.get("metric", float("nan"))
        
        return metric
        
        
        
        # try:
        #     logger.info(f"🔍 Iniciando prueba {trial.number}...")

        #     # Configuración de MLflow
        #     experiment_name = sweeper_config["study_name"]
        #     # model_name = config["model"]
        #     mlflow.set_experiment(experiment_name)

        #     with mlflow.start_run(
        #         run_name=f"{experiment_name}_trial_{trial.number}", nested=True
        #     ):
        #         val_results = {"mAP95_b": 1e10}

        #         mlflow.log_params(config["train"])
        #         mlflow.set_tag("mlflow.note.content", config.get("content", "NA"))

        #         logger.info(
        #             f"🚀 Iniciando entrenamiento con la configuración: {config}"
        #         )

                
        #         for item, value in val_results.items():
        #             mlflow.log_metric(item, value)

        #         if isinstance(config, dict) and "experiment_type" in config:
        #             mlflow.set_tag("experiment_type", config.get("experiment_type"))

        #         # Guardar modelo y artefactos
        #         if isinstance(config, dict) and "model_path" in config:
        #             mlflow.log_artifact(config["model_path"])

        #         if (
        #             isinstance(config, dict)
        #             and "path_results" in config
        #             and os.path.exists(config["path_results"])
        #         ):
        #             mlflow.log_artifacts(config["path_results"])

        #         metric = None
        #         if isinstance(config, dict) and "metric" in config:
        #             metric = config.get("metric", float("nan"))

        #         if metric is None or metric < threshold:
        #             raise optuna.TrialPruned("Rendimiento insuficiente.")

        #         return metric
        # except ValueError as ve:
        #     logger.error(f"Error al seleccionar el modelo: {ve}")
        #     mlflow.log_param("error", str(ve))
        #     return None
        # except Exception as e:

        #     traceback.print_exc()
        #     logger.error(f"Error inesperado: {e}")

        #     mlflow.log_param("error", str(e))
        #     return None
        # finally:
        #     mlflow.end_run()

    logger.debug(
        f"Creando estudio de optimización con {request_config_user['sweeper']['n_trials']} pruebas..."
    )
    study = optuna.create_study(
        direction=sweeper_config.get("direction", "minimize"),
        study_name=sweeper_config.get("study_name", "default_study"),
        sampler=getattr(optuna.samplers, sweeper_config.get("sampler", "TPESampler"))(),
    )
    study.optimize(objective, n_trials=sweeper_config.get("n_trials", 10))

    done = True
    try:
        best_trial = study.best_trial
    except:
        done = False

    try:
        if done:
            RESULT_PATH = f'/models/{sweeper_config.get("study_name", "default_study")}/{request_config_user["type"]}/{request_config_user["task_id"]}'
            best_model_path = (
                f"{RESULT_PATH}/trail_history/trial_{best_trial.number}.pt"
            )

            save_best_model(
                task_id=task_id,
                project_name=sweeper_config.get("study_name", "default_study"),
                model_path=best_model_path,
                version=sweeper_config["version"],
                results_model=f"{RESULT_PATH}/{best_trial.number}/",
                data_path_for_test=request_config_user["train"]["data"],
            )

            custom_logger.info(f"🏆 Mejor modelo guardado en MinIO: {best_model_path}")
        else:
            custom_logger.info(
                "🏆 No se logro entrenar un modelo valido con los parametros estipulados"
            )
    except Exception as e:
        traceback.print_exc()
        logger.error(str(e))


@catch_errors
@hydra.main(config_path="/config_versions", config_name="config", version_base=None)
def main(_):
    logger.info("✅ Worker iniciado. Esperando trabajos en la cola de Redis...")
    custom_logger.info("✅ Worker iniciado. Esperando trabajos en la cola de Redis...")

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

        final_config["config_path"] = config_path
        final_config["task_id"] = task_data["task_id"]

        try:
            with open(config_path, "w") as archivo:
                yaml.dump(final_config, archivo, default_flow_style=False)
        except Exception as e:
            pass

        custom_logger.info(
            f"🚨 Nueva solicitud de entrenamiento recibida: {config_path} -> {final_config}"
        )
        train_model(final_config, task_data["task_id"])

    queue_manager.start()
    queue_manager.wait()


if __name__ == "__main__":

    def handle_signal(signum, frame):
        print(f"Señal {signum} recibida. Finalizando...")
        sys.exit(0)

    try:
        tracemalloc.start()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)  # Captura Ctrl+C

        main()
    except Exception as e:
        print(f"Error inesperado: {e}")
        raise
    finally:
        # Finaliza el seguimiento de memoria
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        print("[ Top 10 memory usage ]")
        for stat in top_stats[:10]:
            print(stat)

        logger.error("Programa finalizado")
