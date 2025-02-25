import os
from loguru import logger
import torch
from ultralytics import RTDETR, YOLO
from datetime import datetime
from glob import glob
from tqdm import tqdm
from ultralytics import settings

# Update a setting
settings.update({"mlflow": True})

# Reset settings to default values
# settings.reset()


#decortador para capturar excepciones
def handle_exception(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            return {"Error": str(e)}
        
    return wrapper
            

# decorador para medir tiempo
def measure_time(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()

        result["time"] = end - start

        return result

    return wrapper


def clean_results(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            return float("nan")
        finally:
            torch.cuda.empty_cache()  # Libera caché de GPU

    return wrapper


def get_datetime():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def obtener_carpeta(ruta):
    # Normalizar la ruta
    ruta_normalizada = os.path.normpath(ruta)

    # Si la ruta normalizada no es un directorio, extraer el directorio
    if not os.path.isdir(ruta_normalizada):
        return os.path.dirname(ruta_normalizada)
    return ruta_normalizada


# @handle_exception
# @measure_time
# @clean_results
def train_yolo(request_config: dict, trial_number: int):
    # Configurar las variables de entorno necesarias para MLflow
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"  # URI del servidor MLflow
    os.environ["MLFLOW_ARTIFACT_URI"] = "s3://mlflow-artifacts/"  # Bucket en MinIO
    
    # Configurar el nombre del experimento y el nombre de la ejecución
    os.environ["MLFLOW_EXPERIMENT_NAME"] = request_config.get("sweeper").get(
        "study_name"
    )
    os.environ["MLFLOW_RUN_NAME"] = request_config.get("task_id")

    logger.info(f"🚀 Iniciando entrenamiento con la configuración: {request_config}")

    # Prepared and train model
    model_name = request_config["model"]

    if request_config["type"] == "yolo":
        model = YOLO(model_name)
    elif request_config["type"] == "rtdetr":
        model = RTDETR(model_name)
    else:
        raise ValueError("Invalid model type specified.")

    # Directorio de resultados
    if "task_id" not in request_config:
        return None

    timestamp = get_datetime()
    request_config["timestamp"] = timestamp

    RESULT_PATH = f'/models/{request_config.get("experiment_name")}/{request_config["type"]}/{request_config["task_id"]}/'
    os.makedirs(f"{RESULT_PATH}/trail_history", exist_ok=True)

    # Entrenamiento
    if "train" in request_config:
        request_config["train"]["project"] = f"{RESULT_PATH}/{trial_number}/"
        request_config["train"][
            "name"
        ] = "train"
        request_config["train"]["verbose"] = True
        request_config["train"]["plots"] = True
        request_config["train"]["exist_ok"] = True
        results = model.train(**request_config["train"])

        request_config["experiment_type"] = str(results.task)
        request_config["train"]["results"] = results.results_dict

    # Validación
    if "val" in request_config:
        request_config["val"]["project"] = f"{RESULT_PATH}/{trial_number}/"
        request_config["val"][
            "name"
        ] = "val"
        request_config["val"]["plots"] = True
        request_config["val"]["verbose"] = False
        request_config["val"]["exist_ok"] = True
        request_config["val"]["batch"] = 1
        validation_results = model.val(**request_config["val"])

        val_results = {
            # "task": str(results.task),
            "precision_b": round(results.results_dict["metrics/precision(B)"], 2),
            "recall_b": round(results.results_dict["metrics/recall(B)"], 2),
            "mAP50_b": round(results.results_dict["metrics/mAP50(B)"], 2),
            "mAP95_b": round(results.results_dict["metrics/mAP50-95(B)"], 2),
            # "f1": round(results.box.f1[0], 2),
            "precision": round(
                validation_results.results_dict["metrics/precision(B)"],
                2,
            ),
            "recall": round(validation_results.results_dict["metrics/recall(B)"], 2),
            # "loss": loss_callback.losses[-1],
        }

        request_config["val"]["results"] = val_results

    if "test" in request_config:
        request_config["test"]["project"] = f"{RESULT_PATH}/{trial_number}/"
        request_config["test"][
            "name"
        ] = "test"
        request_config["test"]["plots"] = True
        request_config["test"]["verbose"] = False
        request_config["test"]["save"] = True
        request_config["test"]["exist_ok"] = True
        request_config["test"]["batch"] = 1
        config_test = request_config["test"]

        path_dataset = obtener_carpeta(request_config["train"]["data"])
        PATH = f"{path_dataset}/test"

        for image_path in tqdm(glob(f"{PATH}/*/*"), desc="Testing", unit="Images"):
            try:
                model.predict(**config_test, source=image_path)
            except:
                pass

        config_test["source"] = PATH
        config_test["name"] = "test_metrics"
        model.val(**config_test)

        request_config["path_results"] = PATH

    # Guardar modelo y artefactos
    metric = val_results["mAP95_b"]
    model_path = f"{RESULT_PATH}/trail_history/trial_{trial_number}.pt"
    model.save(model_path)

    request_config["model_path"] = model_path
    request_config["metric"] = metric
    request_config["model"] = model

    return request_config
