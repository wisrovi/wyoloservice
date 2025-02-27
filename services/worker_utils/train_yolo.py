import os
import traceback
from datetime import datetime
from glob import glob

import torch
from loguru import logger
from tqdm import tqdm
from ultralytics import RTDETR, YOLO, settings

# Update a setting
settings.update({"mlflow": True})

# Reset settings to default values
# settings.reset()


# decortador para capturar excepciones
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
            traceback.print_exc()
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
@clean_results
def train_yolo(request_config: dict, trial_number: int):
    # Configurar las variables de entorno necesarias para MLflow
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = request_config["minio"]["MINIO_ENDPOINT"]
    os.environ["AWS_ACCESS_KEY_ID"] = request_config["minio"]["MINIO_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = request_config["minio"]["MINIO_SECRET_KEY"]
    os.environ["MLFLOW_TRACKING_URI"] = request_config["mlflow"]["MLFLOW_TRACKING_URI"]  # URI del servidor MLflow
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
        return float("nan")

    timestamp = get_datetime()
    request_config["timestamp"] = timestamp

    RESULT_PATH = f'/models/{request_config.get("experiment_name")}/{request_config["type"]}/{request_config["task_id"]}/'
    os.makedirs(f"{RESULT_PATH}/trail_history", exist_ok=True)

    request_config["path_results"] = f"{RESULT_PATH}/{trial_number}/"

    # Entrenamiento
    if "train" in request_config:
        request_config["train"]["project"] = f"{RESULT_PATH}/{trial_number}/"
        request_config["train"]["name"] = f"train_{request_config.get('task_id')}"
        request_config["train"]["verbose"] = True
        request_config["train"]["plots"] = True
        request_config["train"]["exist_ok"] = True
        results = model.train(**request_config["train"])

        request_config["experiment_type"] = str(results.task)
        request_config["train"]["results"] = results.results_dict

    # Validación
    if "val" in request_config:
        request_config["val"]["project"] = f"{RESULT_PATH}/{trial_number}/"
        request_config["val"]["name"] = "val"
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
        request_config["test"]["name"] = "test"
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

    if str(results.task) == "detect" or str(results.task) == "segment":
        metric = round(results.results_dict["metrics/mAP50-95(B)"], 2)
    else:
        # classification,leo el top1
        metric = round(results.results_dict["metrics/accuracy_top1"], 2)

    model_path = f"{RESULT_PATH}/trail_history/trial_{trial_number}.pt"
    model.save(model_path)

    request_config["model_path"] = model_path
    request_config["metric"] = metric
    request_config["model"] = model

    return request_config


@measure_time
@clean_results
def comparar_modelos_yolo(modelo1_path: str, modelo2_path: str, data_path: str):
    """
    Compara dos modelos YOLOv8 usando datos de validación.

    Args:
        modelo1_path (str): Ruta al primer modelo YOLOv8.
        modelo2_path (str): Ruta al segundo modelo YOLOv8.
        data_path (str): Ruta al archivo de datos de validación (YAML).

    Returns:
        str: Ruta del modelo con el mejor mAP, o None si hay un error.
    """
    try:
        # Carga los modelos
        modelo1 = YOLO(modelo1_path)
        modelo2 = YOLO(modelo2_path)

        # Valida los modelos
        resultados1 = modelo1.val(data=data_path)
        resultados2 = modelo2.val(data=data_path)

        # Obtiene el mAP de cada modelo
        map1 = resultados1.results_dict.get(
            "metrics/mAP50-95(B)", 0.0
        )  # Se usa el map50-95 que es el valor más comun.
        map2 = resultados2.results_dict.get("metrics/mAP50-95(B)", 0.0)

        # Compara los mAP
        if map1 > map2:
            # print(
            #     f"El modelo '{modelo1_path}' tiene un mejor mAP ({map1:.4f} > {map2:.4f})."
            # )
            return modelo1_path
        elif map2 > map1:
            print(
                f"El modelo '{modelo2_path}' tiene un mejor mAP ({map2:.4f} > {map1:.4f})."
            )
            return modelo2_path
        else:
            print(f"Ambos modelos tienen el mismo mAP ({map1:.4f}).")
            return modelo1_path  # Devuelve el primero si son iguales
    except Exception as e:
        print(f"Error al comparar modelos: {e}")
        return None
