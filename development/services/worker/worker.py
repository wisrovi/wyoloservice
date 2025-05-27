import argparse
import datetime
import json
import os
import sys
import tempfile
import time
import traceback
from typing import List
import uuid

import hydra
import mlflow
import requests
import yaml

# para informacion inicial
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# from logbook import FileHandler, Logger
from loguru import logger
from omegaconf import OmegaConf

# Cambiar el nombre del proceso
from setproctitle import setproctitle
from states import (
    DEFAULT_CONFIG,
    Eda_calculate,
    OptunaOptimize,
    Start_inform,
    read_user_config,
    results_up_to_minio,
)
from train_yolo.trainer_wrapper import obtener_info_gpu_json
from worker_utils import (
    MinioS3Client,
    SharedResource,
    health,
    validate_yolo_dataset,
    YOLOValidationFailedError,
    PermissionsError,
    DatasetContentError,
    DatasetNotFoundError,
    show_qr_in_terminal,
)
from wpipe.pipe import Pipeline
from wredis.queue import RedisQueueManager
from wredis.hash import RedisHashManager

setproctitle("train_service")

console = Console()


__VERSION__ = "v1.0.12"


CONTROL_HOST = os.getenv("CONTROL_HOST", None)
if CONTROL_HOST is None:
    raise Exception("CONTROL_HOST env var is not set")


DEBUG_MODE = None


# Configura el primer logger: solo errores en un archivo
logger.add(
    "/var/log/worker/error_log.log", level="ERROR", rotation="10 MB", retention="7 days"
)


pipeline = Pipeline()
pipeline.set_steps(
    [
        (read_user_config, "read_config", "v1.0"),
        (Start_inform(), Start_inform.__NAME__, Start_inform.__VERSION__),
        (Eda_calculate(), Eda_calculate.__NAME__, Eda_calculate.__VERSION__),
        (OptunaOptimize(), OptunaOptimize.__NAME__, OptunaOptimize.__VERSION__),
        (results_up_to_minio, "result_to_minio", "v1.0"),
    ]
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Un script que recibe un argumento --config como string y un argumento --epochs como int"
    )
    parser.add_argument("--config", type=str, help="Ruta al archivo de configuración")
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Número de épocas para el entrenamiento (default: 2)",
    )
    return parser.parse_known_args()  # Usa parse_known_args


def remove_argparse_arguments(args):
    # Elimina los argumentos de argparse de sys.argv
    sys.argv = [sys.argv[0]]  # Mantén el nombre del script
    for arg in args:
        if arg.startswith("--config") or arg.startswith("--epochs"):
            continue
        sys.argv.append(arg)


# Analiza los argumentos de argparse ANTES de inicializar Hydra
args, unknown = parse_arguments()
# Elimina los argumentos de argparse de sys.argv
remove_argparse_arguments(unknown)


@hydra.main(config_path="/app", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    global DEFAULT_CONFIG
    global DEBUG_MODE

    show_qr_in_terminal()

    cfg.mlflow.MLFLOW_TRACKING_URI = cfg.mlflow.MLFLOW_TRACKING_URI.replace(
        "localhost", CONTROL_HOST
    )

    cfg.minio.MINIO_ENDPOINT = cfg.minio.MINIO_ENDPOINT.replace(
        "localhost", CONTROL_HOST
    )

    cfg.redis.REDIS_HOST = cfg.redis.REDIS_HOST.replace("localhost", CONTROL_HOST)

    mlflow.set_tracking_uri(cfg.mlflow.MLFLOW_TRACKING_URI)

    # convertir a dict
    cfg = OmegaConf.to_container(cfg, resolve=True)

    DEFAULT_CONFIG.update(cfg)

    DEFAULT_CONFIG["minio"]["MINIO_ID"] = os.getenv("CIFS_USER", "mlflow")
    DEFAULT_CONFIG["minio"]["MINIO_SECRET_KEY"] = os.getenv("CIFS_PASS", "wyoloservice")

    DEFAULT_CONFIG["dvc"]["MINIO_ID"] = os.getenv("CIFS_USER", "mlflow")
    DEFAULT_CONFIG["dvc"]["MINIO_SECRET_KEY"] = os.getenv("CIFS_PASS", "wyoloservice")

    os.makedirs("/config", exist_ok=True)
    with open("/config/config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(DEFAULT_CONFIG))

    MinioS3Client(
        endpoint_url=DEFAULT_CONFIG.get("minio", {}).get("MINIO_ENDPOINT"),
        aws_access_key_id=DEFAULT_CONFIG.get("minio", {}).get("MINIO_ID"),
        aws_secret_access_key=DEFAULT_CONFIG.get("minio", {}).get("MINIO_SECRET_KEY"),
    )

    redis_config = DEFAULT_CONFIG.get("redis", {})

    queue_manager = RedisQueueManager(
        host=redis_config.get("REDIS_HOST"),
        port=redis_config.get("REDIS_PORT"),
        db=redis_config.get("REDIS_DB"),
        verbose=False,
    )

    hash_manager = RedisHashManager(
        host=redis_config.get("REDIS_HOST"),
        port=redis_config.get("REDIS_PORT"),
        db=redis_config.get("REDIS_DB"),
        verbose=False,
    )

    DEBUG_MODE = os.environ.get("debug", None)
    PUBLIC_TOPIC = redis_config.get("TOPIC")
    USER_TOPIC = os.environ.get("USER", None)
    WORKER_HOST_TOPIC = os.environ.get("WORKER_HOST", None)

    results_queue = redis_config.get("RESULT_TOPIC", redis_config.get("TOPIC"))

    info_data = {
        "VERSION": __VERSION__,
        "Results queue": results_queue,
        "Debug mode": "True" if DEBUG_MODE else "False",
    }

    info_data_private = {
        "CONTROL_HOST": CONTROL_HOST,
        "CIFS_USER": os.getenv("CIFS_USER", "mlflow"),
        "CIFS_PASS": os.getenv("CIFS_PASS", "wyoloservice"),
        "NUM_CURRENT_TRAIN": os.getenv("NUM_CURRENT_TRAIN", "1"),
        "MAX_GPU": os.getenv("MAX_GPU", "60"),
    }

    private_topics = []
    public_topics = []

    shared_resource = SharedResource(hash_manager)

    def complete_requests(results):
        try:
            try:
                params = results["train"]["best_trial"].params
                metric = results["train"]["best_metric"]
                best_model_path = results["train"]["best_model_path"]
            except:
                params = "stop and continue in other worker"
                metric = "stop and continue in other worker"
                best_model_path = "stop and continue in other worker"

            queue_manager.publish(
                queue_name=results_queue,
                data={
                    "task_id": results["task_id"],
                    "user_code": results["user_code"],
                    "fitness": results["sweeper"]["fitness"],
                    "minio_url": results.get("minio_url", None),
                    "imgsz": results["train"]["imgsz"],
                    "n_trials": results["sweeper"]["n_trials"],
                    "optimized_params": {
                        "params": params,
                        "best_model_path": best_model_path,
                        "metric": metric,
                    },
                    "optimizer": results["sweeper"]["algorithm"],
                },
            )
        except Exception as e:
            logger.error(f"Can't report results: {e}")
            logger.error(traceback.format_exc())

    def process_requests(task_data: dict):

        # valid if exists the sleep file (created by admin)
        # and check if the sleep time is over
        # if the file exists, the worker is in sleep mode
        # and the file contains the metadata
        # check if the file exists
        sleep_file = "/config/sleep"
        if os.path.exists(sleep_file):
            # read the file in json format
            try:
                with open(sleep_file, "r") as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error reading sleep file: {e}")
                os.remove(sleep_file)
                metadata = None

            # check if the sleep time is over
            if metadata:
                datetime_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                datetime_create = metadata["datetime"]

                elapsed_time = (
                    datetime.datetime.strptime(datetime_now, "%Y-%m-%d %H:%M:%S")
                    - datetime.datetime.strptime(datetime_create, "%Y-%m-%d %H:%M:%S")
                ).total_seconds()

                if elapsed_time > 60 * 30:  # 30 minutes = 60*30 seconds
                    # remove the file
                    logger.info("Sleep time is over, removing file...")
                    # remove the file
                    os.remove(sleep_file)
                else:
                    # check if the sleep time is over
                    logger.info(
                        f"Worker in sleep mode... {metadata}, elapsed time: {elapsed_time}, sleep time: {60 * 30}"
                    )

                    logger.info("Worker awake...")
                    recreate_request(
                        topic=task_data["topic"],
                        args_dict=task_data,
                    )

                    time.sleep(60)  # wait 1 minute before checking again

                    return None

        # validar la cantidad de GPU disponible, si es menor a 2GB, no se ejecuta
        gpu_json_list: List[dict] = obtener_info_gpu_json()
        free = 0
        for gpu_id, gpu in enumerate(gpu_json_list):
            free_memory = int(gpu[f"gpu_{gpu_id}_memoryFree"])

            free += free_memory

        if free < 2 * 1024:
            logger.error("Not enough GPU memory available")
            return None

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                task_data["tempfile"] = temp_dir

                results = shared_resource.execute_process(
                    function=pipeline.run,
                    args_dict=task_data,
                )

                return results

                # results = pipeline.run(task_data)
        except Exception as e:
            traceback.print_exc()

            # token_manager.write_token(token=task_data["task_id"], data=e.args[0])

            queue_manager.publish(
                queue_name=f"{results_queue}_error",
                data={
                    "task_id": results["task_id"],
                    "user_code": results["user_code"],
                    "fitness": results["sweeper"]["fitness"],
                    "minio_url": results.get("minio_url", None),
                    "imgsz": results["train"]["imgsz"],
                    "n_trials": results["sweeper"]["n_trials"],
                    "optimized_params": {
                        "params": results["train"]["best_trial"].params,
                        "best_model_path": results["train"]["best_model_path"],
                        "metric": results["train"]["best_metric"],
                    },
                    "optimizer": results["sweeper"]["algorithm"],
                },
            )

            return None

    def recreate_request(topic, args_dict):
        queue_manager.publish(queue_name=topic, data=args_dict)
        time.sleep(30)

    # Receptores de colas de redis

    # MODE EVALUATE (for dataset validation), only is active when not is in DEBUG_MODE
    if True:
        EVALUATE_TOPIC_RESULTS = "evaluate_results"

        public_topics.append(f"evaluate")
        private_topics.append(f"evaluate_{WORKER_HOST_TOPIC}")

        def _evaluate_yolo_dataset(
            data_yaml: str,
            model_path: str,
            topic: str,
            task_id: str,
            user_code: str = "test",
        ):
            """
            Function simulating a successful training run after dataset validation.

            Args:
                data_yaml (str): Path to the dataset YAML file.
                model_path (str): Path to the model file.
            """

            @validate_yolo_dataset(data_yaml=data_yaml, model_path=model_path)
            def _evaluate():
                """Function simulating a successful training run after dataset validation."""
                logger.info(
                    f"Dataset validation successful. Proceeding with training using model: {model_path}"
                )
                return True

            try:
                results = shared_resource.execute_process(
                    function=_evaluate,
                    args_dict={},
                )
            except YOLOValidationFailedError as e:
                logger.error(f"Error UNEXPECTEDLY caught in valid scenario: {e}")
                queue_manager.publish(
                    queue_name=EVALUATE_TOPIC_RESULTS,
                    data={
                        "task_id": task_id,
                        "user_code": user_code,
                        "data_yaml": data_yaml,
                        "model_path": model_path,
                        "topic": topic,
                        "status": "YOLOValidationFailedError",
                        "message": f"Error UNEXPECTEDLY caught in valid scenario: {e}",
                    },
                )
                return
            except DatasetNotFoundError as e:
                logger.error(f"'DatasetNotFoundError' caught as expected: {e}")
                queue_manager.publish(
                    queue_name=EVALUATE_TOPIC_RESULTS,
                    data={
                        "task_id": task_id,
                        "user_code": user_code,
                        "data_yaml": data_yaml,
                        "model_path": model_path,
                        "topic": topic,
                        "status": "DatasetNotFoundError",
                        "message": f"Dataset not found: {e}",
                    },
                )
                return
            except PermissionsError as e:
                logger.error(f"'PermissionsError' caught as expected: {e}")
                queue_manager.publish(
                    queue_name=EVALUATE_TOPIC_RESULTS,
                    data={
                        "task_id": task_id,
                        "user_code": user_code,
                        "data_yaml": data_yaml,
                        "model_path": model_path,
                        "topic": topic,
                        "status": "PermissionsError",
                        "message": f"Dataset validation failed: {e}",
                    },
                )
                return
            except DatasetContentError as e:
                logger.error(f"'DatasetContentError' caught as expected: {e}")
                queue_manager.publish(
                    queue_name=EVALUATE_TOPIC_RESULTS,
                    data={
                        "task_id": task_id,
                        "user_code": user_code,
                        "data_yaml": data_yaml,
                        "model_path": model_path,
                        "topic": topic,
                        "status": "DatasetContentError",
                        "message": f"Dataset validation failed: {e}",
                    },
                )
                return
            except Exception as e:
                logger.critical(
                    f"Unexpected and unhandled error in valid scenario: {e}"
                )
                queue_manager.publish(
                    queue_name=EVALUATE_TOPIC_RESULTS,
                    data={
                        "task_id": task_id,
                        "user_code": user_code,
                        "data_yaml": data_yaml,
                        "model_path": model_path,
                        "topic": topic,
                        "status": "Exception",
                        "message": f"Dataset validation failed: {e}",
                    },
                )
                return

            if results is None or shared_resource.elapsedtime() < 10:
                recreate_request(
                    topic=topic,
                    args_dict={
                        "data_yaml": data_yaml,
                        "model_path": model_path,
                        "topic": topic,
                    },
                )
            else:
                # Publish the results to the results queue
                queue_manager.publish(
                    queue_name=EVALUATE_TOPIC_RESULTS,
                    data={
                        "task_id": task_id,
                        "user_code": user_code,
                        "data_yaml": data_yaml,
                        "model_path": model_path,
                        "topic": topic,
                        "status": "success",
                        "message": "Dataset validation successful.",
                    },
                )

        @queue_manager.on_message(f"evaluate_{WORKER_HOST_TOPIC}")
        def evaluate_worker_private_mode(task_data: dict):
            logger.debug(f"Received data in evaluate, {task_data}")
            task_id = task_data["task_id"]
            user_code = task_data["user_code"]

            _evaluate_yolo_dataset(
                data_yaml=task_data["data_yaml"],
                model_path=task_data["model_path"],
                topic=f"evaluate_{WORKER_HOST_TOPIC}",
                task_id=task_id,
                user_code=user_code,
            )

        if DEBUG_MODE is None:
            @queue_manager.on_message(f"evaluate")
            def evaluate_worker_public_mode(task_data: dict):
                logger.debug(f"Received data in evaluate, {task_data}")
                task_id = task_data["task_id"]
                user_code = task_data["user_code"]

                _evaluate_yolo_dataset(
                    data_yaml=task_data["data_yaml"],
                    model_path=task_data["model_path"],
                    topic="evaluate",
                    task_id=task_id,
                    user_code=user_code,
                )
                
                # example request
                #task_data = {
                #     "task_id": "test",
                #     "user_code": "test",
                #     "data_yaml": "/app/datasets/valid_dataset.yaml",
                #     "model_path": "/app/yolo_weights/yolov8n.pt",
                # }

    # MODE ADMIN (for sleep), only is active when not is in DEBUG_MODE
    if True:
        private_topics.append(f"admin_{WORKER_HOST_TOPIC}")

        @queue_manager.on_message(f"admin_{WORKER_HOST_TOPIC}")
        def admin_worker(task_data: dict):
            logger.debug(f"Received data in admin, {task_data}")

            if task_data.get("admin", None) == USER_TOPIC:
                logger.info("Activating sleep...")

                # calculate the actual time
                datetime_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # create a file to indicate that the worker is in sleep mode
                metadata = {
                    "task_id": task_data["task_id"],
                    "user_code": task_data["user_code"],
                    "datetime": datetime_now,
                    "end_datetime": "30 minutes",
                    "status": "sleep",
                    "worker": WORKER_HOST_TOPIC,
                    "worker_host": CONTROL_HOST,
                    "user": USER_TOPIC,
                }

                # publish the sleep message to the queue for notification to the admin in the control
                queue_manager.publish(
                    queue_name="sleep_worker",
                    data={"admin": metadata, "datetime": datetime_now},
                )

                with open("/config/sleep", "w") as f:
                    # save metadata in the file in json format
                    metadata_json = json.dumps(metadata)
                    f.write(metadata_json)
                logger.info("Worker in sleep mode...")

    # MODE STOP (for emergency stop), only is active when not is in DEBUG_MODE
    if True:
        private_topics.append(f"stop_{WORKER_HOST_TOPIC}")

        @queue_manager.on_message(f"stop_{WORKER_HOST_TOPIC}")
        def stop_worker(task_data: dict):
            logger.debug(f"Received data in stop, {task_data}")

            user_request = task_data["config_path"]
            with open(user_request, "r") as file:
                config = yaml.safe_load(file)

            if config.get("stop", None) == USER_TOPIC:
                task_id = config.get("task_id", None)
                destinity = config.get("destinity", None)

                metadata = {
                    "task_id": config["task_id"],
                    "user_code": task_data["user_code"],
                    "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "stop",
                    "worker": WORKER_HOST_TOPIC,
                    "worker_host": CONTROL_HOST,
                    "user": USER_TOPIC,
                }

                if task_id:
                    # create a file to indicate that the worker is in stop mode
                    stop_training_file = f"/config/stop_training_{task_id}.txt"
                    with open(stop_training_file, "w") as f:
                        # save metadata in the file in json format
                        metadata_json = json.dumps(metadata)
                        f.write(metadata_json)

                    # this file is used to stop the training process
                    # and the worker will stop when the training is finished
                    # and the file is removed
                    logger.info(f"Worker in stop mode... {metadata}")

                    while os.path.exists(stop_training_file):
                        # wait until the file is removed
                        logger.debug(
                            f"Waiting for the file {stop_training_file} to be removed... (will be removed when the training is finished completely)"
                        )
                        time.sleep(5)

                    # publish the stop message to the queue for notification to the admin in the control
                    logger.info("Stopping worker...")
                    queue_manager.publish(
                        queue_name="stop_worker",
                        data={
                            "stop": metadata,
                            "status": "stop",
                            "datetime": datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        },
                    )
                    if destinity:
                        # To make the process easier, a message is sent to the API,
                        # indicating the task_id, the path of the last model and the path of the original configuration file.
                        if not isinstance(destinity, str):
                            destinity = PUBLIC_TOPIC

                        last_user = task_data["user_code"]
                        train_config = f"/config_versions/{task_id}.yaml"

                        originl_dataset_path = config["train"]["data"]
                        last_model = f"{originl_dataset_path}weights/last.pt"

                        files = {
                            "file": (
                                "config_train.yaml",
                                open(train_config, "rb"),
                                "application/x-yaml",
                            )
                        }
                        headers = {"accept": "application/json"}
                        params = {
                            "user_code": last_user,
                            "resume": True,
                            "task_id": task_id,
                            "last_model": last_model,
                            "destinity": destinity,
                        }
                        url = f"http://{CONTROL_HOST}:23450/recreate/"
                        response = requests.post(
                            url, params=params, headers=headers, files=files
                        )

                        if response.status_code == 200:
                            logger.info(
                                f"Recreate request sent to {destinity} worker: {response.json()}"
                            )
                        else:
                            logger.error(
                                f"Error sending recreate request to {destinity} worker: {response.status_code}, {response.text}"
                            )

                        logger.warning(
                            f"training '{task_id}' recreated for continue in '{destinity}' worker"
                        )
                    # stop the worker
                    os._exit(0)
                else:
                    logger.error("No task_id in stop message")
                    # publish the stop message to the queue for notification to the admin in the control
                    queue_manager.publish(
                        queue_name="stop_worker",
                        data={
                            "stop": metadata,
                            "status": "error",
                            "error": "No task_id in stop message",
                            "datetime": datetime.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        },
                    )

    if USER_TOPIC:
        private_topics.append(USER_TOPIC)

        @queue_manager.on_message(USER_TOPIC)
        def private_worker(task_data: dict):
            logger.debug(f"Received data in {USER_TOPIC}, {task_data}")

            # process_requests devuelve None cuando el semaforo esta ocupado
            results = process_requests(task_data)

            if results is None or shared_resource.elapsedtime() < 10:
                recreate_request(topic=USER_TOPIC, args_dict=task_data)
            else:
                complete_requests(results)

    if WORKER_HOST_TOPIC:
        private_topics.append(WORKER_HOST_TOPIC)

        @queue_manager.on_message(WORKER_HOST_TOPIC)
        def private_worker_2(task_data: dict):
            logger.debug(f"Received data in {WORKER_HOST_TOPIC}, {task_data}")

            # process_requests devuelve None cuando el semaforo esta ocupado
            results = process_requests(task_data)

            if results is None or shared_resource.elapsedtime() < 10:
                recreate_request(topic=WORKER_HOST_TOPIC, args_dict=task_data)
            else:
                complete_requests(results)

    # NOTA: cuando se levanta el worker en modo debug, no funciona la cola "PUBLIC_TOPIC"
    if DEBUG_MODE is None:
        public_topics.append(PUBLIC_TOPIC)

        @queue_manager.on_message(PUBLIC_TOPIC)
        def public_worker(task_data: dict):
            logger.debug(f"Received data in {PUBLIC_TOPIC}, {task_data}")

            # process_requests devuelve None cuando el semaforo esta ocupado
            results = process_requests(task_data)

            if results is None:
                recreate_request(topic=PUBLIC_TOPIC, args_dict=task_data)
            else:
                complete_requests(results)

    else:
        public_topics.append("-")
        logger.info(f"Not activate public topic ({PUBLIC_TOPIC}) in DEBUG_MODE")
    # El worker se inicia con el argumento --config y --epochs

    # el --config es un archivo de configuración
    # que contiene la configuración del entrenamiento
    # y se inicia el worker con esa configuración para el entrenamiento
    # si no se pasa el argumento --config, se inicia el worker normalmente

    # el --epochs es un argumento opcional
    # que se usa para el entrenamiento
    # y se inicia el worker con esa configuración para el entrenamiento
    # si no se pasa el argumento --epochs, epoca por defecto es 2

    # Tabla para la información principal
    table_info = Table(
        title="Información Inicial", show_header=True, header_style="bold magenta"
    )
    table_info.add_column("Campo", style="cyan")
    table_info.add_column("Valor", style="green")

    for key, value in info_data.items():
        table_info.add_row(key, value)

    table_info_private = Table(
        title="Información Privada", show_header=True, header_style="bold magenta"
    )
    table_info_private.add_column("Campo", style="cyan")
    table_info_private.add_column("Valor", style="green")
    for key, value in info_data_private.items():
        table_info_private.add_row(key, value)

    # Tabla para tópicos privados
    table_private = Table(
        title="🔐 Tópicos Privados", show_header=False, border_style="red"
    )
    table_private.add_column("Nombre del tópico", style="yellow")
    for topic in private_topics:
        table_private.add_row(topic)

    # Tabla para tópicos públicos
    table_public = Table(
        title="🌍 Tópicos Públicos", show_header=False, border_style="blue"
    )
    table_public.add_column("Nombre del tópico", style="green")
    for topic in public_topics:
        table_public.add_row(topic)

    # Mostrar todo en consola
    for _ in range(5):
        console.print()

    console.print(table_info)
    console.print(table_info_private)
    console.print()
    console.print(table_private)
    console.print()
    console.print(table_public)
    console.print(
        "[bold green]✅ Health check started with version v1.0.10[/bold green]"
    )

    if args.config:
        config_path = args.config

        # validar si el archivo existe
        if not os.path.exists(config_path):
            logger.error(f"El archivo de configuración no existe: {config_path}")
            return
        # validar si el archivo es un archivo yaml
        if not config_path.endswith(".yaml"):
            logger.error(
                f"El archivo de configuración no es un archivo yaml: {config_path}"
            )
            return

        # trabajar en una carpeta temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            # Cargar la configuración desde el archivo
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
                # validar si el config tiene la clave "train"
                if "train" not in config:
                    logger.error(
                        f"El archivo de configuración no tiene la clave 'train': {config_path}"
                    )
                    return

                config["train"]["epochs"] = args.epochs

            # Guardar la configuración en un archivo temporal
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, "w") as file:
                yaml.dump(config, file)

            logger.info(
                f"Train config path: {args.config} in worker version {__VERSION__}"
            )

            task_data = {
                "task_id": f'test_{str(uuid.uuid4()).replace("-", "")}',
                "config_path": config_path,
                "user_code": "test",
                "db_count": 1,
            }

            results = process_requests(task_data)
            complete_requests(results)
    else:
        # Si no se pasa el argumento --config, se inicia el worker normalmente
        health(__VERSION__)
        queue_manager.start()
        queue_manager.wait()


if __name__ == "__main__":
    main()
