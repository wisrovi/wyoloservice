import os
import sys
import tempfile
import time
import traceback
from typing import List
import uuid

import hydra
import mlflow

# from logbook import FileHandler, Logger
from loguru import logger
from omegaconf import OmegaConf
from wpipe.pipe import Pipeline
from wredis.queue import RedisQueueManager
from wredis.hash import RedisHashManager
from wredis.sortedset import RedisSortedSetManager

from train_yolo.trainer_wrapper import obtener_info_gpu_json, TrainerWrapper
from states import (
    DEFAULT_CONFIG,
    OptunaOptimize,
    read_user_config,
    results_up_to_minio,
    Start_inform,
    Eda_calculate,
)

import uvicorn
from fastapi import FastAPI

app = FastAPI()

from worker_utils import MinioS3Client, ejecutar_en_hilo, SharedResource


__VERSION__ = "v1.0.5"

CONTROL_HOST = os.getenv("CONTROL_HOST", None)
if CONTROL_HOST is None:
    raise Exception("CONTROL_HOST env var is not set")


DEBUG_MODE = None
hash_manager = None
sorted_set_manager = None


# Configura el primer logger: solo errores en un archivo
logger.add(
    "/var/log/worker/error_log.log", level="ERROR", rotation="10 MB", retention="7 days"
)

# Configura el segundo logger: mensajes normales en la consola
# logger.add(
#     sys.stdout,
#     level="INFO",
#     format="<green>{time}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
# )


@app.get("/")
async def read_version():

    global hash_manager
    global sorted_set_manager
    global DEBUG_MODE

    """Devuelve la versión de la API."""

    metadata = {
        other_metadata: os.environ.get(other_metadata, None)
        for other_metadata in TrainerWrapper.worker_metadata
    }
    metadata["__VERSION__"] = __VERSION__
    metadata["debug"] = DEBUG_MODE
    metadata["debug"] = metadata["debug"] if DEBUG_MODE else "False"

    gpu_json_list: List[dict] = obtener_info_gpu_json()
    for gpu_json in gpu_json_list:
        for key, value in gpu_json.items():
            if value is not None:
                if (
                    isinstance(value, int)
                    or isinstance(value, float)
                    or isinstance(value, str)
                    #
                    or isinstance(key, int)
                    or isinstance(key, float)
                    or isinstance(key, str)
                ):
                    metadata[key] = value

    redis_key = (
        "workers"
        + f":{metadata.get('WORKER_HOST', 'noIp')}"
        + f":{metadata.get('USER', str(uuid.uuid4()))}"
    )
    for metadata_key, metadata_value in metadata.items():
        hash_manager.create_hash(
            key=redis_key,
            hash_name=metadata_key,
            value=metadata_value,
            ttl=30,
        )

    gpu_0_memoryFree = int(metadata["gpu_0_memoryFree"])
    gpu_0_memoryTotal = int(metadata["gpu_0_memoryTotal"])
    gpu_0_memoryUsed = int(metadata["gpu_0_memoryUsed"])

    NOT_TO_USE_GPU = 1 - int(os.environ.get("MAX_GPU", 60)) / 100
    if gpu_0_memoryFree > (gpu_0_memoryTotal * NOT_TO_USE_GPU):
        gpu_0_memoryFree = gpu_0_memoryTotal * NOT_TO_USE_GPU
    else:
        available = gpu_0_memoryTotal - (gpu_0_memoryTotal * NOT_TO_USE_GPU)
        gpu_0_memoryFree = max(available - gpu_0_memoryUsed, 0)

    member_name = f'{metadata["WORKER_HOST"]} ({metadata["USER"]})'
    if os.environ.get("debug", None):
        member_name += "[debug]"

    try:
        sorted_set_manager.redis_client.ping()
    except Exception as e:
        pass
        # Detener la aplicación si ocurre un error
        # os._exit(1)  # Fuerza la salida del proceso

    sorted_set_manager.add_to_sorted_set(
        key="available",
        score=gpu_0_memoryFree,
        member=member_name,
        ttl=30,
    )

    return {"version": __VERSION__}


@ejecutar_en_hilo
def health():
    logger.info("Starting health check...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


# Configura el tercer logger: todos los niveles en un archivo separado (completamente independiente)
# FileHandler("/var/log/worker/full_log.log").push_application()
# custom_logger = Logger("wyoloservice")


pipeline = Pipeline()
pipeline.set_steps(
    [
        (read_user_config, "read_config", "v1.0"),
        (Start_inform(), Start_inform.__NAME__, Start_inform.__VERSION__),
        # (Eda_calculate(), Eda_calculate.__NAME__, Eda_calculate.__VERSION__),
        (OptunaOptimize(), OptunaOptimize.__NAME__, OptunaOptimize.__VERSION__),
        (results_up_to_minio, "result_to_minio", "v1.0"),
    ]
)


@hydra.main(config_path="/app", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    global DEFAULT_CONFIG
    global hash_manager
    global sorted_set_manager
    global DEBUG_MODE

    cfg.mlflow.MLFLOW_TRACKING_URI = cfg.mlflow.MLFLOW_TRACKING_URI.replace(
        "localhost", CONTROL_HOST
    )

    cfg.minio.MINIO_ENDPOINT = cfg.minio.MINIO_ENDPOINT.replace(
        "localhost", CONTROL_HOST
    )

    cfg.redis.REDIS_HOST = cfg.redis.REDIS_HOST.replace("localhost", CONTROL_HOST)

    mlflow.set_tracking_uri(cfg.mlflow.MLFLOW_TRACKING_URI)
    logger.info(f"__VERSION__: {__VERSION__}")

    # convertir a dict
    cfg = OmegaConf.to_container(cfg, resolve=True)

    DEFAULT_CONFIG.update(cfg)

    DEFAULT_CONFIG["minio"]["MINIO_ID"] = os.getenv("CIFS_USER", "mlflow")
    DEFAULT_CONFIG["minio"]["MINIO_SECRET_KEY"] = os.getenv("CIFS_PASS", "wyoloservice")

    DEFAULT_CONFIG["dvc"]["MINIO_ID"] = os.getenv("CIFS_USER", "mlflow")
    DEFAULT_CONFIG["dvc"]["MINIO_SECRET_KEY"] = os.getenv("CIFS_PASS", "wyoloservice")

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

    sorted_set_manager = RedisSortedSetManager(
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

    # logger.info(f"Public topic: {PUBLIC_TOPIC}")
    # logger.info(f"Private topic: {USER_TOPIC}")
    logger.info(f"Results queue: {results_queue}")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    # logger.info(f"DEFAULT_CONFIG: {DEFAULT_CONFIG}")

    shared_resource = SharedResource()

    def complete_requests(results):
        try:
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
                        "params": results["train"]["best_trial"].params,
                        "best_model_path": results["train"]["best_model_path"],
                        "metric": results["train"]["best_metric"],
                    },
                    "optimizer": results["sweeper"]["algorithm"],
                },
            )
        except Exception as e:
            logger.error("Can't report results")

    def process_requests(task_data: dict):
        
        # validar la cantidad de GPU disponible, si es menor a 2GB, no se ejecuta
        gpu_json_list: List[dict] = obtener_info_gpu_json()
        free = 0
        for gpu_id, gpu in enumerate(gpu_json_list):
            free_memory = int(gpu[f'gpu_{gpu_id}_memoryFree'])
            
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

    if USER_TOPIC:
        logger.info(f"Activate Private topic: {USER_TOPIC}")

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
        logger.info(f"Activate worker topic: {WORKER_HOST_TOPIC}")

        @queue_manager.on_message(WORKER_HOST_TOPIC)
        def private_worker_2(task_data: dict):
            logger.debug(f"Received data in {WORKER_HOST_TOPIC}, {task_data}")

            # process_requests devuelve None cuando el semaforo esta ocupado
            results = process_requests(task_data)

            if results is None or shared_resource.elapsedtime() < 10:
                recreate_request(topic=WORKER_HOST_TOPIC, args_dict=task_data)
            else:
                complete_requests(results)

    if DEBUG_MODE is None:
        logger.info(f"Activate public topic: {PUBLIC_TOPIC}")

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
        logger.info(f"Not activate public topic ({PUBLIC_TOPIC}) in DEBUG_MODE")

    health()
    queue_manager.start()
    queue_manager.wait()


if __name__ == "__main__":
    main()
