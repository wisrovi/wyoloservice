import datetime
import os
import tempfile
import time
import traceback
from typing import List

import hydra
import mlflow

# from logbook import FileHandler, Logger
from loguru import logger
from omegaconf import OmegaConf

# Cambiar el nombre del proceso
from setproctitle import setproctitle
from train_yolo.trainer_wrapper import obtener_info_gpu_json
from wpipe.pipe import Pipeline
from wredis.hash import RedisHashManager
from wredis.queue import RedisQueueManager
from wredis.sortedset import RedisSortedSetManager

from states import (
    DEFAULT_CONFIG,
    Eda_calculate,
    OptunaOptimize,
    Start_inform,
    read_user_config,
    results_up_to_minio,
)
from worker_utils import MinioS3Client, SharedResource, health

setproctitle("train_service")


__VERSION__ = "v1.0.8"


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


@hydra.main(config_path="/app", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    global DEFAULT_CONFIG
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

    # DEFAULT_CONFIG["minio"]["MINIO_ID"] = os.getenv("CIFS_USER", "mlflow")
    # DEFAULT_CONFIG["minio"]["MINIO_SECRET_KEY"] = os.getenv("CIFS_PASS", "wyoloservice")

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

    DEBUG_MODE = os.environ.get("debug", None)
    PUBLIC_TOPIC = redis_config.get("TOPIC")
    USER_TOPIC = os.environ.get("USER", None)
    WORKER_HOST_TOPIC = os.environ.get("WORKER_HOST", None)

    results_queue = redis_config.get("RESULT_TOPIC", redis_config.get("TOPIC"))

    logger.info(f"Results queue: {results_queue}")
    logger.info(f"Debug mode: {DEBUG_MODE}")

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
            logger.error(f"Can't report results: {e}")
            logger.error(traceback.format_exc())

    def process_requests(task_data: dict):

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

    # MODE STOP
    if DEBUG_MODE is None:
        logger.info(f"Activate stop topic: stop_{WORKER_HOST_TOPIC}")

        @queue_manager.on_message(f"stop_{WORKER_HOST_TOPIC}")
        def stop_worker(task_data: dict):
            logger.debug(f"Received data in stop, {task_data}")

            if task_data.get("stop", None) == USER_TOPIC:
                logger.info("Stopping worker...")
                queue_manager.publish(
                    queue_name="stop_worker",
                    data={
                        "stop": task_data,
                        "datetime": datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    },
                )

                logger.info("Stopping worker...")
                # Detener la aplicación si ocurre un error
                os._exit(0)

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

    # NOTA: cuando se levanta el worker en modo debug, no funciona la cola "PUBLIC_TOPIC"
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
