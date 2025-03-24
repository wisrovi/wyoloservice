import os
import sys
import tempfile
import time
import traceback
import uuid

import hydra
import mlflow
from logbook import FileHandler, Logger
from loguru import logger
from omegaconf import OmegaConf
from wpipe.pipe import Pipeline
from wredis.queue import RedisQueueManager
from wredis.token import RedisTokenManager
from wredis.hash import RedisHashManager

from states import (
    DEFAULT_CONFIG,
    OptunaOptimize,
    read_user_config,
    results_up_to_minio,
    Start_inform,
)

from worker_utils import MinioS3Client, ejecutar_en_hilo, SharedResource


__VERSION__ = "v1.0.0"

CONTROL_HOST = os.getenv("CONTROL_HOST", None)
if CONTROL_HOST is None:
    raise Exception("CONTROL_HOST env var is not set")


# Configura el primer logger: solo errores en un archivo
logger.add(
    "/var/log/worker/error_log.log", level="ERROR", rotation="10 MB", retention="7 days"
)

# Configura el segundo logger: mensajes normales en la consola
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
)


# Configura el tercer logger: todos los niveles en un archivo separado (completamente independiente)
FileHandler("/var/log/worker/full_log.log").push_application()
custom_logger = Logger("wyoloservice")


pipeline = Pipeline()
pipeline.set_steps(
    [
        (read_user_config, "read_config", "v1.0"),
        (Start_inform(), Start_inform.__NAME__, Start_inform.__VERSION__),
        (OptunaOptimize(), OptunaOptimize.__NAME__, OptunaOptimize.__VERSION__),
        (results_up_to_minio, "result_to_minio", "v1.0"),
    ]
)


@hydra.main(config_path="/app", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    global DEFAULT_CONFIG

    cfg.mlflow.MLFLOW_TRACKING_URI = cfg.mlflow.MLFLOW_TRACKING_URI.replace(
        "localhost", CONTROL_HOST
    )

    cfg.minio.MINIO_ENDPOINT = cfg.minio.MINIO_ENDPOINT.replace(
        "localhost", CONTROL_HOST
    )

    cfg.redis.REDIS_HOST = cfg.redis.REDIS_HOST.replace("localhost", CONTROL_HOST)

    mlflow.set_tracking_uri(cfg.mlflow.MLFLOW_TRACKING_URI)
    logger.info(f"MLflow URI: {cfg.mlflow.MLFLOW_TRACKING_URI}")
    logger.info(f"__VERSION__: {__VERSION__}")

    # convertir a dict
    cfg = OmegaConf.to_container(cfg, resolve=True)

    DEFAULT_CONFIG.update(cfg)

    MinioS3Client(
        endpoint_url=cfg.get("minio", {}).get("MINIO_ENDPOINT"),
        aws_access_key_id=cfg.get("minio", {}).get("MINIO_ID"),
        aws_secret_access_key=cfg.get("minio", {}).get("MINIO_SECRET_KEY"),
    )

    redis_config = cfg.get("redis", {})

    queue_manager = RedisQueueManager(
        host=redis_config.get("REDIS_HOST"),
        port=redis_config.get("REDIS_PORT"),
        db=redis_config.get("REDIS_DB"),
    )

    token_manager = RedisTokenManager(
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
    public_topic = redis_config.get("TOPIC")
    private_topic = os.environ.get("USER", None)

    results_queue = redis_config.get("RESULT_TOPIC", redis_config.get("TOPIC"))

    logger.info(f"Public topic: {public_topic}")
    logger.info(f"Private topic: {private_topic}")
    logger.info(f"Results queue: {results_queue}")
    logger.info(f"Debug mode: {DEBUG_MODE}")
    logger.info(f"DEFAULT_CONFIG: {DEFAULT_CONFIG}")

    @ejecutar_en_hilo
    def health():
        worker_metadata = [
            "DEBUG",
            "USER",
            "WORKER_HOST",
            "WORKER_HOSTNAME",
            "WORKER_OS",
            "WORKER_KERNEL_VERSION",
            "WORKER_CPU_CORES",
            "WORKER_GATEWAY",
            "WORKER_NETWORK_INTERFACE",
            "WORKER_DOCKER_VERSION",
            "WORKER_APP_BASE_PATH",
            "WORKER_APP_ENV",
            "WORKER_HOME_DIR",
            "WORKER_CURRENT_DATE",
            "WORKER_CURRENT_TIME",
            "WORKER_GPU_COUNT",
            "WORKER_GPU_MODEL",
            "WORKER_GPU_MEMORY",
        ]
        while True:
            metadata = {
                other_metadata: os.environ.get(other_metadata, None)
                for other_metadata in worker_metadata
            }
            metadata["__VERSION__"] = __VERSION__

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

            time.sleep(20)

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

    if private_topic:

        @queue_manager.on_message(private_topic)
        def private_worker(task_data: dict):
            # process_requests devuelve None cuando el semaforo esta ocupado
            results = process_requests(task_data)

            if results is None or shared_resource.elapsedtime() < 10:
                recreate_request(topic=private_topic, args_dict=task_data)
            else:
                complete_requests(results)

    @queue_manager.on_message(public_topic)
    def public_worker(task_data: dict):

        # si se ha levantado el worker en modo debug no se procesan las peticiones publicas
        if DEBUG_MODE:
            recreate_request(topic=public_topic, args_dict=task_data)
        else:
            # process_requests devuelve None cuando el semaforo esta ocupado
            results = process_requests(task_data)

            if results is None:
                recreate_request(topic=public_topic, args_dict=task_data)
            else:
                complete_requests(results)

    health()
    queue_manager.start()
    queue_manager.wait()


if __name__ == "__main__":
    main()
