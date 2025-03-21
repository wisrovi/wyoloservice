import os
import sys
import tempfile
import time
import traceback

import hydra
import mlflow
from logbook import FileHandler, Logger
from loguru import logger
from omegaconf import OmegaConf
from wpipe.pipe import Pipeline
from wredis.queue import RedisQueueManager
from wredis.token import RedisTokenManager

from states import DEFAULT_CONFIG, OptunaOptimize, read_user_config, results_up_to_minio

from worker_utils import MinioS3Client, ejecutar_en_hilo


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
        verbose=False
    )

    queue_topic = os.environ.get("debug", redis_config.get("TOPIC"))
    results_queue = os.environ.get("redis", {}).get(
        "RESULT_TOPIC", redis_config.get("TOPIC")
    )

    @ejecutar_en_hilo
    def health():
        worker_metadata = [
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

            redis_key = f"workers:{metadata['WORKER_HOST']}:{metadata['USER']}"

            token_manager.write_token(
                token=redis_key,
                data=metadata,
                ttl=30,
            )

            time.sleep(20)

    @queue_manager.on_message(queue_topic)
    def worker(task_data: dict):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                task_data["tempfile"] = temp_dir
                results = pipeline.run(task_data)
        except Exception as e:
            traceback.print_exc()

            token_manager.write_token(token=task_data["task_id"], data=e.args[0])

            return

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

    health()
    queue_manager.start()
    queue_manager.wait()


if __name__ == "__main__":
    main()
