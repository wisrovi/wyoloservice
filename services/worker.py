import json
import sys
import traceback

import hydra
import mlflow
from api.redis_queue import TOPIC, queue_manager
from logbook import FileHandler, Logger
from loguru import logger
from omegaconf import OmegaConf
from worker_utils import (
    DEFAULT_CONFIG,
    process_train_with_optuna,
    read_user_config,
    results_up_to_minio,
)
from wpipe.pipe import Pipeline

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


pipeline = Pipeline()
pipeline.set_steps(
    [
        (read_user_config, "read_config", "v1.0"),
        (process_train_with_optuna, "model_train", "v1.0"),
        (results_up_to_minio, "result_to_minio", "v1.0"),
    ]
)


@queue_manager.on_message(TOPIC)
def worker(task_data: dict):
    try:
        task_data = json.loads(task_data)
    except:
        pass

    try:
        pipeline.run(task_data)
    except Exception as e:
        traceback.print_exc()


@hydra.main(config_path="/app", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    global DEFAULT_CONFIG

    mlflow.set_tracking_uri(cfg.mlflow.MLFLOW_TRACKING_URI)
    logger.info(f"MLflow URI: {cfg.mlflow.MLFLOW_TRACKING_URI}")

    # convertir a dict
    cfg = OmegaConf.to_container(cfg, resolve=True)

    DEFAULT_CONFIG.update(cfg)

    queue_manager.start()
    queue_manager.wait()


if __name__ == "__main__":
    main()
