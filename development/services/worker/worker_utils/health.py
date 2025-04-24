import os
import uuid
from typing import List

import uvicorn
from fastapi import FastAPI
from loguru import logger
from omegaconf import OmegaConf
from setproctitle import setproctitle  # Para cambiar el nombre del proceso
from train_yolo.trainer_wrapper import TrainerWrapper, obtener_info_gpu_json
from wredis.hash import RedisHashManager
from wredis.sortedset import RedisSortedSetManager

from worker_utils.decorators import ejecutar_en_hilo

# Cambiar el nombre del proceso
setproctitle("train_service")


app = FastAPI()
app_version = "v1.0"

CONTROL_HOST = os.getenv("CONTROL_HOST", None)
if CONTROL_HOST is None:
    raise Exception("CONTROL_HOST env var is not set")


DEBUG_MODE = os.environ.get("debug", None)
hash_manager = None
sorted_set_manager = None


@app.get("/")
async def read_version():

    global hash_manager
    global sorted_set_manager

    """Devuelve la versión de la API."""

    metadata = {
        other_metadata: os.environ.get(other_metadata, None)
        for other_metadata in TrainerWrapper.worker_metadata
    }
    metadata["__VERSION__"] = app_version
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
    # if gpu_0_memoryFree > (gpu_0_memoryTotal * NOT_TO_USE_GPU):
    #     gpu_0_memoryFree = gpu_0_memoryTotal * NOT_TO_USE_GPU
    # else:
    #     available = gpu_0_memoryTotal - (gpu_0_memoryTotal * NOT_TO_USE_GPU)
    #     gpu_0_memoryFree = max(available - gpu_0_memoryUsed, 0)

    available = gpu_0_memoryTotal - (gpu_0_memoryTotal * NOT_TO_USE_GPU)
    gpu_0_memoryFree = min(available, gpu_0_memoryFree)

    member_name = f'{metadata["WORKER_HOST"]} ({metadata["USER"]})'
    if os.environ.get("debug", None):
        member_name += "[debug]"
    member_name += f" -> {app_version}"

    try:
        sorted_set_manager.redis_client.ping()
    except Exception as e:
        pass
        # Detener la aplicación si ocurre un error
        # os._exit(1)  # Fuerza la salida del proceso

    sorted_set_manager.add_to_sorted_set(
        key="available",
        score=int(gpu_0_memoryFree),
        member=member_name,
        ttl=30,
    )

    return {"version": app_version}


@ejecutar_en_hilo
def health(version: str = "v1.0"):
    """
    Función para iniciar el servicio de salud.
    Args:
        version (str): Versión de la API.
    """

    global app_version, hash_manager, sorted_set_manager

    app_version = version

    with open("/config/config.yaml", "r") as f:
        config = f.read()
        DEFAULT_CONFIG = OmegaConf.create(config)

    redis_config = DEFAULT_CONFIG.get("redis", {})

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

    logger.info("Starting health check...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    logger.info(f"Health check started with version {version}.")
