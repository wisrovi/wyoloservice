import os
from train_yolo import train
import hydra
from omegaconf import OmegaConf
import yaml
from copy import deepcopy


CONTROL_HOST = os.getenv("CONTROL_HOST", None)
if CONTROL_HOST is None:
    raise Exception("CONTROL_HOST env var is not set")


def merge_configs(default_config, user_config):
    """
    Fusiona dos configuraciones: user_config tiene prioridad sobre default_config.
    Los campos faltantes en user_config se completan con los valores de default_config.

    Args:
        default_config (dict): Configuración predeterminada.
        user_config (dict): Configuración proporcionada por el usuario.

    Returns:
        dict: Configuración final fusionada.
    """
    # Crear una copia profunda de default_config para evitar modificaciones inesperadas

    final_config = deepcopy(default_config)

    # Iterar sobre las claves de user_config y actualizar final_config
    for key, value in user_config.items():
        if (
            isinstance(value, dict)
            and key in final_config
            and isinstance(final_config[key], dict)
        ):
            # Si ambas son diccionarios, fusionar recursivamente
            final_config[key] = merge_configs(final_config[key], value)
        else:
            # Sobrescribir el valor con el proporcionado por el usuario
            final_config[key] = deepcopy(value)

    return final_config


@hydra.main(config_path="/app", config_name="config", version_base=None)
def procesos(cfg: OmegaConf):
    cfg.mlflow.MLFLOW_TRACKING_URI = cfg.mlflow.MLFLOW_TRACKING_URI.replace(
        "localhost", CONTROL_HOST
    )

    cfg.minio.MINIO_ENDPOINT = cfg.minio.MINIO_ENDPOINT.replace(
        "localhost", CONTROL_HOST
    )

    cfg.redis.REDIS_HOST = cfg.redis.REDIS_HOST.replace("localhost", CONTROL_HOST)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    config_path: str = "/demo/config_train.yaml"

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    user_config = merge_configs(
        default_config=cfg,
        user_config=user_config,
    )

    with open("/demo/config.yaml", "w") as f:
        yaml.dump(user_config, f, allow_unicode=True, default_flow_style=False)

    request_config = train.callback(
        config_path="/demo/config.yaml",
        fitness="metrics/accuracy_top1",
        trial_number=1,
    )

    print(request_config)


if __name__ == "__main__":
    procesos()
