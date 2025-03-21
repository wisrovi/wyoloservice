import os
from train_yolo import train
from states.read_user_config import merge_configs
import hydra
from omegaconf import OmegaConf
import yaml
from copy import deepcopy


CONTROL_HOST = os.getenv("CONTROL_HOST", None)
if CONTROL_HOST is None:
    raise Exception("CONTROL_HOST env var is not set")


def complete_config(
    worker_config: OmegaConf, config_path: str = "/demo/config_train.yaml"
):

    worker_config.mlflow.MLFLOW_TRACKING_URI = (
        worker_config.mlflow.MLFLOW_TRACKING_URI.replace("localhost", CONTROL_HOST)
    )

    worker_config.minio.MINIO_ENDPOINT = worker_config.minio.MINIO_ENDPOINT.replace(
        "localhost", CONTROL_HOST
    )

    worker_config.redis.REDIS_HOST = worker_config.redis.REDIS_HOST.replace(
        "localhost", CONTROL_HOST
    )
    worker_config = OmegaConf.to_container(worker_config, resolve=True)

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f)

    user_config = merge_configs(
        default_config=worker_config,
        user_config=user_config,
    )


@hydra.main(config_path="/app", config_name="config", version_base=None)
def procesos(cfg: OmegaConf):

    user_config = complete_config(
        worker_config=cfg,
        config_path="/demo/config_train.yaml",
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
