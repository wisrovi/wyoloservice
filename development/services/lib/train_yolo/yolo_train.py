import os
import uuid
from datetime import datetime

import click
import yaml

from train_yolo.trainer_wrapper import TrainerWrapper


def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_datetime():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


@click.command()
@click.option(
    "--config_path",
    type=click.Path(exists=True),
    required=True,
    default="config.yaml",
    help="Ruta al archivo de configuración.",
)
@click.option(
    "--fitness",
    type=str,
    required=True,
    default="fitness",
    help="variable de fitness a evaluar",
)
@click.option(
    "--trial_number",
    type=int,
    required=True,
    default=0,
    help="Número de la prueba.",
)
def train(config_path: str, fitness: str, trial_number: int):
    request_config = load_config(config_path=config_path)
    request_config["config_path"] = config_path

    trainer = TrainerWrapper(config=request_config)
    trainer.create_model(
        model_name=request_config["model"],
        model_type=request_config["type"],
    )

    if "task_id" not in request_config:
        request_config["task_id"] = str(uuid.uuid4())

    experiment_name = request_config.get("sweeper").get("study_name")
    tempfile = request_config.get("tempfile", "")

    RESULT_PATH = f'{tempfile}/models/{experiment_name}/{request_config["type"]}/{request_config["task_id"]}/'
    os.makedirs(f"{RESULT_PATH}/trail_history", exist_ok=True)
    request_config["path_results"] = f"{RESULT_PATH}/{trial_number}/"

    timestamp = get_datetime()
    request_config["timestamp"] = timestamp

    if request_config["train"]["batch"] > 0:
        better_batch = trainer.get_better_batch(
            batch_to_use=request_config["train"]["batch"]
        )
        if request_config["train"]["batch"] > better_batch:
            request_config["train"]["batch"] = better_batch

    request_config["train"]["project"] = f"{RESULT_PATH}/{trial_number}/"
    request_config["train"]["name"] = f"train_{request_config.get('task_id')}"
    request_config["train"]["verbose"] = True
    request_config["train"]["plots"] = True
    request_config["train"]["exist_ok"] = True

    trainer.config_train = request_config

    if "train" in request_config:
        results = trainer.train(config_train=request_config["train"])

        if results:
            request_config["experiment_type"] = str(results.task)
            request_config["train"]["results"] = results.results_dict

            try:
                print(f'ResultadoFinal:{request_config["train"]["results"][fitness]}')
            except:
                print(f'ResultadoFinal:{request_config["train"]["results"]["fitness"]}')

    return request_config


if __name__ == "__main__":
    """
    EXAMPLE:
        python yolo_train.py --config_path="/datasets/clasificacion/clasificador_arepo_perfil/config_train.yaml" --trial_number=1
    """

    request_config = train()
