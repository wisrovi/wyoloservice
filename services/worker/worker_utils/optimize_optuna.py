import math
import os
import tempfile

import mlflow
import optuna
import yaml
from loguru import logger
from train_yolo import train_run
from ultralytics import YOLO
from tqdm import tqdm

from worker_utils import clean_gpu, load_train_config


def process_train_with_optuna(request_config_user: dict):
    sweeper_config = request_config_user.get("sweeper", {})

    with tqdm(
        total=sweeper_config.get("n_trials", 10),
        desc="Buscando mejores hiperparametros",
        dynamic_ncols=True,
    ) as barra_progreso:

        @clean_gpu
        @load_train_config(config_path=request_config_user.get("config_path"))
        def objective(trial, config=None):
            barra_progreso.update(1)  # Actualiza la barra de progreso en 1 unidad

            barra_progreso.write(f"Impresión {trial.number}")  # Usamos write para imprimir

            try:
                # crear una carpeta temporal
                with tempfile.TemporaryDirectory() as temp_dir:
                    ruta_archivo = os.path.join(temp_dir, "config.yaml")
                    with open(ruta_archivo, "w") as archivo_yaml:
                        yaml.dump(config, archivo_yaml)

                    metric = train_run(
                        config_path=ruta_archivo,
                        trial_number=int(trial.number),
                        verbose=False,
                        fitness=config.get("sweeper", {}).get("fitness", "fitness"),
                    )

                    if (
                        metric is None
                        or not isinstance(metric, float)
                        or math.isnan(metric)
                    ):
                        raise optuna.TrialPruned("Rendimiento insuficiente.")

                return metric
            except Exception as e:
                logger.error(f"Error en la ejecución del objetivo: {e}")
                raise optuna.TrialPruned("Error en la ejecución del objetivo.")

        study = optuna.create_study(
            direction=sweeper_config.get("direction", "minimize"),
            study_name=sweeper_config.get("study_name", "default_study"),
            sampler=getattr(
                optuna.samplers, sweeper_config.get("sampler", "TPESampler")
            )(),
        )
        study.optimize(objective, n_trials=sweeper_config.get("n_trials", 10))

    try:
        best_trial = study.best_trial

        result_path = f'/models/{sweeper_config.get("study_name", "default_study")}/{request_config_user["type"]}/{request_config_user["task_id"]}'
        best_model_path = f"{result_path}/trail_history/trial_{best_trial.number}.pt"

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                ruta_archivo = os.path.join(temp_dir, "model.onnx")
                model = YOLO(best_model_path)
                model.export(format="onnx", path=ruta_archivo)

                mlflow.log_artifact(ruta_archivo, artifact_path="models")
        except:
            logger.error("Can't upload model to mlflow")

        return {
            "train": {
                "best_trial": best_trial,
                "best_model_path": best_model_path,
                "result_path": result_path,
            }
        }

    except:
        logger.error("❌ Error al procesar el entrenamiento.")
        return {}
