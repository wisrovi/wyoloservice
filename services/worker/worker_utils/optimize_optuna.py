import math
import queue

from ultralytics import YOLO
import mlflow
import torch
import traceback

import optuna
from loguru import logger
from worker_utils import clean_gpu, load_train_config, train_yolo


TRADITIONAL_METHOD_TO_USE = False


def process_train_with_optuna(request_config_user: dict):
    sweeper_config = request_config_user.get("sweeper", {})
    to_process_queue = request_config_user["to_process_queue"]
    to_thread_queue = request_config_user["to_thread_queue"]

    @clean_gpu
    @load_train_config(config_path=request_config_user.get("config_path"))
    def objective(trial, config=None):

        if TRADITIONAL_METHOD_TO_USE:
            # metodo tradicional
            response = train_yolo(config, trial_number=trial.number)
        else:
            # metodo usando colas, esto se usa puesto que inicialmente la funcion "process_train_with_optuna"
            # ha sido invocado desde un hilo (por el hilo de recepcion de datos de redis_queue)
            # y dado que "train_yolo" crea hilos y procesos de entrenamiento de YOLO (ultralytics)
            # algunas veces se rompe, debido a que en un hilo no es correcto crear mas hilos o procesos porque puede y de hecho falla
            # por ello el entrenamiento sucede cuando se pone algo en la cola "to_process_queue", que se ejecuta en un proceso paralelo del hilo principal
            # este ejecuta el entrenamiento y pone los resultados del mismo en "to_thread_queue"
            # los cuales son esperados y capturados por "objective" de "process_train_with_optuna" para continuar con el proceso de optuna

            # ejecutar el entrenamiento en un proceso independiente
            to_process_queue.put((config, trial.number))

            # Espera la respuesta del entrenamiento
            response = to_thread_queue.get()

        # procesar los resultados del entrenamiento
        if not isinstance(response, dict) and math.isnan(response):
            raise optuna.TrialPruned("Rendimiento insuficiente.")

        metric = response.get("metric", float("nan"))

        return metric

    study = optuna.create_study(
        direction=sweeper_config.get("direction", "minimize"),
        study_name=sweeper_config.get("study_name", "default_study"),
        sampler=getattr(optuna.samplers, sweeper_config.get("sampler", "TPESampler"))(),
    )
    study.optimize(objective, n_trials=sweeper_config.get("n_trials", 10))

    try:
        best_trial = study.best_trial

        result_path = f'/models/{sweeper_config.get("study_name", "default_study")}/{request_config_user["type"]}/{request_config_user["task_id"]}'
        best_model_path = f"{result_path}/trail_history/trial_{best_trial.number}.pt"

        # try:
        #     model = YOLO(best_model_path)
        #     mlflow.pytorch.log_model(model, "model")
        #     # Convert to ONNX first
        #     model.export(format='onnx', path='model.onnx')

        #     # Then log the ONNX model to MLflow
        #     mlflow.log_artifact('model.onnx', artifact_path="models")
        # except:
        #     logger.error("Can't upload model to mlflow")

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
