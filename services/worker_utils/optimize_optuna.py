import math

from ultralytics import YOLO
import mlflow
import torch
import traceback

import optuna
from loguru import logger
from worker_utils import clean_gpu, load_train_config, train_yolo


def process_train_with_optuna(request_config_user):
    sweeper_config = request_config_user.get("sweeper", {})

    @clean_gpu
    @load_train_config(config_path=request_config_user.get("config_path"))
    def objective(trial, config=None):

        config = train_yolo(config, trial_number=trial.number)

        if not isinstance(config, dict) and math.isnan(config):
            raise optuna.TrialPruned("Rendimiento insuficiente.")

        metric = config.get("metric", float("nan"))

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
            "train":{
                "best_trial"   :best_trial,
                "best_model_path":best_model_path,
                "result_path":result_path
            }
        }
        
    except:
        logger.error("❌ Error al procesar el entrenamiento.")
        return {}
