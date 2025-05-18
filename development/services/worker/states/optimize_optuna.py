import math
import os
import shutil
import tempfile
from functools import wraps

import optuna
import torch
import yaml
from loguru import logger
from tqdm import tqdm
from train_yolo import train, train_run
from ultralytics import YOLO


class OptunaOptimize:
    """Optimizer for model training using Optuna hyperparameter optimization.

    This class provides methods for searching the best model based on hyperparameter
    optimization, converting the best model to ONNX format, and includes decorators for
    configuration loading and GPU memory cleaning.

    Attributes:
        __VERSION__ (str): Version of the optimizer.
    """

    __VERSION__ = "0.1.0"
    __NAME__ = "model_train"
    DEBUG_MODE = os.environ.get("debug", None)
    verbose = False

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def search_best_model(self, request_config_user: dict):
        """Search for the best model based on hyperparameter optimization.

        Creates an Optuna study and optimizes a training objective that loads a YAML configuration,
        applies hyperparameter suggestions, and runs training. The best model is saved and its path is returned.

        Args:
            request_config_user (dict): Dictionary containing user configuration.
                Expected keys include:
                    - "sweeper": dict with sweep settings (e.g., "study_name", "n_trials", "direction").
                    - "config_path": path to the YAML configuration file.
                    - "type": type of model training.
                    - "task_id": identifier for the training task.

        Returns:
            tuple: A tuple containing:
                - best_trial: The best Optuna trial.
                - best_params: Dictionary of best hyperparameters.
                - result_path: Path where trial history and model outputs are saved.
            Returns (None, None, None) if no valid trial is found.
        """
        sweeper_config = request_config_user.get("sweeper", {})

        tempfolder = request_config_user["tempfile"]

        task_id = request_config_user.get("task_id")

        result_path = (
            f'{tempfolder}/models/{sweeper_config.get("study_name", "default_study")}/'
            f'{request_config_user["type"]}/{request_config_user["task_id"]}'
        )
        os.makedirs(f"{result_path}/trail_history/", exist_ok=True)

        def progress_bar_generator():
            with tqdm(
                total=sweeper_config.get("n_trials", 10),
                desc="Searching for best hyperparameters",
                dynamic_ncols=True,
            ) as progress_bar:
                for _ in range(sweeper_config.get("n_trials", 10)):
                    progress_bar.update(1)
                    yield progress_bar, None

        @OptunaOptimize.clean_gpu
        @OptunaOptimize.load_train_config(
            config_path=request_config_user.get("config_path")
        )
        def objective(trial, config=None):
            """Objective function for hyperparameter optimization.

            Loads the training configuration, applies hyperparameter suggestions, and executes the training run.

            Args:
                trial: An Optuna trial object.
                config (dict, optional): The training configuration loaded from the YAML file.

            Returns:
                float: The performance metric produced by the training run.

            Raises:
                optuna.TrialPruned: If the training metric is invalid or an error occurs.
            """

            progress_bar, task = next(progress_bar_generator())

            if task:
                progress_bar.update(
                    task,
                    description=f"[cyan]Trial {trial.number}",
                )
            else:
                progress_bar.write(f"[cyan]Trial {trial.number}")

            try:
                # Create a temporary directory to store a temporary config file
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_config_path = os.path.join(temp_dir, "config.yaml")
                    config["trial_number"] = int(trial.number)
                    config["tempfile"] = tempfolder

                    with open(temp_config_path, "w") as yaml_file:
                        yaml.dump(config, yaml_file)

                    # borrar
                    OptunaOptimize.DEBUG_MODE = False

                    task_id = config.get("task_id")
                    stop_training_file = f"/config/stop_training_{task_id}.txt"
                    if os.path.exists(stop_training_file):
                        raise optuna.TrialPruned("Condition to stop training met.")

                    # Run the training process
                    if OptunaOptimize.DEBUG_MODE:
                        metric = None
                        request_config = train.callback(
                            config_path=temp_config_path,
                            fitness=config.get("sweeper", {}).get("fitness", "fitness"),
                            trial_number=int(trial.number),
                        )
                        metric = request_config["train"]["results"][
                            config.get("sweeper", {}).get("fitness", "fitness")
                        ]
                    else:
                        metric = train_run(
                            config_path=temp_config_path,
                            trial_number=int(trial.number),
                            verbose=self.verbose,
                            fitness=config.get("sweeper", {}).get("fitness", "fitness"),
                        )
                        
                    if os.path.exists(stop_training_file):
                        return 0

                    best_model = f"{result_path}/{int(trial.number)}/train_{config['task_id']}/weights/best.pt"
                    if os.path.exists(best_model):
                        OptunaOptimize.copy_file(
                            origin_path=best_model,
                            destiny_path=f"{result_path}/trail_history/trial_{int(trial.number)}.pt",
                        )

                    if (
                        metric is None
                        or not isinstance(metric, float)
                        or math.isnan(metric)
                    ):
                        raise optuna.TrialPruned("Insufficient performance.")

                return metric
            except Exception as e:
                logger.error(f"Error executing objective: {e}")
                raise optuna.TrialPruned("Error executing objective.")

        study = optuna.create_study(
            direction=sweeper_config.get("direction", "minimize"),
            study_name=sweeper_config.get("study_name", "default_study"),
            sampler=getattr(
                optuna.samplers, sweeper_config.get("sampler", "TPESampler")
            )(),
        )
        study.optimize(objective, n_trials=sweeper_config.get("n_trials", 10))

        # Stop training if the stop file exists
        # This is a workaround to stop the training process
        # when the user sends a stop signal
        # to the worker
        stop_training_file = f"/config/stop_training_{task_id}.txt"
        if os.path.exists(stop_training_file):
            os.remove(stop_training_file)
            logger.info("Training stopped successfully.")

        try:
            best_trial = study.best_trial
            best_params = study.best_params
            best_metric = best_trial.value  # Obtener la mejor métrica

            return best_trial, best_params, best_metric, result_path
        except Exception:
            return None, None, None, None

    def onnx_convert(self, best_model_path: str, imgsz: int):
        """Convert a PyTorch model to ONNX format.

        Loads the best model and exports it to ONNX with the given image size.

        Args:
            best_model_path (str): Path to the best model (.pt file).
            imgsz (int): Image size for the model conversion.

        Returns:
            str: Path to the converted ONNX model, or None if conversion fails.
        """

        model_task = None
        try:
            onnx_path = best_model_path.replace("pt", "onnx")
            model_loaded = YOLO(best_model_path)
            model_task = model_loaded.task

            try:
                model_loaded.export(
                    format="onnx",
                    opset=9,
                    dynamic=False,
                    imgsz=imgsz,
                )
                return onnx_path, model_task
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error exporting model: {str(e)}")
        except Exception as e:
            if self.verbose:
                logger.error(f"Error loading model for conversion: {str(e)}")
        return None, model_task

    def __call__(self, request_config_user: dict):
        """Execute the optimization process and convert the best model.

        Searches for the best hyperparameters, retrieves the corresponding best model,
        and converts it to ONNX format.

        Args:
            request_config_user (dict): Dictionary containing user configuration.

        Returns:
            dict: A dictionary with training results including:
                - best_trial: The best Optuna trial.
                - best_model_path: Path to the best model (.pt file).
                - result_path: Path where outputs are saved.
                - imgsz: Image size used.
                - onnx_path: Path to the converted ONNX model.
            Returns an empty dictionary if the training process fails.
        """

        tempfolder = request_config_user["tempfile"]
        config_path = f"{tempfolder}/{request_config_user['task_id']}.yaml"

        request_config_user["config_path"] = config_path

        best_trial, best_params, best_metric, result_path = self.search_best_model(
            request_config_user
        )

        if best_trial and best_params and best_metric:
            best_model_path = (
                f"{result_path}/trail_history/trial_{best_trial.number}.pt"
            )
            model_converted, model_task = self.onnx_convert(
                best_model_path, imgsz=int(best_params["imgsz"])
            )

            return {
                "train": {
                    "best_trial": best_trial,
                    "best_model_path": best_model_path,
                    "best_metric": best_metric,
                    "result_path": result_path,
                    "imgsz": int(best_params["imgsz"]),
                    "onnx_path": model_converted,
                    "model_task": model_task,
                }
            }
        else:
            logger.error("Error during training process.")
            return {}

    @staticmethod
    def load_train_config(config_path=None):
        """Decorator factory to load and update the training configuration from a YAML file.

        The decorator reads the configuration from the specified file, applies hyperparameter suggestions,
        and injects the updated configuration into the decorated function.

        Args:
            config_path (str, optional): Path to the YAML configuration file.

        Returns:
            function: A decorator that injects the configuration into the wrapped function.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if config_path:
                    logger.info(f"Loading configuration from: {config_path}")
                    try:
                        with open(config_path, "r") as f:
                            config = yaml.safe_load(f)

                        sweeper_config = config.get("sweeper", {})
                        param_config = sweeper_config.get("search_space", {})

                        trial = args[0]
                        if sweeper_config.get("algorithm", "optuna") == "optuna":
                            suggested_params, model = (
                                OptunaOptimize.get_optuna_suggestions(
                                    trial, param_config
                                )
                            )
                            if model:
                                config["model"] = model
                        else:
                            suggested_params = OptunaOptimize.get_ray_suggestions(
                                trial, sweeper_config
                            )

                        config["train"].update(suggested_params)
                        config["experiment_name"] = sweeper_config["study_name"]

                        kwargs["config"] = config
                    except FileNotFoundError:
                        logger.error(f"Configuration file not found: {config_path}")
                        return None
                    except yaml.YAMLError as e:
                        logger.error(f"Error loading YAML file: {e}")
                        return None
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Error in function {func.__name__}: {e}")
                    raise

            return wrapper

        return decorator

    @staticmethod
    def clean_gpu(func):
        """Decorator to clean GPU memory before function execution.

        Calls torch.cuda.empty_cache() to free GPU memory prior to running the wrapped function.

        Args:
            func (function): The function to be decorated.

        Returns:
            function: The wrapped function with GPU memory cleaning.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            torch.cuda.empty_cache()
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def get_optuna_suggestions(trial, sweeper_config):
        """Generate hyperparameter suggestions using Optuna.

        Recursively traverses the provided search space configuration to generate suggestions.

        Args:
            trial: An Optuna trial object.
            sweeper_config (dict): Configuration for the sweeper that should include the search space.

        Returns:
            tuple: A tuple (suggestions, model) where suggestions is a dict of suggested parameters
                   and model is the selected model if provided.

        Raises:
            ValueError: If a parameter type is not supported.
        """
        model = None
        models = sweeper_config.get("model", None)
        if models:
            options = models[1:]
            model = trial.suggest_categorical("models", options)

        train_config = sweeper_config.get("train", {})
        suggestions = {}
        for key, value in train_config.items():
            if isinstance(value, dict):
                suggestions[key] = OptunaOptimize.get_optuna_suggestions(trial, value)[
                    0
                ]
            elif value[0] == "range":
                start, stop, step = value[1:]
                suggestions[key] = trial.suggest_int(key, start, stop, step=step)
            elif value[0] == "choice":
                options = value[1:]
                suggestions[key] = trial.suggest_categorical(key, options)
            elif value[0] == "loguniform":
                low, high = value[1:]
                low, high = float(low), float(high)
                if low >= high:
                    raise ValueError(
                        f"'low' value must be less than 'high' (low={low}, high={high})"
                    )
                suggestions[key] = trial.suggest_loguniform(key, low, high)
            else:
                raise ValueError(f"Unsupported parameter type: {value}")
        return suggestions, model

    @staticmethod
    def copy_file(origin_path: str, destiny_path: str):
        """Copy a file from the source path to the destination path.

        Args:
            origin_path (str): Source file path.
            destiny_path (str): Destination file path.
        """
        try:
            shutil.copy2(origin_path, destiny_path)
            print(f"Archivo copiado de '{origin_path}' a '{destiny_path}'")
        except FileNotFoundError:
            print(f"Error: El archivo '{origin_path}' no existe.")
        except PermissionError:
            print(
                f"Error: No tienes permisos para copiar el archivo a '{destiny_path}'."
            )
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")
