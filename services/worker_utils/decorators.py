from functools import wraps

import torch
import yaml
from loguru import logger
# from ray import tune


def catch_errors(func):
    """
    Decorador para capturar errores y registrarlos en un archivo usando Loguru.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Ejecuta la función
            return func(*args, **kwargs)
        except Exception as e:
            # Registra el error en el archivo de logs
            logger.error(f"Error en la función {func.__name__}: {str(e)}")
            # Opcional: relanza la excepción si quieres que el programa falle
            # raise

    return wrapper


def clean_gpu(func):
    """
    Decorador para capturar errores y registrarlos en un archivo usando Loguru.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()  # Libera caché de GPU
        return func(*args, **kwargs)

    return wrapper


def get_optuna_suggestions(trial, sweeper_config):
    sweeper_config = sweeper_config["train"]

    suggestions = {}
    for key, value in sweeper_config.items():
        if isinstance(value, dict):  # Si es un diccionario, procesa recursivamente
            suggestions[key] = get_optuna_suggestions(trial, value)
        elif value[0] == "range":  # Rango de enteros
            start, stop, step = value[1:]
            suggestions[key] = trial.suggest_int(key, start, stop, step=step)
        elif value[0] == "choice":  # Opciones discretas
            options = value[1:]
            suggestions[key] = trial.suggest_categorical(key, options)
        elif value[0] == "loguniform":  # Distribución log-uniforme
            low, high = value[1:]
            low, high = float(low), float(high)
            if low >= high:
                raise ValueError(
                    f"El valor 'low' debe ser menor que 'high' (low={low}, high={high})"
                )
            suggestions[key] = trial.suggest_loguniform(key, low, high)
        else:
            raise ValueError(f"Tipo de parámetro no soportado: {value}")
    return suggestions


def get_ray_suggestions(trial, sweeper_config):
    search_space = {}

    # Convertir el espacio de búsqueda
    for section, params in sweeper_config.get("search_space", {}).items():
        search_space[section] = {}
        for param, value in params.items():
            distribution, *args = value
            if distribution == "loguniform":
                search_space[section][param] = tune.loguniform(*args)
            elif distribution == "choice":
                search_space[section][param] = tune.choice(args[0])
            elif distribution == "uniform":
                search_space[section][param] = tune.uniform(*args)
            else:
                raise ValueError(f"Distribución no soportada: {distribution}")


def load_train_config(config_path=None):
    """
    Decorador de fábrica para capturar errores y registrarlos en un archivo usando Loguru.
    Permite pasar argumentos al decorador.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if config_path:
                logger.info(f"Cargando configuración desde: {config_path}")
                try:
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)

                    sweeper_config = config.get("sweeper", {})
                    param_config = sweeper_config.get("search_space", {})

                    # Obtener sugerencias de hiperparámetros desde el archivo de configuración
                    trial = args[0]

                    if sweeper_config["algorithm"] == "optuna":
                        suggested_params = get_optuna_suggestions(trial, param_config)
                    else:
                        suggested_params = get_ray_suggestions(trial, sweeper_config)

                    # Actualizar la configuración con las sugerencias
                    config["train"].update(suggested_params)
                    config["experiment_name"] = sweeper_config["study_name"]

                    kwargs["config"] = config
                except FileNotFoundError:
                    logger.error(
                        f"Archivo de configuración no encontrado: {config_path}"
                    )
                    return None  # O manejar el error como sea apropiado.
                except yaml.YAMLError as e:
                    logger.error(f"Error al cargar el archivo YAML: {e}")
                    return None  # O manejar el error como sea apropiado.

            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Error en la función {func.__name__}: {e}")
                raise  # Re-lanzar la excepción para que se propague

        return wrapper

    return decorator
