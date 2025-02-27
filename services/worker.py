import json
import math
import sys
import tempfile
import traceback

import hydra
import mlflow
import optuna

# import ray
import yaml
from api.database import TrainingHistory, db
from api.redis_queue import TOPIC, queue_manager
from logbook import FileHandler, Logger
from loguru import logger
from omegaconf import OmegaConf

# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.search.optuna import OptunaSearch
from worker_utils import (
    MinioS3Client,
    clean_gpu,
    get_ray_suggestions,
    load_train_config,
    train_yolo,
)

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


DEFAULT_CONFIG = {}


# Función para actualizar y completar final_config
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
    from copy import deepcopy

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


def save_best_model(
    task_id,
    model_path,
    version,
    results_model,
    project_name,
    s3: MinioS3Client,
):
    """Registra el mejor modelo en la base de datos y lo sube a MinIO."""

    update_model = True

    # TODO: falta validar el modelo anterior y compararlo con el modelo actual, si es mejor, se reemplaza, de lo contrario solo se suben los resultados del modelo actual

    # if bucket_exists(bucket_name):
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         old_files = list_objects(bucket_name)

    #         temp_file_path = os.path.join(temp_dir, "old.pt")

    #         if data_path and old_files and f"{project_name}.pt" in old_files:
    #             old_model = download_file(
    #                 bucket_name, f"{project_name}.pt", temp_file_path
    #             )

    #             better_model = comparar_modelos_yolo(
    #                 modelo1_path=old_model,
    #                 modelo2_path=model_path,
    #                 data_path=data_path,
    #             )

    #             if better_model == "old.pt":
    #                 update_model = False

    if update_model:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(f"{temp_dir}/{project_name}-v{version}.txt", "w") as f:
                f.write(f"{MinioS3Client.BUCKET_NAME}/{project_name}/{task_id}/*")

            s3.upload_file(
                file_path_local=f"{temp_dir}/{project_name}-v{version}.txt",
                bucket_name=f"better-{MinioS3Client.BUCKET_NAME}",
                name_file_s3=f"{project_name}-v{version}.txt",
            )

            minio_url = s3.upload_file(
                file_path_local=model_path,
                bucket_name=f"better-{MinioS3Client.BUCKET_NAME}",
                name_file_s3=f"{project_name}-v{version}.pt",
            )

        try:
            # TODO: validar que la actualizacion a la base de datos se haga correctamente

            # Verificar si el `task_id` existe en la base de datos
            existing_task = db.get_by_field(task_id=task_id)

            db.update(
                1,
                TrainingHistory(
                    id=existing_task,
                    task_id=f"{task_id}_v{version}",
                    model_path=minio_url,
                    status="completed",
                    recommended_model=minio_url,
                ),
            )
        except Exception as e:
            logger.error(f"❌ Error al actualizar la base de datos: {e}")

    s3.upload_folder(
        folder_path=results_model,
        bucket_name=f"{MinioS3Client.BUCKET_NAME}",
        prefix=f"{project_name}/{task_id}/",
    )

    existing_task = db.get_by_field(task_id=task_id)
    if existing_task is None:
        logger.warning(
            f"⚠️ Advertencia: No se encontró task_id {task_id} en la base de datos."
        )
        return

    logger.info(f"✅ Modelo recomendado guardado en MinIO: {minio_url}")


def process_train_with_ray_tuner(request_config_user, task_id):
    """
    Realiza la optimización de hiperparámetros utilizando Ray Tune.

    Args:
        request_config_user (dict): Configuración proporcionada por el usuario.
        task_id (str): Identificador de la tarea.

    Returns:
        tuple: Mejor trial, ruta del mejor modelo y ruta de resultados.
    """
    sweeper_config = request_config_user.get("sweeper", {})

    # Espacio de búsqueda de hiperparámetros
    search_space = sweeper_config.get("search_space", {})

    # Convertir el espacio de búsqueda a un formato compatible con Ray Tune
    search_space = get_ray_suggestions(None, sweeper_config)

    @load_train_config(config_path=request_config_user.get("config_path"))
    def objective(config):
        """
        Función objetivo que entrena el modelo y reporta la métrica.

        Args:
            config (dict): Conjunto de hiperparámetros a probar.
        """
        try:
            # Actualizar la configuración con el task_id si es necesario
            config["task_id"] = task_id

            # Entrenar el modelo con los hiperparámetros actuales
            train_result = train_yolo(config)

            # Validar el resultado del entrenamiento
            if not isinstance(train_result, dict) or math.isnan(
                train_result.get("metric", float("nan"))
            ):
                raise ValueError("Rendimiento insuficiente.")

            # Obtener la métrica principal
            metric = train_result.get("metric", float("nan"))

            # Reportar la métrica a Ray Tune
            tune.report(metric=metric)

        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            traceback.print_exc()
            tune.report(
                metric=float("inf")
            )  # Reportar un valor alto para descartar este trial

    # Inicializar Ray
    ray.init(ignore_reinit_error=True)

    MODE = {
        "minimize": "min",
        "maximize": "max",
    }

    try:
        mode = MODE[sweeper_config.get("direction", "minimize")]

        # Configurar el scheduler (por ejemplo, ASHA para detener trials poco prometedores)
        scheduler = ASHAScheduler(
            metric="metric",  # Métrica a optimizar
            mode=mode,  # Dirección de optimización
            max_t=sweeper_config.get(
                "max_t", 100
            ),  # Número máximo de epochs o iteraciones
            grace_period=sweeper_config.get(
                "grace_period", 5
            ),  # Período mínimo antes de detener un trial
        )

        # Configurar el algoritmo de búsqueda (usando OptunaSearch como ejemplo)
        search_alg = OptunaSearch(
            metric="metric",
            mode=mode,
            space=search_space,
        )
        search_alg = ConcurrencyLimiter(
            search_alg, max_concurrent=sweeper_config.get("max_concurrent", 4)
        )

        # Ejecutar la optimización
        # TODO: revisar, aun no se logra ejecutar correctamente el tune.run, no entra a la funcion objetivo
        analysis = tune.run(
            objective,
            config=search_space,
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=sweeper_config.get("n_trials", 1),
            resources_per_trial={
                "cpu": sweeper_config.get("cpu_per_trial", 1),
                "gpu": sweeper_config.get("gpu_per_trial", 0),
            },
        )

        # Obtener el mejor trial
        best_trial = analysis.get_best_trial(
            "metric", sweeper_config.get("direction", "minimize")
        )

        # Construir las rutas de resultados
        result_path = f'/models/{sweeper_config.get("study_name", "default_study")}/{request_config_user["type"]}/{request_config_user["task_id"]}'
        best_model_path = f"{result_path}/trail_history/trial_{best_trial.trial_id}.pt"

        return best_trial, best_model_path, result_path

    except Exception as e:
        logger.error(f"❌ Error al procesar el entrenamiento: {e}")
        traceback.print_exc()
        return None, None, None
    finally:
        # Asegurarse de detener Ray al finalizar
        ray.shutdown()


def process_train_with_optuna(request_config_user, task_id):
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

        return best_trial, best_model_path, result_path
    except:
        traceback.print_exc()
        logger.error("❌ Error al procesar el entrenamiento.")
        return


@queue_manager.on_message(TOPIC)
def worker(task_data: dict):
    task_data = json.loads(task_data)

    logger.debug(f"📥 Nueva tarea recibida: {task_data}")
    if "task_id" not in task_data or "config_path" not in task_data:
        logger.error(
            f"⚠️ La tarea recibida no tiene la estructura esperada: {task_data}"
        )
        return

    config_path = task_data["config_path"]

    # Leer el archivo YAML del usuario
    config_path = task_data["config_path"]
    try:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)  # Convertir YAML a dict

            # 🚨 Eliminar `defaults` si existe
            user_config.pop("defaults", None)

            # 🚨 Fusionar con la configuración base
            final_config = DEFAULT_CONFIG.copy()

            # actualizar final_config y añadir los campos que le faltan de DEFAULT_CONFIG
            final_config = merge_configs(DEFAULT_CONFIG, user_config)

    except Exception as e:
        logger.error(f"❌ Error al cargar YAML ({config_path}): {e}")

    final_config["config_path"] = config_path
    final_config["task_id"] = task_data["task_id"]

    try:
        with open(config_path, "w") as archivo:
            yaml.dump(final_config, archivo, default_flow_style=False)
    except Exception as e:
        pass

    try:
        if final_config.get("sweeper", {}).get("algorithm", "optuna") == "ray_tune":
            best_trial, best_model_path, RESULT_PATH = process_train_with_ray_tuner(
                final_config, task_data["task_id"]
            )
        else:
            best_trial, best_model_path, RESULT_PATH = process_train_with_optuna(
                final_config, task_data["task_id"]
            )

    except Exception as e:
        logger.error(f"❌ Error al procesar el entrenamiento: {e}")

    try:
        s3: MinioS3Client = MinioS3Client(
            endpoint_url=final_config.get("minio", {}).get("MINIO_ENDPOINT"),
            aws_access_key_id=final_config.get("minio", {}).get("MINIO_ID"),
            aws_secret_access_key=final_config.get("minio", {}).get("MINIO_SECRET_KEY"),
        )

        sweeper_config = final_config.get("sweeper", {})
        save_best_model(
            task_id=task_data["task_id"],
            project_name=sweeper_config.get("study_name", "default_study"),
            model_path=best_model_path,
            version=sweeper_config["version"],
            results_model=f"{RESULT_PATH}/{best_trial.number}/",
            s3=s3,
        )
    except Exception as e:
        logger.error(f"❌ Error al guardar el mejor modelo: {e}")


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
