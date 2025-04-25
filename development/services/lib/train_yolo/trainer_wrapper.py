import json
import os
import random
import time
from datetime import datetime
from glob import glob
from typing import List

import dvc
import GPUtil
import mlflow
import mlflow.data
import mlflow.data.filesystem_dataset_source
import mlflow.data.http_dataset_source
import yaml
from loguru import logger
from PIL import Image
from slugify import slugify
from ultralytics import RTDETR, YOLO, settings
from ultralytics.utils.autobatch import autobatch
from wredis.hash import RedisHashManager


def obtener_info_gpu_json():
    """
    Obtiene información detallada sobre las GPUs disponibles y la devuelve en formato JSON.
    Maneja el caso en que la propiedad 'processes' no esté disponible.
    """
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return json.dumps({"error": "No se encontraron GPUs disponibles."})

        gpu_info = []
        for gpu in gpus:
            gpu_data = {
                # "gpu_id": gpu.id,
                f"gpu_{gpu.id}_name": gpu.name,
                f"gpu_{gpu.id}_uuid": gpu.uuid,
                f"gpu_{gpu.id}_memoryTotal": gpu.memoryTotal,
                f"gpu_{gpu.id}_memoryFree": gpu.memoryFree,
                f"gpu_{gpu.id}_memoryUsed": gpu.memoryUsed,
                f"gpu_{gpu.id}_load": gpu.load * 100,
                f"gpu_{gpu.id}_temperature": gpu.temperature,
            }
            # Verifica si la propiedad 'processes' existe antes de intentar acceder a ella.
            if hasattr(gpu, "processes"):
                gpu_data["processes"] = [
                    {
                        "pid": process.pid,
                        "name": process.name,
                        "memory": process.memoryUsed,
                    }
                    for process in gpu.processes
                ]
            else:
                gpu_data["processes"] = "Processes information not available."

            gpu_info.append(gpu_data)

        return gpu_info

    except Exception as e:
        return {"error": f"Ocurrió un error al obtener la información de la GPU: {e}"}


class StatusEDA:
    PENDING = 0
    SAVED = 2


class TrainerWrapper:
    # https://github.com/ultralytics/ultralytics/issues/8214
    config = {}
    GPU_USE = 0.6  # procentaje de uso de GPU

    is_configured = False
    model = None

    start_time = 0
    end_time = 0

    firts_epoch = True

    worker_metadata = [
        "debug",
        "USER",
        "WORKER_HOST",
        "WORKER_HOSTNAME",
        "WORKER_OS",
        "WORKER_KERNEL_VERSION",
        "WORKER_CPU_CORES",
        "WORKER_GATEWAY",
        "WORKER_NETWORK_INTERFACE",
        "WORKER_DOCKER_VERSION",
        "WORKER_APP_BASE_PATH",
        "WORKER_APP_ENV",
        "WORKER_HOME_DIR",
        "WORKER_CURRENT_DATE",
        "WORKER_CURRENT_TIME",
        "WORKER_GPU_COUNT",
        "WORKER_GPU_MODEL",
        "WORKER_GPU_MEMORY",
    ]
    hash_manager = None
    status_eda_completed = StatusEDA.PENDING

    def __init__(self, config: dict):

        self.config = config

        # Update a setting
        if "minio" in self.config and "mlflow" in self.config:
            settings.update({"mlflow": True})
        else:
            settings.update({"mlflow": False})

        # Reset settings to default values
        settings.reset()

        if "redis" in self.config:
            redis_config = self.config.get("redis", {})
            self.hash_manager = RedisHashManager(
                host=redis_config.get("REDIS_HOST"),
                port=redis_config.get("REDIS_PORT"),
                db=redis_config.get("REDIS_DB"),
                verbose=False,
            )

    def get_better_batch(self, batch_to_use: int = -1):
        optimal_batch = autobatch(
            model=self.model,
            imgsz=self.config["train"]["imgsz"],
            fraction=self.GPU_USE,
            batch_size=batch_to_use,
        )

        return optimal_batch

    def save_eda(self):
        if self.status_eda_completed == StatusEDA.SAVED:
            return

        dataset = os.path.dirname(self.config.get("train", {}).get("data"))
        hay_folder_report = os.path.exists(f"{dataset}/reports")
        hay_archivos_en_reporte_eda = len(os.listdir(f"{dataset}/reports")) > 0

        if hay_folder_report and hay_archivos_en_reporte_eda:
            mlflow.log_artifacts(f"{dataset}/reports", artifact_path="eda")
            self.status_eda_completed = StatusEDA.SAVED

    def set_config_vars(self):
        if "minio" in self.config and "mlflow" in self.config:
            # Configurar las variables de entorno necesarias para MLflow
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config["minio"][
                "MINIO_ENDPOINT"
            ]
            os.environ["AWS_ACCESS_KEY_ID"] = self.config["minio"]["MINIO_ID"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.config["minio"][
                "MINIO_SECRET_KEY"
            ]
            os.environ["MLFLOW_TRACKING_URI"] = self.config["mlflow"][
                "MLFLOW_TRACKING_URI"
            ]  # URI del servidor MLflow
            os.environ["MLFLOW_ARTIFACT_URI"] = (
                "s3://mlflow-artifacts/"  # Bucket en MinIO
            )

            # Configurar el nombre del experimento y el nombre de la ejecución
            os.environ["MLFLOW_EXPERIMENT_NAME"] = self.config.get("sweeper").get(
                "study_name"
            )
            os.environ["MLFLOW_RUN_NAME"] = self.config.get("task_id")

        self.is_configured = True

    @property
    def config_train(self):
        return self.config

    @config_train.setter
    def config_train(self, new_config: dict):
        self.config = new_config

    def on_train_end(self, trainer):
        if "minio" in self.config and "mlflow" in self.config:
            pytorch_model = trainer.model
            mlflow.pytorch.log_model(pytorch_model, "model")

            metrics = {}
            for key, value in trainer.metrics.items():
                metrics[slugify(key)] = float(value)

            mlflow.log_metrics(metrics)

            self.save_eda()

    def on_train_start(self, trainer):
        if "minio" in self.config and "mlflow" in self.config:
            dvc_path = self.config.get(
                "dvc_data_path", None
            )  # añade esta variable a tu config.

            try:
                if dvc_path:
                    data_url = dvc.api.get_url(dvc_path)
                    data_path = dvc.api.get_data_path(dvc_path)
                else:
                    raise
            except:
                dvc_path = (
                    self.config.get("train", {})
                    .get("data", None)
                    .replace("/datasets/", "")
                )
                data_url = f'http://{os.environ.get("CONTROL_HOST", "localhost")}:23443/files/{dvc_path}'
                data_path = self.config.get("train", {}).get("data", None)

            if data_url:
                try:
                    dataset_source: DatasetSource = HTTPDatasetSource(
                        url=dataset_source_url
                    )
                    dataset_source = mlflow.data.filesystem_dataset_source.FileSystemDatasetSource.from_dict(
                        {"path": data_url}
                    )
                    mlflow.log_input(
                        mlflow.data.Dataset(source=dataset_source, name="dataset_dvc")
                    )
                except:
                    logger.error(f"Error al cargar el dataset {data_url}")
            if data_path:
                try:
                    dataset_source = mlflow.data.http_dataset_source.HttpDatasetSource(
                        {"path": data_path}
                    )
                    mlflow.log_input(
                        mlflow.data.Dataset(
                            source=dataset_source, name="dataset_dvc_local"
                        )
                    )
                except:
                    logger.error(f"Error al cargar el dataset {data_path}")

            # remove batch of self.config
            config_copy = self.config.copy()
            config_copy["train"].pop("batch")

            try:
                gpu_json_list: List[dict] = obtener_info_gpu_json()
                for gpu_json in gpu_json_list:
                    for key, value in gpu_json.items():
                        try:
                            mlflow.set_tag(key, value)
                        except:
                            pass
            except:
                pass

            for key, value in config_copy["train"].items():
                try:
                    mlflow.log_param(key, value)
                except:
                    pass

            mlflow.set_tag(
                "mlflow.note.content", self.config.get("metadata", {}).get("content")
            )
            mlflow.set_tag(
                "documentation",
                self.config.get("metadata", {}).get("documentation", "NA"),
            )

            mlflow.set_tag(
                "author", self.config.get("metadata", {}).get("author", "NA")
            )

            mlflow.set_tag("experiment_type", trainer.model._get_name())
            mlflow.set_tag(
                "version", self.config.get("sweeper", {}).get("version", "NA")
            )
            mlflow.log_artifact(self.config["config_path"])

            mlflow.set_tag(
                "data_source", self.config.get("train", {}).get("data", "NA")
            )

            for other_metadata in self.worker_metadata:
                tag_metadata = os.environ.get(other_metadata, None)
                if tag_metadata:
                    try:
                        mlflow.set_tag(other_metadata, tag_metadata)
                    except:
                        pass

            # Subir 3 imágenes por clase
            self.log_example_images(model_type=trainer.model._get_name())

        self.start_time = time.time()

    def on_epoch_end(self, trainer):
        if self.firts_epoch:
            self.firts_epoch = False
            try:
                mlflow.log_artifacts(
                    self.config["train"]["project"] + "train_" + self.config["task_id"]
                )
            except:
                pass

        self.end_time = time.time()

        elapsed_time = self.end_time - self.start_time

        if "minio" in self.config and "mlflow" in self.config:
            metadata = {
                other_metadata: os.environ.get(other_metadata, None)
                for other_metadata in self.worker_metadata
            }

            trial_number = self.config.get("trial_number", 1)
            total_trails = self.config.get("sweeper", 1).get("n_trials", 10)

            total_epochs = self.config.get("train", {}).get("epochs", 10)

            #
            epoch = trainer.epoch + 1

            task_id = self.config.get("task_id", "noTaskId")

            metadata["TRIAL_NUMBER"] = trial_number
            metadata["TOTAL_TRIALS"] = total_trails
            metadata["TOTAL_EPOCHS"] = total_epochs
            metadata["EPOCH"] = epoch
            metadata["EPOCH_PROGRESS"] = epoch / total_epochs
            metadata["TRIAL_PROGRESS"] = trial_number / total_trails
            metadata["datetime"] = datetime.now().isoformat()
            metadata["task_id"] = task_id
            metadata["DEBUG_MODE"] = self.config.get("debug", "False")
            metadata["elapsed_time"] = round(elapsed_time, 3)

            epoch_count_total = total_epochs * int(total_trails + 1)
            epoch_total_now = int(trial_number + 1) * epoch
            metadata["result_time"] = round(
                elapsed_time * epoch_count_total / epoch_total_now, 3
            )

            gpu_json_list: List[dict] = obtener_info_gpu_json()
            for gpu_json in gpu_json_list:
                for key, value in gpu_json.items():
                    metadata[key] = value

            redis_key = "progress" + f":{task_id}"

            if self.hash_manager:
                for metadata_key, metadata_value in metadata.items():
                    try:
                        self.hash_manager.create_hash(
                            key=redis_key,
                            hash_name=metadata_key,
                            value=metadata_value,
                            ttl=1200,
                        )
                    except:
                        pass

            self.save_eda()

        self.check_stop_training(trainer)

    def check_stop_training(self, trainer):
        # stop training if exists the file: /config/stop_training.txt

        # example for to force stop training
        # if task_id == "_744db150e7514609a7aac96bf459a08d",
        # so, create the file /config/stop_training_744db150e7514609a7aac96bf459a08d.txt
        # touch /config/stop_training_744db150e7514609a7aac96bf459a08d.txt

        task_id = self.config.get("task_id")

        stop_training_file = f"/config/stop_training_{task_id}.txt"
        if os.path.exists(stop_training_file):
            # os.remove(stop_training_file)

            if self.model:
                self.model.stop_training = True
                self.model.stop_training = False
                # self.model = None

            if "minio" in self.config and "mlflow" in self.config:
                # Log the final model in MLflow
                self.on_train_end(trainer)

                # Log the artifacts
                try:
                    mlflow.log_artifacts(
                        self.config["train"]["project"]
                        + "train_"
                        + self.config["task_id"]
                    )
                except:
                    pass

                # copy file:
                # self.config["train"]["project"] + "train_" + self.config["task_id"] + "weights/last.pt"
                # to
                # self.config["train"]{"data"} + "weights/last.pt"
                try:
                    os.makedirs(
                        os.path.join(
                            self.config["train"]["data"],
                            "weights",
                        ),
                        exist_ok=True,
                    )
                    os.system(
                        f"cp {self.config['train']['project']}train_{self.config['task_id']}/weights/last.pt {self.config['train']['data']}weights/last.pt"
                    )
                except:
                    logger.error(
                        f"Error al copiar el modelo final a {self.config['train']['data']}weights/last.pt"
                    )

                # modify the f"/config_versions/{task_id}.yaml" in "model" and put the path of the last.pt
                config_path = f"/config_versions/{task_id}.yaml"
                if os.path.exists(config_path):
                    with open(config_path, "r") as file:
                        config = yaml.safe_load(file)

                    # Update the model path
                    config["model"] = os.path.join(
                        self.config["train"]["data"], "weights", "last.pt"
                    )

                    # add resume=True in config["train"]
                    if "train" in config:
                        config["train"]["resume"] = True
                    # Save the updated config

                    try:
                        # TODO: no deja guardar el archivo por lo que se rompe la ejecucion, quizas en la api cuando se cree toque guardarlo y luego darle permisos 777 al archivo, hay que ver
                        os.remove(config_path)
                        with open(config_path, "w") as file:
                            yaml.dump(config, file)
                    except Exception as e:
                        print(e)
                # Log the final model
                mlflow.log_artifact(
                    os.path.join(
                        self.config["train"]["project"],
                        "train_" + self.config["task_id"],
                        "weights",
                        "last.pt",
                    ),
                    artifact_path="model_final",
                )

                # End the MLflow run
                mlflow.end_run(status="FINISHED")

            raise StopIteration("Entrenamiento detenido por condición de callback.")

    def log_example_images(self, model_type: str):
        images, labels = self.get_images_and_labels(model_type=model_type)

        for i, (label, image_path) in enumerate(zip(labels, images)):
            try:
                image = Image.open(image_path)
                mlflow.log_image(image, f"example_images/{label}/image_{i}.png")
            except Exception as e:
                logger.error(f"Error al cargar la imagen {image_path}: {e}")

    def get_images_and_labels(self, model_type: str, size: int = 5):
        images = []
        labels = []

        if model_type == "ClassificationModel":
            data_path = self.config.get("train", {}).get("data", None)
            if data_path:
                label_list = [
                    path
                    for path in os.listdir(os.path.join(data_path, "train"))
                    if os.path.isdir(os.path.join(data_path, "train", path))
                ]
                for label in label_list:
                    image_paths = glob(os.path.join(data_path, "train", label, "*.jpg"))
                    image_paths = random.sample(
                        image_paths, min(size, len(image_paths))
                    )

                    for image in image_paths:
                        images.append(image)
                        labels.append(label)
        else:
            logger.warning(f"No se han implementado ejemplos para {model_type}.")

        return images, labels

    def train(self, config_train: dict):
        self.set_config_vars()

        if self.model:
            tune = self.config.get("sweeper", {}).get("tune", False)
            if tune:
                if isinstance(tune, bool):
                    tune = 1
                elif isinstance(tune, int):
                    tune = max(1, tune)
                    tune = min(tune, 100)
                else:
                    raise

                grace_period = self.config.get("sweeper", {}).get("grace_period", False)
                epochs = self.config.get("train", {}).get("epochs", False)

                return self.model.tune(
                    **config_train,
                    iterations=tune,
                    use_ray=True,
                    grace_period=min(epochs, grace_period),
                )
            else:
                # read env var MAX_GPU (user defined) for max gpu (value between 0 and 100)
                # if not set, use default value (60%),
                # if set to -1, use 60% of the gpu
                MAX_GPU = float(os.environ.get("MAX_GPU", -5000.1))

                config_train["batch"] = max(MAX_GPU / 100, -1)

                return self.model.train(**config_train)

        logger.warning(
            "No se puede entrenar sin antes configurar con 'set_config_vars'"
        )

    def create_model(self, model_name, model_type):
        if model_type == "yolo":
            model = YOLO(model_name)
        elif model_type == "rtdetr":
            model = RTDETR(model_name)
        else:
            raise ValueError("Invalid model type specified.")

        self.model = model

        # Configura los callbacks
        if "minio" in self.config and "mlflow" in self.config:
            self.model.add_callback("on_train_start", self.on_train_start)
            self.model.add_callback("on_train_end", self.on_train_end)
            self.model.add_callback("on_fit_epoch_end", self.on_epoch_end)

        return model
