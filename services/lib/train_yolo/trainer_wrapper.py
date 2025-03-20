import json
import os
import random
from glob import glob
from typing import List

import dvc
import GPUtil
import mlflow
from loguru import logger
from PIL import Image
from slugify import slugify
from ultralytics import RTDETR, YOLO, settings
from ultralytics.utils.autobatch import autobatch


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


class TrainerWrapper:
    # https://github.com/ultralytics/ultralytics/issues/8214
    config = {}
    GPU_USE = 0.6  # procentaje de uso de GPU

    is_configured = False
    model = None

    def __init__(self, config: dict):

        self.config = config

        # Update a setting
        if "minio" in self.config and "mlflow" in self.config:
            settings.update({"mlflow": True})
        else:
            settings.update({"mlflow": False})

        # Reset settings to default values
        settings.reset()

    def get_better_batch(self, batch_to_use: int = -1):
        optimal_batch = autobatch(
            model=self.model,
            imgsz=self.config["train"]["imgsz"],
            fraction=self.GPU_USE,
            batch_size=batch_to_use,
        )

        return optimal_batch

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

    def on_train_start(self, trainer):
        if "minio" in self.config and "mlflow" in self.config:
            dvc_path = self.config.get(
                "dvc_data_path", None
            )  # añade esta variable a tu config.
            if dvc_path:
                try:
                    data_url = dvc.api.get_url(dvc_path)
                    data_path = dvc.api.get_data_path(dvc_path)
                except:
                    dvc_path = (
                        self.config.get("train", {})
                        .get("data", None)
                        .repalce("/datasets/", "")
                    )
                    data_url = f'http://{os.environ.get("WORKER_HOST", "localhost")}:23443/files/"{dvc_path}"'
                    data_path = self.config.get("train", {}).get("data", None)

                if data_url:
                    try:
                        mlflow.log_input(
                            mlflow.data.Dataset(source=data_url, name="dataset_dvc")
                        )
                    except:
                        logger.error(f"Error al cargar el dataset {data_url}")
                if data_path:
                    try:
                        mlflow.log_input(
                            mlflow.data.Dataset(
                                source=data_path, name="dataset_dvc_local"
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

            mlflow.log_params(config_copy["train"])
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

            worker_metadata = [
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

            for other_metadata in worker_metadata:
                tag_metadata = os.environ.get(other_metadata, None)
                if tag_metadata:
                    try:
                        mlflow.set_tag(other_metadata, tag_metadata)
                    except:
                        pass

            # Subir 3 imágenes por clase
            self.log_example_images(model_type=trainer.model._get_name())

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

        return model
