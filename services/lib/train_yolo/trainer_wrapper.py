import os
import random
from glob import glob

import dvc
import mlflow
from loguru import logger
from PIL import Image
from slugify import slugify
from ultralytics import RTDETR, YOLO, settings
from ultralytics.utils.autobatch import autobatch


class TrainerWrapper:
    # https://github.com/ultralytics/ultralytics/issues/8214
    config = {}
    GPU_USE = 0.6  # procentaje de uso de GPU

    is_configured = False
    model = None

    def __init__(self, config: dict):
        # Update a setting
        settings.update({"mlflow": True})

        # Reset settings to default values
        settings.reset()

        self.config = config

    def get_better_batch(self, batch_to_use: int = -1):
        optimal_batch = autobatch(
            model=self.model,
            imgsz=self.config["train"]["imgsz"],
            fraction=self.GPU_USE,
            batch_size=batch_to_use,
        )

        return optimal_batch

    def set_config_vars(self):
        # Configurar las variables de entorno necesarias para MLflow
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.config["minio"]["MINIO_ENDPOINT"]
        os.environ["AWS_ACCESS_KEY_ID"] = self.config["minio"]["MINIO_ID"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.config["minio"]["MINIO_SECRET_KEY"]
        os.environ["MLFLOW_TRACKING_URI"] = self.config["mlflow"][
            "MLFLOW_TRACKING_URI"
        ]  # URI del servidor MLflow
        os.environ["MLFLOW_ARTIFACT_URI"] = "s3://mlflow-artifacts/"  # Bucket en MinIO

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
        pytorch_model = trainer.model
        mlflow.pytorch.log_model(pytorch_model, "model")
        
        metrics = {}
        for key, value in trainer.metrics.items():
            metrics[slugify(key)] = float(value)            

        mlflow.log_metrics(metrics)

    def on_train_start(self, trainer):

        dvc_path = self.config.get(
            "dvc_data_path", None
        )  # añade esta variable a tu config.
        if dvc_path:
            try:
                data_url = dvc.api.get_url(dvc_path)
                data_path = dvc.api.get_data_path(dvc_path)
                mlflow.log_input(
                    mlflow.data.Dataset(source=data_url, name="dataset_dvc")
                )
                mlflow.log_input(
                    mlflow.data.Dataset(source=data_path, name="dataset_dvc_local")
                )
            except:
                pass

        # remove batch of self.config
        config_copy = self.config.copy()
        config_copy["train"].pop("batch")

        mlflow.log_params(config_copy["train"])
        mlflow.set_tag(
            "mlflow.note.content", self.config.get("metadata", {}).get("content")
        )
        mlflow.set_tag(
            "documentation", self.config.get("metadata", {}).get("documentation", "NA")
        )

        mlflow.set_tag("author", self.config.get("metadata", {}).get("author", "NA"))

        mlflow.set_tag("experiment_type", trainer.model._get_name())
        mlflow.set_tag("version", self.config.get("sweeper", {}).get("version", "NA"))
        mlflow.log_artifact(self.config["config_path"])

        mlflow.set_tag("data_source", self.config.get("train", {}).get("data", "NA"))

        dvc_path = self.config.get("dvc_data_path", None)
        if dvc_path:
            try:
                data_url = dvc.api.get_url(dvc_path)
                data_path = dvc.api.get_data_path(dvc_path)
                mlflow.log_input(
                    mlflow.data.Dataset(source=data_url, name="dataset_dvc")
                )
                mlflow.log_input(
                    mlflow.data.Dataset(source=data_path, name="dataset_dvc_local")
                )
            except:
                pass

        # Subir 3 imágenes por clase
        self.log_example_images(model_type=trainer.model._get_name())

    def log_example_images(self, model_type: str):
        images, labels = self.get_images_and_labels(model_type=model_type)

        for i, (label, image_path) in enumerate(zip(labels, images)):
            image = Image.open(image_path)
            mlflow.log_image(image, f"example_images/{label}/image_{i}.png")

    def get_images_and_labels(self, model_type: str):
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
                    image_paths = random.sample(image_paths, min(3, len(image_paths)))

                    for image in image_paths:
                        images.append(image)
                        labels.append(label)

        return images, labels

    def train(self, config_train: dict):
        self.set_config_vars()

        if self.model:
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
        self.model.add_callback("on_train_start", self.on_train_start)
        self.model.add_callback("on_train_end", self.on_train_end)

        return model
