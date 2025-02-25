import os
import boto3
import hydra
from omegaconf import DictConfig
from botocore.exceptions import ClientError


MINIO_ENDPOINT = "http://minio:9000"
BUCKET_NAME = "models"
MLFLOW_ARTIFACTS_BUCKET = "mlflow-artifacts"


def create_bucket(bucket_name: str, s3: boto3.client):
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' creado exitosamente.")
    except ClientError as e:
        print(f"No se pudo crear el bucket '{bucket_name}': {e}")


def bucket_exists(bucket_name: str, s3: boto3.client) -> bool:
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            print(f"El bucket '{bucket_name}' no existe.")
        elif error_code == "403":
            print(f"No tienes permisos para acceder al bucket '{bucket_name}'.")
        else:
            print(f"Ocurrió un error: {e}")
        return False


def upload_file_to_minio(
    task_id: str,
    s3: boto3.client,
    model_path: str,
    version: int,
):
    """Guarda múltiples versiones del modelo en MinIO y devuelve la URL."""
    model_filename = f"{task_id}_v{version}.pt"

    s3.upload_file(model_path, BUCKET_NAME, model_filename)
    return f"{MINIO_ENDPOINT}/{BUCKET_NAME}/{model_filename}"


# Función para subir una carpeta completa
def upload_folder_to_minio(
    folder_path: str,
    s3: boto3.client,
    bucket_name: str,
    prefix: str = "",
):
    """
    Sube una carpeta completa al bucket, preservando la estructura de directorios.

    :param folder_path: Ruta de la carpeta local.
    :param bucket_name: Nombre del bucket.
    :param prefix: Prefijo opcional para agregar a las claves de los objetos.
    """
    if not os.path.isdir(folder_path):
        print(f"La ruta '{folder_path}' no es una carpeta válida.")
        return

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_file_path = os.path.join(
                root, file
            )  # Ruta completa del archivo local
            relative_path = os.path.relpath(
                local_file_path, folder_path
            )  # Ruta relativa
            s3_object_key = os.path.join(prefix, relative_path).replace(
                "\\", "/"
            )  # Clave del objeto en MinIO

            try:
                s3.upload_file(local_file_path, bucket_name, s3_object_key)
                print(f"Subido: {local_file_path} -> {s3_object_key}")
            except ClientError as e:
                print(f"Error al subir {local_file_path}: {e}")


@hydra.main(config_path="/app", config_name="config", version_base=None)
def get_s3(cfg: DictConfig):
    MINIO_ENDPOINT = cfg.minio.MINIO_ENDPOINT
    MINIO_ID = cfg.minio.MINIO_ID
    MINIO_KEY = cfg.minio.MINIO_SECRET_KEY

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ID,
        aws_secret_access_key=MINIO_KEY,
    )
    
    if not bucket_exists(MLFLOW_ARTIFACTS_BUCKET, s3):
        create_bucket(MLFLOW_ARTIFACTS_BUCKET, s3)
        
    if not bucket_exists(BUCKET_NAME, s3):
        create_bucket(BUCKET_NAME, s3)

    return s3
