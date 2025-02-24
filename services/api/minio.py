import boto3
import datetime

MINIO_ENDPOINT = "http://minio:9000"
BUCKET_NAME = "models"

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
)


def upload_model(task_id: str, file):
    """Sube un modelo con versionado automático."""
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    versioned_filename = f"{task_id}_v{timestamp}.pt"
    s3.upload_fileobj(file.file, BUCKET_NAME, versioned_filename)
    return {"message": "Modelo subido", "filename": versioned_filename}


def list_models():
    """Lista los modelos almacenados en MinIO con sus versiones."""
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    models = sorted([obj["Key"] for obj in response.get("Contents", [])], reverse=True)
    return {"models": models}


def download_model(filename: str):
    """Descarga una versión específica de un modelo."""
    file_path = f"/tmp/{filename}"
    s3.download_file(BUCKET_NAME, filename, file_path)
    return {"message": f"Modelo descargado en {file_path}"}
