import os

import boto3
import botocore
import hydra  # pip install hydra-core
from botocore.exceptions import ClientError
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from omegaconf import DictConfig
from tqdm import tqdm
from loguru import logger


class MinioS3Client:
    MINIO_ENDPOINT = "http://minio:9000"
    BUCKET_NAME = "models"
    MLFLOW_ARTIFACTS_BUCKET = "mlflow-artifacts"

    s3: boto3.client

    def __init__(
        self,
        endpoint_url=None,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        """
        Inicializa el cliente MinIO/S3.

        Args:
            endpoint_url (str): URL del endpoint de MinIO/S3.
            aws_access_key_id (str): ID de acceso de AWS.
            aws_secret_access_key (str): Clave secreta de AWS.
        """
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url or self.MINIO_ENDPOINT,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        if not self.bucket_exists(self.MLFLOW_ARTIFACTS_BUCKET):
            self.create_bucket(self.MLFLOW_ARTIFACTS_BUCKET)

        if not self.bucket_exists(self.BUCKET_NAME):
            self.create_bucket(self.BUCKET_NAME)

        if not self.validate_connection():
            os._exit(1)  # Fuerza la salida del proceso

    def validate_connection(self):
        """
        Valida si la conexión al servicio S3 es exitosa.
        Retorna True si la conexión es válida, False en caso contrario.
        """
        try:
            # Intenta listar los buckets disponibles
            self.s3.list_buckets()
            # print("Conexión exitosa: Se puede acceder al servicio S3.")
            return True
        except NoCredentialsError:
            print("Error: No se encontraron credenciales válidas.")
        except PartialCredentialsError:
            print("Error: Las credenciales están incompletas.")
        except ClientError as e:
            # Captura errores específicos del cliente S3
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "InvalidAccessKeyId":
                logger.error("Error: La clave de acceso MinIO (Access Key ID) no es válida.")
            elif error_code == "SignatureDoesNotMatch":
                logger.error("Error: La clave secreta MinIO (Secret Access Key) no coincide.")
            elif error_code == "AuthorizationHeaderMalformed":
                logger.error(
                    "Error: El endpoint o las credenciales están mal configurados."
                )
            else:
                logger.error(f"Error desconocido: {e}")
        except Exception as e:
            # Captura cualquier otro error inesperado
            logger.error(f"Error inesperado: {e}")

        return False

    def object_exists(self, bucket_name: str, object_key: str) -> bool:
        """Verifica si un objeto existe en el bucket."""
        try:
            self.s3.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(
                    f"❌ El archivo '{object_key}' no existe en el bucket '{bucket_name}'."
                )
            else:
                print(f"⚠️ Error desconocido: {e}")
            return False

    def download_file(
        self, bucket_name: str, object_key: str, local_file_path: str
    ) -> bool:
        """
        Descarga un archivo de un bucket de S3.

        Args:
            bucket_name (str): El nombre del bucket.
            object_key (str): La clave del objeto (ruta del archivo en S3).
            local_file_path (str): La ruta local donde se guardará el archivo descargado.

        Returns:
            bool: True si la descarga fue exitosa, False en caso contrario.
        """
        if not self.object_exists(bucket_name, object_key):
            return False  # No descargar si no existe

        try:
            self.s3.download_file(bucket_name, object_key, local_file_path)
            print(f"Archivo descargado exitosamente en: {local_file_path}")
            return True
        except botocore.exceptions.NoCredentialsError:
            print("Error: Credenciales no encontradas.")
        except botocore.exceptions.PartialCredentialsError:
            print("Error: Credenciales incompletas.")
        except self.s3.exceptions.NoSuchKey:
            print(
                f"Error: El archivo '{object_key}' no existe en el bucket '{bucket_name}'."
            )
        except Exception as e:
            print(f"Error desconocido: {e}")
        return False

    def create_folder_and_upload_file(
        self,
        bucket_name: str,
        folder_name: str,
        local_file_path: str,
        s3_file_name: str,
    ):
        """
        Crea una carpeta en un bucket de S3 y sube un archivo a esa carpeta.

        Args:
            bucket_name (str): El nombre del bucket.
            folder_name (str): El nombre de la carpeta a crear.
            local_file_path (str): La ruta local del archivo a subir.
            s3_file_name (str): El nombre del archivo en S3.
        """
        try:
            # Crea la carpeta (en S3, las "carpetas" son objetos con un prefijo)
            self.s3.put_object(Bucket=bucket_name, Key=f"{folder_name}/")
            print(f"Carpeta '{folder_name}' creada en el bucket '{bucket_name}'.")

            # Sube el archivo a la carpeta
            self.s3.upload_file(
                local_file_path, bucket_name, f"{folder_name}/{s3_file_name}"
            )
            print(
                f"Archivo '{local_file_path}' subido a '{folder_name}/{s3_file_name}' en el bucket '{bucket_name}'."
            )
        except ClientError as e:
            print(f"Error al crear la carpeta o subir el archivo: {e}")

    def list_objects(self, bucket_name: str, prefix: str = "") -> list:
        """
        Lista los objetos (archivos y carpetas) en un bucket de S3 y retorna una lista de claves.

        Args:
            bucket_name (str): El nombre del bucket.
            prefix (str, optional): Prefijo para filtrar los objetos. Defaults to ''.

        Returns:
            list: Una lista de claves de objetos, o None si hay un error.
        """
        try:
            response = self.s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if "Contents" in response:
                return [obj["Key"] for obj in response["Contents"]]
            else:
                return []  # Retorna una lista vacía si no hay objetos.
        except ClientError as e:
            print(f"Error al listar objetos en el bucket '{bucket_name}': {e}")
            return None  # Retorna None en caso de error.

    def create_bucket(self, bucket_name: str):
        """
        Crea un bucket en S3.

        Args:
            bucket_name (str): El nombre del bucket.
        """
        try:
            self.s3.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' creado exitosamente.")
        except ClientError as e:
            print(f"No se pudo crear el bucket '{bucket_name}': {e}")

    def bucket_exists(self, bucket_name: str) -> bool:
        """
        Verifica si un bucket existe.

        Args:
            bucket_name (str): El nombre del bucket.

        Returns:
            bool: True si el bucket existe, False en caso contrario.
        """
        try:
            self.s3.head_bucket(Bucket=bucket_name)
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

    def upload_file(
        self, file_path_local: str, name_file_s3: str, bucket_name: str = BUCKET_NAME
    ) -> str:
        """
        Sube un archivo a un bucket de S3.

        Args:
            file_path_local (str): Ruta local del archivo.
            name_file_s3 (str): Nombre del archivo en S3.
            bucket_name (str): Nombre del bucket.

        Returns:
            str: URL del archivo subido.
        """
        if not self.bucket_exists(bucket_name):
            self.create_bucket(bucket_name)

        self.s3.upload_file(file_path_local, bucket_name, name_file_s3)
        return f"{self.MINIO_ENDPOINT}/{bucket_name}/{name_file_s3}"

    def upload_folder(self, folder_path: str, bucket_name: str, prefix: str = ""):
        """
        Sube una carpeta completa al bucket, preservando la estructura de directorios.

        Args:
            folder_path (str): Ruta de la carpeta local.
            bucket_name (str): Nombre del bucket.
            prefix (str): Prefijo opcional para agregar a las claves de los objetos.
        """
        if not os.path.isdir(folder_path):
            print(f"La ruta '{folder_path}' no es una carpeta válida.")
            return

        if not self.bucket_exists(bucket_name):
            self.create_bucket(bucket_name)

        for root, dirs, files in tqdm(os.walk(folder_path)):
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
                    self.s3.upload_file(local_file_path, bucket_name, s3_object_key)
                    # print(f"Subido: {local_file_path} -> {s3_object_key}")
                except ClientError as e:
                    print(f"Error al subir {local_file_path}: {e}")


@hydra.main(config_path="/app", config_name="config", version_base=None)
def initialize_minio_client(cfg: DictConfig) -> MinioS3Client:
    """
    Inicializa el cliente MinIO/S3 utilizando la configuración de Hydra.

    Args:
        cfg (DictConfig): Configuración de Hydra.

    Returns:
        MinioS3Client: Instancia del cliente MinIO/S3.
    """
    minio_client = MinioS3Client(
        endpoint_url=cfg.minio.MINIO_ENDPOINT,
        aws_access_key_id=cfg.minio.MINIO_ID,
        aws_secret_access_key=cfg.minio.MINIO_SECRET_KEY,
    )
    return minio_client
