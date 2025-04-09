import os
from pydantic import BaseModel, Field, ConfigDict
from wsqlite import WSQLite
from datetime import datetime


class TrainingHistory(BaseModel):
    id: int = Field(primary_key=True)
    task_id: str
    status: str
    user_code: str
    timestamp: str = datetime.utcnow().isoformat()
    config_path: str
    model_path: str = None
    cpu_usage: float = None
    ram_usage: float = None
    gpu_usage: float = None
    loss: float = None
    precision: float = None
    recall: float = None
    map50: float = None
    map50_95: float = None
    recommended_model: str = None
    experiment_name: str = None

    model_config = ConfigDict(protected_namespaces=())


# Ruta donde se guardará la base de datos
DB_PATH = "/database/mlflow.db"

# Verificar si la carpeta /database existe, si no, crearla
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


db = WSQLite(TrainingHistory, "/database/mlflow.db")
