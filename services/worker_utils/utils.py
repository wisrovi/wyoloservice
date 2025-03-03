from loguru import logger
import json

import yaml


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


DEFAULT_CONFIG = {}


def read_user_config(task_data: dict):
    try:
        task_data = json.loads(task_data)
    except:
        pass

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

            final_config["config_path"] = config_path
            final_config["task_id"] = task_data["task_id"]
    except Exception as e:
        logger.error(f"❌ Error al cargar YAML ({config_path}): {e}")

    try:
        with open(config_path, "w") as archivo:
            yaml.dump(final_config, archivo, default_flow_style=False)
    except Exception as e:
        pass

    return final_config
