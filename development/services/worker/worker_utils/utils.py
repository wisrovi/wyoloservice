import shutil


def copiar_archivo(ruta_origen, ruta_destino):
    """
    Copia un archivo de una ruta a otra.

    Args:
        ruta_origen (str): La ruta del archivo de origen.
        ruta_destino (str): La ruta del archivo de destino.
    """
    try:
        shutil.copy2(ruta_origen, ruta_destino)
        print(f"Archivo copiado de '{ruta_origen}' a '{ruta_destino}'")
    except FileNotFoundError:
        print(f"Error: El archivo '{ruta_origen}' no existe.")
    except PermissionError:
        print(f"Error: No tienes permisos para copiar el archivo a '{ruta_destino}'.")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


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


