try:
    import customtkinter as ctk
    from tkinter import filedialog
except:
    pass

import time
import subprocess
import requests
import sys
from wredis.streams import RedisStreamManager
import os

GLOBAL_RESPONSE = None

def obtener_usuario():
    """Obtiene el nombre de usuario usando 'whoami'."""
    try:
        resultado = subprocess.run(
            ["whoami"], capture_output=True, text=True, check=True
        )
        return resultado.stdout.strip()
    except subprocess.CalledProcessError:
        return "usuario_desconocido"

def cargar_configuracion():
    """Carga la configuración desde control_host.env."""
    try:
        with open("control_host.env", "r") as f:
            config = {}
            for line in f:
                key, value = line.strip().split("=")
                config[key] = value
            return config
    except FileNotFoundError:
        return None

def seleccionar_archivo():
    """Abre un diálogo para seleccionar un archivo."""
    filename = filedialog.askopenfilename()
    return filename

def enviar_archivo(filename=None):
    global GLOBAL_RESPONSE

    """Envía el archivo seleccionado a la API."""
    config = cargar_configuracion()
    if not config:
        print("Archivo de configuración no encontrado.")
        return

    if not filename:
        filename = seleccionar_archivo()
        if not filename:
            return  # El usuario canceló la selección del archivo

    try:
        files = {
            "file": (
                "config_train.yaml",
                open(filename, "rb"),
                "application/x-yaml",
            )
        }

        headers = {"accept": "application/json"}
        params = {"user_code": obtener_usuario()}
        url = f"http://{config['CONTROL_HOST']}:23450/train/"

        response = requests.post(url, params=params, headers=headers, files=files)

        print(response.status_code)

        GLOBAL_RESPONSE = response.json()
        response.raise_for_status()  # Lanza una excepción para códigos de error HTTP

        print("Archivo enviado correctamente.")
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar el archivo: {e}")
        return
    except FileNotFoundError:
        print("Archivo no encontrado.")
        return

    print(GLOBAL_RESPONSE)

    stream_manager = RedisStreamManager(
        host=config["CONTROL_HOST"],
        verbose=False,
        port=23438,
    )
    task_id = GLOBAL_RESPONSE["task_id"]
    
    print("Streaming activated in", f"stream:{task_id}")

    @stream_manager.on_message(
        stream_name=f"stream:{task_id}", group_name=f"stream:{task_id}", consumer_name=f"stream:{task_id}"
    )
    def process_message(data):
        print(data.get("value"))

    # Mantener el programa activo para consumir mensajes
    # stream_manager.wait()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Ejecución desde la línea de comandos con un path de archivo
        filename = sys.argv[1]
        if os.path.exists(filename):
            enviar_archivo(filename)
            while True:
            	time.sleep(15)
        else:
            print(f"El archivo '{filename}' no existe.")
    else:
        # Ejecución normal con interfaz gráfica
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        ventana = ctk.CTk()
        ventana.title("user - wyoloservice")

        boton_enviar = ctk.CTkButton(
            ventana, text="Seleccionar y Enviar Archivo", command=enviar_archivo
        )
        boton_enviar.grid(row=0, column=0, padx=20, pady=20)

        label_error = ctk.CTkLabel(ventana, text="", fg_color="transparent", text_color="red")
        label_error.grid(row=1, column=0)

        ventana.mainloop()
