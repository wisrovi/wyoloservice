try:
    from tkinter import filedialog

    import customtkinter as ctk
except:
    pass

import curses
import os
import signal  # Importa el módulo signal
import subprocess
import sys
import tempfile
import time

import yaml
from colorama import Fore, Style, init

# Inicializa colorama
init()


import requests
from wredis.streams import RedisStreamManager

GLOBAL_RESPONSE, task_id, terminal = None, None, True


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


def enviar_archivo(filename=None, streaming: bool = True):
    global GLOBAL_RESPONSE, task_id

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

    if streaming:
        stream_manager = RedisStreamManager(
            host=config["CONTROL_HOST"],
            verbose=False,
            port=23438,
        )
        task_id = GLOBAL_RESPONSE["task_id"]

        @stream_manager.on_message(
            stream_name=f"stream:{task_id}",
            group_name=f"stream:{task_id}",
            consumer_name=f"stream:{task_id}",
        )
        def process_message(data):
            if terminal:
                # si la impresión es por terminal
                print(data.get("value"))

    # Mantener el programa activo para consumir mensajes

    # stream_manager.wait()


def pedir_parametros():
    print("\n" + Fore.CYAN + "¡Hola! Por favor, introduce los siguientes parámetros:\n" + Style.RESET_ALL)

    # Solicitar hostname
    while True:
        hostname = input(Fore.GREEN + "Hostname: " + Style.RESET_ALL)
        if hostname:  # Comprobar que no esté vacío
            break
        print(Fore.RED + "¡Error! Por favor, introduce un valor válido para el hostname." + Style.RESET_ALL)

    # Solicitar host IP
    while True:
        host_ip = input(Fore.GREEN + "Host IP: " + Style.RESET_ALL)
        if host_ip:  # Comprobar que no esté vacío
            break
        print(Fore.RED + "¡Error! Por favor, introduce una dirección IP válida." + Style.RESET_ALL)

    # Solicitar destination
    while True:
        destination = input(Fore.GREEN + "Destination: " + Style.RESET_ALL)
        if destination:  # Comprobar que no esté vacío
            break
        print(Fore.RED + "¡Error! Por favor, introduce un destino válido." + Style.RESET_ALL)

    # Mostrar los parámetros
    print(Fore.YELLOW + "\nLos parámetros introducidos son:" + Style.RESET_ALL)
    print(f"{Fore.BLUE}Hostname: {hostname}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Host IP: {host_ip}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Destination: {destination}{Style.RESET_ALL}")
    print(Fore.MAGENTA + "\nGracias por introducir los parámetros.\n" + Style.RESET_ALL)

    return hostname, host_ip, destination


# Función que maneja el evento de Control+C
def manejar_ctrl_c(signum, frame):
    global terminal

    terminal = False

    print("Control + C presionado, enviando archivo...")

    hostname, host_ip, destination = pedir_parametros()

    # crear una carpeta temporal
    with tempfile.TemporaryDirectory() as temp_dir:
        # crear un archivo temporal
        """
        debug: stop_192.168.1.137
        stop: wisrovi
        task_id: 49eed38feb964cc3ba862ad1edd73042

        # destinity for continuation
        # destinity: true # for public queue
        """

        content_json = {
            "debug": f"stop_{host_ip}",
            "stop": hostname,
            "task_id": task_id,
        }

        if destination and len(destination) > 5:
            content_json["destinity"] = destination

        # Guardar el contenido en un archivo temporal
        temp_file_path = os.path.join(temp_dir, "stop.yaml")
        with open(temp_file_path, "w") as temp_file:
            yaml.dump(content_json, temp_file)
        # Enviar el archivo temporal
        enviar_archivo(temp_file_path, streaming=False)

    terminal = True


# Función que maneja el evento de Control+D
def manejar_ctrl_d():
    print("Control + D presionado, enviando archivo...")


def manejar_teclas(event):
    if event.state == 4:  # 4 significa que la tecla Ctrl está presionada
        if event.keysym == "c":
            manejar_ctrl_c()
        elif event.keysym == "d":
            manejar_ctrl_d()


def by_terminal(stdscr):
    # stdscr.clear()
    # stdscr.refresh()

    print(
        "Presiona Ctrl+C para detener el entrenamiento remoto y continuarlo en otro worker\n"
    )
    print("Presiona Ctrl+D para detener el entrenamiento remoto sin continuarlo\n")
    print("Presiona Ctrl+Z para salir de este programa.\n")

    # Ejecución desde la línea de comandos con un path de archivo
    filename = sys.argv[1]
    if os.path.exists(filename):
        hostname, host_ip, destination = pedir_parametros()
        # enviar_archivo(filename)

        while True:
            # key = stdscr.getch()  # Espera la entrada de teclado

            # if key == 3:  # Ctrl+C (ASCII 3)
            #     print("Control + C presionado, enviando archivo...\n")
            #     manejar_ctrl_c(0, 0)  # Llama a la función para manejar Ctrl+C

            # elif key == 4:  # Ctrl+D (ASCII 4)
            #     print("Control + D presionado, enviando archivo...\n")
            #     manejar_ctrl_d()

            # elif key == 26:  # Ctrl+Z (ASCII 26)
            #     print("Saliendo...\n")
            #     break  # Sale del bucle y finaliza el programa

            # stdscr.refresh()

            time.sleep(15)  # Espera 15 segundo antes de la siguiente iteración
    else:
        print(f"El archivo '{filename}' no existe.")


if __name__ == "__main__":
    # Añadir el evento de captura de teclas a la ventana
    print("Iniciando el programa...")

    if len(sys.argv) > 1:
        signal.signal(signal.SIGINT, manejar_ctrl_c)  # Captura la señal de Ctrl+C
        # signal.signal(signal.SIGQUIT, manejar_ctrl_d)
        # signal.signal(signal.SIGTSTP, manejar_ctrl_d)
        # signal.signal(signal.SIGTERM, manejar_ctrl_d)
        # signal.signal(signal.SIGCONT, manejar_ctrl_d)

        by_terminal(None)
        # try:
        #     curses.wrapper(by_terminal)  # Inicializa curses y llama a la función main
        # except Exception as e:
        #     print(f"Error: {e}")
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

        label_error = ctk.CTkLabel(
            ventana, text="", fg_color="transparent", text_color="red"
        )
        label_error.grid(row=1, column=0)

        ventana.bind("<Key>", manejar_teclas)

        ventana.mainloop()
