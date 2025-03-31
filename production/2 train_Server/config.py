import customtkinter as ctk
import subprocess
import os

if os.path.exists("control_host.env"):
    print("No need to set environment variables, because they are already set.")
    exit()


def obtener_usuario():
    """Obtiene el nombre de usuario usando 'whoami'."""
    try:
        resultado = subprocess.run(['whoami'], capture_output=True, text=True, check=True)
        return resultado.stdout.strip()
    except subprocess.CalledProcessError:
        return "usuario_desconocido"

def crear_archivo():
    control_host = entry_control_host.get()
    cifs_user = entry_cifs_user.get()
    cifs_pass = entry_cifs_pass.get()
    modo = menu_modo.get() if opciones_avanzadas_visible else "Público"
    num_current_train = entry_num_current_train.get() if opciones_avanzadas_visible else "1"
    max_gpu = entry_max_gpu.get() if opciones_avanzadas_visible else "60"

    if control_host and cifs_user and cifs_pass:
        with open("control_host.env", "w") as f:
            f.write(f"CONTROL_HOST={control_host}\n")
            f.write(f"CIFS_USER={cifs_user}\n")
            f.write(f"CIFS_PASS={cifs_pass}\n")
            if modo == "Privado":
                usuario = obtener_usuario()
                f.write(f"debug={usuario}\n")
            f.write(f"NUM_CURRENT_TRAIN={num_current_train}\n")
            f.write(f"MAX_GPU={max_gpu}\n")
        ventana.destroy()
    else:
        label_error.configure(text="Por favor, completa todos los campos.")

def toggle_opciones_avanzadas():
    global opciones_avanzadas_visible
    opciones_avanzadas_visible = not opciones_avanzadas_visible
    if opciones_avanzadas_visible:
        menu_modo.grid(row=4, column=0, columnspan=2, pady=10)
        label_num_current_train.grid(row=5, column=0, padx=10, pady=10)
        entry_num_current_train.grid(row=5, column=1, padx=10, pady=10)
        label_max_gpu.grid(row=6, column=0, padx=10, pady=10)
        entry_max_gpu.grid(row=6, column=1, padx=10, pady=10)
    else:
        menu_modo.grid_forget()
        label_num_current_train.grid_forget()
        entry_num_current_train.grid_forget()
        label_max_gpu.grid_forget()
        entry_max_gpu.grid_forget()

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

ventana = ctk.CTk()
ventana.title("Configuración")

ctk.CTkLabel(ventana, text="CONTROL_HOST:").grid(row=0, column=0, padx=10, pady=10)
entry_control_host = ctk.CTkEntry(ventana, placeholder_text="ip del host")
entry_control_host.grid(row=0, column=1, padx=10, pady=10)

ctk.CTkLabel(ventana, text="CIFS_USER:").grid(row=1, column=0, padx=10, pady=10)
entry_cifs_user = ctk.CTkEntry(ventana, placeholder_text="usuario")
entry_cifs_user.grid(row=1, column=1, padx=10, pady=10)
entry_cifs_user.insert(0, "wisrovi")  # Default value

ctk.CTkLabel(ventana, text="CIFS_PASS:").grid(row=2, column=0, padx=10, pady=10)
entry_cifs_pass = ctk.CTkEntry(ventana, show="*", placeholder_text="contraseña")
entry_cifs_pass.grid(row=2, column=1, padx=10, pady=10)
entry_cifs_pass.insert(0, "wyoloservice")  # Default value

# Menú desplegable para modo público/privado (inicialmente oculto)
menu_modo = ctk.CTkOptionMenu(ventana, values=["Público", "Privado"])
menu_modo.set("Público")

# Etiqueta y entrada para NUM CURRENT TRAIN (inicialmente ocultos)
label_num_current_train = ctk.CTkLabel(ventana, text="NUM CURRENT TRAIN:")
entry_num_current_train = ctk.CTkEntry(ventana)
entry_num_current_train.insert(0, "1")  # Valor predeterminado

# Etiqueta y entrada para MAX GPU (inicialmente ocultos)
label_max_gpu = ctk.CTkLabel(ventana, text="MAX GPU (%):")
entry_max_gpu = ctk.CTkEntry(ventana)
entry_max_gpu.insert(0, "60")  # Valor predeterminado

# Botón para mostrar/ocultar opciones avanzadas
opciones_avanzadas_visible = False
boton_opciones_avanzadas = ctk.CTkButton(ventana, text="Opciones Avanzadas", command=toggle_opciones_avanzadas)
boton_opciones_avanzadas.grid(row=3, column=0, columnspan=2, pady=10)

boton_crear = ctk.CTkButton(ventana, text="Crear archivo", command=crear_archivo)
boton_crear.grid(row=7, columnspan=2, pady=20)

label_error = ctk.CTkLabel(ventana, text="", fg_color="transparent", text_color="red")
label_error.grid(row=8, columnspan=2)

ventana.mainloop()
