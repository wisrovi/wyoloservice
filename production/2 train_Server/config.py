import customtkinter as ctk
import subprocess

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
    modo = menu_modo.get()

    if control_host and cifs_user and cifs_pass:
        with open("control_host.env", "w") as f:
            f.write(f"CONTROL_HOST={control_host}\n")
            f.write(f"CIFS_USER={cifs_user}\n")
            f.write(f"CIFS_PASS={cifs_pass}\n")
            if modo == "Privado":
                usuario = obtener_usuario()
                f.write(f"debug={usuario}\n")
        ventana.destroy()
    else:
        label_error.configure(text="Por favor, completa todos los campos.")

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

ventana = ctk.CTk()
ventana.title("Configuración")

ctk.CTkLabel(ventana, text="CONTROL_HOST:").grid(row=0, column=0, padx=10, pady=10)
entry_control_host = ctk.CTkEntry(ventana)
entry_control_host.grid(row=0, column=1, padx=10, pady=10)

ctk.CTkLabel(ventana, text="CIFS_USER:").grid(row=1, column=0, padx=10, pady=10)
entry_cifs_user = ctk.CTkEntry(ventana)
entry_cifs_user.grid(row=1, column=1, padx=10, pady=10)

ctk.CTkLabel(ventana, text="CIFS_PASS:").grid(row=2, column=0, padx=10, pady=10)
entry_cifs_pass = ctk.CTkEntry(ventana, show="*")
entry_cifs_pass.grid(row=2, column=1, padx=10, pady=10)

# Menú desplegable para modo público/privado
menu_modo = ctk.CTkOptionMenu(ventana, values=["Público", "Privado"])
menu_modo.grid(row=3, column=0, columnspan=2, pady=10)
menu_modo.set("Público")  # Valor predeterminado

boton_crear = ctk.CTkButton(ventana, text="Crear archivo", command=crear_archivo)
boton_crear.grid(row=4, columnspan=2, pady=20)

label_error = ctk.CTkLabel(ventana, text="", fg_color="transparent", text_color="red")
label_error.grid(row=5, columnspan=2)

ventana.mainloop()
