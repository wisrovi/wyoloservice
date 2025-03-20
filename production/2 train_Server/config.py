import customtkinter as ctk

def crear_archivo():
    control_host = entry_control_host.get()
    cifs_user = entry_cifs_user.get()
    cifs_pass = entry_cifs_pass.get()

    if control_host and cifs_user and cifs_pass:
        with open("control_host.env", "w") as f:
            f.write(f"CONTROL_HOST={control_host}\n")
            f.write(f"CIFS_USER={cifs_user}\n")
            f.write(f"CIFS_PASS={cifs_pass}\n")
        ventana.destroy()
    else:
        label_error.configure(text="Por favor, completa todos los campos.")

ctk.set_appearance_mode("System")  # Modo claro/oscuro automático
ctk.set_default_color_theme("blue")  # Tema de color predeterminado

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

boton_crear = ctk.CTkButton(ventana, text="Crear archivo", command=crear_archivo)
boton_crear.grid(row=3, columnspan=2, pady=20)

label_error = ctk.CTkLabel(ventana, text="", fg_color="transparent", text_color="red")
label_error.grid(row=4, columnspan=2)

ventana.mainloop()
