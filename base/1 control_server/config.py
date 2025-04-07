try:
    import customtkinter as ctk
    from tkinter import filedialog
except:
    pass

import customtkinter as ctk
import subprocess
import os

if os.path.exists("control_host.env"):
    print("No need to set environment variables, because they are already set.")
    exit()


def get_ip():
    """Gets the host IP using 'hostname -I'."""
    try:
        # hostname -I | awk '{print $1}'
        result = subprocess.run(
            ["hostname", "-I", "|", "awk", "'{print", "$1}'"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split()[
            0
        ]  # Takes the first IP if there are multiple
    except subprocess.CalledProcessError:
        try:
            result = subprocess.run(
                ["hostname", "-I"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().split()[
                0
            ]  # Takes the first IP if there are multiple
        except subprocess.CalledProcessError:
            return "127.0.0.1"  # Default local IP if it fails


def select_folder():
    """Opens a file explorer to select the folder."""
    selected_folder = filedialog.askdirectory()
    if selected_folder:
        entry_folder_path.delete(0, ctk.END)
        entry_folder_path.insert(0, selected_folder)


def create_file():
    folder_path = entry_folder_path.get()
    username = entry_username.get()
    password = entry_password.get()
    redis_commander = checkbox_redis_commander.get() if advanced_options_visible else 0
    control_host = get_ip()  # Gets the host IP

    if folder_path and username and password:
        with open("control_host.env", "w") as f:
            f.write(f"FOLDER_SHARED={folder_path}\n")
            f.write(f"USERNAME={username}\n")
            f.write(f"PASSWORD={password}\n")
            f.write(f"REDIS_COMMANDER={redis_commander}\n")
            f.write(f"CONTROL_HOST={control_host}\n")  # Saves the host IP

        subprocess.run(["export", f'FOLDER_SHARED="{folder_path}"'])
        subprocess.run(["export", f'USERNAME="{username}"'])
        subprocess.run(["export", f'PASSWORD="{password}"'])
        subprocess.run(["export", f'REDIS_COMMANDER="{redis_commander}"'])
        subprocess.run(["export", f'CONTROL_HOST="{control_host}"'])
        subprocess.run(["chmod", "777", "control_host.env"])

        window.destroy()
    else:
        error_label.configure(text="Please complete all fields.")


def toggle_advanced_options():
    global advanced_options_visible
    advanced_options_visible = not advanced_options_visible
    if advanced_options_visible:
        checkbox_redis_commander.grid(row=4, column=0, columnspan=2, pady=10)
    else:
        checkbox_redis_commander.grid_forget()


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("Configuration")

ctk.CTkLabel(window, text="Folder Path:").grid(row=0, column=0, padx=10, pady=10)
entry_folder_path = ctk.CTkEntry(window, placeholder_text="/path/to/folder", width=350)
entry_folder_path.grid(row=0, column=1, padx=10, pady=10)
entry_folder_path.insert(0, "/media/training/Train_service_an1")  # Default value

explore_button = ctk.CTkButton(window, text="Explore", command=select_folder)
explore_button.grid(row=0, column=2, padx=10, pady=10)

ctk.CTkLabel(window, text="Username:").grid(row=1, column=0, padx=10, pady=10)
entry_username = ctk.CTkEntry(window, placeholder_text="wisrovi")
entry_username.grid(row=1, column=1, padx=10, pady=10)
entry_username.insert(0, "wisrovi")  # Default value

ctk.CTkLabel(window, text="Password:").grid(row=2, column=0, padx=10, pady=10)
entry_password = ctk.CTkEntry(window, show="*", placeholder_text="wyoloservice")
entry_password.grid(row=2, column=1, padx=10, pady=10)
entry_password.insert(0, "wyoloservice")  # Default value

# Checkbox for Redis Commander (initially hidden)
checkbox_redis_commander = ctk.CTkCheckBox(window, text="Activate Redis Commander")

# Button to show/hide advanced options
advanced_options_visible = False
advanced_options_button = ctk.CTkButton(
    window, text="Advanced Options", command=toggle_advanced_options
)
advanced_options_button.grid(row=3, column=0, columnspan=2, pady=10)

create_button = ctk.CTkButton(window, text="Create File", command=create_file)
create_button.grid(row=5, columnspan=3, pady=20)

error_label = ctk.CTkLabel(window, text="", fg_color="transparent", text_color="red")
error_label.grid(row=6, columnspan=3)

window.mainloop()
