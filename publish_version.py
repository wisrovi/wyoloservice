# necesito ejecutar estos comados en la terminal
# pip install --upgrade build

# ahora abrir el archivo pyproject.toml y cambiar la version del proyecto incrementando el numero de la version y luego ejecutar el siguiente comando


# python setup.py sdist bdist_wheel
# twine upload dist/*


import subprocess


try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    # pip install tomli
    import tomli as tomllib  # Python <3.11
    
import tomli_w

with open("pyproject.toml", "rb") as file:
    config_project = tomllib.load(file)

actual_version = config_project["project"]["version"]

# incrementar la version
version = actual_version.split(".")

# hay 3 partes en la version: major, minor, patch
# cada parte se incrementa en 1, cuando la parte anterior llega a 9, se incrementa la siguiente parte en 1
# si la parte anterior es 0, se incrementa en 1

major = int(version[0])
minor = int(version[1])
patch = int(version[2])

if patch < 9:
    patch += 1
else:
    patch = 0
    if minor < 9:
        minor += 1
    else:
        minor = 0
        major += 1

new_version = f"{major}.{minor}.{patch}"

# actualizar la version en el archivo pyproject.toml
config_project["project"]["version"] = new_version

with open("pyproject.toml", "wb") as file:
    # guardar el archivo con la nueva version
    tomli_w.dump(config_project, file)

print(f"Version actual: {actual_version}")

command = "python setup.py sdist bdist_wheel"

subprocess.run(command, shell=True)

command = "twine upload dist/*"

subprocess.run(command, shell=True)

