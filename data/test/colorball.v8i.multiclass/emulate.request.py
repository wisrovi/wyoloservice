import requests

# URL del endpoint
url = "http://api_user:8000/train/"

# Parámetros de la solicitud
params = {"user_code": "color_ball"}

# Archivo a enviar
files = {
    "file": (
        "config_train.yaml",
        open(
            "/datasets/clasificacion/colorball.v8i.multiclass/config_train.yaml", "rb"
        ),
        "application/x-yaml",
    )
}

# Encabezados de la solicitud
headers = {"accept": "application/json"}

# Realizar la solicitud POST
response = requests.post(url, params=params, headers=headers, files=files)

# Imprimir la respuesta
print(response.status_code)
print(response.json())
