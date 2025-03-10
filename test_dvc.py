import dvc.api
import cv2
import io
from PIL import Image
import numpy as np


def obtener_imagenes_dvc(repo=".", path="images.dvc", rev="main"):
    """
    Obtiene imágenes desde un archivo DVC y las carga con OpenCV.

    Args:
        repo (str): Ruta al repositorio DVC.
        path (str): Ruta al archivo .dvc.
        rev (str): Revisión (rama, etiqueta, commit).

    Returns:
        list: Lista de imágenes cargadas con OpenCV.
    """
    try:
        urls = dvc.api.get_url(
            path=path, repo=repo, rev=rev, remote="storage"
        )  # obtengo la url del remote storage.
        paths = dvc.api.get_deps(
            path=path, repo=repo, rev=rev
        )  # obtengo los paths de las imagenes.
        imagenes = []
        for p in paths:
            path_image = p["path"]
            data = dvc.api.read(
                path=path_image, repo=repo, rev=rev, remote="storage", mode="rb"
            )  # leo la imagen en binario desde el remote storage.
            imagen_pil = Image.open(io.BytesIO(data))  # abro la imagen con PIL
            imagen_cv2 = cv2.cvtColor(
                np.array(imagen_pil), cv2.COLOR_RGB2BGR
            )  # convierto la imagen a cv2
            imagenes.append(imagen_cv2)
        return imagenes

    except Exception as e:
        print(f"Error al obtener las imágenes de DVC: {e}")
        return []


# Ejemplo de uso
imagenes = obtener_imagenes_dvc(
    path="data/datasets/clasificacion/colorball.v8i.multiclass.dvc"
)

if imagenes:
    for i, imagen in enumerate(imagenes):
        cv2.imshow(f"Imagen {i}", imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
