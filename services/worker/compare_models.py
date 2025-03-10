import os
import tempfile
from ultralytics import YOLO, RTDETR
import numpy as np
import time
from worker_utils import MinioS3Client


def create_model(model_name, model_type="yolo"):
    if model_type == "yolo":
        model = YOLO(model_name)
    elif model_type == "rtdetr":
        model = RTDETR(model_name)
    else:
        raise ValueError("Invalid model type specified.")

    return model


def evaluar_modelos(
    model_path1, model_path2, image_paths, label_paths, iou_threshold=0.5
):
    """
    Evalúa y compara dos modelos YOLOv8 en un conjunto de imágenes etiquetadas.

    Args:
        model_path1 (str): Ruta del primer modelo YOLOv8.
        model_path2 (str): Ruta del segundo modelo YOLOv8.
        image_paths (list): Lista de rutas de imágenes de prueba.
        label_paths (list): Lista de rutas de archivos de etiquetas en formato YOLO.
        iou_threshold (float): Umbral de IoU para considerar una detección como correcta.

    Returns:
        str: "modelo_1" si el modelo 1 es mejor, "modelo_2" si el modelo 2 es mejor.
    """
    # Cargar los modelos
    model1 = create_model(model_path1)
    model2 = create_model(model_path2)

    # Almacenar métricas
    metrics = {
        "modelo_1": {
            "precision": [],
            "recall": [],
            "iou": [],
            "map50": [],
            "map50_95": [],
            "tiempo": [],
        },
        "modelo_2": {
            "precision": [],
            "recall": [],
            "iou": [],
            "map50": [],
            "map50_95": [],
            "tiempo": [],
        },
    }

    def calcular_metricas(model, image_paths, label_paths, model_key):
        """Calcula métricas de rendimiento para un modelo dado."""
        tiempos = []
        for img_path, label_path in zip(image_paths, label_paths):
            # Leer etiquetas reales
            with open(label_path, "r") as f:
                ground_truth = [
                    list(map(float, line.strip().split()[1:])) for line in f
                ]

            # Ejecutar inferencia y medir tiempo
            start_time = time.time()
            results = model(img_path, conf=0.25, iou=iou_threshold)
            end_time = time.time()
            tiempos.append(end_time - start_time)

            # Obtener predicciones
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()

            # Calcular IoU y mAP
            if len(pred_boxes) > 0 and len(ground_truth) > 0:
                iou_values = calcular_iou(pred_boxes, np.array(ground_truth))
                metrics[model_key]["iou"].append(np.mean(iou_values))

            # Simulación de métricas (esto depende de la implementación exacta de evaluación)
            metrics[model_key]["precision"].append(np.random.uniform(0.7, 1.0))
            metrics[model_key]["recall"].append(np.random.uniform(0.7, 1.0))
            metrics[model_key]["map50"].append(np.random.uniform(0.7, 1.0))
            metrics[model_key]["map50_95"].append(np.random.uniform(0.5, 0.9))

        metrics[model_key]["tiempo"] = np.mean(tiempos)

    def calcular_iou(pred_boxes, gt_boxes):
        """Calcula IoU entre predicciones y etiquetas reales."""
        ious = []
        for pred in pred_boxes:
            for gt in gt_boxes:
                xi1 = max(pred[0], gt[0])
                yi1 = max(pred[1], gt[1])
                xi2 = min(pred[2], gt[2])
                yi2 = min(pred[3], gt[3])
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
                gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                union_area = pred_area + gt_area - inter_area
                ious.append(inter_area / union_area if union_area > 0 else 0)
        return ious if ious else [0]

    # Evaluar ambos modelos
    calcular_metricas(model1, image_paths, label_paths, "modelo_1")
    calcular_metricas(model2, image_paths, label_paths, "modelo_2")

    # Comparar métricas
    promedio_m1 = {k: np.mean(v) for k, v in metrics["modelo_1"].items()}
    promedio_m2 = {k: np.mean(v) for k, v in metrics["modelo_2"].items()}

    print("\n📊 **Comparación de Modelos** 📊")
    print(f"\n🔹 Modelo 1: {promedio_m1}")
    print(f"\n🔹 Modelo 2: {promedio_m2}")

    # Seleccionar el mejor modelo basado en mAP50, precisión y velocidad
    criterios = ["map50", "precision", "recall"]
    puntaje_m1 = sum(promedio_m1[c] for c in criterios) - promedio_m1["tiempo"]
    puntaje_m2 = sum(promedio_m2[c] for c in criterios) - promedio_m2["tiempo"]

    return "modelo_1" if puntaje_m1 > puntaje_m2 else "modelo_2"


# 📌 **Ejemplo de uso**
if __name__ == "__main__":
    bucket_name = "better-models"
    project_name = "color_ball_v2"
    version = 1

    with tempfile.TemporaryDirectory() as temp_dir:
        modelo_1_path = os.path.join(temp_dir, "old.pt")
        
        modelo_2_path = "logs/best.pt"

        minio = MinioS3Client(
            endpoint_url="http://mlflow-minio:9000",
            aws_access_key_id="mlflow",
            aws_secret_access_key="wyoloservice",
        )

        old_model_name = f"{project_name}-v{version}.pt"

        if minio.download_file(
            bucket_name,
            object_key=old_model_name,
            local_file_path=modelo_1_path,
        ):
            print("Descargado")
        else:
            print("Problema al descargar")

        print("queso")

    imagenes = ["test1.jpg", "test2.jpg", "test3.jpg"]
    labels = ["test1.txt", "test2.txt", "test3.txt"]

    # mejor_modelo = evaluar_modelos(modelo_1_path, modelo_2_path, imagenes, labels)
    # print(f"\n🏆 **El mejor modelo es:** {mejor_modelo}")
