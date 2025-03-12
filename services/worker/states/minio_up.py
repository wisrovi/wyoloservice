import os
import tempfile
import time
from loguru import logger
import numpy as np
from train_yolo import TrainingHistory, db
from worker_utils import MinioS3Client
from ultralytics import YOLO, RTDETR


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

    # Evaluar ambos modelos    # if model1.task == model2.task:
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

    return "m1" if puntaje_m1 > puntaje_m2 else "m2"


def save_best_model(
    task_id,
    model_path,
    version,
    results_model,
    project_name,
    s3: MinioS3Client,
):
    """Registra el mejor modelo en la base de datos y lo sube a MinIO."""

    update_model = True
    minio_url = None

    # TODO: falta validar el modelo anterior y compararlo con el modelo actual, si es mejor, se reemplaza, de lo contrario solo se suben los resultados del modelo actual

    # if bucket_exists(bucket_name):
    #     with tempfile.TemporaryDirectory() as temp_dir:
    #         old_files = list_objects(bucket_name)

    #         temp_file_path = os.path.join(temp_dir, "old.pt")

    #         if data_path and old_files and f"{project_name}.pt" in old_files:
    #             old_model = download_file(
    #                 bucket_name, f"{project_name}.pt", temp_file_path
    #             )

    #             better_model = comparar_modelos_yolo(
    #                 modelo1_path=old_model,
    #                 modelo2_path=model_path,
    #                 data_path=data_path,
    #             )

    #             if better_model == "old.pt":
    #                 update_model = False

    # if model_converted:
    #             mlflow.log_artifact(model_converted, artifact_path="models")

    if update_model:
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(f"{temp_dir}/{project_name}-v{version}.txt", "w") as f:
                f.write(f"{MinioS3Client.BUCKET_NAME}/{project_name}/{task_id}/*")

            s3.upload_file(
                file_path_local=f"{temp_dir}/{project_name}-v{version}.txt",
                bucket_name=f"better-{MinioS3Client.BUCKET_NAME}",
                name_file_s3=f"{project_name}-v{version}.txt",
            )

            minio_url = s3.upload_file(
                file_path_local=model_path,
                bucket_name=f"better-{MinioS3Client.BUCKET_NAME}",
                name_file_s3=f"{project_name}-v{version}.pt",
            )

            # TODO: subir el ONNXa better-models
            # s3.upload_file(
            #     file_path_local=f"{temp_dir}/{project_name}-v{version}.txt",
            #     bucket_name=f"better-{MinioS3Client.BUCKET_NAME}",
            #     name_file_s3=f"{project_name}-v{version}.txt",
            # )

        try:
            # TODO: validar que la actualizacion a la base de datos se haga correctamente

            # Verificar si el `task_id` existe en la base de datos
            existing_task = db.get_by_field(task_id=task_id)

            db.update(
                1,
                TrainingHistory(
                    id=existing_task,
                    task_id=f"{task_id}_v{version}",
                    model_path=minio_url,
                    status="completed",
                    recommended_model=minio_url,
                ),
            )
        except Exception as e:
            logger.error("❌ Error al actualizar la base de datos")

    s3.upload_folder(
        folder_path=results_model,
        bucket_name=f"{MinioS3Client.BUCKET_NAME}",
        prefix=f"{project_name}/{task_id}/",
    )

    logger.info(f"✅ Modelo recomendado guardado en MinIO: {minio_url}")

    return {"minio_url": minio_url}


def results_up_to_minio(task_data: dict):
    the_new_model_is_better = True

    try:
        s3_minio: MinioS3Client = MinioS3Client(
            endpoint_url=task_data.get("minio", {}).get("MINIO_ENDPOINT"),
            aws_access_key_id=task_data.get("minio", {}).get("MINIO_ID"),
            aws_secret_access_key=task_data.get("minio", {}).get("MINIO_SECRET_KEY"),
        )

        sweeper_config = task_data.get("sweeper", {})

        with tempfile.TemporaryDirectory() as temp_dir:
            modelo_1_path = os.path.join(temp_dir, "old.pt")

            project_name = sweeper_config.get("study_name")
            version = sweeper_config.get("version")
            old_model_name = f"{project_name}-v{version}.pt"

            if s3_minio.download_file(
                bucket_name="better-models",
                object_key=old_model_name,
                local_file_path=modelo_1_path,
            ):
                modelo_2_path = task_data["train"]["best_model_path"]
                print("Descargado el modelo anterior")

                try:
                    # TODO: terminar esta funcion
                    compare_results = evaluar_modelos(
                        model_path1=modelo_1_path,
                        model_path2=modelo_2_path,
                        image_paths=[],
                        label_paths=[],
                        iou_threshold=0.5,
                    )
                    if compare_results == "m1":
                        the_new_model_is_better = False

                except Exception as e:
                    logger.error("Error al comparar modelos")
                    the_new_model_is_better = True

                # Aca se compara el modelo descargado (anterior modelo) y el nuevo modelo
                # si el nuevo modelo es mejor que el anterior, se reemplaza
                # sino es mejor, se termina el proceso de este estado

        if the_new_model_is_better:
            results_train = task_data["train"]
            best_trial = results_train["best_trial"]
            best_model_path = results_train["best_model_path"]
            RESULT_PATH = results_train["result_path"]

            save_best_model(
                task_id=task_data["task_id"],
                project_name=sweeper_config.get("study_name", "default_study"),
                model_path=best_model_path,
                version=sweeper_config["version"],
                results_model=f"{RESULT_PATH}/{best_trial.number}/",
                s3=s3_minio,
            )
    except Exception as e:
        logger.error(f"❌ Error al guardar el mejor modelo: {e}")

    return {}
