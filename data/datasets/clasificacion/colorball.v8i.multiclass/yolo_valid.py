import os
from datetime import datetime

import cv2
import fiftyone as fo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO


class YOLODataValidator:
    def __init__(self, image_folder, label_folder, pdf_name, data_yaml, verbose=False):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.pdf_name = pdf_name
        self.data_yaml = data_yaml
        self.report_folder = "reports"
        self.verbose = verbose
        os.makedirs(self.report_folder, exist_ok=True)
        self._configure_logger()

    def _configure_logger(self):
        logger.remove()
        log_level = "DEBUG" if self.verbose else "INFO"
        logger.add(
            os.path.join(self.report_folder, "validation.log"),
            level=log_level,
            rotation="1 day",
            retention="7 days",
        )
        logger.add(lambda msg: print(msg, end=""), level=log_level)

    def analyze_class_distribution(self):
        class_counts = {}

        if os.path.exists(self.label_folder):
            for label_file in os.listdir(self.label_folder):
                if label_file.endswith(".txt"):
                    with open(os.path.join(self.label_folder, label_file), "r") as f:
                        for line in f:
                            class_id = int(line.split()[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        else:
            self.label_folder = [
                folder
                for folder in os.listdir(self.image_folder)
                if os.path.isdir(os.path.join(self.image_folder, folder))
            ]
            for label_file in self.label_folder:
                class_counts[label_file] = len(
                    os.listdir(os.path.join(self.image_folder, label_file))
                )

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.xlabel("Class ID")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.savefig(os.path.join(self.report_folder, "class_distribution.png"))
        plt.close()

        return class_counts

    def analyze_image_sizes(self):
        widths, heights = [], []
        for image_file in os.listdir(self.image_folder):
            if image_file.endswith((".jpg", ".png")):
                with Image.open(os.path.join(self.image_folder, image_file)) as img:
                    widths.append(img.width)
                    heights.append(img.height)

        plt.figure(figsize=(10, 6))
        sns.histplot(widths, bins=30, color="blue", label="Width")
        sns.histplot(heights, bins=30, color="red", label="Height")
        plt.xlabel("Pixels")
        plt.ylabel("Frequency")
        plt.title("Image Size Distribution")
        plt.legend()
        plt.savefig(os.path.join(self.report_folder, "image_size_distribution.png"))
        plt.close()

        return widths, heights

    def analyze_bbox_areas(self):
        areas = []

        if self.label_folder is None:
            for label_file in os.listdir(self.label_folder):
                if label_file.endswith(".txt"):
                    with open(os.path.join(self.label_folder, label_file), "r") as f:
                        for line in f:
                            _, x_center, y_center, box_width, box_height = map(
                                float, line.split()
                            )
                            areas.append(box_width * box_height)

        plt.figure(figsize=(10, 6))
        sns.histplot(areas, bins=30, color="green")
        plt.xlabel("Bounding Box Area (normalized)")
        plt.ylabel("Frequency")
        plt.title("Bounding Box Area Distribution")
        plt.savefig(os.path.join(self.report_folder, "bbox_area_distribution.png"))
        plt.close()

        return areas

    def analyze_aspect_ratios(self):
        aspect_ratios = []
        for image_file in os.listdir(self.image_folder):
            if image_file.endswith((".jpg", ".png")):
                with Image.open(os.path.join(self.image_folder, image_file)) as img:
                    aspect_ratios.append(img.width / img.height)

        plt.figure(figsize=(10, 6))
        sns.histplot(aspect_ratios, bins=30, color="purple")
        plt.xlabel("Aspect Ratio (Width/Height)")
        plt.ylabel("Frequency")
        plt.title("Image Aspect Ratio Distribution")
        plt.savefig(os.path.join(self.report_folder, "aspect_ratio_distribution.png"))
        plt.close()

        return aspect_ratios

    def detect_duplicates_and_overlaps(self):
        if self.label_folder is None:
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.YOLOv5Dataset,
                data_path=self.image_folder,
                labels_path=self.label_folder,
                yaml_path=self.data_yaml,
            )
        else:

            class Image_classification(BaseModel):
                filepath: str
                filename: str

            class Dataset:
                dataset = []

                def from_dir(self, data_path):
                    self.dataset = []
                    for class_name in [
                        folder
                        for folder in os.listdir(data_path)
                        if os.path.isdir(os.path.join(data_path, folder))
                    ]:
                        for image_path in os.listdir(
                            os.path.join(data_path, class_name)
                        ):
                            self.dataset.append(
                                Image_classification(
                                    filepath=os.path.join(
                                        data_path, class_name, image_path
                                    ),
                                    filename=image_path,
                                )
                            )
                    return self.dataset

            dataset = Dataset().from_dir(self.image_folder)

        image_hashes = {}
        duplicates = []
        for sample in dataset:
            try:
                img = cv2.imread(sample.filepath)
            except Exception:
                continue

            img_hash = hash(img.tobytes())
            if img_hash in image_hashes:
                duplicates.append((sample.filename, image_hashes[img_hash]))
            else:
                image_hashes[img_hash] = sample.filename
        return duplicates

    def validate_yolo_format(self, data_yaml):
        if self.label_folder is None:
            model = YOLO("yolov8n.pt")
        else:
            model = YOLO("yolov8n-cls.pt")
            
        # model.check_dataset(data_yaml)
        
        # for split in ["val", "test", "train"]:
        #     model.load_data(data_yaml, split=split, visualize=True, save_dir=self.report_folder)
            
        model.tune(data=data_yaml, epochs=1, iterations=3, batch=1, imgsz=640)
        
        model.train(data=data_yaml, epochs=1, imgsz=640, batch=1)

        results = model.val(
            data=data_yaml,
            save=True,
            verbose=False,
            plots=False,
            project=os.path.join(self.report_folder, "runs"),
            name="val",
        )
        return results.results_dict

    def validate_image_quality(self):
        corrupt_images, small_images = [], []
        for image_file in os.listdir(self.image_folder):
            if image_file.endswith((".jpg", ".png")):
                image_path = os.path.join(self.image_folder, image_file)
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        corrupt_images.append(image_file)
                    else:
                        height, width, _ = img.shape
                        if width < 64 or height < 64:
                            small_images.append((image_file, width, height))
                except Exception:
                    corrupt_images.append(image_file)
        return corrupt_images, small_images

    def generate_example_mosaics(self, num_mosaics=3):
        image_files = [
            f for f in os.listdir(self.image_folder) if f.endswith((".jpg", ".png"))
        ]

        if self.label_folder is None:
            label_files = {
                f.replace(".jpg", ".txt").replace(".png", ".txt"): f
                for f in os.listdir(self.label_folder)
                if f.endswith(".txt")
            }

        mosaics = []
        for mosaic_idx in range(num_mosaics):
            mosaic_images = []
            for _ in range(9):
                if not image_files:
                    break
                img_file = image_files.pop(0)

                if isinstance(self.label_folder, list):
                    label_file = img_file
                else:
                    label_file = img_file.replace(".jpg", ".txt").replace(
                        ".png", ".txt"
                    )
                    if label_file not in label_files:
                        continue

                img_path, label_path = os.path.join(
                    self.image_folder, img_file
                ), os.path.join(self.label_folder, label_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                with open(label_path, "r") as f:
                    for line in f:
                        class_id, x_center, y_center, box_width, box_height = map(
                            float, line.split()
                        )
                        h, w, _ = img.shape
                        x1 = int((x_center - box_width / 2) * w)
                        y1 = int((y_center - box_height / 2) * h)
                        x2 = int((x_center + box_width / 2) * w)
                        y2 = int((y_center + box_height / 2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            img,
                            str(int(class_id)),
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                mosaic_images.append(img)
            if mosaic_images:
                mosaic = np.zeros_like(mosaic_images[0])
                rows = []
                for row_idx in range(3):
                    row = np.hstack(mosaic_images[row_idx * 3 : (row_idx + 1) * 3])
                    rows.append(row)
                mosaic = np.vstack(rows)
                mosaic_filename = os.path.join(
                    self.report_folder, f"mosaic_{mosaic_idx}.png"
                )
                cv2.imwrite(mosaic_filename, mosaic)
                mosaics.append(mosaic_filename)
        return mosaics

    def analyze_bbox_aspect_ratios(self):
        aspect_ratios = []

        if self.label_folder is None:
            for label_file in os.listdir(self.label_folder):
                if label_file.endswith(".txt"):
                    with open(os.path.join(self.label_folder, label_file), "r") as f:
                        for line in f:
                            _, _, _, box_width, box_height = map(float, line.split())
                            if box_height != 0:  # Evitar división por cero
                                aspect_ratios.append(box_width / box_height)
        plt.figure(figsize=(10, 6))
        sns.histplot(aspect_ratios, bins=30, color="orange")
        plt.xlabel("Bounding Box Aspect Ratio (Width/Height)")
        plt.ylabel("Frequency")
        plt.title("Bounding Box Aspect Ratio Distribution")
        plt.savefig(
            os.path.join(self.report_folder, "bbox_aspect_ratio_distribution.png")
        )
        plt.close()
        return aspect_ratios

    def analyze_bbox_center_positions(self):
        x_centers, y_centers = [], []

        if self.label_folder is None:
            for label_file in os.listdir(self.label_folder):
                if label_file.endswith(".txt"):
                    with open(os.path.join(self.label_folder, label_file), "r") as f:
                        for line in f:
                            _, x_center, y_center, _, _ = map(float, line.split())
                            x_centers.append(x_center)
                            y_centers.append(y_center)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(x_centers, bins=30, color="purple")
        plt.xlabel("Bounding Box X Center (normalized)")
        plt.ylabel("Frequency")
        plt.title("Bounding Box X Center Distribution")
        plt.subplot(1, 2, 2)
        sns.histplot(y_centers, bins=30, color="blue")
        plt.xlabel("Bounding Box Y Center (normalized)")
        plt.ylabel("Frequency")
        plt.title("Bounding Box Y Center Distribution")
        plt.savefig(
            os.path.join(self.report_folder, "bbox_center_position_distribution.png")
        )
        plt.close()
        return x_centers, y_centers

    def analyze_bbox_width_height(self):
        widths, heights = [], []
        if self.label_folder is None:
            for label_file in os.listdir(self.label_folder):
                if label_file.endswith(".txt"):
                    with open(os.path.join(self.label_folder, label_file), "r") as f:
                        for line in f:
                            _, _, _, box_width, box_height = map(float, line.split())
                            widths.append(box_width)
                            heights.append(box_height)

        plt.figure(figsize=(8, 8))
        plt.scatter(widths, heights, alpha=0.5)
        plt.xlabel("Bounding Box Width (normalized)")
        plt.ylabel("Bounding Box Height (normalized)")
        plt.title("Bounding Box Width vs Height")
        plt.grid(True)
        plt.savefig(os.path.join(self.report_folder, "bbox_width_height.png"))
        plt.close()
        return widths, heights

    # 8. Generar reporte en PDF
    def generate_pdf_report(
        self,
        class_distribution,
        image_sizes,
        bbox_areas,
        aspect_ratios,
        duplicates,
        validation_results,
        corrupt_images,
        small_images,
        bbox_aspect_ratios,
        bbox_center_positions,
        bbox_width_height,
    ):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_title("Reporte de Validación de Datos YOLO")
        pdf.cell(200, 10, txt="Reporte de Validación de Datos YOLO", ln=True, align="C")
        pdf.ln(10)
        pdf.cell(
            200,
            10,
            txt=f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ln=True,
        )
        pdf.ln(10)
        pdf.add_page()
        pdf.cell(200, 10, txt="Distribución de Clases:", ln=True)
        pdf.image(
            os.path.join(self.report_folder, "class_distribution.png"),
            x=10,
            y=pdf.get_y(),
            w=180,
        )
        pdf.ln(100)
        pdf.add_page()
        pdf.cell(200, 10, txt="Distribución de Tamaños de Imágenes:", ln=True)
        pdf.image(
            os.path.join(self.report_folder, "image_size_distribution.png"),
            x=10,
            y=pdf.get_y(),
            w=180,
        )
        pdf.ln(100)
        pdf.add_page()
        pdf.cell(200, 10, txt="Distribución de Áreas de Bounding Boxes:", ln=True)
        pdf.image(
            os.path.join(self.report_folder, "bbox_area_distribution.png"),
            x=10,
            y=pdf.get_y(),
            w=180,
        )
        pdf.ln(100)
        pdf.add_page()
        pdf.cell(200, 10, txt="Distribución de Relaciones de Aspecto:", ln=True)
        pdf.image(
            os.path.join(self.report_folder, "aspect_ratio_distribution.png"),
            x=10,
            y=pdf.get_y(),
            w=180,
        )
        pdf.ln(100)
        pdf.add_page()
        pdf.cell(
            200,
            10,
            txt="Distribución de Relaciones de Aspecto de Bounding Boxes:",
            ln=True,
        )
        pdf.image(
            os.path.join(self.report_folder, "bbox_aspect_ratio_distribution.png"),
            x=10,
            y=pdf.get_y(),
            w=180,
        )
        pdf.ln(100)
        pdf.add_page()
        pdf.cell(
            200,
            10,
            txt="Distribución de Posiciones Centrales de Bounding Boxes:",
            ln=True,
        )
        pdf.image(
            os.path.join(self.report_folder, "bbox_center_position_distribution.png"),
            x=10,
            y=pdf.get_y(),
            w=180,
        )
        pdf.ln(100)
        pdf.add_page()
        pdf.cell(
            200,
            10,
            txt="Gráfico de Dispersión de Ancho vs. Alto de Bounding Boxes:",
            ln=True,
        )
        pdf.image(
            os.path.join(self.report_folder, "bbox_width_height.png"),
            x=10,
            y=pdf.get_y(),
            w=180,
        )
        pdf.ln(100)
        pdf.add_page()
        pdf.cell(200, 10, txt="Validación de Calidad de Imágenes:", ln=True)
        if corrupt_images:
            pdf.cell(200, 10, txt="Imágenes corruptas encontradas:", ln=True)
            for img in corrupt_images:
                pdf.cell(200, 10, txt=f"- {img}", ln=True)
        else:
            pdf.cell(200, 10, txt="No se encontraron imágenes corruptas.", ln=True)
        if small_images:
            pdf.cell(
                200, 10, txt="Imágenes con dimensiones pequeñas encontradas:", ln=True
            )
            for img, width, height in small_images:
                pdf.cell(200, 10, txt=f"- {img} ({width}x{height})", ln=True)
        else:
            pdf.cell(
                200,
                10,
                txt="No se encontraron imágenes con dimensiones pequeñas.",
                ln=True,
            )
        pdf.add_page()
        pdf.cell(200, 10, txt="Mosaicos de Imágenes de Ejemplo:", ln=True)
        mosaics = self.generate_example_mosaics()
        for mosaic_file in mosaics:
            pdf.image(mosaic_file, x=10, y=pdf.get_y(), w=180)
            pdf.ln(130)
            pdf.add_page()
        pdf.cell(200, 10, txt="Imágenes Duplicadas:", ln=True)
        if duplicates:
            for dup in duplicates:
                pdf.cell(200, 10, txt=f"- {dup[0]} y {dup[1]}", ln=True)
        else:
            pdf.cell(200, 10, txt="No se encontraron imágenes duplicadas.", ln=True)
        pdf.ln(10)
        pdf.add_page()
        pdf.cell(200, 10, txt="Resultados de Validación con Ultralytics:", ln=True)
        pdf.multi_cell(190, 8, txt=str(validation_results))
        pdf.ln(15)
        pdf.set_y(-15)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(
            0,
            10,
            f"© {datetime.now().year} wisrovi rodriguez. Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            0,
            0,
            "C",
        )
        report_path = os.path.join(self.report_folder, self.pdf_name)
        pdf.output(report_path)
        logger.info(f"Reporte generado en {report_path}")

    def run_validation(self, data_yaml):
        self.data_yaml = data_yaml
        try:
            logger.info("Iniciando validación...")
            logger.info("Validando distribución de clases...")
            class_distribution = self.analyze_class_distribution()
            logger.info("Analizando tamaños de imágenes...")
            image_widths, image_heights = self.analyze_image_sizes()
            logger.info("Analizando áreas de bounding boxes...")
            bbox_areas = self.analyze_bbox_areas()
            logger.info("Analizando relaciones de aspecto de imágenes...")
            aspect_ratios = self.analyze_aspect_ratios()
            logger.info("Detectando imágenes duplicadas y overlaps...")
            duplicates = self.detect_duplicates_and_overlaps()
            logger.info("Validando formato de datos con Ultralytics...")
            validation_results = self.validate_yolo_format(data_yaml)
            logger.info("Validando calidad de imágenes...")
            corrupt_images, small_images = self.validate_image_quality()
            logger.info("Analizando relaciones de aspecto de bounding boxes...")
            bbox_aspect_ratios = self.analyze_bbox_aspect_ratios()
            logger.info("Analizando posiciones centrales de bounding boxes...")
            bbox_center_positions = self.analyze_bbox_center_positions()
            logger.info("Analizando ancho vs alto de bounding boxes...")
            bbox_width_height = self.analyze_bbox_width_height()
            logger.info("Generando reporte en PDF...")
            self.generate_pdf_report(
                class_distribution,
                (image_widths, image_heights),
                bbox_areas,
                aspect_ratios,
                duplicates,
                validation_results,
                corrupt_images,
                small_images,
                bbox_aspect_ratios,
                bbox_center_positions,
                bbox_width_height,
            )
            logger.info("Validación completada.")
        except Exception as e:
            logger.exception(f"Error durante la validación: {e}")


if __name__ == "__main__":

    image_folder = "train/images"  # para deteccion o segmentacion
    data_yaml = "data.yaml"  # para deteccion o segmentacion

    image_folder = "train"  # para clasificacion

    label_folder = "train/labels"
    pdf_name = "validation_report.pdf"

    validator = YOLODataValidator(image_folder, label_folder, pdf_name, data_yaml)

    # validator.run_validation(data_yaml)  # para deteccion o segmentacion
    validator.run_validation("./")  # para clasificacion


# pip install fiftyone
# pip install fpdf
