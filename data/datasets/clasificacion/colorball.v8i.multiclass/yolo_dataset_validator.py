"""
pip install --upgrade pip setuptools wheel
pip install fpdf2
"""

from ultralytics import YOLO
import os
import yaml
import glob
import cv2
from loguru import logger
from fpdf import FPDF


class YOLODatasetValidator:
    dataset_type = None

    def __init__(
        self,
        dataset_path: str,
        log_file: str = "dataset_validation.log",
        min_img_size: int = 640,
    ):
        self.dataset_path = dataset_path
        self.dataset_yaml = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(self.dataset_yaml) is False:
            self.dataset_type = "classification"

        self.log_file = log_file
        self.min_img_size = min_img_size

        # Configurar loguru
        logger.add(
            self.log_file,
            rotation="1 day",
            level="INFO",
            format="{time} {level} {message}",
        )

        # Cargar la configuración del dataset
        if self.dataset_type  is None:
            with open(self.dataset_yaml, "r") as f:
                self.dataset_config = yaml.safe_load(f)
        else:
            self.dataset_config = {
                "names": os.listdir(self.dataset_path),
                "train": os.path.join(self.dataset_path, "train"),
                "val": os.path.join(self.dataset_path, "val"),
                "test": os.path.join(self.dataset_path, "test"),
            }
        

        self.model = YOLO("yolov8n.pt")

    def check_class_balance(self):
        """Verificar el balance de clases en el dataset."""
        class_counts = {name: 0 for name in self.dataset_config["names"]}
        for split in ["train", "val", "test"]:
            if split in self.dataset_config:
                image_files = glob.glob(
                    os.path.join(self.dataset_config[split], "*.txt")
                )
                for file in image_files:
                    with open(file, "r") as anno:
                        for line in anno.readlines():
                            class_idx = int(line.split()[0])
                            class_counts[self.dataset_config["names"][class_idx]] += 1
        return class_counts

    def check_images(self):
        """Verificar la calidad y validez de las imágenes."""
        corrupted_images = []
        low_quality_images = []
        for split in ["train", "val", "test"]:
            if split in self.dataset_config:
                image_files = glob.glob(
                    os.path.join(self.dataset_config[split], "*.jpg")
                ) + glob.glob(os.path.join(self.dataset_config[split], "*.png"))
                for img_path in image_files:
                    img = cv2.imread(img_path)
                    if img is None:
                        corrupted_images.append(img_path)
                    elif (
                        img.shape[0] < self.min_img_size
                        or img.shape[1] < self.min_img_size
                    ):
                        low_quality_images.append(img_path)
        return corrupted_images, low_quality_images

    def check_annotations(self):
        """Verificar la validez de las anotaciones."""
        invalid_annotations = []
        for split in ["train", "val", "test"]:
            if split in self.dataset_config:
                label_files = glob.glob(
                    os.path.join(self.dataset_config[split], "*.txt")
                )
                for label_file in label_files:
                    with open(label_file, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            try:
                                parts = line.strip().split()
                                class_idx = int(parts[0])
                                if class_idx < 0 or class_idx >= len(
                                    self.dataset_config["names"]
                                ):
                                    invalid_annotations.append(
                                        f"Invalid class index {class_idx} in {label_file}"
                                    )
                                if len(parts) != 5:
                                    invalid_annotations.append(
                                        f"Invalid annotation format in {label_file}"
                                    )
                            except Exception as e:
                                invalid_annotations.append(
                                    f"Error in {label_file}: {e}"
                                )
        return invalid_annotations

    def generate_pdf_report(self, validation_results):
        """Generar un informe en PDF con los resultados de la validación."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Dataset Validation Report", ln=True, align="C")
        pdf.ln(10)

        # Balance de clases
        pdf.cell(200, 10, txt="Class Balance:", ln=True)
        for class_name, count in validation_results["class_balance"].items():
            pdf.cell(200, 10, txt=f"{class_name}: {count}", ln=True)
        pdf.ln(5)

        # Imágenes corruptas
        pdf.cell(200, 10, txt="Corrupted Images:", ln=True)
        for img_path in validation_results["corrupted_images"]:
            pdf.cell(200, 10, txt=img_path, ln=True)
        pdf.ln(5)

        # Imágenes de baja calidad
        pdf.cell(200, 10, txt="Low Quality Images:", ln=True)
        for img_path in validation_results["low_quality_images"]:
            pdf.cell(200, 10, txt=img_path, ln=True)
        pdf.ln(5)

        # Anotaciones inválidas
        pdf.cell(200, 10, txt="Invalid Annotations:", ln=True)
        for annotation in validation_results["invalid_annotations"]:
            pdf.cell(200, 10, txt=annotation, ln=True)
        pdf.ln(5)

        # Si todo es válido, indicar que el dataset es apto para entrenamiento
        if validation_results["is_valid"]:
            pdf.cell(200, 10, txt="Dataset is valid for training.", ln=True)
        else:
            pdf.cell(200, 10, txt="Dataset is not valid for training.", ln=True)

        # Guardar el reporte
        pdf.output(os.path.join(self.dataset_path, "validation_report.pdf"))

    def __call__(self, params: dict = None):
        """Realizar todas las validaciones del dataset y generar el informe."""
        validation_results = {
            "class_balance": self.check_class_balance(),
            "corrupted_images": [],
            "low_quality_images": [],
            "invalid_annotations": [],
            "is_valid": True,
        }

        # Verificar imágenes y anotaciones
        (
            validation_results["corrupted_images"],
            validation_results["low_quality_images"],
        ) = self.check_images()
        validation_results["invalid_annotations"] = self.check_annotations()

        # Determinar si el dataset es válido
        if (
            validation_results["corrupted_images"]
            or validation_results["low_quality_images"]
            or validation_results["invalid_annotations"]
        ):
            validation_results["is_valid"] = False

        # Generar el reporte en PDF
        self.generate_pdf_report(validation_results)

        return validation_results


if __name__ == "__main__":
    # Inicializa el validador con la ruta del dataset
    validator = YOLODatasetValidator("./")
    # Llama al validador
    validation_results = validator()
    # Revisa los resultados
    print(validation_results)
