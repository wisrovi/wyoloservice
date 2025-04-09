from pathlib import Path
from setuptools import setup, find_packages

# this_directory = Path(__file__).parent
# long_description = (this_directory / "README.md").read_text()

setup(
    name="train_yolo",  # El nombre de tu paquete
    version="0.1.0",  # La versión de tu paquete
    description="trainer yolo",
    author="William Steve Rodriguez Villamizar",
    author_email="wrodriguez@ecapturedtech.com",
    packages=find_packages(),
    install_requires=[
        "boto3",
        "loguru",
        "mlflow",
        "dvc"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Build Tools",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.8",  # Requiere Python 3.6 o superior
    long_description_content_type="text/markdown",
    long_description="# AIDIAGNOST states",  # long_description,
    license="MIT",
    url="https://github.com/cimacorporate/001-aidiagnost",
)