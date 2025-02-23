try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    # pip install tomli
    import tomli as tomllib  # Python <3.11

from pathlib import Path
from setuptools import setup, find_packages

with open("pyproject.toml", "rb") as file:
    config_project = tomllib.load(file)


print(config_project)


# leer el archivo requerimientos.txt en una lista
with open("requirements.txt", "r") as file:
    requirements = file.readlines()
    # sino existe el archivo o esta vacio, se asigna una lista vacia
    if not requirements:
        requirements = []


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name=config_project["project"]["name"],  # Nombre del paquete en PyPI
    version=config_project["project"]["version"],
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Build Tools",
        "Intended Audience :: Developers",
    ],
    description=config_project["project"]["description"],
    long_description_content_type="text/markdown",
    long_description=long_description,
    url=f"https://github.com/wisrovi/{config_project['project']['name']}",
    author="William Steve Rodriguez Villamizar",
    author_email="wisrovi.rodriguez@gmail.com",
    license="MIT",
    python_requires=">=3.6, <3.12",  # Requiere Python >=3.6 y <3.10
)

