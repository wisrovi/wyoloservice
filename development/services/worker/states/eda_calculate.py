import functools
import os
import subprocess
import threading
import time

from loguru import logger


def ejecutar_en_hilo(func):
    """
    Decorador para ejecutar una función en un hilo separado.
    """

    @functools.wraps(func)
    def envoltorio(*args, **kwargs):
        hilo = threading.Thread(
            target=func,
            args=args,
            kwargs=kwargs,
            daemon=True,
        )
        hilo.start()
        return hilo

    return envoltorio


class Eda_calculate:

    __NAME__ = "Eda_calculate"
    __VERSION__ = "v1.0"

    @ejecutar_en_hilo
    def eda(self):
        dataset = os.path.dirname(self.config.get("train", {}).get("data"))
        project = self.config.get("sweeper", {}).get("study_name")
        author = self.config.get("metadata", {}).get("author")
        content = self.config.get("metadata", {}).get("content")

        args = [
            "python",
            "/app/worker/eda.py",
            "--dataset",
            f"{dataset}",
            "--project",
            f'"{project}"',
            "--author",
            f'"{author}"',
            "--content",
            f'"{content}"',
        ]

        with subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combinar stderr con stdout
            text=True,
            bufsize=1,  # Line-buffered (1 línea a la vez)
            cwd=os.getcwd(),
        ) as process:

            # Leer la salida en tiempo real
            for line in process.stdout:
                logger.debug(line)

    def __call__(self, args_dict: dict):
        self.config = args_dict

        try:
            self.eda()
        except Exception as e:
            logger.error(f"Error al ejecutar EDA: {e}")
            return {"eda": False}

        for _ in range(15):
            print("Waiting for EDA to finish...")
            time.sleep(1)

        return {"eda": True}
