from tqdm import tqdm
from ultralytics import YOLO
import yaml
import os
from glob import glob


MODEL = "yolov8n-seg.pt"


def load_config():
    with open("/demo/config_train.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config_train = load_config()["train"]

    model = YOLO(MODEL)
    model.train(**config_train)

    if "val" in load_config():
        config_val = load_config()["val"]
        model.val(**config_val)

    if "test" in load_config():
        config_test = load_config()["test"]

        CWD = os.getcwd()
        for image_path in tqdm(glob(f"{CWD}/test/*/*"), desc="Testing", unit="Images"):
            model.predict(**config_test, source=image_path)
