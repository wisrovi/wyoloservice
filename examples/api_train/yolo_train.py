from tqdm import tqdm
from ultralytics import YOLO
import yaml
import os
from glob import glob


MODEL = "yolov8n-cls.pt"
# MODEL = "/dataset/results/train/weights/best.pt"


def load_config():
    with open("config_train.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config_train = load_config()["train"]
    config_test = load_config()["test"]
    config_val = load_config()["val"]
    
    MODEL = load_config()["model"]

    model = YOLO(MODEL)
    # model.train(**config_train)
    # model.val(**config_val)

    CWD = os.getcwd()
    images = glob(f"{CWD}/test/*/*")
    images = [image for image in images if not image.endswith("txt")]
    for image_path in tqdm(images, desc="Testing", unit="Images"):
        model.predict(**config_test, source=image_path)
