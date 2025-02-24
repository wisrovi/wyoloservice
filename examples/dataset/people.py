# !pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="eE8kJi3wvnkEsw6FydSA")
project = rf.workspace("leo-ueno").project("people-detection-o4rdr")
version = project.version(8)
dataset = version.download("yolov8")
