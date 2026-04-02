import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("gazxard").project("pipe-crack-detection")
version = project.version(1)
dataset = version.download("yolov8")
