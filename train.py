from ultralytics import YOLO
import cv2
import numpy as np


def train_the_model():
    # Load a model
    model = YOLO("yolov8s-seg.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    # it will save the best.pt modulw in runs/weight/best.pt
    model.train(
        data=r"data.yaml", # This is your yaml after creating data using roboflow
        epochs=100,
        imgsz=640,
    )  # train the model
    
    metrics = model.val()  # evaluate model performance on the validation set
    results = model(r"D:\learning\final pipe counting module\slats_segmentation-1\valid\images\Copy-of-image6_png_jpg.rf.838beecc04ef8f8d924e7f0f1c630c6a.jpg", retina_masks=True)  # predict on an image

    print(results[0].masks)


train_the_model()