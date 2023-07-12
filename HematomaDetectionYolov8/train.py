from ultralytics import YOLO
import torch
   
if __name__ == "__main__":    
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)


    # Use the model
    model.train(
        data="config.yaml",
        epochs=20,
        device='0',
        patience=5,
        batch=8,
        imgsz=(960, 1280),
        save=True,
        pretrained='yolov8m.pt',
        val=True,
        lrf=0.001,
        )  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("custom_data/images/test/dicom-004.png")  # predict on an image
    # path = model.export(format='torchscript')