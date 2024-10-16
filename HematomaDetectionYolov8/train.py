from ultralytics import YOLO
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
   
if __name__ == "__main__":    
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

    params = {
        'batch': 8,
        'lr': 0.0437,
        'epochs': 20,
    }

    print("MODEL INFO: ")  # print model YAML
    print(model.info())  # print model information
    print('#################')
        
    model.train(
        data="config.yaml",
        epochs=params['epochs'],
        device='0',
        batch = params['batch'],
        lr0 = params['lr'],
        imgsz=(320,320),
        save=True,
        pretrained='yolov8m.pt',
        val=True,
            # lrf=0.001,
            )  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    metrics
    # print(metrics)  # print all metrics, remove .print() to save to results.txt
    # results = model("custom_data/images/test/dicom-004.png")  # predict on an image
    # path = model.export(format='torchscript')