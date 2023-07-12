from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO('C:/Users/kkotkar1/Desktop/HematomaDetection/yolov8/runs/detect/train2/weights/best.pt')  # load a custom model

    # Validate the model
    metrics = model.val(
        split='test',
        device='0',
        iou=0.9,
        )  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category
