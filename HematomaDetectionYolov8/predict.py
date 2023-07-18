from ultralytics import YOLO


model = YOLO("C:/Users/kkotkar1/Desktop/HematomaDetection/yolov8/runs/detect/train4/weights/best.pt")

# results = model(
#     "C:/Users/kkotkar1/Desktop/HematomaDetection/yolov8/custom_data/images/test",
#     save=True,
#     max_det=1,
#     save_crop=True,
#     device="0",
#     )

results = model(
    "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25",
    save=True,
    max_det=1,
    save_crop=True,
    device="0",
    )


# results = model("C:/Users/kkotkar1/Desktop/CropAllDicoms/DicomSlicesPNG", save=True)
