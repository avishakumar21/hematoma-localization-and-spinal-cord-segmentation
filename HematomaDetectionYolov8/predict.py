from ultralytics import YOLO

def run_predict(imagepath):

    model = YOLO("runs/detect/train3/weights/best.pt")

    results = model(
        imagepath,
        save=True,
        max_det=1,
        device="0",
        save_txt=True,
        conf=0.7,
        )

# results = model("C:/Users/kkotkar1/Desktop/CropAllDicoms/DicomSlicesPNG", save=True)

if __name__ == "__main__":
    ippath = 'custom_data_prepost/images/test'
    run_predict(ippath)