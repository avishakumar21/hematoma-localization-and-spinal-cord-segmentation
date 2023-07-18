from ultralytics import YOLO
import csv

def extract_coordinates(oppath, results):
    # Open the CSV file in write mode
    with open(oppath, 'w', newline='') as csvfile:
        fieldnames = ['File Name', 'X1', 'Y1', 'X2', 'Y2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        for i, result in enumerate(results):
            try:
            # print(f'Coordinates are: {result.boxes.xyxy[0]}')
                filename = 'zoomed_dicom-'+str(i+1).zfill(3)+'.png'
                res = result.boxes.xyxy[0]  
                writer.writerow({'File Name': filename, 'X1': float(res[0]), 'Y1': float(res[1]), 'X2': float(res[2]), 'Y2': float(res[3])})
            except:
                print(f'No hematoma detected in {filename}')
                writer.writerow({'File Name': filename, 'X1': 0.0, 'Y1': 0.0, 'X2': 0.0, 'Y2': 0.0})
                continue

if __name__ == '__main__':
    model = YOLO("C:/Users/kkotkar1/Desktop/HematomaDetection/yolov8/runs/detect/train4/weights/best.pt")

    results = model(
        # "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25/",
        "C:/Users/kkotkar1/Desktop/CropAllDicoms/DicomSlicesPNG",
        save=True,
        max_det=1,
        # save_crop=True,
        device="0",
        save_txt=True,
        )
    
    # extract_coordinates("C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25/coordinates.csv", results)
    extract_coordinates("C:/Users/kkotkar1/Desktop/HematomaDetection/yolov8/runs/detect/predict17/coordinates.csv", results)
