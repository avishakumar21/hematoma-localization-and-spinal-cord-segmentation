import csv
import os

def extract_coordinates(oppath, results):
    # Open the CSV file in write mode
    # if not os.path.exists(oppath):
    #     os.makedirs(oppath)
    with open(oppath, 'w', newline='') as csvfile:
        fieldnames = ['File Name', 'X1', 'Y1', 'X2', 'Y2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        for i, result in enumerate(results):
            # print(result)
            # input()
            try:
            # print(f'Coordinates are: {result.boxes.xyxy[0]}')
                filename = result[4]
                filename = filename.replace('.txt', '.png')
                # res = result.boxes.xywh[0]  
                writer.writerow({'File Name': filename, 'X1': float(result[0]), 'Y1': float(result[1]), 'X2': float(result[2]), 'Y2': float(result[3])})
            except:
                print(f'No hematoma detected in {filename}')
                writer.writerow({'File Name': filename, 'X1': 0.0, 'Y1': 0.0, 'X2': 0.0, 'Y2': 0.0})
                continue

def calculate_coordinates(file_path, image_width, image_height):
    # print(os.listdir(file_path))
    coordinates = []
    for file in os.listdir(file_path):
        filename = file_path + '/' + file
        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split(' ')
            class_id = int(data[0])
            x_center = float(data[1])
            y_center = float(data[2])
            width = float(data[3])
            height = float(data[4])

            # Convert normalized coordinates to absolute coordinates
            x1 = int((x_center - width/2) * image_width)
            y1 = int((y_center - height/2) *  image_height)
            x2 = int((x_center + width/2) * image_width)
            y2 = int((y_center + height/2) * image_height)
            
            coordinates.append((x1, y1, x2, y2, file))

    return coordinates

# Specify the path to the text file
text_file_path = "C:/Users/kkotkar1/Desktop/HematomaDetection/yolov8/runs/detect/predict16/labels"

# Specify the image width and height (in pixels)
image_width = 1280
image_height = 960

# Call the function to calculate the coordinates
result = calculate_coordinates(text_file_path, image_width, image_height)
extract_coordinates('C:/Users/kkotkar1/Desktop/HematomaDetection/yolov8/runs/detect/predict16/CalculatedCoordinates.csv', result)

# # Print the calculated coordinates
# for i, coordinate in enumerate(result, 1):
#     print(f"Coordinate {i}: X = {coordinate[0]}, Y = {coordinate[1]}") 
