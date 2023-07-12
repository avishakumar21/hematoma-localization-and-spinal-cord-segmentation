import cv2
import numpy as np
import pydicom as pd
import os

def get_bounding_box_coordinates(image_path):
    image = image_path

    mask = np.asarray(np.where((image[:,:,0] > 127 ) & (image[:,:,1] > 127) & (image[:,:,2] <= 127), 255, 0), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h

def convert_to_yolo_format(image_width, image_height, min_x, min_y, width, height):
    # Calculate bounding box center coordinates
    bbox_width = width
    bbox_height = height
    bbox_center_x = min_x + (bbox_width / 2)
    bbox_center_y = min_y + (bbox_height / 2)

    # Normalize coordinates between 0 and 1
    norm_bbox_center_x = bbox_center_x / image_width
    norm_bbox_center_y = bbox_center_y / image_height
    norm_bbox_width = bbox_width / image_width
    norm_bbox_height = bbox_height / image_height

    return norm_bbox_center_x, norm_bbox_center_y, norm_bbox_width, norm_bbox_height

def write_yolo_coordinates_to_file(filename, yolo_bbox):
        with open(filename, 'w') as file:
            file.write(f"0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}")

def entire_directory(input_dir, output_dir):
    files = os.listdir(input_dir)
    for file in files: 
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file.replace('.dcm', '.txt'))
        img = pd.dcmread(input_file).pixel_array
        img_h = img.shape[0]
        img_w = img.shape[1]
        x, y, w, h = get_bounding_box_coordinates(img)
        yolo_bbox = convert_to_yolo_format(img_w, img_h, x, y, w, h)
        write_yolo_coordinates_to_file(output_file, yolo_bbox)
    
    return 0




if __name__ == "__main__":
     
    input_dir = "C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData/GroundTruthFixedCropDicom"
    output_dir = "C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData/LabelsFixedCrop"
    
    entire_directory(input_dir, output_dir)
    print("Done!")