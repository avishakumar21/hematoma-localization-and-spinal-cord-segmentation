import os
import xml.etree.ElementTree as ET
from PIL import Image
import shutil



def move_xml_files(xml_path, train_path, val_path, test_path):
    index = 0
    total_size = len([entry for entry in os.listdir(xml_path) if os.path.isfile(os.path.join(xml_path, entry))])
    train_size = round(0.6*total_size)
    val_size = train_size + round(0.2*total_size)
    test_size = train_size + val_size + round(0.2*total_size)

    for image_filename in os.listdir(xml_path):
        if image_filename.lower().endswith(('.xml')):
            image_path = os.path.join(xml_path, image_filename)

            index += 1
            if index <= train_size:
                shutil.move(image_path, os.path.join(train_path, image_filename))
            if index > train_size and index <= val_size:
                shutil.move(image_path, os.path.join(val_path, image_filename))
            if index > val_size and index <= test_size:
                shutil.move(image_path, os.path.join(test_path, image_filename))


def move_png_files(png_path, train_path, val_path, test_path):
    index = 0
    total_size = len([entry for entry in os.listdir(png_path) if os.path.isfile(os.path.join(png_path, entry))])
    train_size = 0.6*total_size
    val_size = 0.2*total_size 
    test_size = 0.2*total_size 

    for image_filename in os.listdir(png_path):
        if image_filename.lower().endswith(('.png')):
            image_path = os.path.join(png_path, image_filename)

            index += 1
            if index <= train_size:
                shutil.move(image_path, os.path.join(train_path, image_filename))
            if index > train_size and index <= val_size:
                shutil.move(image_path, os.path.join(val_path, image_filename))
            if index > val_size and index <= test_size:
                shutil.move(image_path, os.path.join(test_path, image_filename))

if __name__ == "__main__":
    png_folder = "C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/Pre Injury PNG"
    xml_folder = "C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/pre injury xml"
    train_folder = "C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/train"
    val_folder =  "C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/val"
    test_folder =  "C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/test"

    move_xml_files(xml_folder, train_folder, val_folder, test_folder)
    move_png_files(png_folder, train_folder, val_folder, test_folder)
