import os
import xml.etree.ElementTree as ET
from PIL import Image

def create_true_negative_xml(image_filename, image_path, save_folder):
    image = Image.open(image_path)
    width, height = image.size

    root = ET.Element("annotation")
    folder = ET.SubElement(root, "folder")
    folder.text = "images"

    filename = ET.SubElement(root, "filename")
    filename.text = image_filename

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(root, "size")
    width_element = ET.SubElement(size, "width")
    width_element.text = str(width)
    height_element = ET.SubElement(size, "height")
    height_element.text = str(height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    tree = ET.ElementTree(root)
    xml_filename = os.path.splitext(image_filename)[0] + ".xml"
    xml_path = os.path.join(save_folder, xml_filename)
    tree.write(xml_path)

def create_true_negative_annotations(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_filename in os.listdir(input_folder):
        if image_filename.lower().endswith(('.png')):
            image_path = os.path.join(input_folder, image_filename)
            print(f"Processing: {image_filename}")
            create_true_negative_xml(image_filename, image_path, output_folder)
            print(f"XML created for: {image_filename}")

if __name__ == "__main__":
    input_folder = "C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/Pre Injury PNG"
    output_folder = "C:/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/pre injury xml"

    create_true_negative_annotations(input_folder, output_folder)
