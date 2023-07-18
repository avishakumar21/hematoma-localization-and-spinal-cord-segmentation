import os
import xml.etree.ElementTree as ET

# Directory paths
image_dir = '/Users/avishakumar/Documents/dicom_images/TrainPNG/'
txt_dir = '/Users/avishakumar/Documents/dicom_images/Labels/'
output_dir = '/Users/avishakumar/Documents/dicom_images/annotations/'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each image in the dataset
for image_name in os.listdir(image_dir):
    if image_name.endswith('.png'):
        # Get the image file path
        image_path = os.path.join(image_dir, image_name)
        print(image_path)
        
        # Get the corresponding text file path
        txt_file = os.path.join(txt_dir, os.path.splitext(image_name)[0] + '.txt')
        # Read bounding box coordinates from the text file
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        # Create XML annotation file path
        xml_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.xml')

        # Create XML root element
        root = ET.Element("annotation")

        # Add image path
        ET.SubElement(root, "path").text = image_path

        # Add image dimensions (modify accordingly based on your PNG image source)
        image_width = 1280  # replace with actual image width
        image_height = 960  # replace with actual image height
        image = ET.SubElement(root, "size")
        ET.SubElement(image, "width").text = str(image_width)
        ET.SubElement(image, "height").text = str(image_height)

        # Iterate over each line in the text file
        for line in lines:
            # Parse the line to get bounding box coordinates
            class_num, x_min, y_min, width, height = line.split()

            # Add bounding box coordinates
            object = ET.SubElement(root, "object")
            ET.SubElement(object, "name").text = "hematoma"  
            bbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bbox, "xmin").text = x_min
            ET.SubElement(bbox, "ymin").text = y_min
            ET.SubElement(bbox, "xmax").text = x_max
            ET.SubElement(bbox, "ymax").text = y_max

        # Create and save the XML file
        tree = ET.ElementTree(root)
        tree.write(xml_path)

        print(f"Annotation file created: {xml_path}")
