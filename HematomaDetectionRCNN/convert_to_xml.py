import os
import xml.etree.ElementTree as ET

# This file is meant to be used with both pre injury and post injury images and labels 
# The pre injury text files should be empty 

# Directory paths
image_dir  = '/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/images/test'
txt_dir    = '/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/labels/test'
output_dir = '/Users/akumar80/Documents/Avisha Kumar Lab Work/hematoma localization/ModelData/annotations/test'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over each image in the dataset
for image_name in os.listdir(image_dir):
    if image_name.endswith('.png'):
        # Get the image file path
        image_path = os.path.join(image_dir, image_name)
        #print(image_path)
        
        # Get the corresponding text file path
        txt_file = os.path.join(txt_dir, os.path.splitext(image_name)[0] + '.txt')
        #print(txt_file)
        # Read bounding box coordinates from the text file
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        # Create XML annotation file path
        xml_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.xml')

        # Create XML root element
        root = ET.Element("annotation")
        folder = ET.SubElement(root, "folder")
        folder.text = "images"
        filename = ET.SubElement(root, "filename")
        filename.text = image_name
        # Add image path
        ET.SubElement(root, "path").text = image_path
        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        # Add image dimensions (modify accordingly based on your PNG image source)
        image_width = 1280  
        image_height = 960  
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(image_width)
        ET.SubElement(size, "height").text = str(image_height)

        depth = ET.SubElement(size, "depth")
        depth.text = "3"

        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"

        # Iterate over each line in the text file
        for line in lines:
            # Parse the line to get bounding box coordinates              
            class_num, x_min, y_min, width, height = line.split()
            xmin = float(x_min)*int(image_width)
            xmax = (float(x_min) + float(width))*int(image_width)

            # Add bounding box coordinates
            object = ET.SubElement(root, "object")
            ET.SubElement(object, "name").text = "hematoma"  
            bbox = ET.SubElement(object, "bndbox")

            ET.SubElement(bbox, "xmin").text = str(xmin)
            ET.SubElement(bbox, "ymin").text = 0
            ET.SubElement(bbox, "xmax").text = 0 #(x_min + width)*image_width
            ET.SubElement(bbox, "ymax").text = 0

        # Create and save the XML file
        tree = ET.ElementTree(root)
        tree.write(xml_path)

        #print(f"Annotation file created: {xml_path}")
