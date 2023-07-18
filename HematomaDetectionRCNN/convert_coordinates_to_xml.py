import os
import pandas as pd
import xml.etree.ElementTree as ET

# Load the CSV file
df = pd.read_csv('/Users/avishakumar/Documents/dicom_images/CalculatedCoordinates.csv')

# Function to create an XML file for a given row of the dataframe
def create_xml(row, folder, img_path, img_width, img_height, img_depth):
    # Create the root element
    root = ET.Element("annotation")

    # Create and append the folder, filename, and path elements
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = row["File Name"]
    ET.SubElement(root, "path").text = img_path

    # Create and append the source element
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    # Create and append the size element
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = str(img_depth)

    # Create and append the segmented element
    ET.SubElement(root, "segmented").text = "0"

    # Create and append the object element
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = "hematoma"
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"

    # Create and append the bndbox element
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = str(row["X1"])
    ET.SubElement(bndbox, "ymin").text = str(row["Y1"])
    ET.SubElement(bndbox, "xmax").text = str(row["X2"])
    ET.SubElement(bndbox, "ymax").text = str(row["Y2"])

    # Create the tree from the root element and write it to an XML file
    tree = ET.ElementTree(root)
    xml_filename = os.path.join(folder, os.path.splitext(row["File Name"])[0] + ".xml")
    tree.write(xml_filename)

# Iterate over the rows of the dataframe and create an XML file for each
for _, row in df.iterrows():
    create_xml(row, "/Users/avishakumar/Documents/dicom_images/AllDicomPNG", os.path.join("/Users/avishakumar/Documents/dicom_images/AllDicomPNG", row["File Name"]), 1280, 960, 3)
