# code to crop all dicom files to 960x640

import os
from PIL import Image
import pydicom

def crop_dicom_images(ippath, oppath):
    # Create the output folder if it doesn't exist
    if not os.path.exists(oppath):
        os.makedirs(oppath)

    # Iterate over the DICOM files in the folder
    for i, filename in enumerate(os.listdir(ippath)):
        if filename.endswith('.dcm'):
            filepath = os.path.join(ippath, filename)

            # Read the DICOM file
            dcm = pydicom.dcmread(filepath)

            # Get the pixel data
            pixel_data = dcm.pixel_array

            # Crop the image to the desired size (960x640)
            image = Image.fromarray(pixel_data).crop((193, 121, 1153, 841))

            # Save the image as PNG with the same name as the DICOM file
            png_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(oppath, png_filename)
            image.save(output_path)
            if i % 100 == 0:
                print(f'{i+1} Images Cropped')


if __name__ == '__main__':
    # Specify the folder containing DICOM files and the output folder for PNG images
    ippath = "C:/Users/kkotkar1/Desktop/DicomScaling/DicomSlices"
    oppath = "C:/Users/kkotkar1/Desktop/DicomScaling/CroppedImages"

    # Call the function to crop DICOM files and save as PNG
    crop_dicom_images(ippath, oppath)
