import os
import pydicom
from PIL import Image

dicom_dir = 'C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData/AllDicom'

png_dir = 'C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData/AllDicomPNG'

dicom_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]

for i,dicom_file in enumerate(dicom_files):
    dicom_path = os.path.join(dicom_dir, dicom_file)
    dicom_data = pydicom.read_file(dicom_path)

    dicom_image = dicom_data.pixel_array
    image = Image.fromarray(dicom_image)

    png_file = os.path.splitext(dicom_file)[0] + '.png'
    png_path = os.path.join(png_dir, png_file)
    image.save(png_path, format='png')
    if (i%1000 == 0):
        print(f'Image {i} converted')

print('Done')