import os
import pydicom
from PIL import Image


if __name__ == '__main__':

    ippath = 'C:/Users/kunal/Desktop/This PC/HEPIUS/Pre Injury Sagittal B Mode/frames'

    oppath = 'C:/Users/kunal/Desktop/This PC/HEPIUS/Pre Injury Sagittal B Mode/png'

    dicom_files = [f for f in os.listdir(ippath) if f.endswith('.dcm')]

    os.makedirs(oppath, exist_ok=True)

    for i,dicom_file in enumerate(dicom_files):
        dicom_path = os.path.join(ippath, dicom_file)
        dicom_data = pydicom.read_file(dicom_path)

        dicom_image = dicom_data.pixel_array
        image = Image.fromarray(dicom_image)

        png_file = os.path.splitext(dicom_file)[0] + '.png'
        png_path = os.path.join(oppath, png_file)
        
        image.convert('L').save(png_path, format='png')
        if (i%1000 == 0):
            print(f'Image {i} converted')

    print('Done')