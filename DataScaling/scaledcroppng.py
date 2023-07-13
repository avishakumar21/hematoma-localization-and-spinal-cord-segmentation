from PIL import Image 
import os
import csv

def getxyxy(filename, csvpath):
    with open(csvpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # print(row[0])
            if row[0] == filename:
                print(row[0], row[1], row[2], row[3], row[4])
                input()
                return float(row[1]), float(row[2]), float(row[3]), float(row[4])          
    return 0, 0, 0, 0


def scale_crop(ippath, oppath, csvpath):
    # Create the output folder if it doesn't exist
    if not os.path.exists(oppath):
        os.makedirs(oppath) 
    for i, filename in enumerate(os.listdir(ippath)):
        if filename.endswith('.png'):
            filepath = os.path.join(ippath, filename)

            # Read the image
            img = Image.open(filepath)
            print(filepath)
            # input()
            # Get the coordinates
            left, top, right, bottom = getxyxy(filepath, csvpath)

            # Crop the image to the desired size (960x640)
            image = img.crop((left, top, right, bottom))

            # Save the image as PNG with the same name as the DICOM file
            png_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(oppath, png_filename)
            img.show()
            input()
            image.save(output_path)
            if i % 100 == 0:
                print(f'{i+1} Images Cropped')


if __name__ == '__main__':
    ippath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25/"
    oppath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledCropped25"
    csvpath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25/coordinates.csv"

    # Call the function to crop PNGs
    scale_crop(ippath, oppath, csvpath)