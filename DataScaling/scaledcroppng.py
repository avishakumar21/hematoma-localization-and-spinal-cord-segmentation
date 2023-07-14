from PIL import Image 
import os
import csv
import math

def getcropcoordinates(x1, y1, x2, y2):
    X = 690
    Y = 274
    if y1 < 300:
        y1 = y1+250
        y2 = y2+250
    xdiff =  (690 - (x2 - x1))/2
    ydiff =  (274 - (y2 - y1))
    # if (xdiff)%2 != 0:
    #     xdiff = xdiff - 1
    # if (ydiff)%2 != 0:
    #     ydiff = ydiff - 1
    left = x1 - math.ceil(xdiff)
    right = x2 + math.floor(xdiff)
    # left = x1 - (xdiff)/2
    # right = x2 + (xdiff)/2
    top = y1 - round((ydiff)*0.75)-1
    bottom = y2 + round((ydiff)*0.25)


    return left, top, right, bottom

def getxyxy(filename, csvpath):
    with open(csvpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
                # print(row[0])
                # print(filename)
                # input()
            if row[0] == filename:
                    # print(row[0], row[1], row[2], row[3], row[4])
                    # input()
                x1, y1, x2, y2 = int(round(float(row[1]))), int(round(float(row[2]))), int(round(float(row[3]))), int(round(float(row[4])))
                print(f'Name:{filename}, X1:{x1}, Y1:{y1}, X2:{x2}, Y2:{y2}')
                return x1, y1, x2, y2
            else:
                continue       
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
            # print(filepath)
            # input()
            # Get the coordinates
            x1, y1, x2, y2 = getxyxy(filename, csvpath)
            if x1 == 0:
                continue
            left, top, right, bottom = getcropcoordinates(x1, y1, x2, y2)
            print(f'Left:{left}, Top:{top}, Right:{right}, Bottom:{bottom}')
            # print(left, top, right, bottom)
            png_filename = filename.replace('zoomed_', 'scaled-')
            output_path = os.path.join(oppath, png_filename)
            # Crop the image to the desired size (690x275)
            image = img.crop((left, top, right, bottom))
            image.save(output_path)
            # input()
            # Save the image as PNG with the same name as the DICOM file
            # png_filename = os.path.splitext(filename)[0] + '.png'
            # output_path = os.path.join(oppath, png_filename)   
            # image.show()
            # input()
            # image.save(output_path)
            # if i % 100 == 0:
                # print(f'{i+1} Images Cropped')
            print(f'{i+1} Images Cropped')
            


if __name__ == '__main__':
    # ippath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25"
    
    ippath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledCropTest"
    oppath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledCropped25"
    csvpath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25/coordinates.csv"

    # Call the function to crop PNGs
    scale_crop(ippath, oppath, csvpath)