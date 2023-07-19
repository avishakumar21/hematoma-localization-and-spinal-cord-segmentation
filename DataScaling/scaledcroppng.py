from PIL import Image 
import os
import csv
import math
import numpy as np 

def check_consecutive_black_columns(crop_coords, og_img):
    width = crop_coords[2] - crop_coords[0]
    # print(width)
    image = og_img
    # input()
    x1, y1, x2, y2 = crop_coords
    print(f'Original Crop: Left:{x1}, Top:{y1}, Right:{x2}, Bottom:{y2}')
    # input()
    for column_index in range(int(math.ceil(width/2)), x1-1, -1):
        black_column_left = all(image.getpixel((column_index, y)) == 0 for y in range(y1,y2+1))
        # print(black_column_left)
        if black_column_left:
                print(f'Black column in left found at: {column_index}')
                diff = column_index - x1 - 1
                x1 = column_index + 1
                x2 = x2 + diff
                break
        # else:
            # print('No black column in left found')
    for column_index in range(int(math.floor(width/2)), x2+1):
        # print(column_index)
        black_column_left = all(image.getpixel((column_index, y)) == 0 for y in range(y1,y2+1))
        if black_column_left:
                print(f'Black column in right found at: {column_index}')
                diff = x2 - column_index
                x1 = x1 - diff - 1
                x2 = column_index - 1
                break
        # else:
            # print('No black column in right found')
    print(f'Modified Crop: Left:{x1}, Top:{y1}, Right:{x2}, Bottom:{y2}')
    # input()
    print(f'Width:{x2-x1}, Height:{y2-y1}')
    return (x1, y1, x2, y2)

def getcropcoordinates(bbox):
    x1, y1, x2, y2 = bbox
    X = 690
    Y = 274
    xdiff =  (690 - (x2 - x1))/2
    ydiff =  (274 - (y2 - y1))
    left = x1 - math.ceil(xdiff)
    right = x2 + math.floor(xdiff)
    top = y1 - round((ydiff)*0.65)-1
    bottom = y2 + round((ydiff)*0.35)

    return (left, top, right, bottom)

def getxyxy(filename, csvpath):
    with open(csvpath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if row[0] == filename:
                x1, y1, x2, y2 = int(round(float(row[1]))), int(round(float(row[2]))), int(round(float(row[3]))), int(round(float(row[4])))
                print(f'Name:{filename}, X1:{x1}, Y1:{y1}, X2:{x2}, Y2:{y2}')
                return (x1, y1, x2, y2)
            else:
                continue       
    return (0, 0, 0, 0)


def scale_crop(ippath, oppath, csvpath):
    # Create the output folder if it doesn't exist
    if not os.path.exists(oppath):
        os.makedirs(oppath) 
    c = 0
    for i, filename in enumerate(os.listdir(ippath)):
        if filename.endswith('.png'):
            c += 1
            filepath = os.path.join(ippath, filename)

            # Read the image
            og_img = Image.open(filepath)
            # print(filepath)
            # input()
            # Get the coordinates
            bbox = getxyxy(filename, csvpath)
            if bbox[0] == 0:
                continue
            crop_coords = getcropcoordinates(bbox)
            # print(crop_coords)
            # exit()
            crop_coords = check_consecutive_black_columns(crop_coords, og_img)
            print(f'Left:{crop_coords[0]}, Top:{crop_coords[1]}, Right:{crop_coords[2]}, Bottom:{crop_coords[3]}')
            # print(left, top, right, bottom)
            png_filename = filename.replace('zoomed_', 'scaled-')
            output_path = os.path.join(oppath, png_filename)
            # Crop the image to the desired size (690x275)
            image = og_img.crop(crop_coords)
            image.save(output_path)
            if c % 100 == 0:
                print(f'{c} Images Cropped')
            


if __name__ == '__main__':
    # ippath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25"
    
    ippath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25"
    oppath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledCropped25NewTest"
    csvpath = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25/CalculatedCoordinates.csv"

    # Call the function to crop PNGs
    scale_crop(ippath, oppath, csvpath)