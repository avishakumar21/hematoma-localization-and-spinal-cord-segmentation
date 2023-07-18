# code to scale the images to 1280x960 as per scaling factor in each dicom file
import os
import csv
from PIL import Image
import numpy as np
from scipy.ndimage import zoom


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def apply_zoom(ippath, csvpath, oppath, scaling_depth):
    # Read the CSV file
    with open(csvpath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Iterate over the CSV rows
        for i, row in enumerate(reader):
            filename = row['File Name']
            filename = filename.replace('.dcm', '.png')
            field_depth = float(row['Depth of Scan Field'])

            # Construct the file path for the image
            # image_path = os.path.join(ippath, filename).replace('.dcm', '.png')
            image_path = os.path.join(ippath, filename)

            # Open the image using PIL
            image = Image.open(image_path)

            # Calculate the zoom factor based on scaling depth

            zoom_factor = field_depth / scaling_depth

            zoomed_image = clipped_zoom(np.array(image), zoom_factor)

            zoomed_image = Image.fromarray(zoomed_image)

            # Save the zoomed image
            # output_path = os.path.join(oppath, f"zoomed_{filename}").replace('.dcm', '.png')
            output_path = os.path.join(oppath, f"zoomed_{filename}")
            image_path = os.path.join(ippath, filename)
            zoomed_image.save(output_path)
            if i % 100 == 0:
                print(f'{i} Images Zoomed and Saved')

if __name__ == '__main__':
    # Specify the image directory and the CSV file path
    scale_depths = [20, 25, 30, 35, 40]

    for i in scale_depths:
        print(f'Zooming Images for Scaling Depth {str(i)}')

        ippath = "C:/Users/kkotkar1/Desktop/DicomScaling/DicomSlicesPNG"
        oppath = f"C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages{str(i)}"
        csvpath = 'C:/Users/kkotkar1/Desktop/DicomScaling/focusdepthdata.csv'

        # Call the function to apply zoom on images
        apply_zoom(ippath, csvpath, oppath, i)
        print(f'Images Zoomed and Saved for Scaling Depth {str(i)}')



