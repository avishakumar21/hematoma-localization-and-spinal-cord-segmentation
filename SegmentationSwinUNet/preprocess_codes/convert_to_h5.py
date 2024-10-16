import os
import numpy as np
import h5py
from PIL import Image
from masks import convert_mask
from images import convert_image

# Assuming grayscale images and masks are in separate directories with matching filenames
# image_dir = '../data/full_dataset/val_images'
# mask_dir = '../data/full_dataset/val_masks'
# output_dir = '../data/full_dataset/test_vol_h5_full'

image_dir = '../data/full_dataset/test_images'
mask_dir = '../data/full_dataset/test_masks'
output_dir = '../data/full_dataset/test_vol_h5_full'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convert_to_h5(image_path, mask_path, output_path):
    # print(f"Converting {image_path} and {mask_path} to {output_path}")
    # Read the grayscale image
    with Image.open(image_path) as img:
        # grayscale_array = np.array(img).astype(np.float32) / 255.0  # Normalize if necessary
        grayscale_array = convert_image(image_path)
        grayscale_array = np.expand_dims(grayscale_array, axis=0)
        # print(grayscale_array.shape)

    # Load the mask image
    with Image.open(mask_path) as mask:
        # mask_array = np.array(mask).astype(np.float32) / 255.0  # Normalize if necessary
        mask_array = convert_mask(mask_path)
        mask_array = np.expand_dims(mask_array, axis=0)
        # print(mask_array.shape)

    # Save the arrays to an H5 file
    with h5py.File(output_path, 'w') as h5_file:
        h5_file.create_dataset('image', data=grayscale_array)
        h5_file.create_dataset('label', data=mask_array)

# Loop through all files in the image directory
for i, filename in enumerate(os.listdir(image_dir)):
    if filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy.h5'))
        
        convert_to_h5(image_path, mask_path, output_path)
        print(f"Converted {filename} to npy.h5 format")
        # if i % 100 == 0 and i == len(os.listdir(image_dir)) - 1:
            # print(f"Converted {i} images to npy.h5 format")
