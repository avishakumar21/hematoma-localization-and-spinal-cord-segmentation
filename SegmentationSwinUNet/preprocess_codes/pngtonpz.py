import os
import numpy as np
# from PIL import Image
from masks import convert_mask
from images import convert_image

def load_images_and_masks(image_dir, mask_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # images = []
    # masks = []
    filenames = os.listdir(image_dir)
    for filename in filenames:
        if filename.endswith('.png'):
            # Load image
            image_path = os.path.join(image_dir, filename)
            image = convert_image(image_path)
            # Load mask
            mask_filename = filename
            mask_path = os.path.join(mask_dir, mask_filename)
            mask = convert_mask(mask_path)
        np.savez(os.path.join(save_dir,filename.split('.')[0]), image=image, label=mask)
    
if __name__ == '__main__':
    # Directories
    # image_dir = '../data/small_dataset/train_images'
    # mask_dir = '../data/small_dataset/train_masks'
    # save_dir = '../data/train_npz_small'

    image_dir = '../../SegmentationData/full_dataset/train_images'
    mask_dir = '../../SegmentationData/full_dataset/train_masks'
    save_dir = '../../SegmentationData/full_dataset/train_npz_full'

    # Load images and masks
    load_images_and_masks(image_dir, mask_dir, save_dir)
    print("Images and masks saved")
