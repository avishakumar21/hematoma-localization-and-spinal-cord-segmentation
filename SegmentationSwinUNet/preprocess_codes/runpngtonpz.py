import os
from pngtonpz import load_images_and_masks

if __name__ == '__main__':
    train_image_dir = '../small_dataset/train_images/'
    train_mask_dir = '../small_dataset/train_masks/'
    train_save_dir = '../small_dataset/train_npz/'
    # test_image_dir = '../small_dataset/test_images/'
    # test_mask_dir = '../small_dataset/test_masks/'
    # test_save_dir = '../small_dataset/test_npz/'
    val_image_dir = '../small_dataset/val_images/'
    val_mask_dir = '../small_dataset/val_masks/'
    val_save_dir = '../small_dataset/val_npz/'

    # Load images and masks
    # load_images_and_masks(train_image_dir, train_mask_dir, train_save_dir)
    print("Train Images and masks saved")
    # load_images_and_masks(test_image_dir, test_mask_dir, test_save_dir)
    # print("Test Images and masks saved")
    load_images_and_masks(val_image_dir, val_mask_dir, val_save_dir)
    print("Val Images and masks saved")
    