import os
import random
import shutil

def copy_random_images(source_dir, destination_dir, num_images):
    # Get a list of all image files in the source directory
    image_files = [file for file in os.listdir(source_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Check if the number of available image files is less than the desired number
    if len(image_files) < num_images:
        print(f"There are only {len(image_files)} image files available, less than {num_images}.")
        return
    
    # Select 5 random images
    selected_images = random.sample(image_files, num_images)
    
    # Copy selected images to the destination directory
    for image in selected_images:
        source_path = os.path.join(source_dir, image)
        destination_path = os.path.join(destination_dir, image)
        shutil.copy(source_path, destination_path)
        print(f"Copied {image} to {destination_dir}")

if __name__ == '__main__':
    # Specify the source and destination directories
    source_directory = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25"
    destination_directory = "C:/Users/kkotkar1/Desktop/DicomScaling/ScaledCropTest"
    num_images = 5

    # Call the function to copy 5 random images
    copy_random_images(source_directory, destination_directory, num_images)
