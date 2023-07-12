import os
import random
import shutil

# Set the paths to the source and destination directories
src_dir = 'C:/Users/kkotkar1/Desktop/CropAllDicoms/DicomSlices'
dst_dir = 'C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData/OriginalDicom'

# Get a list of all files in the source directory
file_list = os.listdir(src_dir)

# Select 400 random files from the file list
random_files = random.sample(file_list, 400)
random.shuffle(random_files)
# Copy the selected files to the destination directory
for count,file_name in enumerate(random_files):
    count1 = str(count+1).zfill(3)
    src_file = os.path.join(src_dir, file_name)
    dst_file = os.path.join(dst_dir, f"dicom-{count1}.dcm")
    shutil.copy(src_file, dst_file)