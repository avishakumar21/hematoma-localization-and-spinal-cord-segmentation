import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = "C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData"

output_dir = "C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData/ModelData"

test_size = 0.2
val_size = 0.2

image_files = os.listdir(os.path.join(source_dir, "TrainPNG"))

train_img_dir = os.path.join(output_dir, "images", "train")
train_label_dir = os.path.join(output_dir, "labels", "train")
test_img_dir = os.path.join(output_dir, "images", "test")
test_label_dir = os.path.join(output_dir, "labels", "test")
val_img_dir = os.path.join(output_dir, "images", "val")
val_label_dir = os.path.join(output_dir, "labels", "val")

train_files, test_val_files = train_test_split(image_files, test_size=(test_size+val_size), random_state=42)
test_files, val_files = train_test_split(test_val_files, test_size=(val_size/(test_size+val_size)), random_state=42)

for file in train_files:
    image_path = os.path.join(source_dir, "TrainPNG", file)
    label_path = os.path.join(source_dir, "Labels", file.replace(".png", ".txt"))
    shutil.copy(image_path, os.path.join(train_img_dir, file))
    shutil.copy(label_path, os.path.join(train_label_dir, file.replace(".png", ".txt")))

for file in test_files:
    image_path = os.path.join(source_dir, "TrainPNG", file)
    label_path = os.path.join(source_dir, "Labels", file.replace(".png", ".txt"))
    shutil.copy(image_path, os.path.join(test_img_dir, file))
    shutil.copy(label_path, os.path.join(test_label_dir, file.replace(".png", ".txt")))

for file in val_files:
    image_path = os.path.join(source_dir, "TrainPNG", file)
    label_path = os.path.join(source_dir, "Labels", file.replace(".png", ".txt"))
    shutil.copy(image_path, os.path.join(val_img_dir, file))
    shutil.copy(label_path, os.path.join(val_label_dir, file.replace(".png", ".txt")))

print("Done!")