from PIL import Image

def is_grayscale(image_path):
    image = Image.open(image_path)
    if image.mode == 'L':
        return True
    else:
        return False

def is_rgb(image_path):
    image = Image.open(image_path)
    if image.mode == 'RGB':
        return True
    else:
        return False

# Usage
image_path = 'C:/Users/kkotkar1/Desktop/DicomScaling/ScaledImages25/zoomed_dicom-001.png'

if is_grayscale(image_path):
    print("Image is grayscale.")
elif is_rgb(image_path):
    print("Image is RGB.")
else:
    print("Image is neither grayscale nor RGB.")
