import numpy as np
from PIL import Image
# import cv2

# Load the image
def convert_mask(image_path):
    # image_path = '/mnt/data/predict4_scaled-A0001_frame1.png'
    image = Image.open(image_path)
    # image = image.resize((512, 512))
    image = image.resize((128, 128))

    # Convert the image into a numpy array
    image_array = np.array(image)

    # Define the colors mapping to numbers
    color_to_number = {
        (0, 0, 0): 0, # background
        (128, 0, 128): 1, # blood
        (128, 0, 0): 2, # dura
        (0, 128, 0): 3, # pia
        (128, 128, 0): 4, # csf
        (0, 0, 128): 5, # spinal cord
        (0, 128, 128): 6, # hematoma
        (64, 0, 0): 7, # dura pia complex
        (128, 128, 128): 8, # extradural
        (192, 0, 0): 9 # dura bone complex
    }

    # Initialize an empty array for the output
    number_array = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=int)

    # Map each pixel to the corresponding number
    for color, number in color_to_number.items():
        # Create a mask for pixels matching the current color
        mask = np.all(image_array == color, axis=-1)
        # Assign the number to the corresponding positions in the number_array
        number_array[mask] = number

    return number_array

# Save the numpy array to a file
# np.save('/mnt/data/converted_image_array.npy', number_array)



