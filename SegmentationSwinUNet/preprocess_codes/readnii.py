import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def readnii(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    return data

def save_png(data, save_path, is_img, color_map):
    try:
        for i in range(data.shape[2]):
            if is_img:
                image = Image.fromarray((data[:,:,i] * 255).astype(np.uint8))
            else:
                mask_array = data[:,:,i]

                # Create an RGB image using the color map
                rgb_image = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
                for i in range(mask_array.shape[0]):
                    for j in range(mask_array.shape[1]):
                        rgb_image[i, j, :] = color_map[mask_array[i, j]]

                # Convert numpy array to PIL Image
                image = Image.fromarray(rgb_image)
                
                # Rotate the image by 270 degrees clockwise
                rotated_image = image.rotate(-90, expand=True)

                # Resize the image to a height of 275 and width of 690
                resized_image = rotated_image.resize((690, 275), Image.Resampling.LANCZOS)

                # Display the resized image
                # resized_image.show()

                # Save the resized image
                resized_image.save(save_path.replace('.nii.gz', f'.png'))

    except Exception as e:
        print(f'Exception occured: {e}')
        print(f'Error in file: {save_path}')
        print(f'Error in data shape: {data.shape}')    

def run_multiple():
    # file_path = r'../predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224'
    file_path = r'C:\Users\kkotkar1\Desktop\TransUNet\predictions\TU_Synapse128\TU_pretrain_R50-ViT-B_16_skip3_epo200_bs32_lr0.021308864223836394_128'

    color_map = { 
        0: (0, 0, 0),       # Black # background
        # 1: (255, 0, 0),     # Red # blood
        # 2: (0, 255, 0),     # Green
        # 3: (0, 0, 255),     # Blue
        # 4: (255, 255, 0),   # Yellow
        # 5: (255, 0, 255),   # Magenta
        # 6: (0, 255, 255),   # Cyan
        # 7: (255, 255, 255), # White
        # 8: (128, 128, 128), # Gray
        # 9: (255, 165, 0)    # Orange (Adding a new color for value 9)
        1: (128, 0, 128), # blood # Purple
        2: (128, 0, 0), # dura # Maroon
        3: (0, 128, 0), # pia # Green
        4: (128, 128, 0), # csf # Olive
        5: (0, 0, 128), # spinal cord # Navy
        6: (0, 128, 128), # hematoma # Teal
        7: (64, 0, 0), # dura pia complex # Brown
        8: (128, 128, 128), # extradural # Silver
        9: (192, 0, 0) # dura bone complex # Dark Red
    }

    for i, file in enumerate(os.listdir(file_path)):
        save_folder = r'pngs'
        save_path = os.path.join(file_path, save_folder)
        is_img = False
        if not file.endswith('.nii.gz'):
            continue
        data = readnii(os.path.join(file_path, file))
        if file.endswith('gt.nii.gz'):
            save_folder = r'gt'
            save_path = os.path.join(save_path, save_folder)
        elif file.endswith('pred.nii.gz'):
            save_folder = r'pred'
            save_path = os.path.join(save_path, save_folder)
        elif file.endswith('img.nii.gz'):
            is_img = True
            save_folder = r'img'
            save_path = os.path.join(save_path, save_folder)
        save_png(data, os.path.join(save_path, file), is_img, color_map)

        if i % 50 == 0:
            print(f'Processed {i} files')
    
    print('Done!')
        

if __name__ == "__main__":
    run_multiple()