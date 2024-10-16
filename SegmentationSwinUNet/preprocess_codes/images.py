import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def normalize(x):
    mn = x.min()
    mx = x.max()
    mx -= mn
    x = ((x - mn) / mx)
    return x.astype(np.float32)

def convert_image(img_path):
    img = Image.open(img_path).convert('L')
    # img = img.resize((512, 512))
    img = img.resize((128, 128))
    img = np.array(img)
    img = normalize(img)

    # img1 = np.expand_dims(img, axis=0)

    # print(f'Image Shape: {img1.shape}')
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img

if __name__ == '__main__':
    b = np.load(r'C:\Users\kkotkar1\Desktop\TransUNet\data\Synapse\train_npz\case0005_slice056.npz')
    img = np.array(Image.open(r'C:\Users\kkotkar1\Desktop\TransUNet\spinaldata\images\predict4_scaled-A0001_frame1.png').convert('L'))

    image = b['image']
    img1 = Image.fromarray(image)

    print('Synapse Image')
    print(image.shape)
    for i in image:
        for j in i:
            print('%.5f' % j)
    input()

    print('Our Image')
    print(img.shape)

    # normalize the values in the image
    mn = img.min()
    mx = img.max()
    mx -= mn
    img = ((img - mn) / mx)
    img = img.astype(np.float32)
    print(img)
    for i in img:
        print(i)