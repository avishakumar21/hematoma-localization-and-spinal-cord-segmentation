import numpy as np
from PIL import Image

def normalize(x):
    mn = x.min()
    mx = x.max()
    mx -= mn
    x = ((x - mn) / mx) * 255
    return x.astype(np.uint8)

b = np.load(r'C:\Users\kkotkar1\Desktop\TransUNet\data\Synapse\train_npz\case0040_slice163.npz')
print(b.files)

image = normalize(b['image'])
# print(image)
print(image.shape)
for i in image:
    # for j in i:
    #     print(j)
    print(i)
img = Image.fromarray(image)
img.show()

# mask = b['label']
# mask = normalize(mask)
# img = Image.fromarray(mask)
# img.show()
# print(b['label'])
# for i in b['label']:
#     print(i)
# print(b['label'])

# img = Image.fromarray(b['image'].astype('uint8'))
# img.show()

# mask = b['label'].astype(np.uint8)

# rgb = np.dstack((mask, mask, mask))

# print(mask.shape)
# for i in mask:
#     # print(i)
#     for j in i:
#         # print(j.dtype)
#         # input()
#         if j != 0:
#             j = 255
#             print("performed conversion")
#         # for k in j:
#         #     print(k)
#         # print(j)
#     print(i)


# mask = Image.fromarray(mask)
# mask.show()