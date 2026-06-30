import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# 1_, 2_ Dataset has different size of image
# 1_ = 420,580,3 -> Need to add 96,0,0
# 2_ = 512,512,3 -> Need to add 0,68,0

def process_img(img, size):
    to_tensor = T.ToTensor()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = to_tensor(img)
    img = F.pad(img, size, mode='reflect')
    return img

# cv2 img load testing
img = cv2.imread('DataSet/1_/test/1.tif')
print(img.shape)

img2 = cv2.imread('DataSet/2_/masks/bjorke_1.png')
print(img2.shape)

# pure padding 0 add
# img = cv2.copyMakeBorder(img, 46, 46, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
# print(img.shape)
# plt.imshow(img)
# plt.show()

# img_path = "DataSet/1_/test/1.tif"
# img = Image.open(img_path)
# img = img.convert('RGB')
# img = np.array(img)

# Reflect padding
# padding_size = (0,0,46,46)
# img = F.pad(img, padding_size, mode='reflect')
# print(img.shape)
# plt.imshow(img.permute(1,2,0))
# plt.show()

# use def
img = process_img(img, (0,0,46,46))
img2 =process_img(img2, (34,34,0,0))

plt.imshow(img.permute(1,2,0))
plt.show()
plt.imshow(img2.permute(1,2,0))
plt.show()

print(img.shape)
print(img2.shape)