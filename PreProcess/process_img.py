import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import glob
import pathlib

# 1_, 2_ Dataset has different size of image
# 1_ = 420,580,3 -> Need to add 96,0,0
# 2_ = 512,512,3 -> Need to add 0,68,0

# def process_img(img, size):
#     to_tensor = T.ToTensor()
#     img = to_tensor(img)
#     img = F.pad(img, size, mode='reflect')
#     return img

# cv2 img load testing
# img = cv2.imread('DataSet/1_/test/1.tif')
# print(img.shape)
# Get Parent folder name
# print(pathlib.Path('DataSet/1_/test/1.tif').parent.name)

# img2 = cv2.imread('DataSet/2_/masks/bjorke_1.png')
# print(img2.shape)

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
# img = process_img(img, (0,0,46,46))
# img2 =process_img(img2, (34,34,0,0))

# plt.imshow(img.permute(1,2,0))
# plt.show()
# plt.imshow(img2.permute(1,2,0))
# plt.show()

# print(img.shape)
# print(img2.shape)


# try glob
# file_path = "DataSet/1_/test/*.tif"
# files = glob.glob(file_path)
# print(files)


# load Files
file_path = ["DataSet/1_/test/*.tif", "DataSet/1_/train/*.tif"]
files = sorted({p for file_path in file_path for p in glob.glob(file_path)})

# 1_ Folder Process
# for file in files :
#     parent_folder = pathlib.Path(file).parent.name
#     img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#     img = process_img(img, (0,0,46,46)) 
#     np.save(f"DataSet/Processed/1_/{parent_folder}/{pathlib.Path(file).name}", img)

# 2_ Folder Process
# Every one has mask we need to split the data, train and test  
# file_path = ["DataSet/2_/frames/*.png","DataSet/2_/masks/*.png"]

# files = sorted({p for file_path in file_path for p in glob.glob(file_path)})
# split_files = 0.3
# split_index = int(len(files) * split_files)

# train_files = files[:split_index]
# test_files = files[split_index:]

# for file in train_files:
#     img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#     img = process_img(img, (34,34,0,0))
#     np.save(f"DataSet/Processed/2_/train/{pathlib.Path(file).name}", img)

# for file in test_files:
#     img = cv2.imread(file)
#     img = process_img(img, (34,34,0,0))
#     np.save(f"DataSet/Processed/2_/test/{pathlib.Path(file).name}", img)


# ----------------------------------------------------------------------------------------
# need to redesign the process_img function

# if we do imread, we will need to convert the image to grayscale
# and procees the image diffrent from common and mask

# load_img = cv2.imread('DataSet/2_/frames/bjorke_1.png', cv2.IMREAD_GRAYSCALE)
# print(load_img.shape)
# plt.imshow(load_img, cmap='gray')
# plt.show()

file_path = ['DataSet/1_/train/*.tif', 'DataSet/1_/test/*.tif']

common_files = [
    file for folder in file_path for file in glob.glob(folder) if 'mask' not in file
]

mask_files = [
        file for folder in file_path for file in glob.glob(folder) if 'mask' in file
]

print(mask_files)