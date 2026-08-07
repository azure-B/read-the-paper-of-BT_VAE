import os
import cv2
import torch
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

#common_file Set up
for file in common_files:

    # load file
    parent_folder = pathlib.Path(file).parent.name
    proceesed = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # normalization
    proceesed = torch.from_numpy(proceesed).float()
    proceesed = proceesed.unsqueeze(0) / 255

    # Save np
    np.save(f"DataSet/Processed/1_/{parent_folder}/{pathlib.Path(file).name}", proceesed)

for file in mask_files :
    
    #load file
    parent_folder = pathlib.Path(file).parent.name
    proceesed = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    #cv2 Error
    _, proceesed = cv2.threshold(proceesed, 128, 1, cv2.THRESH_BINARY)
    proceesed = torch.from_numpy(proceesed).float()
    proceesed = proceesed.unsqueeze(0)

    # Save np
    np.save(f"DataSet/Processed/1_/{parent_folder}/{pathlib.Path(file).name}", proceesed)
 
frame_files = sorted(glob.glob('DataSet/2_/frames/*.png'))
 
out_dir2 = 'DataSet/Processed/2_/train'
os.makedirs(out_dir2, exist_ok=True)
 
count = 0
skip = 0
 
for file in frame_files:
 
    # masks 폴더에서 같은 이름 찾기
    mask_file = 'DataSet/2_/masks/' + pathlib.Path(file).name
 
    if not os.path.exists(mask_file):
        skip += 1
        continue
 
    # ----- image : grayscale -> [0,1] -> (1,H,W)
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0) / 255
 
    # ----- mask : threshold 128 -> 0/1 -> (1,H,W)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
    mask = torch.from_numpy(mask).float()
    mask = mask.unsqueeze(0)
 
    # ----- Save np
    name = pathlib.Path(file).name
    np.save(f"{out_dir2}/{name}", img)
    np.save(f"{out_dir2}/{name.replace('.png', '_mask.png')}", mask)
 
    count += 1
 
print(f"[COVID] {count}쌍 저장 -> {out_dir2} (마스크 없어 건너뜀 {skip})")
 
 
# ----------------------------------------------------------------------------------------
# 검증 : 제대로 저장됐는지 확인
# shape / 값 범위 / positive-negative 개수
 
check_list = [
    ('DataSet/Processed/1_/train', 'tif'),
    ('DataSet/Processed/2_/train', 'png'),
]
 
os.makedirs('DataSet/Processed/1_/train', exist_ok=True)
os.makedirs('DataSet/Processed/1_/test', exist_ok=True)

for check_dir, ext in check_list:
 
    files = sorted(glob.glob(f"{check_dir}/*.npy"))
    img_files = [f for f in files if '_mask.' not in pathlib.Path(f).name]
 
    print(f"\n[검증] {check_dir} : 파일 {len(files)}개 ({len(img_files)}쌍)")
 
    if len(img_files) == 0:
        continue
 
    shapes = set()
    n_pos = 0
    fg_list = []
 
    for f in img_files:
        mf = f.replace(f'.{ext}.npy', f'_mask.{ext}.npy')
        if not os.path.exists(mf):
            continue
 
        m = np.load(mf)
        shapes.add(m.shape)
 
        if m.sum() > 0:
            n_pos += 1
            fg_list.append(m.mean())
 
    x = np.load(img_files[0])
    y = np.load(img_files[0].replace(f'.{ext}.npy', f'_mask.{ext}.npy'))
 
    print(f"  shape : {shapes}")
    print(f"  positive {n_pos} / negative {len(img_files) - n_pos}")
    if len(fg_list) > 0:
        print(f"  positive 전경비율 : 평균 {np.mean(fg_list):.4f} "
              f"(min {np.min(fg_list):.4f}, max {np.max(fg_list):.4f})")
    print(f"  x : {x.shape} {x.dtype} range [{x.min():.3f}, {x.max():.3f}]")
    print(f"  y : {y.shape} {y.dtype} 고유값 {np.unique(y)}")
 