from torch.utils.data import Dataset
import numpy as np
import torch
import glob

class get_data(Dataset):
    def __init__(self, folder, only_p = True):
        self.all_files = glob.glob(folder + '/*.npy')
        self.img_files = [file for file in self.all_files if 'mask' not in file]
        self.mask_files = [file.replace('.tif.npy','_mask.tif.npy') for file in self.img_files]

        if only_p:
            paired = [(img_f, mask_f) for img_f, mask_f in zip(self.img_files, self.mask_files) if np.load(mask_f).sum() > 0]
            self.img_files = [p[0] for p in paired]
            self.mask_files = [p[1] for p in paired]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self,i):
        x = np.load(self.img_files[i])
        y = np.load(self.mask_files[i])
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

if __name__ == '__main__':
    data = get_data('DataSet/Processed/1_/train')
    x,y = data[0]

    print(x.shape)
    print(y.shape)
    print("mask", y.min(), y.max())