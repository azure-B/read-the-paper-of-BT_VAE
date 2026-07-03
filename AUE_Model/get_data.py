from torch.utils.data import Dataset, DataLoader
import glob, numpy as np, torch

class NpyDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(folder + "/*.npy") 
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        x = np.load(self.files[i])
        return torch.from_numpy(x).float()        
