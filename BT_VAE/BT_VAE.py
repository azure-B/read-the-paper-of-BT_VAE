import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader,random_split

from get_data import get_data
# import glob

@dataclass
class Config:
    stride: int = 2
    kernel_size: int = 3
    padding: int = 1
    output_padding: int = 1
    z_dim: int = 32  
    lr: float = 1e-3
    lr2: float = 1e-3
    alpha: float = 0.99
    beta: float = 1e-5
    eps: float = 1e-5
    pre_epoch: int = 30
    tune_epoch: int = 30
    batch_size: int = 64

class BT_VAE(nn.Module):
    def __init__(self, z_dim=32, stride=2, kernel_size = 3, padding = 1, output_padding = 1):
        super().__init__()
        self.z_dim = z_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding),
                nn.ReLU()
            )

        # BT pre_trained
        # we want to matrix output
        # shape ( batch, 32, 105, 145 )
        self.encoder = nn.Sequential(
           conv_block(1,16),
           conv_block(16,32)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 16, kernel_size, stride, padding, output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size, stride, padding, output_padding),
            nn.Sigmoid()
        )

        self.log_var = nn.Conv2d( 32, z_dim, kernel_size = 1 )
        self.mu = nn.Conv2d( 32, z_dim, kernel_size = 1 )


    def encode(self, x):
        x_left = torch.rot90(x, k=1, dims=(-2, -1))
        
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)

        BT = self.encoder(x_left)
        BT_mu = self.mu(BT)

        return mu, log_var, BT_mu

    def decode(self, h):
        h = self.decoder(h)
        h = F.interpolate( h, size=( 420, 580 ), mode='bilinear' )
        return h

    def get_CM(self, BT_mu, mu):
        shape = BT_mu.shape[1]

        BT_mu = BT_mu.permute(0, 2, 3, 1)
        BT_mu = BT_mu.reshape(-1, shape)

        N = BT_mu.shape[0]

        mu = mu.permute(0, 2, 3, 1)
        mu = mu.reshape(-1, shape)

        BT_norm = (BT_mu - BT_mu.mean(0)) / (BT_mu.std(0) + 1e-8)
        common_norm = (mu - mu.mean(0)) / (mu.std(0) + 1e-8)

        CM = torch.matmul(BT_norm.T, common_norm) / N
        return CM

    def reparameter(self, log_var, mu):
        std = torch.exp(log_var*0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var, BT_mu = self.encode(x)    
        z = self.reparameter(log_var,mu)
        CM = self.get_CM(BT_mu,mu)
        
        recon = self.decode(z)

        return CM, z, recon, log_var, mu
        
    @staticmethod
    def get_BT_VAE_Loss(CM):
        shape = CM.shape[0]
        eye = torch.eye(shape, device=CM.device)
        BT_loss = 0
        w = 0.2

        for i in range(shape):
            for j in range(shape):
                if i == j:
                    BT_loss += (eye[i][j] - CM[i][j]).pow(2)
                else:
                    BT_loss += (eye[i][j] - CM[i][j]).pow(2) * w

        BT_loss = BT_loss / (shape * shape)
        return BT_loss  

    @staticmethod
    def get_VAE_Loss(recon, x, log_var, mu, dice, alpha, beta):
        recon_loss = F.binary_cross_entropy(recon, x, reduction='mean')
        recon_loss = (alpha * dice) + (recon_loss * (1-alpha))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        print(f'recon : {recon_loss}, kl : {kl_loss * beta}')

        return recon_loss + kl_loss * beta

    @staticmethod
    def dice_loss(recon, y, eps):
        dice = (recon * y).sum() *2
        dice = (dice + eps) / ((recon.sum() + y.sum()) + eps)
        return 1 - dice


# model Testing
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BT_VAE(
        z_dim=Config.z_dim,
        stride=Config.stride,
        kernel_size=Config.kernel_size,
        padding=Config.padding,
    )

    model.to(device)

    # glob always out list
    # if i want to input just one file,
    # i can use np.load
    # x = np.load('DataSet/Processed/1_/test/1.tif.npy')

    # npy to tensor
    # x = torch.tensor(x).float()
    # x = x.unsqueeze(0)
    # x = x.to(device)
    # print(x.shape)


    # print(x.shape)
    # CM, z, recon, log_var, mu = model(x)
    # print("CM:", CM.shape)
    # print("recon:", recon.shape)
    # loss = model.VAE_Loss(recon, x, log_var, mu, beta=Config.beta)
    # print(loss.item)

    data = get_data('DataSet/Processed/1_/train')
    
    total_len = len(data)

    train_len = int(len(data)*0.7)
    val_len = int(len(data)-train_len)

    train_data, val_data = random_split(data, [train_len, val_len])

    train_loader = DataLoader(train_data, batch_size = Config.batch_size, shuffle=True )
    val_loader = DataLoader(val_data, batch_size = Config.batch_size, shuffle=False )

    op1 = torch.optim.Adam(model.parameters(), lr=Config.lr)


    for epoch in range(Config.pre_epoch):
        total_loss = 0
        total_BT_loss = 0
        count = 0

        for x,y in train_loader:
            x = x.to(device)
            
            CM, z, recon, log_var, mu = model(x)
            loss = model.get_BT_VAE_Loss(CM)
            
            op1.zero_grad()
            loss.backward()
            op1.step()
            total_loss += loss.item()
            
            count += 1    
            total_BT_loss += loss.item()
        print(f"epoch {epoch}, BT: { total_BT_loss / count }")

    op2 = torch.optim.Adam(model.parameters(), lr=Config.lr2)

    for epoch in range(Config.tune_epoch):
        total_loss = 0
        total_VAE_loss = 0
        count = 0

        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)


            CM, z, recon, log_var, mu = model(x)
            dice = model.dice_loss(recon, y, Config.eps)

            loss = model.get_VAE_Loss(recon, y, log_var, mu, dice, Config.alpha, Config.beta)

            op2.zero_grad()
            loss.backward()
            op2.step()
            total_VAE_loss += loss.item()

            count += 1    
        print(f"epoch {epoch}, VAE: { total_VAE_loss / count }")

    model.eval()

    with torch.no_grad():
        count = 0
        total_loss = 0
        for x, y in val_loader:            
            x = x.to(device)
            y = y.to(device)
            CM, z, recon, log_var, mu = model(x)
            dice = model.dice_loss(recon, y, Config.eps)

            total_loss += model.get_VAE_Loss(recon, y, log_var, mu, dice, Config.alpha, Config.beta).item()
            count += 1

        print(f"total_loss : {total_loss/count}")
            


