import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# import glob

class BT_VAE(nn.Module):
    def __init__(self, z_dim=32, stride=2, kernel_size = 3, padding = 1, output_padding = 1):
        super().__init__()
        self.z_dim = z_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        # BT pre_trained
        # we want to matrix output
        # shape ( batch, 32, 105, 145 )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size , stride , padding ),
            nn.ReLU(),
            nn.Conv2d(16, z_dim, kernel_size, stride, padding),
            nn.ReLU(),
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
        loss = 0
        w = 0.2

        for i in range(shape):
            for j in range(shape):
                if i == j:
                    loss += (eye[i][j] - CM[i][j]).pow(2)
                else:
                    loss += (eye[i][j] - CM[i][j]).pow(2) * w

        return loss

    @staticmethod
    def VAE_Loss(recon, x, log_var, mu):
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        beta = 1

        return recon_loss + kl_loss * beta

    @staticmethod 
    def get_Loss(VAE_Loss,BT_Loss):
        return VAE_Loss + BT_Loss

    

# model Testing
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BT_VAE(z_dim = 32)
    model.to(device)
    op = torch.optim.Adam(model.parameters(), lr=1e-3)

    # glob always out list
    # if i want to input just one file,
    # i can use np.load
    x = np.load('DataSet/Processed/1_/test/1.tif.npy')

    # npy to tensor
    x = torch.tensor(x).float()
    x = x.unsqueeze(0)
    x = x.to(device)
    print(x.shape)


    # print(x.shape)
    CM, z, recon, log_var, mu = model(x)
    print("CM:", CM.shape)
    print("recon:", recon.shape)