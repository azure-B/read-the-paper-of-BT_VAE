import torch
import torch.nn as nn
import numpy as np

class VAE(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.mu = nn.Linear(32 * 105 * 145, z_dim)
        self.log_var = nn.Linear(32 * 105 * 145, z_dim)
        self.fc = nn.Linear(z_dim, 32 * 105 * 145)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def decode(self, z):
        h = self.fc(z)
        h = h.view(-1, 32, 105, 145)
        return self.decoder(h)

    def reparameter(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameter(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


# model Testing 

# model = VAE(z_dim=32)
# model.eval()
 
# x = np.load("DataSet/Processed/1_/train/1_1.tif.npy")
# x = torch.from_numpy(x).float().unsqueeze(0)
# print("numpy shape : ", x.shape)

# with torch.no_grad():
#     h = model.encoder(x)
#     print("encoder : ", h.shape)

#     recon, mu, log_var = model(x)
#     print("recon: ", recon.shape)
#     print("mu: ", mu.shape)   
#     print("log_var: ", log_var.shape)

#     print("\n입력 x   min/max :", x.min().item(), x.max().item())
#     print("복원 recon min/max :", recon.min().item(), recon.max().item())