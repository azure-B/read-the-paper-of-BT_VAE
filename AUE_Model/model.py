import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import glob

# Get Data Testing
from get_data import NpyDataset
from torch.utils.data import Dataset, DataLoader


# Let's fallow the workflow
# i will Enter the step for example # 1.. # 2.. 

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

        
    # 1. we insert input image to encoder
    def encode(self, x):
        # 2. encoder has 2 output mu and log_var
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def decode(self, z):
        h = self.fc(z)
        h = h.view(-1, 32, 105, 145)
        return self.decoder(h)

    # 2. encode's outputs are doing reparameter
    # .. why log_var?
    def reparameter(self, mu, log_var):
        # we process the log_var like log σ^2, it is log_varience
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        # z = μ + σ·ε
        # μ = Mean, σ = Varience, ε do like dice
        # we need to update this model to mu and eps, but if we don't have ε, they can't do it
        return mu + eps * std

    
    # processing step
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameter(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    @staticmethod
    def get_loss(recon, x, mu, log_var):
        recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
        # kl_loss is a error Normal distribution and mu,Log_var's distribution
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_loss


# model Testing
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(z_dim=32)
    model = model.to(device)

    op = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = NpyDataset("DataSet/Processed/1_/test")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True, 
        num_workers=4,
        drop_last=True,
    )


    for epoch in range(50):
        for x in loader:
            x = x.to(device)
            recon, mu, log_var = model(x)
            loss = model.get_loss(recon, x, mu, log_var)
            op.zero_grad()
            loss.backward()
            op.step()
        
        print(f"epoch {epoch}: loss = {loss.item():.2f}")

# with torch.no_grad():
#     h = model.encoder(x)
#     print("encoder : ", h.shape)

#     recon, mu, log_var = model(x)
#     loss = model.get_loss(recon,x,mu,log_var)

#     print("recon: ", recon.shape)
#     print("mu: ", mu.shape)   
#     print("log_var: ", log_var.shape)

#     print("\n입력 x   min/max :", x.min().item(), x.max().item())
#     print("복원 recon min/max :", recon.min().item(), recon.max().item())
