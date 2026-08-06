import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from get_data import get_data


@dataclass
class Config:
    stride: int = 2
    kernel_size: int = 3
    padding: int = 1
    output_padding: int = 0
    z_dim: int = 32
    lr: float = 1e-3
    lr2: float = 1e-3
    beta: float = 0.001
    pos_weight: float = 30.0     # 배경:전경 ≈ 97:3 → 약 30배
    dice_smooth: float = 1.0     # dice smooth 항 (기존 1e-5는 너무 작음)
    eps: float = 1e-5            # BT KL용
    pre_epoch: int = 5
    tune_epoch: int = 50
    batch_size: int = 16
    log_every: int = 20          # 디버깅 로그 주기 (배치 단위)


class BT_VAE(nn.Module):
    def __init__(self, z_dim=128, stride=2, kernel_size=3, padding=1, output_padding=1):
        super().__init__()
        self.z_dim = z_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size, stride, padding),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size, stride, padding),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size, stride, padding),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size, stride, padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 출력은 logits (Sigmoid 제거!)
        # 확률이 필요하면 밖에서 torch.sigmoid() 적용
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, kernel_size, stride, padding, output_padding),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size, stride, padding, output_padding),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size, stride, padding, output_padding),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size, stride, padding, output_padding)
            # nn.Sigmoid()  # 제거: BCEWithLogits가 내부에서 처리
        )

        self.log_var = nn.Conv2d(256, z_dim, kernel_size=1)
        self.mu = nn.Conv2d(256, z_dim, kernel_size=1)

    def encode(self, x):
        x_left = torch.rot90(x, k=1, dims=(-2, -1))

        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)

        BT = self.encoder(x_left)
        BT_log_var = self.log_var(BT)
        BT_mu = self.mu(BT)

        return mu, log_var, BT_mu, BT_log_var

    def decode(self, h):
        h = self.decoder(h)
        h = F.interpolate(h, size=(420, 580), mode='bilinear')
        return h  # logits

    def get_CM(self, BT_mu, mu):
        shape = BT_mu.shape[1]

        BT_mu = BT_mu.permute(0, 2, 3, 1).reshape(-1, shape)
        N = BT_mu.shape[0]
        mu = mu.permute(0, 2, 3, 1).reshape(-1, shape)

        BT_norm = (BT_mu - BT_mu.mean(0)) / (BT_mu.std(0) + 1e-8)
        common_norm = (mu - mu.mean(0)) / (mu.std(0) + 1e-8)

        CM = torch.matmul(BT_norm.T, common_norm) / N
        return CM

    def reparameter(self, log_var, mu):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var, BT_mu, BT_log_var = self.encode(x)
        z = self.reparameter(log_var, mu)
        CM = self.get_CM(BT_mu, mu)

        logits = self.decode(z)

        return CM, z, logits, log_var, mu, BT_log_var, BT_mu

    @staticmethod
    def get_BT_KL_loss(BT_mu, BT_log_var, mu, log_var, eps=1e-8, tau=1):
        shape = mu.shape[1]

        BT_mu = BT_mu.permute(0, 2, 3, 1).reshape(-1, shape)
        BT_var_flat = BT_log_var.exp().permute(0, 2, 3, 1).reshape(-1, shape)

        mu = mu.permute(0, 2, 3, 1).reshape(-1, shape)
        var_flat = log_var.exp().permute(0, 2, 3, 1).reshape(-1, shape)

        BT_mean = BT_mu.mean(dim=0).unsqueeze(1)
        BT_var = BT_var_flat.mean(dim=0) + BT_mu.var(dim=0)
        BT_var = BT_var.unsqueeze(1) + eps

        mean = mu.mean(dim=0).unsqueeze(0)
        var = var_flat.mean(dim=0) + mu.var(dim=0)
        var = var.unsqueeze(0) + eps

        kl = 0.5 * torch.log(var / BT_var) + (BT_var + (mean - BT_mean).pow(2)) / (2 * var) - 0.5

        shape = kl.shape[0]
        eye = torch.eye(shape, device=kl.device)

        diag_loss = (kl.diagonal() ** 2).sum()
        off_diag_loss = ((kl * (1 - eye) - tau * (1 - eye)) ** 2).sum()

        return (diag_loss + off_diag_loss) / (shape * shape)

    @staticmethod
    def get_BT_VAE_Loss(CM, kl_loss, w=0.2):
        shape = CM.shape[0]
        eye = torch.eye(shape, device=CM.device)
        # 이중 for 루프 벡터화 (결과 동일, 훨씬 빠름)
        weight = eye + (1 - eye) * w
        BT_loss = ((eye - CM).pow(2) * weight).sum() / (shape * shape)
        return BT_loss + kl_loss

    @staticmethod
    def get_VAE_Loss(logits, y, log_var, mu, beta, pos_weight, smooth=1.0, verbose=False):
        # BCE (pos_weight로 전경 픽셀 가중) — logits 입력
        recon_loss = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight, reduction='mean')

        # Dice — sigmoid 확률로 계산
        prob = torch.sigmoid(logits)
        inter = (prob * y).sum(dim=[1, 2, 3]) * 2
        dice = (inter + smooth) / (prob.sum(dim=[1, 2, 3]) + y.sum(dim=[1, 2, 3]) + smooth)
        dice_loss = (1 - dice).mean()

        # KL
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        if verbose:
            print(f"bce: {recon_loss.item():.4f}, dice: {dice_loss.item():.4f}, "
                  f"kl: {kl_loss.item():.4f}, mu_std: {mu.std().item():.3f}")

        return recon_loss + dice_loss + kl_loss * beta


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BT_VAE(
        z_dim=Config.z_dim,
        stride=Config.stride,
        kernel_size=Config.kernel_size,
        padding=Config.padding,
        output_padding=Config.output_padding,
    )
    model.to(device)

    data = get_data('DataSet/Processed/1_/train', only_p=True)

    train_len = int(len(data) * 0.7)
    val_len = len(data) - train_len
    train_data, val_data = random_split(data, [train_len, val_len])

    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size, shuffle=False)

    # ── 주의: 이전 체크포인트(BT_best.pth / VAE_best.pth)는
    # Sigmoid 있던 구조 + 붕괴된 가중치라서 로드하지 말 것.
    # 처음부터 학습해야 함.

    # ── BT 사전학습 (원인 분리를 위해 비활성. VAE 단독 학습이 되는 걸 확인한 후 재검토)
    # op1 = torch.optim.Adam(model.parameters(), lr=Config.lr)
    # for epoch in range(Config.pre_epoch):
    #     ...

    # ── VAE 학습
    op2 = torch.optim.Adam(model.parameters(), lr=Config.lr2)
    best_loss = float('inf')
    pos_weight = torch.tensor([Config.pos_weight], device=device)

    for epoch in range(Config.tune_epoch):
        model.train()
        total_VAE_loss = 0
        count = 0

        for x, y in tqdm(train_loader, desc=f"epoch {epoch}"):
            x = x.to(device)
            y = y.to(device)

            CM, z, logits, log_var, mu, BT_log_var, BT_mu = model(x)

            verbose = (count % Config.log_every == 0)
            if verbose:
                prob = torch.sigmoid(logits)
                print(f"\nmax: {prob.max().item():.3f}, "
                      f">0.5 비율: {(prob > 0.5).float().mean().item():.4f}, "
                      f"y 전경 비율: {(y > 0.5).float().mean().item():.4f}")

            loss = model.get_VAE_Loss(
                logits, y, log_var, mu,
                Config.beta, pos_weight, Config.dice_smooth, verbose=verbose)

            op2.zero_grad()
            loss.backward()
            op2.step()

            total_VAE_loss += loss.item()
            count += 1

        common = total_VAE_loss / count

        if common < best_loss:
            best_loss = common  # (수정) 에폭 평균 기준으로 best 갱신
            torch.save(model.state_dict(), 'VAE_best.pth')

        print(f"epoch {epoch}, VAE: {common:.4f}")

    # ── 검증 (학습에 안 쓴 val_data 사용)
    model.eval()
    with torch.no_grad():
        count = 0
        total_loss = 0
        total_dice = 0
        for x, y in tqdm(val_loader, desc="validation"):
            x = x.to(device)
            y = y.to(device)

            CM, z, logits, log_var, mu, BT_log_var, BT_mu = model(x)

            total_loss += model.get_VAE_Loss(
                logits, y, log_var, mu,
                Config.beta, pos_weight, Config.dice_smooth).item()

            # dice score(높을수록 좋음)도 따로 집계
            prob = torch.sigmoid(logits)
            inter = (prob * y).sum(dim=[1, 2, 3]) * 2
            dice = (inter + 1.0) / (prob.sum(dim=[1, 2, 3]) + y.sum(dim=[1, 2, 3]) + 1.0)
            total_dice += dice.mean().item()
            count += 1

        print(f"val loss : {total_loss / count:.4f}, val dice : {total_dice / count:.4f}")