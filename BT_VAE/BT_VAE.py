import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

from get_data import get_data


@dataclass
class Config:
    stride: int = 2
    kernel_size: int = 3
    padding: int = 1
    output_padding: int = 0
    z_dim: int = 32
    out_size: tuple = (420, 580)

    # VAE 학습
    lr2: float = 1e-3
    beta: float = 0.001
    pos_weight: float = 30.0
    dice_smooth: float = 1.0
    tune_epoch: int = 50
    batch_size: int = 16

    # BT 사전학습
    lr: float = 1e-3
    pre_epoch: int = 20
    bt_batch_size: int = 64      # OOM 나면 32로
    proj_dim: int = 512
    lambd: float = 0.005

    log_every: int = 20


# ──────────────────────────────────────────────
# [수정 2] BT 전용 Dataset 래퍼: x만 반환 (라벨 유무/형식과 무관하게 동작)
# ──────────────────────────────────────────────
class ImageOnly(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        # (x, y) 튜플이면 x만, 아니면 그대로
        return item[0] if isinstance(item, (tuple, list)) else item


# ──────────────────────────────────────────────
# [수정 3] 샘플별 독립 augmentation
# ──────────────────────────────────────────────
def bt_augment(x):
    B, C, H, W = x.shape
    device = x.device

    # horizontal flip — 샘플별 p=0.5
    flip = torch.rand(B, 1, 1, 1, device=device) < 0.5
    x = torch.where(flip, torch.flip(x, dims=[-1]), x)

    # random resized crop — 샘플별 (scale 0.7~1.0)
    out = torch.empty_like(x)
    for i in range(B):
        s = 0.7 + 0.3 * torch.rand(1).item()
        ch, cw = int(H * s), int(W * s)
        top = torch.randint(0, H - ch + 1, (1,)).item()
        left = torch.randint(0, W - cw + 1, (1,)).item()
        crop = x[i:i + 1, :, top:top + ch, left:left + cw]
        out[i:i + 1] = F.interpolate(crop, size=(H, W),
                                     mode='bilinear', align_corners=False)
    x = out

    # 밝기 스케일 — 샘플별 (0.8~1.2배)
    scale = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=device)
    x = x * scale

    # 가우시안 노이즈 (원소별)
    x = x + torch.randn_like(x) * 0.03

    return x.clamp(0, 1)  # 입력이 [0,1] 정규화라는 가정. z-score면 clamp 제거할 것


class BT_VAE(nn.Module):
    def __init__(self, z_dim=32, stride=2, kernel_size=3, padding=1,
                 output_padding=0, proj_dim=512, out_size=(420, 580)):
        super().__init__()
        self.z_dim = z_dim
        self.out_size = out_size

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

        # 출력은 logits (Sigmoid 없음)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, kernel_size, stride, padding, output_padding),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size, stride, padding, output_padding),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size, stride, padding, output_padding),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size, stride, padding, output_padding)
        )

        self.log_var = nn.Conv2d(256, z_dim, kernel_size=1)
        self.mu = nn.Conv2d(256, z_dim, kernel_size=1)

        # ── BT 사전학습 전용
        self.projector = nn.Sequential(
            nn.Linear(256, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
        )
        self.bn = nn.BatchNorm1d(proj_dim, affine=False)

    # ────────────── BT 사전학습 ──────────────
    def bt_loss(self, x1, x2, lambd=0.005):
        h1 = self.encoder(x1).mean(dim=[2, 3])   # GAP → (B, 256)
        h2 = self.encoder(x2).mean(dim=[2, 3])

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        B = z1.shape[0]
        c = (self.bn(z1).T @ self.bn(z2)) / B

        on_diag = (torch.diagonal(c) - 1).pow(2).sum()
        n = c.shape[0]
        off_diag = (c.flatten()[:-1].view(n - 1, n + 1)[:, 1:]).pow(2).sum()

        return on_diag + lambd * off_diag, on_diag.item(), off_diag.item()

    # ────────────── VAE ──────────────
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def decode(self, h):
        h = self.decoder(h)
        return F.interpolate(h, size=self.out_size,
                             mode='bilinear', align_corners=False)  # logits

    def reparameter(self, log_var, mu):
        std = torch.exp(log_var * 0.5)
        return mu + torch.randn_like(std) * std

    # [수정 1] 추론/검증은 sample=False로 결정적 경로 (z = mu)
    def forward(self, x, sample=True):
        mu, log_var = self.encode(x)
        z = self.reparameter(log_var, mu) if sample else mu
        logits = self.decode(z)
        return logits, mu, log_var

    @staticmethod
    def get_VAE_Loss(logits, y, log_var, mu, beta, pos_weight, smooth=1.0, verbose=False):
        recon_loss = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight, reduction='mean')

        prob = torch.sigmoid(logits)
        inter = (prob * y).sum(dim=[1, 2, 3]) * 2
        dice = (inter + smooth) / (prob.sum(dim=[1, 2, 3]) + y.sum(dim=[1, 2, 3]) + smooth)
        dice_loss = (1 - dice).mean()

        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        if verbose:
            print(f"bce: {recon_loss.item():.4f}, dice: {dice_loss.item():.4f}, "
                  f"kl: {kl_loss.item():.4f}, mu_std: {mu.std().item():.3f}")

        return recon_loss + dice_loss + kl_loss * beta


# ══════════════════════════════════════════════
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BT_VAE(
        z_dim=Config.z_dim,
        stride=Config.stride,
        kernel_size=Config.kernel_size,
        padding=Config.padding,
        output_padding=Config.output_padding,
        proj_dim=Config.proj_dim,
        out_size=Config.out_size,
    ).to(device)

    # ────────────────────────────────────────
    # 1) BT 사전학습
    # ────────────────────────────────────────
    RUN_BT = True   # BT 유무 비교 실험 시 토글

    if RUN_BT:
        bt_data = ImageOnly(get_data('DataSet/Processed/1_/train', only_p=False))
        bt_loader = DataLoader(bt_data, batch_size=Config.bt_batch_size,
                               shuffle=True, drop_last=True,
                               num_workers=2, pin_memory=True)

        # BT에서 실제로 학습되는 파라미터만 (encoder + projector)
        op1 = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.projector.parameters()),
            lr=Config.lr)

        for epoch in range(Config.pre_epoch):
            model.train()
            total, tot_on, tot_off, count = 0, 0, 0, 0
            for x in tqdm(bt_loader, desc=f"BT epoch {epoch}"):
                x = x.to(device)
                x1, x2 = bt_augment(x), bt_augment(x)

                loss, on_d, off_d = model.bt_loss(x1, x2, Config.lambd)

                op1.zero_grad()
                loss.backward()
                op1.step()
                total += loss.item()
                tot_on += on_d
                tot_off += off_d
                count += 1

            print(f"BT epoch {epoch}: loss {total / count:.4f} "
                  f"(on_diag {tot_on / count:.4f}, off_diag {tot_off / count:.4f})")

        torch.save(model.state_dict(), 'BT_pretrained.pth')

    # ────────────────────────────────────────
    # 2) VAE 학습 (fine-tune)
    # ────────────────────────────────────────
    data = get_data('DataSet/Processed/1_/train', only_p=True)

    train_len = int(len(data) * 0.7)
    val_len = len(data) - train_len
    train_data, val_data = random_split(
        data, [train_len, val_len],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    # 시각화용 고정 샘플 (val에서 한 번만 뽑아 매 epoch 같은 이미지로 비교)
    fixed_x, fixed_y = next(iter(val_loader))
    fixed_x, fixed_y = fixed_x.to(device), fixed_y.to(device)

    op2 = torch.optim.Adam(model.parameters(), lr=Config.lr2)
    best_dice = 0.0
    pos_weight = torch.tensor([Config.pos_weight], device=device)

    for epoch in range(Config.tune_epoch):
        model.train()
        total_VAE_loss = 0
        count = 0

        for x, y in tqdm(train_loader, desc=f"epoch {epoch}"):
            x = x.to(device)
            y = y.to(device)

            logits, mu, log_var = model(x)   # 학습은 sample=True (기본)

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

        # ── validation dice: 결정적 경로(sample=False), per-image 집계
        model.eval()
        dices = []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vlogits, _, _ = model(vx, sample=False)   # [수정 1]
                p = torch.sigmoid(vlogits)
                inter = (p * vy).sum(dim=[1, 2, 3]) * 2
                d = (inter + 1.0) / (p.sum(dim=[1, 2, 3]) + vy.sum(dim=[1, 2, 3]) + 1.0)
                dices.extend(d.cpu().tolist())            # 이미지별로 모아서
        val_dice = sum(dices) / len(dices)                # 전체 평균 (배치 크기 편향 제거)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'VAE_best.pth')
        torch.save(model.state_dict(), 'last.pth')

        print(f"epoch {epoch}, train loss: {common:.4f}, "
              f"val dice: {val_dice:.4f} (best {best_dice:.4f})")

        # 5 epoch마다 고정 val 샘플로 예측 시각화
        if epoch % 5 == 0:
            with torch.no_grad():
                fl, _, _ = model(fixed_x, sample=False)
                prob = torch.sigmoid(fl[0, 0]).cpu().numpy()
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].imshow(fixed_x[0, 0].cpu(), cmap='gray'); ax[0].set_title('input')
            ax[1].imshow(fixed_y[0, 0].cpu(), cmap='gray'); ax[1].set_title('GT')
            ax[2].imshow(prob, cmap='gray'); ax[2].set_title(f'pred (epoch {epoch})')
            plt.savefig(f'pred_epoch{epoch}.png'); plt.close()
            model.train()