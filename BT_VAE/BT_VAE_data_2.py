"""
BT-VAE 재현본 (학습 스크립트)
논문: 박민영 외, "Barlow Twins 기반 VAE를 활용한 제한된 레이블 환경의 의료영상 분할",
      디지털콘텐츠학회논문지 Vol.27 No.5, pp.1391-1399, 2026.

논문에 명시되지 않아 추정한 부분은 [추정] 주석으로 표시.
get_data.py (개선판)와 함께 사용.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm

from get_data import get_data, split_811, take_labeled, positive_negative_split


@dataclass
class Config:
    # ── 데이터
    data_dir: str = 'DataSet/Processed/2_/train'
    in_size: tuple = (512, 512)    # [추정] 잠재 8×8 → 6단 다운샘플 역산. 정사각이라 rot90도 안전
    label_ratio: float = 0.3       # 논문 실험 1: 0.1 / 0.3 / 0.5 / 0.7 / 0.9
    seed: int = 42

    # ── 구조
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    output_padding: int = 1
    z_dim: int = 1024              # 논문: 잠재 공간 1024×8×8

    # ── 사전학습 (논문: AdamW, lr 3e-4, wd 1e-4, batch 16, 200 epoch)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    pre_epoch: int = 200

    # ── 미세조정 (논문: lr 1e-4, batch 16, 400 epoch, α 0.99)
    lr2: float = 1e-4
    tune_epoch: int = 400
    alpha: float = 0.99            # 논문 식(4): L = α·Dice + (1−α)·BCE  ※ 에폭 램프 없이 고정

    # ── 손실 하이퍼파라미터
    lam: float = 0.2               # 논문 식(3) λ
    tau: float = 1.0               # [추정] 식(2) τ 미기재
    beta: float = 0.001            # [추정] KL 정규화 가중치 미기재

    batch_size: int = 16           # OOM 시 8로 (논문값 이탈이므로 기록할 것)
    eps: float = 1.0
    log_every: int = 20
    vis_every: int = 20            # 예측 시각화 저장 주기(epoch)

    run_bt: bool = True
    bt_ckpt: str = 'BT_pretrained_2.pth'


class BT_VAE(nn.Module):
    def __init__(self, z_dim=1024, stride=2, kernel_size=3, padding=1,
                 output_padding=1, in_size=(256, 256)):
        super().__init__()
        self.z_dim = z_dim
        self.in_size = in_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size, stride, padding),
            nn.ReLU(),
        )

        # 논문 2-2: 평균 벡터 / 표준편차 벡터를 표현 학습 대상으로 사용
        self.mu = nn.Conv2d(1024, z_dim, kernel_size=1)
        self.log_var = nn.Conv2d(1024, z_dim, kernel_size=1)

        # [추정] 디코더 채널 구성 미기재. 인코더 대칭으로 구성
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, kernel_size, stride, padding, output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size, stride, padding, output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size, stride, padding, output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size, stride, padding, output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size, stride, padding, output_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size, stride, padding, output_padding),
            nn.Sigmoid()   # 논문 식(6) BCE가 확률 입력을 전제하므로 유지
        )

    # ────────────────────────── 1단계: 사전학습 ──────────────────────────
    def encode_bt(self, x):
        """논문 2-2: 좌측 90도 / 우측 90도 회전 두 뷰를 가중치 공유 인코더에 통과.
        재매개변수화 샘플링 없이 분포 파라미터 자체를 표현 벡터로 사용."""
        x_left = torch.rot90(x, k=1, dims=(-2, -1))
        x_right = torch.rot90(x, k=-1, dims=(-2, -1))

        h1 = self.encoder(x_left)
        mu1, log_var1 = self.mu(h1), self.log_var(h1)

        h2 = self.encoder(x_right)
        mu2, log_var2 = self.mu(h2), self.log_var(h2)

        return mu1, log_var1, mu2, log_var2

    @staticmethod
    def get_CM(mu1, mu2):
        """두 뷰의 평균 벡터로부터 채널 간 상호 상관 행렬 C_mu 산출."""
        C = mu1.shape[1]

        a = mu1.permute(0, 2, 3, 1).reshape(-1, C)
        b = mu2.permute(0, 2, 3, 1).reshape(-1, C)
        N = a.shape[0]

        a = (a - a.mean(0)) / (a.std(0) + 1e-8)
        b = (b - b.mean(0)) / (b.std(0) + 1e-8)

        return torch.matmul(a.T, b) / N

    @staticmethod
    def get_BT_KL_loss(mu1, log_var1, mu2, log_var2, tau=1.0, eps=1e-8):
        """논문 식(2): L_kl = Σ_i C²_kl,ii + Σ_i Σ_{j≠i} (C_kl,ij − τ)²
        동일 채널 쌍의 KL은 0으로, 상이 채널 쌍의 KL은 τ로 수렴시킨다."""
        C = mu1.shape[1]

        m1 = mu1.permute(0, 2, 3, 1).reshape(-1, C)
        v1 = log_var1.exp().permute(0, 2, 3, 1).reshape(-1, C)
        m2 = mu2.permute(0, 2, 3, 1).reshape(-1, C)
        v2 = log_var2.exp().permute(0, 2, 3, 1).reshape(-1, C)

        mean1 = m1.mean(dim=0).unsqueeze(1)                        # (C,1)
        var1 = (v1.mean(dim=0) + m1.var(dim=0)).unsqueeze(1) + eps

        mean2 = m2.mean(dim=0).unsqueeze(0)                        # (1,C)
        var2 = (v2.mean(dim=0) + m2.var(dim=0)).unsqueeze(0) + eps

        kl = 0.5 * torch.log(var2 / var1) + (var1 + (mean2 - mean1).pow(2)) / (2 * var2) - 0.5

        eye = torch.eye(C, device=kl.device)
        diag_loss = (kl.diagonal() ** 2).sum()
        off_diag_loss = ((kl * (1 - eye) - tau * (1 - eye)) ** 2).sum()

        return (diag_loss + off_diag_loss) / (C * C)

    @staticmethod
    def get_BT_mu_loss(CM, lam=0.2):
        """논문 식(3): L_mu = Σ_i (1 − C_mu,ii)² + λ Σ_i Σ_{j≠i} C²_mu,ij
        ※ 원본의 이중 for문과 수학적으로 동일. z_dim=1024에서는 벡터화 필수."""
        C = CM.shape[0]
        eye = torch.eye(C, device=CM.device)
        weight = eye + (1 - eye) * lam
        return ((eye - CM).pow(2) * weight).sum() / (C * C)

    def bt_loss(self, x, lam=0.2, tau=1.0, diag=False):
        mu1, log_var1, mu2, log_var2 = self.encode_bt(x)

        CM = self.get_CM(mu1, mu2)
        l_kl = self.get_BT_KL_loss(mu1, log_var1, mu2, log_var2, tau)
        l_mu = self.get_BT_mu_loss(CM, lam)

        stats = {}
        if diag:
            C = CM.shape[0]
            eye = torch.eye(C, device=CM.device).bool()
            stats = {
                'C_ii': CM[eye].mean().item(),          # 목표 1.0 (불변성)
                'C_ij': CM[~eye].abs().mean().item(),   # 목표 0.0 (탈상관)
                'mu_std': mu1.std().item(),
            }
        return l_kl + l_mu, l_kl.item(), l_mu.item(), stats

    # ────────────────────────── 2단계: 미세조정 ──────────────────────────
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def reparameter(self, log_var, mu):
        std = torch.exp(log_var * 0.5)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.decoder(z)
        return F.interpolate(h, size=self.in_size, mode='bilinear', align_corners=False)

    def forward(self, x, sample=True):
        mu, log_var = self.encode(x)
        # [추정] 평가 시 결정적 추론(z=mu). 논문 미기재이나 지표 재현성 확보를 위해 사용
        z = self.reparameter(log_var, mu) if sample else mu
        return self.decode(z), mu, log_var

    @staticmethod
    def dice_loss(pred, y, eps=1e-5):
        """논문 식(5)"""
        inter = (pred * y).sum(dim=[1, 2, 3]) * 2
        dice = (inter + eps) / (pred.sum(dim=[1, 2, 3]) + y.sum(dim=[1, 2, 3]) + eps)
        return (1 - dice).mean()

    @staticmethod
    def get_VAE_Loss(pred, y, mu, log_var, dice, alpha, beta, verbose=False):
        """논문 식(4): L = α·Dice + (1−α)·BCE  (α 고정)
        + 잠재 분포 안정화를 위한 KL 정규화"""
        bce = F.binary_cross_entropy(pred, y, reduction='mean')
        seg_loss = alpha * dice + (1 - alpha) * bce

        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        if verbose:
            print(f"  dice {dice.item():.4f} | bce {bce.item():.4f} | "
                  f"kl {kl_loss.item():.4f} | mu_std {mu.std().item():.3f} | "
                  f">0.5 {(pred > 0.5).float().mean().item():.4f} "
                  f"(y {(y > 0.5).float().mean().item():.4f})")

        return seg_loss + beta * kl_loss


# ══════════════════════════ 평가 ══════════════════════════
@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    """논문 2-3 3): Dice, IoU, Precision, Recall (이미지 단위 평균).

    빈 마스크(negative) 샘플은 GT·예측이 모두 비면 Dice/IoU를 1로 정의한다.
    Precision/Recall은 정의되지 않으므로 positive 샘플에 대해서만 집계한다.
    """
    was_training = model.training
    model.eval()

    sum_d = sum_i = 0.0
    sum_p = sum_r = 0.0
    n_all = n_pos = 0
    neg_total = 0          # negative 샘플 수
    neg_fp_imgs = 0        # 그중 한 픽셀이라도 오탐한 이미지 수
    neg_fp_pixels = 0.0    # negative 샘플의 평균 오탐 픽셀 수

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred, _, _ = model(x, sample=False)

        pb = (pred > thr).float()
        yb = (y > 0.5).float()

        tp = (pb * yb).sum(dim=[1, 2, 3])
        fp = (pb * (1 - yb)).sum(dim=[1, 2, 3])
        fn = ((1 - pb) * yb).sum(dim=[1, 2, 3])
        e = 1e-7

        is_pos = yb.sum(dim=[1, 2, 3]) > 0
        both_empty = (yb.sum(dim=[1, 2, 3]) == 0) & (pb.sum(dim=[1, 2, 3]) == 0)

        d = 2 * tp / (2 * tp + fp + fn + e)
        i = tp / (tp + fp + fn + e)
        d = torch.where(both_empty, torch.ones_like(d), d)
        i = torch.where(both_empty, torch.ones_like(i), i)

        sum_d += d.sum().item()
        sum_i += i.sum().item()
        n_all += x.shape[0]

        if is_pos.any():
            sum_p += (tp[is_pos] / (tp[is_pos] + fp[is_pos] + e)).sum().item()
            sum_r += (tp[is_pos] / (tp[is_pos] + fn[is_pos] + e)).sum().item()
            n_pos += int(is_pos.sum().item())

        neg = ~is_pos
        if neg.any():
            neg_total += int(neg.sum().item())
            neg_fp_imgs += int((fp[neg] > 0).sum().item())
            neg_fp_pixels += fp[neg].sum().item()

    if was_training:
        model.train()

    return {
        'dice': sum_d / max(n_all, 1),
        'iou': sum_i / max(n_all, 1),
        'precision': sum_p / max(n_pos, 1),
        'recall': sum_r / max(n_pos, 1),
        'neg_fp_rate': neg_fp_imgs / neg_total if neg_total else float('nan'),
        'neg_fp_pixels': neg_fp_pixels / neg_total if neg_total else float('nan'),
        'n_all': n_all, 'n_pos': n_pos, 'n_neg': neg_total,
    }


def fmt(m):
    s = (f"Dice {m['dice']:.4f} IoU {m['iou']:.4f} "
         f"Prec {m['precision']:.4f} Rec {m['recall']:.4f}")
    if m['n_neg']:
        s += f" | neg 오탐률 {m['neg_fp_rate']:.3f} ({m['neg_fp_pixels']:.0f}px)"
    return s


@torch.no_grad()
def save_vis(model, x, y, epoch, path='vis'):
    """고정 샘플의 input / GT / pred 저장"""
    os.makedirs(path, exist_ok=True)
    was_training = model.training
    model.eval()
    pred, _, _ = model(x, sample=False)
    if was_training:
        model.train()

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(x[0, 0].cpu(), cmap='gray'); ax[0].set_title('input')
    ax[1].imshow(y[0, 0].cpu(), cmap='gray'); ax[1].set_title('GT')
    ax[2].imshow(pred[0, 0].cpu(), cmap='gray'); ax[2].set_title(f'pred (ep {epoch})')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.savefig(f'{path}/pred_epoch{epoch:03d}.png', dpi=80)
    plt.close()


# ══════════════════════════ 메인 ══════════════════════════
if __name__ == '__main__':
    torch.manual_seed(Config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BT_VAE(
        z_dim=Config.z_dim,
        stride=Config.stride,
        kernel_size=Config.kernel_size,
        padding=Config.padding,
        output_padding=Config.output_padding,
        in_size=Config.in_size,
    ).to(device)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ── 데이터: 논문은 negative 포함 전체 사용 (only_p=False), 8:1:1 분할
    data = get_data(Config.data_dir, only_p=False, size=Config.in_size)
    train_all, val_data, test_data = split_811(data, seed=Config.seed)
    print(f"split: train {len(train_all)} / val {len(val_data)} / test {len(test_data)}")

    # 사전학습: train 전체 (레이블 미사용)
    bt_loader = DataLoader(train_all, batch_size=Config.batch_size, shuffle=True,
                           drop_last=True, num_workers=2, pin_memory=True)

    # 미세조정: 논문 실험 1의 레이블 비율만큼만 사용
    train_labeled = take_labeled(train_all, Config.label_ratio, seed=Config.seed)
    print(f"레이블 {Config.label_ratio*100:.0f}%: {len(train_labeled)}장으로 미세조정")

    train_loader = DataLoader(train_labeled, batch_size=Config.batch_size, shuffle=True,
                              drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=Config.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # 시각화용 고정 샘플
    fx, fy = next(iter(val_loader))
    fx, fy = fx.to(device), fy.to(device)

    # ─────────────── 1단계: Barlow Twins 사전학습 ───────────────
    if Config.run_bt:
        if os.path.exists(Config.bt_ckpt):
            model.load_state_dict(torch.load(Config.bt_ckpt, map_location=device))
            print(f"{Config.bt_ckpt} 로드 — 사전학습 생략")
        else:
            op1 = torch.optim.AdamW(model.parameters(), lr=Config.lr,
                                    weight_decay=Config.weight_decay)

            for epoch in range(Config.pre_epoch):
                model.train()
                tot = tot_kl = tot_mu = 0.0
                cnt = 0

                for x, _ in tqdm(bt_loader, desc=f"BT {epoch}"):
                    x = x.to(device, non_blocking=True)

                    loss, l_kl, l_mu, _ = model.bt_loss(x, Config.lam, Config.tau)

                    op1.zero_grad()
                    loss.backward()
                    op1.step()

                    tot += loss.item(); tot_kl += l_kl; tot_mu += l_mu; cnt += 1

                print(f"BT epoch {epoch}: loss {tot/cnt:.4f} "
                      f"(L_kl {tot_kl/cnt:.4f}, L_mu {tot_mu/cnt:.4f})")

            torch.save(model.state_dict(), Config.bt_ckpt)

    # ─────────────── 2단계: VAE 미세조정 ───────────────
    op2 = torch.optim.AdamW(model.parameters(), lr=Config.lr2,
                            weight_decay=Config.weight_decay)
    best_dice = 0.0
    tag = f"covid_r{int(Config.label_ratio*100)}{'_bt' if Config.run_bt else ''}"
    
    for epoch in range(Config.tune_epoch):
        model.train()
        tot, cnt = 0.0, 0

        for x, y in tqdm(train_loader, desc=f"VAE {epoch}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred, mu, log_var = model(x)
            dice = model.dice_loss(pred, y, Config.eps)

            loss = model.get_VAE_Loss(pred, y, mu, log_var, dice,
                                      Config.alpha, Config.beta,
                                      verbose=(cnt % Config.log_every == 0))

            op2.zero_grad()
            loss.backward()
            op2.step()

            tot += loss.item(); cnt += 1

        m = evaluate(model, val_loader, device)

        if m['dice'] > best_dice:
            best_dice = m['dice']
            torch.save(model.state_dict(), f'VAE_best_{tag}.pth')
        torch.save(model.state_dict(), f'last_{tag}.pth')

        print(f"epoch {epoch} | train {tot/cnt:.4f} | val {fmt(m)} (best {best_dice:.4f})")

        if epoch % Config.vis_every == 0:
            save_vis(model, fx, fy, epoch, path=f'vis_{tag}')

    # ─────────────── 최종 테스트 ───────────────
    model.load_state_dict(torch.load(f'VAE_best_{tag}.pth', map_location=device))

    tm = evaluate(model, test_loader, device)
    print(f"\n[TEST · 레이블 {Config.label_ratio*100:.0f}%] {fmt(tm)}")

    # 논문 실험 2: positive / negative 분리 평가
    pos_set, neg_set = positive_negative_split(test_data)
    if len(pos_set):
        pm = evaluate(model, DataLoader(pos_set, batch_size=Config.batch_size), device)
        print(f"[TEST · positive {len(pos_set)}장] {fmt(pm)}")
    if len(neg_set):
        nm = evaluate(model, DataLoader(neg_set, batch_size=Config.batch_size), device)
        print(f"[TEST · negative {len(neg_set)}장] "
              f"오탐률 {nm['neg_fp_rate']:.3f}, 평균 오탐 {nm['neg_fp_pixels']:.0f}px")