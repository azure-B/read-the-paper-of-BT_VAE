"""
get_data.py — BT-VAE 논문 재현용 데이터 로더

논문: 박민영 외, JDCS Vol.27 No.5, 2026.
- BP Dataset 5,635쌍 전체 사용 (negative 포함) → only_p 기본값 False
- 8:1:1 분할, 레이블 비율 10~90% 실험 지원
"""

import glob
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


class get_data(Dataset):
    """
    Args:
        folder      : .npy 파일 폴더
        only_p      : True면 마스크가 비어있지 않은 샘플만 사용.
                      논문 재현에는 False (negative 포함이 실험 2의 전제)
        size        : (H, W)로 리사이즈. None이면 원본 유지.
                      논문의 잠재 1024×8×8을 맞추려면 (256, 256)
        binarize    : 마스크를 0/1로 이진화 (리사이즈 보간 잔여값 제거)
    """

    def __init__(self, folder, only_p=False, size=(256, 256), binarize=True):
        self.size = size
        self.binarize = binarize

        # sorted() 필수: glob 순서는 파일시스템 의존이라
        # seed를 고정해도 실행 환경마다 split이 달라진다
        all_files = sorted(glob.glob(os.path.join(folder, '*.npy')))

        img_files = [f for f in all_files if 'mask' not in os.path.basename(f)]
        def to_mask(f):
                    body = f[:-4]                     # '.npy' 제거 → '.../bjorke_1.png'
                    stem, ext = body.rsplit('.', 1)   # '.../bjorke_1', 'png'
                    return f"{stem}_mask.{ext}.npy"

        mask_files = [to_mask(f) for f in img_files]
        # 짝이 실제로 존재하는 것만 (누락 시 학습 도중 죽는 것 방지)
        pairs = [(i, m) for i, m in zip(img_files, mask_files) if os.path.exists(m)]
        if len(pairs) < len(img_files):
            print(f"[get_data] 마스크 누락 {len(img_files) - len(pairs)}개 제외")

        # 각 샘플의 positive 여부를 한 번만 계산해 보관
        # (실험 2에서 positive/negative 분리 평가에 사용)
        self.is_positive = [bool(np.load(m).sum() > 0) for _, m in pairs]

        if only_p:
            pairs = [p for p, pos in zip(pairs, self.is_positive) if pos]
            self.is_positive = [True] * len(pairs)

        self.img_files = [p[0] for p in pairs]
        self.mask_files = [p[1] for p in pairs]

        n_pos = sum(self.is_positive)
        print(f"[get_data] {folder}: 총 {len(self.img_files)}장 "
              f"(positive {n_pos}, negative {len(self.img_files) - n_pos})")

    def __len__(self):
        return len(self.img_files)

    def _to_chw(self, a):
        """(H,W) → (1,H,W), (H,W,1) → (1,H,W). 이미 (1,H,W)면 그대로."""
        t = torch.from_numpy(np.ascontiguousarray(a)).float()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[-1] == 1:
            t = t.permute(2, 0, 1)
        return t

    def __getitem__(self, i):
        x = self._to_chw(np.load(self.img_files[i]))
        y = self._to_chw(np.load(self.mask_files[i]))

        if self.size is not None and x.shape[-2:] != self.size:
            # 배치 차원을 임시로 붙여 interpolate
            x = F.interpolate(x.unsqueeze(0), size=self.size,
                              mode='bilinear', align_corners=False).squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=self.size,
                              mode='nearest').squeeze(0)

        if self.binarize:
            y = (y > 0.5).float()

        return x, y


# ══════════════════════════════════════════════════════════
def split_811(dataset, seed=42):
    """논문 2-3: train/validation/test = 8:1:1"""
    g = np.random.default_rng(seed)
    idx = g.permutation(len(dataset))

    n_tr = int(len(dataset) * 0.8)
    n_va = int(len(dataset) * 0.1)

    return (Subset(dataset, idx[:n_tr].tolist()),
            Subset(dataset, idx[n_tr:n_tr + n_va].tolist()),
            Subset(dataset, idx[n_tr + n_va:].tolist()))


def take_labeled(train_subset, ratio, seed=42):
    """논문 실험 1: 레이블 데이터 비율(10~90%)만큼 train에서 추출.

    사전학습은 전체 train으로(레이블 불필요), 미세조정은 이 부분집합으로 수행한다.
    """
    g = np.random.default_rng(seed)
    n = len(train_subset)
    k = max(1, int(round(n * ratio)))
    sel = g.permutation(n)[:k].tolist()
    return Subset(train_subset, sel)


def positive_negative_split(subset):
    """실험 2용: 평가 세트를 positive / negative로 분리.

    negative(빈 마스크)에서는 Dice가 정의되지 않으므로
    오탐 면적·오탐 발생률 같은 별도 지표로 봐야 한다.
    """
    base = subset.dataset
    while isinstance(base, Subset):          # 중첩 Subset 대응
        idx_map = base.indices
        base = base.dataset
    flags = base.is_positive

    def resolve(s):
        idx = s.indices
        while isinstance(s.dataset, Subset):
            s = s.dataset
            idx = [s.indices[i] for i in idx]
        return idx

    real_idx = resolve(subset)
    pos = [i for i, r in enumerate(real_idx) if flags[r]]
    neg = [i for i, r in enumerate(real_idx) if not flags[r]]
    return Subset(subset, pos), Subset(subset, neg)


if __name__ == '__main__':
    data = get_data('DataSet/Processed/1_/train', only_p=False, size=(256, 256))
    x, y = data[0]
    print('x', x.shape, x.min().item(), x.max().item())
    print('y', y.shape, y.min().item(), y.max().item())

    tr, va, te = split_811(data)
    print(f'split: train {len(tr)}, val {len(va)}, test {len(te)}')

    tr30 = take_labeled(tr, 0.3)
    print(f'레이블 30%: {len(tr30)}')