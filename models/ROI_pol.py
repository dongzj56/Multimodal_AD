import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ROIPooling3D(nn.Module):
    """
    将 3D-UNet 解码特征 (B,64,96,112,96) → 裁 Pad → ROI 均值池化
    输出 (B, R, 64)，默认 R=94 (AAL 1-94).
    """
    def __init__(self,
                 atlas_path : str,
                 roi_range  : tuple = (1, 94)   # 仅保留 1–94 号 ROI
                 ) -> None:
        super().__init__()

        # ---------- 1. 读 atlas ----------
        atlas   = nib.load(atlas_path).get_fdata().astype(int)    # (91,109,91)
        self.DHW = atlas.shape                                    # 原尺寸

        # ---------- 2. 生成 one-hot ----------
        roi_ids   = np.arange(roi_range[0], roi_range[1] + 1)   # [1,94]
        atlas_max = int(atlas.max())                            # 120
        oh = F.one_hot(torch.from_numpy(atlas),
                    num_classes=atlas_max + 1)[..., roi_ids] # 先 one-hot 再选通道
        oh = oh.permute(3,0,1,2).float()                       # (R,D,H,W)
        self.register_buffer("onehot", oh)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat : (B, 64, 96,112,96) —— Pad 后特征图
        return: (B, R, 64)        —— ROI × 通道均值
        """
        D, H, W = self.DHW                       # 91,109,91
        feat = feat[..., :D, :H, :W]             # 裁 Pad → (B,64,91,109,91)

        B, C = feat.size(0), feat.size(1)
        R    = self.onehot.size(0)               # 94

        # -------- 1⃣ 展平成 (B,C,N) & (R,N) --------
        feat_flat = feat.reshape(B, C, -1)       # (B,64,N)
        mask_flat = self.onehot.reshape(R, -1)   # (94,N)

        # -------- 2⃣ 逐批次 ROI 求和 --------
        # num[b,r,c] = Σ_n mask[r,n] * feat[b,c,n]
        num = torch.einsum("rn,bcn->brc", mask_flat, feat_flat)  # (B,R,64)

        # -------- 3⃣ ROI 体素数 --------
        den = mask_flat.sum(dim=1).clamp_min(1e-6)               # (R,)

        return num / den[None, :, None]          # 广播除 → (B,R,64)

import torch
import torch.nn as nn

class ROIClassifier(nn.Module):
    """
    输入  : roi_feat  (B, R, 64)   —— 94 × 64 的脑区均值特征
    输出  : logits    (B, num_cls) —— 二/多分类 logits
    结构  : Flatten → FC(6016→512) → ReLU → Dropout → FC(512→num_cls)
    """
    def __init__(self, R: int = 94, num_cls: int = 2, hidden: int = 512):
        super().__init__()
        in_dim = R * 64                         # 94*64=6016
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_cls)
        )

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        x = roi_feat.flatten(1)                 # (B, 6016)
        return self.mlp(x)                      # (B, num_cls)
