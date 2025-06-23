import os, json, time, csv, numpy as np
from multiprocessing import freeze_support

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast          # 新 API
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, auc)

from monai.data import Dataset
# from models.unet3d import UNet3D
from datasets.ADNI import ADNI, ADNI_transform


# -------------------- 配置 --------------------
def load_cfg(path="/data/coding/Multimodal_AD/config/config.json"):
    with open(path) as f: return json.load(f)

class Cfg:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)
        self.n_splits   = getattr(self, 'n_splits', 5)
        self.batch_size = getattr(self, 'batch_size', 2)
        self.lr         = getattr(self, 'lr', 1e-5)
        self.num_epochs = getattr(self, 'num_epochs', 100)
        self.device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.fp16       = getattr(self, 'fp16', True)

# ----------------- 指标函数 -------------------
def calculate_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp + 1e-8)
    return {
        'ACC': accuracy_score(y_true, y_pred),
        'PRE': precision_score(y_true, y_pred, zero_division=0),
        'SEN': recall_score(y_true, y_pred, zero_division=0),
        'SPE': spe,
        'F1' : f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_score),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'cm' : np.array([[tn, fp], [fn, tp]])
    }

cfg = Cfg(load_cfg())
writer = SummaryWriter(cfg.checkpoint_dir)

# 数据划分
full_ds = ADNI(cfg.label_file, cfg.mri_dir, cfg.task, cfg.augment).data_dict
train_val, test_ds = train_test_split(
    full_ds, test_size=0.2, random_state=42,
    stratify=[d['label'] for d in full_ds])

# 再划分验证集
train_ds, val_ds = train_test_split(
    train_val, test_size=0.2, random_state=42,
    stratify=[d['label'] for d in train_val])


tr_tf, vl_tf = ADNI_transform(augment=cfg.augment)
tr_loader = DataLoader(Dataset(train_ds, tr_tf),
                       batch_size=cfg.batch_size, shuffle=True,
                       num_workers=4, pin_memory=True)
vl_loader = DataLoader(Dataset(val_ds, vl_tf),
                       batch_size=cfg.batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础卷积块：Conv3D + BN + ReLU（可选）
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# 上采样 + 融合
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if necessary to match size due to rounding
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3DClassifier(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        self.up4 = UpBlock(base_ch * 16, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch)

        self.avgpool = nn.AdaptiveAvgPool3d(1)  # 输出 [B, C, 1, 1, 1]
        self.classifier = nn.Linear(base_ch, num_classes)  # 最后一层通道为 base_ch

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bn = self.bottleneck(self.pool4(e4))

        d4 = self.up4(bn, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)  # shape: [B, base_ch, D, H, W]

        x = self.avgpool(d1)       # [B, base_ch, 1, 1, 1]
        x = torch.flatten(x, 1)    # [B, base_ch]
        out = self.classifier(x)   # [B, num_classes]
        return out


model = UNet3DClassifier(in_ch=cfg.in_channels, num_classes=2).to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
scaler = GradScaler(enabled=cfg.fp16)

best_auc = -np.inf
for epoch in range(1, cfg.num_epochs + 1):
    t0 = time.time()

    # -------- Train --------
    model.train(); yt, yp, ys = [], [], []
    for batch in tr_loader:
        x = batch['MRI'].to(cfg.device)
        y = batch['label'].to(cfg.device).long().view(-1)

        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=cfg.fp16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        prob = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
        pred = out.argmax(1).detach().cpu().numpy()
        yt.extend(y.cpu().numpy())
        yp.extend(pred)
        ys.extend(prob)

    tr_met = calculate_metrics(yt, yp, ys)

    # -------- Validation --------
    model.eval(); yt, yp, ys = [], [], []
    with torch.no_grad():
        for batch in vl_loader:
            x = batch['MRI'].to(cfg.device)
            y = batch['label'].to(cfg.device).long().view(-1)

            with autocast(device_type='cuda', enabled=cfg.fp16):
                out = model(x)
                loss = criterion(out, y)

            prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(1).cpu().numpy()
            yt.extend(y.cpu().numpy())
            yp.extend(pred)
            ys.extend(prob)

    vl_met = calculate_metrics(yt, yp, ys)
    scheduler.step()

    print(f"Epoch {epoch:03d} | "
          f"Train ACC={tr_met['ACC']:.4f} F1={tr_met['F1']:.4f} AUC={tr_met['AUC']:.4f} | "
          f"Val ACC={vl_met['ACC']:.4f} F1={vl_met['F1']:.4f} AUC={vl_met['AUC']:.4f} | "
          f"time={time.time()-t0:.1f}s")

    if vl_met['AUC'] > best_auc:
        best_auc = vl_met['AUC']
        torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "best_model.pth"))
        print("✅ Saved best model.")
