"""
train_unet3d_cls.py
---------------------------------------
用 UNet3D 主干做 MRI 二分类 (AD vs CN)
"""

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
from models.unet3d import UNet3D
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
        self.fp16       = getattr(self, 'fp16', True)
        self.device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

# --------------- 分类模型封装 ------------------
class UNet3DClassifier(nn.Module):
    def __init__(self, in_ch=1, feat_ch=64, num_classes=2):
        super().__init__()
        self.backbone = UNet3D(in_channels=in_ch, num_classes=feat_ch)
        self.pool     = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc       = nn.Linear(feat_ch, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)

# -------------------- 训练 --------------------
def train():
    cfg = Cfg(load_cfg())
    writer = SummaryWriter(cfg.checkpoint_dir)

    # 数据划分
    full_ds = ADNI(cfg.label_file, cfg.mri_dir, cfg.task, cfg.augment).data_dict
    train_val, test_ds = train_test_split(
        full_ds, test_size=0.2, random_state=42,
        stratify=[d['label'] for d in full_ds])

    labels = [d['label'] for d in train_val]
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_val, labels), 1):
        print(f"\n=== Fold {fold}/{cfg.n_splits} ===")
        tr_data = [train_val[i] for i in tr_idx]
        vl_data = [train_val[i] for i in vl_idx]

        tr_tf, vl_tf = ADNI_transform(augment=cfg.augment)
        tr_loader = DataLoader(Dataset(tr_data, tr_tf),
                               batch_size=cfg.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
        vl_loader = DataLoader(Dataset(vl_data, vl_tf),
                               batch_size=cfg.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

        # 模型与优化器
        model = UNet3DClassifier(in_ch=cfg.in_channels, num_classes=2).to(cfg.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        scaler = GradScaler(enabled=cfg.fp16)

        best_auc = -np.inf
        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()
            # -------- Train --------
            model.train(); yt,yp,ys = [],[],[]
            for batch in tr_loader:
                x = batch['MRI'].to(cfg.device, non_blocking=True)
                y = batch['label'].to(cfg.device).long().view(-1)
                optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.fp16):
                    out = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                prob = torch.softmax(out,1)[:,1].detach().cpu().numpy()
                ys.extend(prob); yp.extend(out.argmax(1).cpu().numpy()); yt.extend(y.cpu().numpy())
            tr_met = calculate_metrics(yt, yp, ys)

            # -------- Val --------
            model.eval(); yt,yp,ys = [],[],[]
            with torch.no_grad():
                for batch in vl_loader:
                    x = batch['MRI'].to(cfg.device, non_blocking=True)
                    y = batch['label'].to(cfg.device).long().view(-1)
                    with autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.fp16):
                        out = model(x)
                        loss = criterion(out, y)
                    prob = torch.softmax(out,1)[:,1].cpu().numpy()
                    ys.extend(prob); yp.extend(out.argmax(1).cpu().numpy()); yt.extend(y.cpu().numpy())
            vl_met = calculate_metrics(yt, yp, ys)

            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()

            # 日志 & 打印
            writer.add_scalar(f'fold{fold}/val/AUC', vl_met['AUC'], epoch)
            print(f"Fold{fold} Ep{epoch:03d} | "
                  f"TR  ACC={tr_met['ACC']:.4f} PRE={tr_met['PRE']:.4f} "
                  f"SEN={tr_met['SEN']:.4f} SPE={tr_met['SPE']:.4f} "
                  f"F1={tr_met['F1']:.4f} AUC={tr_met['AUC']:.4f} "
                  f"MCC={tr_met['MCC']:.4f} | "
                  f"VL  ACC={vl_met['ACC']:.4f} PRE={vl_met['PRE']:.4f} "
                  f"SEN={vl_met['SEN']:.4f} SPE={vl_met['SPE']:.4f} "
                  f"F1={vl_met['F1']:.4f} AUC={vl_met['AUC']:.4f} "
                  f"MCC={vl_met['MCC']:.4f} | "
                  f"lr={lr_now:.6f} time={time.time()-t0:.1f}s")

            # 保存最优
            if vl_met['AUC'] > best_auc:
                best_auc = vl_met['AUC']
                torch.save(model.state_dict(),
                           os.path.join(cfg.checkpoint_dir, f"best_fold{fold}.pth"))

        # fold 结束
    writer.close()
    print("\n=== CV complete ===")

    # -------- 测试阶段 --------
    test_models(cfg, test_ds)
    print("\nTraining finished!")

# ------------------ 测试函数 -------------------
def test_models(cfg, test_data):
    device = cfg.device
    test_tf = ADNI_transform(augment=False)[1]
    test_loader = DataLoader(Dataset(test_data, test_tf),
                             batch_size=cfg.batch_size, shuffle=False)

    all_metrics, all_probs, all_labels = [], [], []

    for fold in range(1, cfg.n_splits + 1):
        model = UNet3DClassifier(in_ch=cfg.in_channels, num_classes=2).to(device)
        ckpt = torch.load(os.path.join(cfg.checkpoint_dir, f"best_fold{fold}.pth"),
                          map_location=device)
        model.load_state_dict(ckpt)
        model.eval()

        probs, labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['MRI'].to(device)
                y = batch['label'].cpu().numpy()
                labels.extend(y)
                out = model(x)
                probs.extend(torch.softmax(out,1)[:,1].cpu().numpy())

        all_probs.extend(probs); all_labels.extend(labels)
        preds = (np.array(probs) > 0.5).astype(int)
        metrics = calculate_metrics(labels, preds, probs)
        all_metrics.append(metrics)

        print(f"\n=== Fold {fold} Test Metrics ===")
        for k in ['ACC','PRE','SEN','SPE','F1','AUC','MCC']:
            print(f"{k}: {metrics[k]:.4f}")
        print("Confusion Matrix:\n", metrics['cm'])

    print("\n=== Final Test Results ===")
    for k in ['ACC','PRE','SEN','SPE','F1','AUC','MCC']:
        vals = [m[k] for m in all_metrics]
        print(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# --------------------- main -------------------
if __name__ == "__main__":
    freeze_support()
    train()
