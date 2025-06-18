# train_UNet3D.py  ——  用 UNet3D 训练 ADNI 数据（分割/分类皆可）
# ---------------------------------------------------------------
import os, json, time, csv, numpy as np
from multiprocessing import freeze_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split, StratifiedKFold

from monai.data import Dataset
from datasets.ADNI import ADNI, ADNI_transform          # ← 保持不变
from models.unet3d import UNet3D                               # ← NEW：导入 3D U-Net
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             f1_score, precision_score, recall_score,
                             confusion_matrix, roc_curve, auc)

# ------------ 配置读取，与原脚本一致 ----------------
def load_config(path="config/config_unet.json"):
    with open(path) as f: return json.load(f)

class Config:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)
        self.weight_decay = getattr(self, 'weight_decay', 1e-4)
        self.dropout_rate = getattr(self, 'dropout_rate', 0.5)
        self.n_splits = getattr(self, 'n_splits', 5)
        self.print_config()

    def print_config(self):
        print("Configuration Parameters:\n" + "=" * 40)
        for k, v in vars(self).items():
            print(f"{k}: {v}")
        print("=" * 40)

# ------------- 生成 UNet3D（可选分类头） --------------
def generate_unet(cfg, device=torch.device('cpu')):
    """返回一个适配分割 / 分类的 UNet3D"""
    model = UNet3D(
        in_channels=cfg.in_channels,      # 如 1（MRI）或 3（多模态堆叠）
        num_classes=cfg.nb_class       # 若做分割 = 类别数；若做分类 = 特征通道数
    ).to(device)

    if not cfg.seg_task:  # —— 分类任务：UNet 编码后再接一个全局池化线性层
        model = nn.Sequential(
            model,                                 # UNet3D backbone
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Dropout(p=cfg.dropout_rate),
            nn.Linear(cfg.nb_class, 2)          # AD vs CN
        ).to(device)
    return model

# ---------- 指标计算（保持与原脚本一致） --------------
def calculate_metrics(y_true, y_pred, y_score):
    return {
        'acc'     : accuracy_score(y_true, y_pred),
        'auc'     : roc_auc_score(y_true, y_score),
        'f1'      : f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall'  : recall_score(y_true, y_pred, zero_division=0),
        'cm'      : confusion_matrix(y_true, y_pred)
    }

# ---------------- 主训练入口 --------------------------
def train():
    torch.manual_seed(42); np.random.seed(42)
    cfg = Config(load_config())

    # 1）数据切分
    dataset = ADNI(cfg.label_file, cfg.mri_dir, cfg.task, cfg.augment).data_dict
    tr_val, test_data = train_test_split(dataset, test_size=0.2, random_state=42,
                                         stratify=[d['label'] for d in dataset])
    labels = [d['label'] for d in tr_val]

    # 2）日志
    writer = SummaryWriter(cfg.checkpoint_dir)
    csv_path = os.path.join(cfg.checkpoint_dir, 'cv_results.csv')
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['fold', 'epoch',
                                'tr_acc', 'tr_auc', 'tr_loss',
                                'vl_acc', 'vl_auc', 'vl_loss', 'lr'])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 3）交叉验证
    kf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(tr_val, labels), 1):
        print(f"\n=== Fold {fold}/{cfg.n_splits} ===")
        train_data = [tr_val[i] for i in train_idx]
        val_data   = [tr_val[i] for i in val_idx]

        tf_tr, tf_vt = ADNI_transform(augment=cfg.augment)
        ds_tr = Dataset(train_data, transform=tf_tr)
        ds_vl = Dataset(val_data,   transform=tf_vt)
        loader_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
        loader_vl = DataLoader(ds_vl, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

        # ---- 建模 ----
        model = generate_unet(cfg, device)          # ← 替换为 UNet3D
        if cfg.seg_task:                            # 分割 or 分类 损失
            criterion = nn.CrossEntropyLoss()
        else:
            # 分类同以前：类别权重
            class_counts  = np.bincount([d['label'] for d in train_data])
            class_weights = torch.tensor(1.0 / class_counts,
                                         dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)

        # 学习率调度（保持原逻辑）...
        warm = max(1, min(10, int(cfg.num_epochs*0.1))); total = cfg.num_epochs
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, 0.1, 1.0, total_iters=warm),
                CosineAnnealingLR(optimizer, T_max=total-warm,
                                  eta_min=cfg.lr*1e-4)
            ],
            milestones=[warm]
        )

        # 训练循环（与原脚本相同，只改 forward / loss 逻辑）
        best_metric, patience, no_improve = -np.inf, 5, 0
        for epoch in range(1, cfg.num_epochs+1):
            t0 = time.time()
            # --- train ---
            model.train(); tr_loss=0; y_true=y_pred=y_score=[]
            for batch in loader_tr:
                x = batch['MRI'].to(device)
                y = batch['label'].to(device).long().squeeze()
                out = model(x)
                loss = criterion(out, y)
                tr_loss += loss.item()
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                probs = torch.softmax(out, 1)[:,1].detach().cpu().numpy()
                preds = out.argmax(1).detach().cpu().numpy()
                y_true.extend(y.cpu().numpy()); y_score.extend(probs); y_pred.extend(preds)

            tr_metrics = calculate_metrics(y_true, y_pred, y_score)
            tr_loss /= len(loader_tr)

            # --- val ---
            model.eval(); vl_loss=0; v_true=v_pred=v_score=[]
            with torch.no_grad():
                for batch in loader_vl:
                    x = batch['MRI'].to(device)
                    y = batch['label'].to(device).long().squeeze()
                    out = model(x)
                    loss = criterion(out, y) if not cfg.seg_task else criterion(out, y.unsqueeze(1))
                    vl_loss += loss.item()
                    probs = torch.softmax(out, 1)[:,1].cpu().numpy()
                    v_score.extend(probs); v_pred.extend(out.argmax(1).cpu().numpy())
                    v_true.extend(y.cpu().numpy())
            vl_metrics = calculate_metrics(v_true, v_pred, v_score)
            vl_loss /= len(loader_vl)

            # --- 日志 & lr ---
            lr_now = scheduler.get_last_lr()[0]; scheduler.step()
            writer.add_scalar(f'fold{fold}/train/acc', tr_metrics['acc'], epoch)
            writer.add_scalar(f'fold{fold}/val/acc',  vl_metrics['acc'], epoch)
            # ... 其他 scalar 省略 ...

            print(f"Fold{fold} Ep{epoch:03d} | tr_acc={tr_metrics['acc']:.3f} "
                  f"vl_acc={vl_metrics['acc']:.3f} lr={lr_now:.6f} "
                  f"time={time.time()-t0:.1f}s")

            # 早停
            metric_now = 0.5*vl_metrics['auc'] + 0.5*vl_metrics['acc']
            if metric_now > best_metric:
                best_metric, no_improve = metric_now, 0
                torch.save(model.state_dict(),
                           os.path.join(cfg.checkpoint_dir,
                                        f"best_fold{fold}.pth"))
            else:
                no_improve += 1
                if no_improve >= patience: break
        # fold end
    writer.close()

if __name__ == "__main__":
    freeze_support()
    train()
