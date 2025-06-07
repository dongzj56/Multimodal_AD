# test_feature_extraction.py
import os, torch
from torch.utils.data import Dataset, DataLoader
from models.ImageEncoder import image_encoder18
from datasets.ADNI import ADNI
from monai.transforms import (
    EnsureChannelFirstd, ScaleIntensityd, EnsureTyped,
    RandFlipd, RandRotated, RandZoomd, Compose
)
import pandas as pd

# ---------- 0. 后加载 Transform（无 LoadImaged） ----------
def postload_transform(augment=False):
    keys = ['MRI']
    tfm = [
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityd(keys=keys),
        EnsureTyped(keys=keys),
    ]
    if augment:
        tfm.extend([
            RandFlipd(keys=keys, prob=0.3, spatial_axis=0),
            RandRotated(keys=keys, prob=0.3, range_x=0.05),
            RandZoomd(keys=keys, prob=0.3, min_zoom=0.95, max_zoom=1.0),
        ])
    return Compose(tfm)

# ---------- 1. 基本参数 ----------
dataroot       = r'C:\Users\dongz\Desktop\adni_dataset\test'
label_filename = r'C:\Users\dongz\Desktop\adni_dataset\test\ADNI_902.csv'
task           = 'ADCN'          # 可改 'ADCN' 等
device         = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- 2. 构建 ADNI 数据集 ----------
print("[INFO] Loading ADNI dataset...")
base_ds = ADNI(label_file=label_filename, mri_dir=dataroot, task=task)
print(f"[INFO] Dataset loaded. Total samples: {len(base_ds)}")

# ---------- 3. 包装数据集并应用 Transform ----------
train_transform = postload_transform(augment=False)

class WrappedADNI(Dataset):
    def __init__(self, baseds, transform):
        self.baseds = baseds
        self.transform = transform
    def __len__(self):
        return len(self.baseds)
    def __getitem__(self, idx):
        img, label = self.baseds[idx]
        img = self.transform({'MRI': img})['MRI']
        return img, int(label)

ds = WrappedADNI(base_ds, train_transform)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
print(f"[INFO] DataLoader ready. Total batches: {len(loader)}")

# ---------- 4. 初始化 ImageEncoder ----------
model = image_encoder18(in_channels=1, global_pool=False).to(device)
model.eval()
print("[INFO] ImageEncoder initialized (global_pool=True)")

# ---------- 5. 注册 hook，记录特征图形状 ----------
print("[INFO] Capturing feature map shapes from forward pass...")
shape_records = []
def get_hook(name):
    def hook(module, inp, out):
        shape_records.append({'layer': name, 'shape': list(out.shape)})
    return hook

hooks = []
for name, module in model.named_modules():
    if len(list(module.children())) == 0:
        hooks.append(module.register_forward_hook(get_hook(name)))

# ---------- 6. 执行一次推理，收集特征图形状 ----------
with torch.no_grad():
    sample, _ = next(iter(loader))
    _ = model(sample.to(device))
for h in hooks:
    h.remove()

shape_df = pd.DataFrame(shape_records)
shape_csv = 'feature_map_shapes.csv'
shape_df.to_csv(shape_csv, index=False)
print(f"[INFO] Feature-map shapes saved to: {shape_csv}")

# ---------- 7. 提取特征向量并保存 ----------
print("[INFO] Starting feature extraction for all batches...")
feature_rows = []
with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(loader):
        print(f"[Progress] Processing batch {batch_idx + 1}/{len(loader)}")
        feats = model(imgs.to(device))            # [B, 512]
        feats = feats.cpu().numpy()
        for f_vec, lbl in zip(feats, labels):
            feature_rows.append({'label': lbl, **{f'v{i}': f_vec[i] for i in range(512)}})

feat_df  = pd.DataFrame(feature_rows)
feat_csv = 'adni_features.csv'
feat_df.to_csv(feat_csv, index=False)
print(f"[INFO] Encoded features saved to: {feat_csv}")
print(f"[INFO] Saved feature shape: {feat_df.shape}")
