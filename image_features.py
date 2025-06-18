# extract_unet_feats.py  -----------------------------------------
import os, json, csv, torch, pandas as pd
from torch.utils.data import DataLoader
from torchsummary import summary
from datasets.ADNI import ADNI, ADNI_transform
from models.unet3d import UNet3D
from sklearn.model_selection import train_test_split
from monai.data import Dataset
import torch.nn.functional as F            # 🔸
import nibabel as nib                      # 🔸
import numpy as np                         # 🔸

# ---------- 1. 读取 config ----------
with open("/data/coding/Multimodal_AD/config/config.json") as f:
    cfg = json.load(f)

BATCH = cfg.get("batch_size", 4)
TASK  = cfg.get("task",       "ADCN")
LABEL = cfg["label_file"]
MRI_DIR = cfg["mri_dir"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device = {device}\n")

# ---------- 2. DataLoader ----------
full_list = ADNI(label_file=LABEL, mri_dir=MRI_DIR,
                 task=TASK, augment=False).data_dict

train_data, test_data = train_test_split(
    full_list, test_size=0.2, random_state=42,
    stratify=[d["label"] for d in full_list])

_, tf_te = ADNI_transform(augment=False)
ds_test = Dataset(data=test_data, transform=tf_te)

loader = DataLoader(ds_test, batch_size=BATCH,
                    shuffle=False, num_workers=4, pin_memory=True)

# ---------- 3. UNet3D (完整解码器) ----------
model = UNet3D(in_channels=1, num_classes=1).to(device)
model.eval()

summary(model,
        input_size=(1, cfg.get("input_D", 96),
                       cfg.get("input_H", 112),
                       cfg.get("input_W", 96)),
        device=device.type)


# ---------- 4-A. Hook：模型最终输出（1 通道，保持原功能） ----------
feat_bank = {}
def grab_out(_, __, output):
    feat_bank["out"] = output.detach().cpu()     # (B,1,D,H,W)
model.register_forward_hook(grab_out)

# ---------- 4-B. Hook：抓 64-通道特征图（ROI 池化用） 🔸 ----------
feat64_bank = {}
def grab_64(_, __, output):
    feat64_bank["x"] = output.detach().cpu()     # (B,64,D',H',W')
model.s_block1.conv2.register_forward_hook(grab_64)

# ---------- 4-C. 预载 AAL 模板 & JSON LUT 🔸 ----------
AAL_NII  = "/data/coding/test_dataset/AAL_space-MNI152NLin6_res-2x2x2.nii/AAL_space-MNI152NLin6_res-2x2x2.nii"
AAL_JSON = "/data/coding/test_dataset/AAL_space-MNI152NLin6_res-2x2x2.nii/AAL_space-MNI152NLin6_res-2x2x2.json"

aal_img   = nib.load(AAL_NII)
aal_data  = aal_img.get_fdata().astype(int)      # (91,109,91)
roi_ids   = np.unique(aal_data); roi_ids = roi_ids[roi_ids > 0]
R         = len(roi_ids)

# 读 LUT→名称
try:
    lut = {int(k):v["label"] for k,v in
           json.load(open(AAL_JSON))["rois"].items()}
except Exception:
    lut = {i: f"ROI{i}" for i in roi_ids}
roi_names = [lut.get(i, f"ROI{i}") for i in roi_ids]

# 预计算 one-hot 掩膜 (R,D,H,W)  → 放 CPU，与 feat64 同设备
onehot = F.one_hot(torch.from_numpy(aal_data).long(),
                   num_classes=roi_ids.max()+1)[...,1:]
onehot = onehot.permute(3,0,1,2).float()         # (R,D,H,W)

# ---------- 5-A. 原 voxel CSV ----------
os.makedirs("/data/coding/Multimodal_AD/output", exist_ok=True)
csv_path = "/data/coding/Multimodal_AD/output/features.csv"
header_written = False
writer_f = open(csv_path, "w", newline='')
writer   = csv.writer(writer_f)

# ---------- 5-B. ROI 池化 CSV 🔸 ----------
roi_csv = "/data/coding/Multimodal_AD/output/roi_features.csv"
roi_header_done = False
roi_f  = open(roi_csv, "w", newline='')
roi_w  = csv.writer(roi_f)

for step, batch in enumerate(loader, 1):
    vol        = batch["MRI"].to(device)
    subj_list  = batch["Subject"]

    _ = model(vol)                          # forward → 两个 hook 触发
    feats1 = feat_bank["out"]               # (B,1,D,H,W)  keep as is
    feats64= feat64_bank["x"]               # (B,64,96,112,96)

    # ---------- 裁剪 64-ch 到原尺寸 🔸 ----------
    _,_,D,H,W = feats1.shape
    feats64 = feats64[..., :D, :H, :W]      # remove right/bottom pad
    B,C,_,_,_ = feats64.shape

    # ---------- ROI 平均池化 (B,R,C) 🔸 ----------
    num = (feats64[:,None,:,:,:,:] *
           onehot[None,:,None,:,:,:]).sum((-1,-2,-3))
    den =  onehot[None,:,None,:,:,:].sum((-1,-2,-3)).clamp_min(1e-6)
    roi_feat = (num / den)                  # (B,R,C)

    # ---------- shape log ----------
    print(f"step {step:03d} | input torch.Size{tuple(vol.shape)} "
          f"→ feat64 torch.Size{tuple(feats64.shape)} "
          f"→ roi torch.Size{tuple(roi_feat.shape)}")

    # ---------- 保存 voxel-level 展平 (原功能) ----------
    feats_flat = feats1.view(B, -1).numpy()
    if not header_written:
        writer.writerow(["Subject_ID"] +
                        [f"f{i}" for i in range(feats_flat.shape[1])])
        header_written = True
    for sid, vec in zip(subj_list, feats_flat):
        writer.writerow([sid] + vec.tolist())

    # ---------- 保存 ROI 池化结果 🔸 ----------
    if not roi_header_done:
        roi_w.writerow(["Subject_ID"] +
                       [f"{name}_c{c}"
                        for name in roi_names
                        for c in range(C)])
        roi_header_done = True
    for sid, rvec in zip(subj_list,
                         roi_feat.permute(0,2,1).reshape(B,-1).numpy()):
        roi_w.writerow([sid] + rvec.tolist())

    print(f"[{step:03d}/{len(loader)}]  voxel_feat={feats1.shape}  "
          f"roi_feat={roi_feat.shape}")

writer_f.close(); roi_f.close()
print("\n✓ voxel CSV :", csv_path)
print("✓ ROI   CSV :", roi_csv)
# ---------------------------------------------------------------
