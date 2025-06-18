# extract_unet_feats.py  -----------------------------------------
import os, json, csv, time, torch, pandas as pd
from torch.utils.data import DataLoader
from torchsummary import summary
from datasets.ADNI import ADNI, ADNI_transform
from models.unet3d import UNet3D
from sklearn.model_selection import train_test_split
from monai.data import Dataset          # MONAI 的通用 Dataset
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
full_list = ADNI(label_file=LABEL,
                 mri_dir=MRI_DIR,
                 task=TASK,
                 augment=False).data_dict

train_data, test_data = train_test_split(
    full_list,
    test_size=0.2,
    random_state=42,
    stratify=[d["label"] for d in full_list]
)

_, tf_te = ADNI_transform(augment=False)
ds_test = Dataset(data=test_data, transform=tf_te)

loader = DataLoader(
    ds_test,
    batch_size=BATCH,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ---------- 3. UNet3D (完整解码器) ----------
model = UNet3D(in_channels=1, num_classes=1).to(device)
model.eval()

summary(model, input_size=(1, cfg.get("input_D",96),
                               cfg.get("input_H",112),
                               cfg.get("input_W",96)),
        device=device.type)

# ---------- 4. Hook：抓取 s_block1.conv2 输出 ----------
feat_bank = {}
def grab_features(_, __, output):
    feat_bank["x"] = output.detach().cpu()   # (B,64,D,H,W)

model.s_block1.conv2.register_forward_hook(grab_features)

# ---------- 5. 推理 & 保存 CSV ----------
os.makedirs("/data/coding/Multimodal_AD/output", exist_ok=True)
csv_path = "/data/coding/Multimodal_AD/output/features.csv"
header_written = False
with open(csv_path, "w", newline='') as fcsv:
    writer = csv.writer(fcsv)

    for step, batch in enumerate(loader, 1):              # ← 修改处
        vol        = batch["MRI"].to(device)              # (B,1,D,H,W)
        subj_list  = batch["Subject"]                     # list[str]

        _ = model(vol)                                    # forward → hook
        feats = feat_bank["x"]                            # (B,64,D,H,W)

        B,C,D,H,W = feats.shape
        feats_flat = feats.view(B, -1).numpy()            # 展平

        if not header_written:
            writer.writerow(["Subject_ID"] +
                            [f"f{i}" for i in range(feats_flat.shape[1])])
            header_written = True

        for sid, vec in zip(subj_list, feats_flat):
            writer.writerow([sid] + vec.tolist())

        print(f"[{step:03d}/{len(loader)}]  "
              f"in={tuple(vol.shape)}  feat={tuple(feats.shape)}  "
              f"csv_row_len={feats_flat.shape[1]}")

print(f"\n[Done] Feature CSV saved to {csv_path}")
# ---------------------------------------------------------------
