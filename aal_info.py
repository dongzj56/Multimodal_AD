import nibabel as nib
import numpy as np

# ---------- 1. 读取 AAL 模板 ----------
aal_path = rf"C:\Users\dongzj\Desktop\Multimodal_AD\adni_dataset\AAL_space-MNI152NLin6_res-2x2x2.nii\AAL_space-MNI152NLin6_res-2x2x2.nii"
img   = nib.load(aal_path)
data  = img.get_fdata()          # 浮点 ndarray
total = data.size

# ---------- 2. 定义三个掩膜 ----------
mask0        = np.isclose(data, 0.0)
mask_1_94    = np.logical_and(data >= 1,  data <= 94)    # 1‒94
mask_95_120  = np.logical_and(data >= 95, data <= 120)   # 95‒120

cnt0        = np.count_nonzero(mask0)
cnt_1_94    = np.count_nonzero(mask_1_94)
cnt_95_120  = np.count_nonzero(mask_95_120)

# ---------- 3. 输出 ----------
print(f"总  体  素 数 : {total}")
print(f"标签 0   体素 : {cnt0}   ({cnt0/total:.4%})")
print(f"标签 1-94 体素 : {cnt_1_94}   ({cnt_1_94/total:.4%})")
print(f"标签 95-120体素 : {cnt_95_120}   ({cnt_95_120/total:.4%})")
