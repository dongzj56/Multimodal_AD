#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Overlay AAL ROI (41-42, bilateral hippocampus) on an MNI-aligned MRI,
then save as PNG & interactive HTML.
"""

import os, numpy as np, nibabel as nib
from nilearn import image, plotting

# -------------------------------------------------
# 1. 路径（按需修改）
# -------------------------------------------------
mri_path = "/data/coding/test_dataset/MRI/003_S_4524.nii"
aal_path = "/data/coding/test_dataset/AAL_space-MNI152NLin6_res-2x2x2.nii/AAL_space-MNI152NLin6_res-2x2x2.nii"

output_dir = "/data/coding/Multimodal_AD/output"
os.makedirs(output_dir, exist_ok=True)
png_file  = os.path.join(output_dir, "mri_hippocampus_overlay.png")
html_file = os.path.join(output_dir, "mri_hippocampus_overlay.html")

# -------------------------------------------------
# 2. 读入 MRI & AAL atlas
# -------------------------------------------------
mri_img = nib.load(mri_path)
aal_img = nib.load(aal_path)
aal_data = aal_img.get_fdata()

# -------------------------------------------------
# 3. 生成海马 ROI 掩膜（41 L-Hippocampus, 42 R-Hippocampus）
# -------------------------------------------------
roi_ids   = [41, 42]
mask_data = np.isin(aal_data, roi_ids).astype(np.uint8)
mask_img  = nib.Nifti1Image(mask_data, affine=aal_img.affine)

# 若 atlas 与 MRI 分辨率不同 → 把掩膜重采样到 MRI 网格
if mri_img.header.get_zooms() != aal_img.header.get_zooms():
    mask_img = image.resample_to_img(mask_img, mri_img, interpolation="nearest")

# -------------------------------------------------
# 4-A. 静态 PNG 叠加
# -------------------------------------------------
display = plotting.plot_roi(
    roi_img=mask_img,
    bg_img=mri_img,
    cmap="autumn",
    alpha=0.7,
    title="Bilateral Hippocampus (red overlay)"
)
display.savefig(png_file, dpi=300)
display.close()
print(f"静态 PNG 已保存: {png_file}")

# -------------------------------------------------
# 4-B. 交互式 HTML 叠加
# -------------------------------------------------
view = plotting.view_img(
    mask_img,
    bg_img=mri_img,
    cmap="autumn",
    opacity=0.7,
    symmetric_cmap=False,
    title="Bilateral Hippocampus (interactive view)"
)
view.save_as_html(html_file)
print(f"交互式 HTML 已保存: {html_file}")
