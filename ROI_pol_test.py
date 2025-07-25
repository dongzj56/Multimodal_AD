#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atlas 查询工具
--------------
1. 读取标签图 (AAL / Hammers 等) NIfTI + JSON LUT
2. 交互式可视化
3. 支持：
   • voxel → ROI 名称
   • world → 最近 ROI 质心 + 真实落点标签
"""

import numpy as np
import nibabel as nib
import torch
from nilearn import plotting
from nilearn.datasets import load_mni152_template
import json
from pathlib import Path

# -------------------------------------------------
# 1) 读取 JSON LUT（NeuroParc / BIDS 风格）
# -------------------------------------------------
def load_aal_json_lut(json_path: str, return_center=False, return_size=False):
    """
    解析 JSON
    ----------
    Parameters
    ----------
    json_path : str
        JSON 文件路径
    return_center , return_size : bool
        是否一并返回质心坐标 / ROI 体素数

    Returns
    -------
    lut : dict[int, str]
        标签号 → 名称
    centers : dict[int, tuple]  (可选)
    sizes   : dict[int, int]    (可选)
    """
    p = Path(json_path)
    with open(p, "r", encoding="utf-8") as f:
        js = json.load(f)

    lut, centers, sizes = {}, {}, {}
    for k, v in js["rois"].items():
        idx = int(k)
        if idx == 0 or v["label"] in (None, "null"):
            continue                  # 跳过背景或空 ROI
        lut[idx]      = v["label"]
        centers[idx]  = tuple(v["center"]) if v["center"] else None
        sizes[idx]    = v["size"]

    if return_center or return_size:
        return lut, centers, sizes
    return lut


# -------------------------------------------------
# 2) 最近 ROI 质心映射
# -------------------------------------------------
def nearest_roi(world_xyz, centers: dict):
    """
    给定 MNI 坐标 → 找到最近 ROI 质心
    过滤 center is None 的 ROI，避免 TypeError
    """
    w = np.asarray(world_xyz)
    # 仅对有效质心计算欧氏距离
    valid = ((k, np.asarray(c)) for k, c in centers.items() if c is not None)
    lab, dist = min(((k, np.linalg.norm(w - c)) for k, c in valid),
                    key=lambda t: t[1])
    return lab, dist


# -------------------------------------------------
# 3) 主程序
# -------------------------------------------------
if __name__ == "__main__":
    # ======= 修改为你本地的实际路径 =======
    atlas_nii = "/data/coding/test_dataset/AAL_space-MNI152NLin6_res-2x2x2.nii/AAL_space-MNI152NLin6_res-2x2x2.nii"
    lut_file  = "/data/coding/test_dataset/AAL_space-MNI152NLin6_res-2x2x2.nii/AAL_space-MNI152NLin6_res-2x2x2.json"
    # =====================================

    bg_template = load_mni152_template(resolution=2)

    # ---------- 加载 Atlas ----------
    img  = nib.load(atlas_nii)
    data = img.get_fdata().astype(int)
    torch_data = torch.from_numpy(data).long()   # 如需 PyTorch 处理可用

    lut, centers, sizes = load_aal_json_lut(lut_file,
                                            return_center=True,
                                            return_size=True)

    labels = np.unique(data)
    print(f"标签总数: {labels.size} (含 0 背景)")
    print(f"最大标签号: {labels.max()}")
    print(f"Atlas 体素网格: {img.shape}, 体素大小: {np.abs(np.diag(img.affine)[:3])} mm")

    # ---------- 交互式查看 ----------
    view = plotting.view_img(
        atlas_nii,
        bg_img=bg_template,
        cmap="tab20",
        threshold=0,
        opacity=0.55,
        title="AAL Atlas (2 mm)"
    )
    try:
        view.open_in_browser()
    except RuntimeError:
        out_html = "/data/coding/Multimodal_AD/output/atlas_view.html"
        view.save_as_html(out_html)
        print(f"无 GUI 环境，交互式 HTML 已保存到: {out_html}")

    # -------------------------------------------------
    # 4) 查询函数
    # -------------------------------------------------
    def query_voxel(i, j, k):
        """体素索引 → ROI 名称"""
        if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]):
            print("索引越界")
            return
        val = int(data[i, j, k])
        print(f"[Voxel] ({i},{j},{k}) → label {val}: {lut.get(val, '背景/未知')}")
        return val

    def query_world(x, y, z):
        """
        世界坐标查询
        1. 输出该点所属标签
        2. 寻找最近 ROI 质心并展示距离、名称等信息
        """
        world = (x, y, z)

        # 1) 真实标签
        ijk = np.round(np.linalg.inv(img.affine) @ [*world, 1])[:3].astype(int)
        true_lab = None
        if (ijk >= 0).all() and (ijk < data.shape).all():
            true_lab = int(data[tuple(ijk)])

        # 2) 最近质心
        lab_cen, dist = nearest_roi(world, centers)
        cen_xyz = centers[lab_cen]
        size    = sizes[lab_cen]
        name    = lut[lab_cen]

        # ---------- 输出 ----------
        print("\n=== World 查询结果 ===")
        print(f"输入坐标            : ({x:.1f}, {y:.1f}, {z:.1f}) mm")
        print(f"落点体素索引        : {tuple(ijk)}")
        print(f"该体素标签          : {true_lab} ({lut.get(true_lab, '背景/未知')})")
        print("—— 最近 ROI 质心 ——")
        print(f"ROI 编号            : {lab_cen}")
        print(f"ROI 名称            : {name}")
        print(f"ROI 质心 (mm)       : ({cen_xyz[0]:.1f}, {cen_xyz[1]:.1f}, {cen_xyz[2]:.1f})")
        print(f"距质心距离          : {dist:.2f} mm")
        print(f"ROI 体素数          : {size}")
        print("========================\n")
        return lab_cen, dist

    # -------------------------------------------------
    # 5) Demo
    # -------------------------------------------------
    print("\n######### DEMO #########")
    query_voxel(75, 97, 50)
    query_world(-34, -20, -18)   # 旧版左海马参考坐标
    query_world(-27, -18, -24)   # AAL 左海马质心
    query_world(11, -80, 24)     # 枕叶示例
