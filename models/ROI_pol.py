import torch, nibabel as nib, numpy as np
from nilearn import plotting
from pathlib import Path
# 改成 datasets 下的接口
from nilearn.datasets import load_mni152_template



# ---------------------------------------------
# 读取 LUT（txt → {index: name}）
# ---------------------------------------------
def load_aal3_lut(txt_path: str, keep_fullname=False):
    lut = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln[0] == '#':
                continue
            parts = ln.split()
            idx   = int(parts[0])
            name  = parts[-1] if keep_fullname else parts[1]
            lut[idx] = name
    return lut

# ---------------------------------------------
# 主程序
# ---------------------------------------------
if __name__ == '__main__':

    # === 路径自行替换 ===
    atlas_nii   = rf'C:\Users\dongzj\Desktop\adni_dataset\AAL3v1.nii'
    lut_txt     = rf'C:\Users\dongzj\Desktop\adni_dataset\AAL3v1.nii.txt'
    # bg_template = plotting.load_mni152_template(resolution=2)  # 2 mm 背景 T1
    bg_template = load_mni152_template(resolution=2)   # 2 mm 背景 T1

    # ---------- 1. 载入 atlas & LUT ----------
    img   = nib.load(atlas_nii)
    data  = torch.from_numpy(img.get_fdata()).long()
    lut   = load_aal3_lut(lut_txt, keep_fullname=False)
    print(f'Atlas shape: {data.shape}, 体素大小 ~{np.abs(np.diag(img.affine)[:3])} mm')

    # ---------- 2. 交互式可视化 ----------
    view = plotting.view_img(
        atlas_nii,
        bg_img=bg_template,
        cmap='tab20',
        threshold=0,    # 显示全部标签
        opacity=0.55,
        symmetric_cmap=False,
        title='AAL3  (1.5 mm, 112×136×112)'
    )
    # Jupyter 环境直接 display(view)
    view.open_in_browser()            # 脚本环境自动打开默认浏览器

    # ---------- 3. 实用函数：查询体素 / 世界坐标 ----------
    def query_voxel(i, j, k):
        """给体素索引(i,j,k) → 打印标签号+名称"""
        if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]):
            print('索引越界')
            return
        val = int(data[i, j, k])
        print(f'Voxel ({i},{j},{k}) -> label {val}: {lut.get(val, "背景/未知")}')
        return val

    def query_world(x, y, z, img, data, lut):
        import numpy as np
        ijk = np.round(np.linalg.inv(img.affine) @ [x, y, z, 1])[:3].astype(int)
        if (ijk < 0).any() or (ijk >= data.shape).any():
            print('坐标超出模板范围')
            return None
        val = int(data[tuple(ijk)])
        print(f'World ({x:.1f},{y:.1f},{z:.1f}) mm -> voxel {tuple(ijk)} '
            f'→ label {val}: {lut.get(val, "背景/未知")}')
        return val


    # ---------- 4. Demo ----------
    query_voxel(1, 81, 50)
    query_voxel(2, 81, 50)
    query_voxel(3, 81, 50)
    query_voxel(4, 81, 50)
    query_voxel(5, 81, 50)
    query_voxel(6, 81, 50)
    query_voxel(7, 81, 50)
    query_voxel(8, 81, 50)
    query_voxel(9, 81, 50)
    query_voxel(10, 81, 50)
    query_world(-34, -20, -18,img,data,lut)        # 例：左海马常见坐标
