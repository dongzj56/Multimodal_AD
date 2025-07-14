import os
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def inspect_nii_images(
    input_dir: str,
    output_dir: str = None,
    batch_size: int = 4,
    slice_axis: int = 2,
    dpi: int = 100
):
    """
    批量检查 NIfTI 影像，将每 batch_size 个文件在同一画布展示并标注元信息。

    参数：
    - input_dir:    输入目录，搜索 .nii 和 .nii.gz
    - output_dir:   如果不为 None，则保存每画布为 PNG；否则 plt.show()
    - batch_size:   每张画布显示多少张图（默认 4）
    - slice_axis:   沿哪个轴截取中间层（默认 z 轴=2）
    - dpi:          保存 PNG 的分辨率
    """
    input_path = Path(input_dir)
    files = sorted(input_path.glob('*.nii')) + sorted(input_path.glob('*.nii.gz'))
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 下找不到 .nii 或 .nii.gz 文件")

    # 如果需要保存，创建目录
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    for idx in range(0, len(files), batch_size):
        batch = files[idx:idx + batch_size]
        rows = cols = int(batch_size**0.5)
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten()

        for ax, fpath in zip(axes, batch):
            # 加载图像
            img = nib.load(str(fpath))
            data = img.get_fdata(caching='unchanged', dtype=np.float32)

            # 取中间切片
            mid = data.shape[slice_axis] // 2
            if slice_axis == 0:
                slc = data[mid, :, :]
            elif slice_axis == 1:
                slc = data[:, mid, :]
            else:
                slc = data[:, :, mid]

            # 显示
            ax.imshow(slc.T, cmap='gray', origin='lower')

            # 文件大小、体素尺寸、数据类型
            size_mb = os.path.getsize(fpath) / (1024**2)
            voxel_size = img.header.get_zooms()[:3]
            dtype = img.header.get_data_dtype()
            title = (
                f"{fpath.name}\n"
                f"Size: {size_mb:.2f} MB\n"
                f"Voxel: {voxel_size} mm³\n"
                f"Dtype: {dtype}"
            )
            ax.set_title(title, fontsize=10)
            ax.axis('off')

        # 隐藏多余子图
        for ax in axes[len(batch):]:
            ax.axis('off')

        plt.tight_layout()

        if output_dir:
            out_file = out_path / f"inspection_{idx//batch_size+1:03d}.png"
            plt.savefig(out_file, dpi=dpi)
            plt.close(fig)
            print(f"[Saved] {out_file}")
        else:
            plt.show()


if __name__ == "__main__":
    inspect_nii_images(
        input_dir=rf"C:\Users\dongz\Desktop\test_dataset\PET",
        output_dir=None,  # 或者设为 None 直接弹窗显示
        batch_size=4,
        slice_axis=2,
        dpi=150
    )
