import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import time
import logging
from datasets.ADNI import ADNI, ADNI_transform
from models.Resnet3D import ResNet3D
from tqdm import tqdm

# 配置日志
logging.basicConfig(format='[%(asctime)s]  %(message)s',
                    datefmt='%d.%m %H:%M:%S',
                    level=logging.INFO)

def print_dataset_info(dataset, name):
    """打印数据集统计信息"""
    labels = [label for _, _, label in dataset]
    class_counts = np.bincount(labels)
    logging.info(
        f"{name} | Total: {len(dataset)} | Class 0: {class_counts[0]} | Class 1: {class_counts[1]}"
    )

def train_model(task='ADCN', num_epochs=50, batch_size=4, lr=1e-3):
    # 初始化模型（双通道输入）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet3D(in_channels=2, num_classes=2).to(device)  # 修改输入通道为2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 加载数据
    dataroot = rf'C:\Users\dongz\Desktop\adni_dataset'
    dataset = ADNI(
        label_file=f'{dataroot}/ADNI.csv',
        mri_dir=f'{dataroot}/MRI',
        pet_dir=f'{dataroot}/PET',
        task=task,
        augment=True
    )

    # 按6:2:2划分数据集
    total = len(dataset)
    train_size = int(0.6 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子保证可重复性
    )

    # 打印数据集信息
    logging.info("\n===== Dataset Information =====")
    print_dataset_info(train_dataset, "Train")
    print_dataset_info(val_dataset, "Validation")
    print_dataset_info(test_dataset, "Test")
    logging.info("==============================\n")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        # 训练阶段（使用MRI+PET双模态）
        for mri, pet, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100):
            # 拼接MRI和PET为双通道输入 [batch, 2, depth, H, W]
            inputs = torch.cat([mri.unsqueeze(1), pet.unsqueeze(1)], dim=1).float().to(device)
            targets = labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for mri, pet, labels in tqdm(val_loader, desc="Validation", ncols=100):
                inputs = torch.cat([mri.unsqueeze(1), pet.unsqueeze(1)], dim=1).float().to(device)
                targets = labels.long().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        # 计算指标
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))

        # 打印日志
        epoch_time = time.time() - start_time
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"AUC: {val_auc:.4f} | Acc: {val_acc:.4f}"
        )

    # 最终测试（可选）
    # 可在此处添加测试集评估代码，方法同验证阶段

if __name__ == "__main__":
    train_model(task='ADCN', num_epochs=50, batch_size=4, lr=1e-3)