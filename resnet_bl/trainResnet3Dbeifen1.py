import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import time
import logging
from datasets.ADNI import ADNI, ADNI_transform
from models.Resnet3D import ResNet3D
from tqdm import tqdm
from monai.data import Dataset

# 配置日志
logging.basicConfig(format='[%(asctime)s]  %(message)s',
                    datefmt='%d.%m %H:%M:%S',
                    level=logging.INFO)


def print_dataset_info(dataset, name):
    """打印数据集统计信息"""
    # 从 dataset 中提取每个字典的 'label' 值，并确保是整数
    labels = [int(item['label']) for item in dataset]

    # 统计每个标签出现的次数
    class_counts = np.bincount(labels)

    # 记录统计信息，打印出 Class 0 和 Class 1 的数量
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
    ).data_dict

    print(type(dataset))
    print(len(dataset))
    print(dataset)

    # 首先划分出 60% 的训练集
    train_data, val_data= train_test_split(dataset, test_size=0.2, random_state=42)

    print(type(train_data))
    print(len(train_data))
    print(train_data)

    # 根据索引从ADNI_data中取出对应的数据
    # train_data = [dataset[i] for i in train_idx]
    # val_data = [dataset[i] for i in val_idx]

    # print(type(train_data))

    train_transforms, val_transforms = ADNI_transform()
    # 输出 train_transform 中的每个转换操作的信息
    print("Train Transform Operations:")
    for i, transform in enumerate(train_transforms.transforms):
        print(f"Operation {i + 1}: {transform.__class__.__name__}")
        print(f"Parameters: {transform}")
        print("-" * 50)

    print("\nTest Transform Operations:")
    for i, transform in enumerate(val_transforms.transforms):
        print(f"Operation {i + 1}: {transform.__class__.__name__}")
        print(f"Parameters: {transform}")
        print("-" * 50)

    # 重新定义数据集
    train_dataset = Dataset(data=train_data, transform=train_transforms)
    val_dataset = Dataset(data=val_data, transform=val_transforms)

    # 打印数据集信息
    logging.info("\n===== Dataset Information =====")
    print_dataset_info(train_dataset, "Train")
    print_dataset_info(val_dataset, "Validation")
    logging.info("==============================\n")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('train_loader...:')
    print(type(train_loader))
    print(len(train_loader))
    print(train_loader)

    # for data in train_loader:
    #     print(data)  # 打印返回的数据
    #     break  # 只查看第一个批次

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        # 训练阶段（使用MRI+PET双模态）
        for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100):
            mri = data['MRI']
            pet = data['PET']
            labels = data['label']

            # 拼接MRI和PET为双通道输入 [batch, 2, depth, H, W]
            inputs = torch.cat([mri.unsqueeze(1), pet.unsqueeze(1)], dim=1).float().to(device)
            inputs = inputs.squeeze(2)
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
            for data in tqdm(val_loader, desc="Validation", ncols=100):
                mri = data['MRI']
                pet = data['PET']
                labels = data['label']

                inputs = torch.cat([mri.unsqueeze(1), pet.unsqueeze(1)], dim=1).float().to(device)
                inputs = inputs.squeeze(2)
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