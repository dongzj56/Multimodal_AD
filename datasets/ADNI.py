import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    EnsureTyped
)


# 定义数据类
class ADNI(Dataset):
    """
    用于处理ADNI数据集的类，包含读取MRI和PET数据、标签处理、图像预处理等功能。
    """

    def __init__(self, label_file, mri_dir, pet_dir, task='ADCN', augment=False):
        """
        初始化ADNI数据集类，读取数据和标签文件，并生成数据字典。

        :param label_file: 标签文件路径（包含 Group 和 Subject ID 等信息）
        :param mri_dir: MRI图像所在目录
        :param pet_dir: PET图像所在目录
        :param task: 任务类型，用于选择不同标签类别
        :param augment: 是否进行数据增强
        """
        self.label = pd.read_csv(label_file)  
        self.mri_dir = mri_dir
        self.pet_dir = pet_dir
        self.task = task
        self.augment = augment

        self._process_labels()
        self._build_data_dict()

    def _process_labels(self):
        """
        根据指定的任务从标签 CSV 文件中提取数据标签。
        """
        if self.task == 'ADCN':
            self.labels = self.label[(self.label['Group'] == 'AD') | (self.label['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'AD': 1}
        if self.task == 'CNEMCI':
            self.labels = self.label[(self.label['Group'] == 'CN') | (self.label['Group'] == 'EMCI')]
            self.label_dict = {'CN': 0, 'EMCI': 1}
        if self.task == 'LMCIAD':
            self.labels = self.label[(self.label['Group'] == 'LMCI') | (self.label['Group'] == 'AD')]
            self.label_dict = {'LMCI': 0, 'AD': 1}
        if self.task == 'EMCILMCI':
            self.labels = self.label[(self.label['Group'] == 'EMCI') | (self.label['Group'] == 'LMCI')]
            self.label_dict = {'EMCI': 0, 'LMCI': 1}
        # 可根据需要添加其他任务类别

    def _build_data_dict(self):
        """
        根据标签信息和文件路径构建数据字典，存储每个患者的MRI、PET图像路径和标签。
        """
        subject_list = self.labels['Subject ID'].tolist()  # 提取所有患者的ID
        label_list = self.labels['Group'].tolist()  # 提取所有患者的组别标签
        self.data_dict = []
        for subject, group in zip(subject_list, label_list):
            self.data_dict.append({
                'MRI': os.path.join(self.mri_dir, f'{subject}.nii'),
                'PET': os.path.join(self.pet_dir, f'{subject}.nii'),
                'label': self.label_dict[group],
                'Subject': subject
            })

    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.data_dict)

    def __getitem__(self, idx):
        """
        根据索引获取一个样本，返回MRI和PET图像及其标签。
        """
        sample = self.data_dict[idx]  # 获取样本信息
        label = sample['label']  # 提取标签
        result = {}  # 初始化结果字典

        # 加载MRI图像
        mri_path = sample['MRI']
        mri_img = LoadImaged(keys=['MRI'])({'MRI': mri_path})['MRI']
        result['MRI'] = mri_img

        # 加载PET图像
        pet_path = sample['PET']
        pet_img = LoadImaged(keys=['PET'])({'PET': pet_path})['PET']
        result['PET'] = pet_img

        return result['MRI'], result['PET'], label

    def print_dataset_info(self, start=0, end=None):
        """
        打印数据集结构和指定范围的样本信息，以表格形式展示。

        :param start: 起始样本索引，默认为0
        :param end: 结束样本索引，默认为None，表示打印到数据集的最后一个样本
        """
        print(f"Dataset Structure:\n{'=' * 40}")
        print(f"Total Samples: {len(self)}")
        print(f"Task: {self.task}")

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        if end is None:
            end = len(self)

        sample_data = []
        for sample in self.data_dict[start:end]:
            sample_data.append([
                sample['MRI'],
                sample['PET'],
                sample['label'],
                sample['Subject']
            ])

        df = pd.DataFrame(sample_data, columns=["MRI", "PET", "Label", "Subject"])
        print(df)
        print(f"\n{'=' * 40}")


# =============================================================================
# 用于数据预处理的转换函数，适用于MRI和PET数据
# =============================================================================
def ADNI_transform(augment=False):
    if augment:
        # 如果进行数据增强，进行一系列的图像预处理和增强操作
        train_transform = Compose([
            LoadImaged(keys=['MRI', 'PET']),  # 加载 MRI 和 PET 图像
            EnsureChannelFirstd(keys=['MRI', 'PET']),  # 确保通道维度为第一个
            ScaleIntensityd(keys=['MRI', 'PET']),  # 标准化图像强度
            # 图像增强操作
            RandFlipd(keys=['MRI', 'PET'], prob=0.3, spatial_axis=0),  # 随机翻转
            RandRotated(keys=['MRI', 'PET'], prob=0.3, range_x=0.05),  # 随机旋转
            RandZoomd(keys=['MRI', 'PET'], prob=0.3, min_zoom=0.95, max_zoom=1),  # 随机缩放
            EnsureTyped(keys=['MRI', 'PET'])  # 确保数据类型正确
        ])
    else:
        # 如果不进行数据增强，仅进行基本的图像预处理
        train_transform = Compose([
            LoadImaged(keys=['MRI', 'PET']),
            EnsureChannelFirstd(keys=['MRI', 'PET']),
            ScaleIntensityd(keys=['MRI', 'PET']),
            EnsureTyped(keys=['MRI', 'PET'])
        ])

    # 测试时使用相同的预处理（不进行增强）
    test_transform = Compose([
        LoadImaged(keys=['MRI', 'PET']),
        EnsureChannelFirstd(keys=['MRI', 'PET']),
        ScaleIntensityd(keys=['MRI', 'PET']),
        EnsureTyped(keys=['MRI', 'PET'])
    ])

    return train_transform, test_transform


def main():
    """
    主函数，测试ADNI数据集类的功能
    """
    dataroot = rf'C:\Users\dongz\Desktop\adni_dataset'
    label_filename = rf'C:\Users\dongz\Desktop\adni_dataset\ADNI.csv'
    mri_dir = os.path.join(dataroot, 'MRI')
    pet_dir = os.path.join(dataroot, 'PET')
    task = 'ADCN'
    augment = False

    # 创建ADNI数据集对象
    adni_dataset = ADNI(label_file=label_filename,
                        mri_dir=mri_dir,
                        pet_dir=pet_dir,
                        task=task,
                        augment=augment)

    print(f'Dataset size: {len(adni_dataset)}')

    # 拆分数据集为训练集和测试集（80%训练，20%测试）
    train_data, test_data = train_test_split(adni_dataset.data_dict, test_size=0.2, random_state=42)

    # 创建训练集和测试集对象
    train_dataset = ADNI(label_file=label_filename,
                         mri_dir=mri_dir,
                         pet_dir=pet_dir,
                         task=task,
                         augment=augment)
    train_dataset.data_dict = train_data  # 设置训练集数据

    test_dataset = ADNI(label_file=label_filename,
                        mri_dir=mri_dir,
                        pet_dir=pet_dir,
                        task=task,
                        augment=augment)
    test_dataset.data_dict = test_data  # 设置测试集数据

    print(f'Train Dataset size: {len(train_dataset)}')
    print(f'Test Dataset size: {len(test_dataset)}')

    sample_mri, sample_pet, sample_label = train_dataset[0]
    print(f'Sample MRI shape: {sample_mri.shape}, Sample PET shape: {sample_pet.shape}, Label: {sample_label}')

    train_dataset.print_dataset_info(start=0, end=5)
    test_dataset.print_dataset_info(start=0, end=5)

    # 获取数据增强操作
    train_transform, test_transform = ADNI_transform(augment=False)

    # 输出 train_transform 中的每个转换操作的信息
    print("Train Transform Operations:")
    for i, transform in enumerate(train_transform.transforms):
        print(f"Operation {i + 1}: {transform.__class__.__name__}")
        print(f"Parameters: {transform}")
        print("-" * 50)

    print("\nTest Transform Operations:")
    for i, transform in enumerate(test_transform.transforms):
        print(f"Operation {i + 1}: {transform.__class__.__name__}")
        print(f"Parameters: {transform}")
        print("-" * 50)


if __name__ == '__main__':
    main()






