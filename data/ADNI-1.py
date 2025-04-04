'''第一版数据集加载代码'''
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
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

class ADNIDataset(Dataset):
    """
    用于处理ADNI数据集的类，包含读取数据、标签处理、图像预处理等功能。
    支持三个模态数据：MRI、PET图像和表格数据（如 ADNIMERGE.xlsx 中的生物样本监测、临床检查数据）。
    """

    def __init__(self, csv_file, mri_dir, pet_dir, table_file, task='ADCN', augment=False):
        """
        初始化ADNI数据集类，读取CSV文件、Excel表格数据并生成数据字典。

        :param csv_file: 标签文件路径（包含 Group 和 Subject ID 信息）
        :param mri_dir: MRI图像所在目录
        :param pet_dir: PET图像所在目录
        :param table_file: 表格数据文件路径（例如 "ADNIMERGE.xlsx"）
        :param task: 任务类型，用于选择不同标签类别
        :param augment: 是否启用数据增强
        """
        self.csv = pd.read_csv(csv_file)         # 读取标签 CSV 文件
        self.mri_dir = mri_dir                     # MRI 数据目录
        self.pet_dir = pet_dir                     # PET 数据目录
        self.table_df = pd.read_excel(table_file)  # 读取包含所有患者表格数据的 Excel 文件
        self.task = task                           # 任务类型
        self.augment = augment                     # 是否进行数据增强

        # 根据任务设置标签字典和筛选标签
        self._process_labels()
        # 构建数据字典：每个患者的 MRI、PET 路径、表格数据及标签
        self._build_data_dict()

    def _process_labels(self):
        """
        根据指定的任务从标签 CSV 文件中提取数据标签。
        """
        if self.task == 'ADCN':
            self.labels = self.csv[(self.csv['Group'] == 'AD') | (self.csv['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'AD': 1}
        elif self.task == 'pMCIsMCI':
            self.labels = self.csv[(self.csv['Group'] == 'pMCI') | (self.csv['Group'] == 'sMCI')]
            self.label_dict = {'sMCI': 0, 'pMCI': 1}
        elif self.task == 'EMCILMCI':
            self.labels = self.csv[(self.csv['Group'] == 'EMCI') | (self.csv['Group'] == 'LMCI')]
            self.label_dict = {'EMCI': 0, 'LMCI': 1}
        elif self.task == 'MCICN':
            self.labels = self.csv[(self.csv['Group'] == 'pMCI') | (self.csv['Group'] == 'sMCI') |
                                   (self.csv['Group'] == 'MCI') | (self.csv['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'sMCI': 1, 'pMCI': 1, 'MCI': 1}

    def _build_data_dict(self):
        """
        根据标签信息和文件路径构建数据字典，存储每个患者的 MRI、PET 图像路径、表格数据及标签。
        """
        subject_list = self.labels['Subject ID'].tolist()  # 提取所有患者的 ID
        label_list = self.labels['Group'].tolist()          # 提取所有患者的组别标签
        self.data_dict = []
        for subject, group in zip(subject_list, label_list):
            # 在表格数据中查找对应的行（假设Excel中有 "Subject ID" 列）
            rows = self.table_df[self.table_df['Subject ID'] == subject]
            if not rows.empty:
                table_data = rows.iloc[0].to_dict()  # 将对应行转换为字典
            else:
                table_data = None
            self.data_dict.append({
                'MRI': os.path.join(self.mri_dir, f'{subject}.nii'),
                'PET': os.path.join(self.pet_dir, f'{subject}.nii'),
                'TABLE': table_data,  # 表格数据以字典形式存储
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
        根据索引获取一个样本，包括 MRI、PET 图像、表格数据和标签。
        """
        sample = self.data_dict[idx]
        mri_path = sample['MRI']
        pet_path = sample['PET']
        table_data = sample['TABLE']  # 已经在构建数据字典时加载，直接使用字典数据
        label = sample['label']

        # 加载 MRI 和 PET 图像
        mri_img = LoadImaged(keys=['MRI'])({'MRI': mri_path})['MRI']
        pet_img = LoadImaged(keys=['PET'])({'PET': pet_path})['PET']

        # 对图像模态进行预处理（表格数据通常无需数据增强）
        if self.augment:
            transform = self.get_augmentation_transform()
        else:
            transform = self.get_basic_transform()
        transformed = transform({'MRI': mri_img, 'PET': pet_img})

        return transformed['MRI'], transformed['PET'], table_data, label

    def get_augmentation_transform(self):
        """
        返回包括数据增强操作的转换流程（仅针对图像模态）。
        """
        return Compose([
            EnsureChannelFirstd(keys=['MRI', 'PET']),
            ScaleIntensityd(keys=['MRI', 'PET']),
            RandFlipd(keys=['MRI', 'PET'], prob=0.3, spatial_axis=0),
            RandRotated(keys=['MRI', 'PET'], prob=0.3, range_x=0.05),
            RandZoomd(keys=['MRI', 'PET'], prob=0.3, min_zoom=0.95, max_zoom=1),
            EnsureTyped(keys=['MRI', 'PET'])
        ])

    def get_basic_transform(self):
        """
        返回基本的预处理转换流程（仅针对图像模态）。
        """
        return Compose([
            EnsureChannelFirstd(keys=['MRI', 'PET']),
            ScaleIntensityd(keys=['MRI', 'PET']),
            EnsureTyped(keys=['MRI', 'PET'])
        ])

    def print_dataset_info(self, start=0, end=None):
        """
        打印数据集结构和指定范围的样本信息，以表格形式展示。

        :param start: 起始样本索引，默认为 0
        :param end: 结束样本索引，默认为 None，表示打印到数据集最后一个样本
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
            # 对表格数据，只计算除 'Subject ID' 外的变量数量
            table_dict = sample['TABLE']
            if table_dict is not None:
                count_vars = len(table_dict) - (1 if 'Subject ID' in table_dict else 0)
            else:
                count_vars = None

            sample_data.append([
                sample['MRI'],
                sample['PET'],
                count_vars,      # 输出表格数据中变量的数量
                sample['label'],
                sample['Subject']
            ])

        df = pd.DataFrame(sample_data, columns=["MRI", "PET", "Table Vars Count", "Label", "Subject"])
        print(df)
        print(f"\n{'=' * 40}")


def main():
    """
    主函数，测试ADNI数据集类的功能
    """
    dataroot = rf'C:\Users\dongz\Desktop\adni_dataset'
    label_filename = rf'C:\Users\dongz\Desktop\adni_dataset\ADNI.csv'
    mri_dir = os.path.join(dataroot, 'MRI')
    pet_dir = os.path.join(dataroot, 'PET')
    table_file = os.path.join(dataroot, rf'TABLE\ADNIMERGE.xlsx')  # 表格数据存放在Excel文件中
    task = 'ADCN'

    adni_dataset = ADNIDataset(csv_file=label_filename,
                               mri_dir=mri_dir,
                               pet_dir=pet_dir,
                               table_file=table_file,
                               task=task,
                               augment=True)

    print(f'Dataset size: {len(adni_dataset)}')
    sample_mri, sample_pet, sample_table, sample_label = adni_dataset[0]
    print(f'Sample MRI shape: {sample_mri.shape}, PET shape: {sample_pet.shape}, Label: {sample_label}')
    print("Sample Table Data:")
    print(sample_table)  # 输出对应患者的表格数据（字典形式）

    adni_dataset.print_dataset_info(start=0, end=5)

if __name__ == '__main__':
    main()
