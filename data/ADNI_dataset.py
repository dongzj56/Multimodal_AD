'''数据加载代码'''
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
    支持三个模态数据：MRI、PET图像和表格数据（例如 ADNIMERGE.xlsx 中的生物样本监测、临床检查数据）。
    """

    def __init__(self, csv_file, mri_dir, pet_dir, table_file, task='ADCN', augment=False, data_use='all'):
        """
        初始化ADNI数据集类，读取CSV文件、Excel表格数据并生成数据字典。

        :param csv_file: 标签文件路径（包含 Group 和 Subject ID 信息）
        :param mri_dir: MRI图像所在目录
        :param pet_dir: PET图像所在目录
        :param table_file: 表格数据文件路径（例如 "ADNIMERGE.xlsx"）
        :param task: 任务类型，用于选择不同标签类别
        :param augment: 是否启用数据增强
        :param data_use: 模态选择，可选值:
                         'all'   - 使用 MRI、PET 以及表格数据
                         'image' - 只使用 MRI 和 PET 影像
                         'mri'   - 只使用 MRI 影像
                         'pet'   - 只使用 PET 影像
        """
        self.csv = pd.read_csv(csv_file)         # 读取标签 CSV 文件
        self.mri_dir = mri_dir                     # MRI 数据目录
        self.pet_dir = pet_dir                     # PET 数据目录
        self.table_df = pd.read_excel(table_file)  # 读取包含所有患者表格数据的 Excel 文件
        self.task = task                           # 任务类型
        self.augment = augment                     # 是否进行数据增强
        self.data_use = data_use.lower()           # 模态选择

        self._process_labels()
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
        根据索引获取一个样本，根据 data_use 参数返回不同模态数据：
            'all'   : 返回 MRI、PET、表格数据及标签
            'image' : 返回 MRI、PET 影像及标签
            'mri'   : 返回 MRI 影像及标签
            'pet'   : 返回 PET 影像及标签
        """
        sample = self.data_dict[idx]
        label = sample['label']
        result = {}
        if self.data_use in ['all', 'image', 'mri']:
            mri_path = sample['MRI']
            mri_img = LoadImaged(keys=['MRI'])({'MRI': mri_path})['MRI']
            result['MRI'] = mri_img
        if self.data_use in ['all', 'image', 'pet']:
            pet_path = sample['PET']
            pet_img = LoadImaged(keys=['PET'])({'PET': pet_path})['PET']
            result['PET'] = pet_img
        if self.data_use == 'all':
            result['TABLE'] = sample['TABLE']

        # 根据 data_use 设置预处理流程（仅针对图像模态）
        if self.data_use in ['all', 'image', 'mri', 'pet']:
            if self.augment:
                transform = self.get_augmentation_transform()
            else:
                transform = self.get_basic_transform()
            # 构造临时字典，包含需要预处理的图像数据
            image_data = {}
            if 'MRI' in result:
                image_data['MRI'] = result['MRI']
            if 'PET' in result:
                image_data['PET'] = result['PET']
            transformed = transform(image_data)
            if 'MRI' in transformed:
                result['MRI'] = transformed['MRI']
            if 'PET' in transformed:
                result['PET'] = transformed['PET']

        # 返回不同模态的数据
        if self.data_use == 'all':
            return result['MRI'], result['PET'], result['TABLE'], label
        elif self.data_use == 'image':
            return result['MRI'], result['PET'], label
        elif self.data_use == 'mri':
            return result['MRI'], label
        elif self.data_use == 'pet':
            return result['PET'], label
        else:
            raise ValueError("data_use 参数必须为 'all', 'image', 'mri' 或 'pet'")

    def get_augmentation_transform(self):
        """
        返回包括数据增强操作的转换流程（仅针对图像模态）。
        """
        if self.data_use in ['all', 'image']:
            keys = ['MRI', 'PET']
        elif self.data_use == 'mri':
            keys = ['MRI']
        elif self.data_use == 'pet':
            keys = ['PET']
        else:
            raise ValueError("data_use 参数错误")
        return Compose([
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityd(keys=keys),
            RandFlipd(keys=keys, prob=0.3, spatial_axis=0),
            RandRotated(keys=keys, prob=0.3, range_x=0.05),
            RandZoomd(keys=keys, prob=0.3, min_zoom=0.95, max_zoom=1),
            EnsureTyped(keys=keys)
        ])

    def get_basic_transform(self):
        """
        返回基本的预处理转换流程（仅针对图像模态）。
        """
        if self.data_use in ['all', 'image']:
            keys = ['MRI', 'PET']
        elif self.data_use == 'mri':
            keys = ['MRI']
        elif self.data_use == 'pet':
            keys = ['PET']
        else:
            raise ValueError("data_use 参数错误")
        return Compose([
            EnsureChannelFirstd(keys=keys),
            ScaleIntensityd(keys=keys),
            EnsureTyped(keys=keys)
        ])

    def print_dataset_info(self, start=0, end=None):
        """
        打印数据集结构和指定范围的样本信息，以表格形式展示。

        :param start: 起始样本索引，默认为 0
        :param end: 结束样本索引，默认为 None，表示打印到数据集的最后一个样本
        """
        print(f"Dataset Structure:\n{'=' * 40}")
        print(f"Total Samples: {len(self)}")
        print(f"Task: {self.task}")
        print(f"Data Use: {self.data_use}")

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

            # 根据 data_use 决定是否显示 MRI、PET
            if self.data_use in ['all', 'image', 'mri']:
                mri_path = sample['MRI']
            else:
                mri_path = "N/A"

            if self.data_use in ['all', 'image', 'pet']:
                pet_path = sample['PET']
            else:
                pet_path = "N/A"

            # 如果 data_use == 'all'，才显示表格变量数量，否则显示 "N/A"
            if self.data_use == 'all':
                table_count_str = str(count_vars)
            else:
                table_count_str = "N/A"

            sample_data.append([
                mri_path,
                pet_path,
                table_count_str,
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
    table_file = os.path.join(dataroot, r'TABLE\ADNIMERGE.xlsx')  # 表格数据存放在Excel文件中
    task = 'ADCN'

    # 在主函数中自定义 data_use 参数，可选值: 'all', 'image', 'mri', 'pet'
    data_use = 'image'  # 例如只使用MRI数据

    adni_dataset = ADNIDataset(csv_file=label_filename,
                               mri_dir=mri_dir,
                               pet_dir=pet_dir,
                               table_file=table_file,
                               task=task,
                               augment=True,
                               data_use=data_use)

    print(f'Dataset size: {len(adni_dataset)}')
    if data_use == 'all':
        sample_mri, sample_pet, sample_table, sample_label = adni_dataset[0]
        print(f'Sample MRI shape: {sample_mri.shape}, PET shape: {sample_pet.shape}, Label: {sample_label}')
        print("Sample Table Data:")
        print(sample_table)
    elif data_use == 'image':
        sample_mri, sample_pet, sample_label = adni_dataset[0]
        print(f'Sample MRI shape: {sample_mri.shape}, PET shape: {sample_pet.shape}, Label: {sample_label}')
    elif data_use == 'mri':
        sample_mri, sample_label = adni_dataset[0]
        print(f'Sample MRI shape: {sample_mri.shape}, Label: {sample_label}')
    elif data_use == 'pet':
        sample_pet, sample_label = adni_dataset[0]
        print(f'Sample PET shape: {sample_pet.shape}, Label: {sample_label}')

    adni_dataset.print_dataset_info(start=0, end=5)

if __name__ == '__main__':
    main()
