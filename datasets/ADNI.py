'''数据加载代码'''

import os
import pandas as pd
from cv2 import transform
from sympy.physics.continuum_mechanics import Truss
from torch.utils.data import Dataset
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    EnsureTyped,
    SpatialPadd
)

from AdaptiveNormal import adaptive_normal  # 导入自适应归一化函数


# 定义数据类
class ADNI(Dataset):
    """
    用于处理ADNI数据集的类，包含读取数据、标签处理、图像预处理等功能。
    支持三个模态数据：MRI、PET图像和表格数据（生物样本数据、临床检查数据）。
    """

    def __init__(self, csv_file, mri_dir, pet_dir, table_file, task='ADCN', augment=False, data_use='img',model='Ours'):
        """
        初始化ADNI数据集类，读取数据和标签文件，并生成数据字典。

        :param csv_file: 标签文件路径（包含 Group 和 Subject ID 等信息）
        :param mri_dir: MRI图像所在目录
        :param pet_dir: PET图像所在目录
        :param table_file: 表格数据文件路径
        :param task: 任务类型，用于选择不同标签类别
        :param augment: 预处理是否启用数据增强
        :param data_use: 模态选择，可选值:
                         'all'   - 使用 MRI、PET 以及表格数据
                         'img' - 只使用 MRI 和 PET 影像
                         'mri'   - 只使用 MRI 影像
                         'pet'   - 只使用 PET 影像
        :param model: 所用模型，可以用来调整不同数据集
        """
        self.csv = pd.read_csv(csv_file)         # 读取标签 CSV 文件
        self.mri_dir = mri_dir                     # MRI 数据目录
        self.pet_dir = pet_dir                     # PET 数据目录
        self.table_df = pd.read_excel(table_file)  # 读取包含所有患者表格数据的 Excel 文件
        self.task = task                           # 任务类型
        self.augment = augment                     # 是否进行数据增强
        self.data_use = data_use.lower()           # 模态选择
        self.model = model

        self._process_labels()
        self._build_data_dict()

    # 有问题，待完善
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

    def __getitem__(self, idx):  # 定义 __getitem__ 方法，通过索引获取一个样本数据
        """
        根据索引获取一个样本，根据 data_use 参数返回不同模态数据：
            'all'   : 返回 MRI、PET、表格数据及标签
            'img' : 返回 MRI、PET 影像及标签
            'mri'   : 返回 MRI 影像及标签
            'pet'   : 返回 PET 影像及标签
        根据模型调整预处理：主要适配图像维度
        """
        sample = self.data_dict[idx]  # 从数据字典中根据索引获取对应的样本信息
        label = sample['label']  # 从样本信息中提取标签
        result = {}  # 初始化一个空字典，用于存储加载后的数据

        if self.data_use in ['all', 'img', 'mri']:  # 如果 data_use 参数包含 MRI 模态（'all'、'img' 或 'mri'）
            mri_path = sample['MRI']  # 获取样本中 MRI 图像的路径
            mri_img = LoadImaged(keys=['MRI'])({'MRI': mri_path})['MRI']  # 使用 LoadImaged 函数加载 MRI 图像
            result['MRI'] = mri_img  # 将加载后的 MRI 图像存入结果字典中
        if self.data_use in ['all', 'img', 'pet']:  # 如果 data_use 参数包含 PET 模态（'all'、'img' 或 'pet'）
            pet_path = sample['PET']  # 获取样本中 PET 图像的路径
            pet_img = LoadImaged(keys=['PET'])({'PET': pet_path})['PET']  # 使用 LoadImaged 函数加载 PET 图像
            result['PET'] = pet_img  # 将加载后的 PET 图像存入结果字典中
        if self.data_use == 'all':  # 如果 data_use 参数为 'all'
            result['TABLE'] = sample['TABLE']  # 将样本中的表格数据存入结果字典中

        # 根据 data_use 设置预处理流程（仅针对图像模态）
        if self.data_use in ['all', 'img', 'mri', 'pet']:  # 如果需要预处理图像数据（所有包含图像数据的情况）
            # if self.augment:  # 如果启用了数据增强
            #     transform = self.get_augmentation_transform()  # 获取数据增强的转换流程
            # else:
            #     transform = self.get_basic_transform()  # 否则获取基本的预处理转换流程
            if self.model == 'Ours':
                transform = self.ADNI_transform()
            elif self.model == 'ADVIT':
                transform = self.ADNI_transform_ADVIT()

            # 构造临时字典，包含需要预处理的图像数据
            image_data = {}  # 初始化一个空字典，用于存储待预处理的图像数据
            if 'MRI' in result:  # 如果结果字典中包含 MRI 数据
                image_data['MRI'] = result['MRI']  # 将 MRI 图像数据加入临时字典
            if 'PET' in result:  # 如果结果字典中包含 PET 数据
                image_data['PET'] = result['PET']  # 将 PET 图像数据加入临时字典

            transformed = transform(image_data)  # 对临时字典中的图像数据应用预处理转换

            if 'MRI' in transformed:  # 如果预处理后的数据中包含 MRI 数据
                result['MRI'] = transformed['MRI']  # 更新结果字典中的 MRI 数据为预处理后的数据
            if 'PET' in transformed:  # 如果预处理后的数据中包含 PET 数据
                result['PET'] = transformed['PET']  # 更新结果字典中的 PET 数据为预处理后的数据

        # 返回不同模态的数据，根据 data_use 参数返回相应的数据组合
        if self.data_use == 'all':  # 如果 data_use 参数为 'all'
            return result['MRI'], result['PET'], result['TABLE'], label  # 返回 MRI、PET、表格数据及标签
        elif self.data_use == 'img':  # 如果 data_use 参数为 'img'
            return result['MRI'], result['PET'], label  # 返回 MRI、PET 图像及标签
        elif self.data_use == 'mri':  # 如果 data_use 参数为 'mri'
            return result['MRI'], label  # 仅返回 MRI 图像及标签
        elif self.data_use == 'pet':  # 如果 data_use 参数为 'pet'
            return result['PET'], label  # 仅返回 PET 图像及标签
        else:
            raise ValueError("data_use 参数必须为 'all', 'img', 'mri' 或 'pet'")  # 如果 data_use 参数无效，则抛出错误

    def AdaptiveNormalization(self, keys):
        """
        自适应归一化方法的转换函数
        """

        def transform(data):
            for key in keys:
                img = data[key]  # 获取当前图像
                img = adaptive_normal(img)  # 使用自适应归一化
                data[key] = img  # 更新数据字典中的图像
            return data

        return transform

    def ADNI_transform(self):
        '''
        训练数据集可选，测试数据集必须augment=False
        '''
        if self.augment == True:
            """
            返回包括数据增强操作的转换流程（仅针对图像模态）。
            """

            print('data augmentation...')
            if self.data_use in ['all', 'img']:
                keys = ['MRI', 'PET']
            elif self.data_use == 'mri':
                keys = ['MRI']
            elif self.data_use == 'pet':
                keys = ['PET']
            else:
                raise ValueError("data_use 参数错误")
            return Compose([
                EnsureChannelFirstd(keys=keys),  # 确保图像的通道维度位于第一个位置，通常是为了符合神经网络输入格式的要求（例如 CHW: Channel, Height, Width）
                # 可选择线性归一化和自适应归一化
                # ScaleIntensityd(keys=keys),  # 对图像进行强度标准化，将图像的强度值缩放到一个统一的范围，通常是 [0, 1] 或 [-1, 1]，便于后续处理和训练
                self.AdaptiveNormalization(keys=keys),
                RandFlipd(keys=keys, prob=0.3, spatial_axis=0), # 随机翻转图像，prob=0.3 表示有 30% 的概率进行翻转，spatial_axis=0 表示沿着 X 轴（左右方向）翻转
                RandRotated(keys=keys, prob=0.3, range_x=0.05), # 随机旋转图像，prob=0.3 表示有 30% 的概率进行旋转，range_x=0.05 表示旋转角度范围为 -5% 到 +5%
                RandZoomd(keys=keys, prob=0.3, min_zoom=0.95, max_zoom=1), # 随机缩放图像，prob=0.3 表示有 30% 的概率进行缩放，min_zoom=0.95 表示最小缩放比例为 95%，max_zoom=1 表示最大缩放比例为 100%
                EnsureTyped(keys=keys)  # 确保图像数据类型为 PyTorch Tensor 或其他适用于模型输入的类型，通常是 float32 或 float64 类型

            ])

        else:
            """
            返回基本的预处理转换流程（仅针对图像模态）。
            """

            print('no data augmentation...')
            if self.data_use in ['all', 'img']:
                keys = ['MRI', 'PET']
            elif self.data_use == 'mri':
                keys = ['MRI']
            elif self.data_use == 'pet':
                keys = ['PET']
            else:
                raise ValueError("data_use 参数错误")
            return Compose([
                EnsureChannelFirstd(keys=keys), # 确保图像的通道维度位于第一个位置，通常是为了符合神经网络输入格式的要求（例如 CHW: Channel, Height, Width）
                # ScaleIntensityd(keys=keys), # 对图像进行强度标准化，将图像的强度值缩放到一个统一的范围，通常是 [0, 1] 或 [-1, 1]，便于后续处理和训练
                self.AdaptiveNormalization(keys=keys),
                EnsureTyped(keys=keys) # 确保图像数据类型为 PyTorch Tensor 或其他适用于模型输入的类型，通常是 float32 或 float64 类型
            ])

    # 暂时用不到，待完善
    def ADNI_transform_ADVIT(self):
        # 不进行数据增强，只进行基本预处理
        train_transform = Compose([
            LoadImaged(keys=['MRI', 'PET']),
            EnsureChannelFirstd(keys=['MRI', 'PET']),
            ScaleIntensityd(keys=['MRI', 'PET']),
            SpatialPadd(keys=['MRI', 'PET'], spatial_size=(128, 128, 79)),  # 填充图像到指定大小
            EnsureTyped(keys=['MRI', 'PET'])
        ])
        test_transform = Compose([
            LoadImaged(keys=['MRI', 'PET']),
            EnsureChannelFirstd(keys=['MRI', 'PET']),
            ScaleIntensityd(keys=['MRI', 'PET']),
            SpatialPadd(keys=['MRI', 'PET'], spatial_size=(128, 128, 79)),
            EnsureTyped(keys=['MRI', 'PET'])
        ])
        return train_transform, test_transform

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
            if self.data_use in ['all', 'img', 'mri']:
                mri_path = sample['MRI']
            else:
                mri_path = "N/A"

            if self.data_use in ['all', 'img', 'pet']:
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
    augment = False
    model = 'Ours'

    # 在主函数中自定义 data_use 参数，可选值: 'all', 'img', 'mri', 'pet'
    data_use = 'img'  # 例如只使用MRI数据

    adni_dataset = ADNI(csv_file=label_filename,
                               mri_dir=mri_dir,
                               pet_dir=pet_dir,
                               table_file=table_file,
                               task=task,
                               augment=augment,
                               data_use=data_use,
                               model = model)

    print(f'Dataset size: {len(adni_dataset)}')

    if data_use == 'all':
        sample_mri, sample_pet, sample_table, sample_label = adni_dataset[0]
        print(f'Sample MRI shape: {sample_mri.shape}, PET shape: {sample_pet.shape}, Label: {sample_label}')
        print("Sample Table Data:")
        print(sample_table)
    elif data_use == 'img':
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
