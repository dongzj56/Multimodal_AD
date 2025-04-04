import pandas as pd  # 导入pandas库，用于数据处理
import os  # 导入os库，用于操作文件和目录
from monai.transforms import (  # 从MONAI库导入一些图像预处理和增强操作
    EnsureChannelFirstd,  # 确保输入图像的通道维度在第一个位置
    Compose,  # 将多个转换操作组合成一个
    LoadImaged,  # 加载图像文件
    SaveImaged,  # 保存图像文件
    ScaleIntensityd,  # 对图像进行强度标准化
    SpatialCropd,  # 裁剪图像的空间尺寸
    SpatialPadd,  # 对图像进行空间填充
    RandFlipd,  # 随机翻转图像
    EnsureTyped,  # 确保图像数据类型一致
    RandRotated,  # 随机旋转图像
    RandZoomd,  # 随机缩放图像
)


# 定义ADNI类，用于加载和处理ADNI数据集
class ADNI:
    def __init__(self, dataroot, label_filename, task):
        # 读取CSV文件，包含ADNI数据集的标签信息
        self.csv = pd.read_csv(os.path.join(dataroot, label_filename))
        self.labels = None  # 标签数据初始化为None
        self.label_dict = None  # 标签字典初始化为None
        self.data_dict = None  # 数据字典初始化为None
        mri_dir = os.path.join(dataroot, 'MRI')  # MRI图像目录路径
        pet_dir = os.path.join(dataroot, 'PET')  # PET图像目录路径

        # 根据指定任务获取数据和标签
        if task == 'ADCN':
            # 获取AD（阿尔茨海默病）和CN（正常控制组）的样本
            self.labels = self.csv[(self.csv['Group'] == 'AD') | (self.csv['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'AD': 1}  # 标签字典，CN为0，AD为1
        if task == 'pMCIsMCI':
            # 获取pMCI（进展型轻度认知障碍）和sMCI（稳定型轻度认知障碍）的样本
            self.labels = self.csv[(self.csv['Group'] == 'pMCI') | (self.csv['Group'] == 'sMCI')]
            self.label_dict = {'sMCI': 0, 'pMCI': 1}  # 标签字典，sMCI为0，pMCI为1
        if task == 'MCICN':
            # 获取pMCI、sMCI、MCI（轻度认知障碍）和CN的样本
            self.labels = self.csv[
                (self.csv['Group'] == 'pMCI') | (self.csv['Group'] == 'sMCI') | (self.csv['Group'] == 'MCI') | (
                            self.csv['Group'] == 'CN')]
            self.label_dict = {'CN': 0, 'sMCI': 1, 'pMCI': 1, 'MCI': 1}  # 标签字典，CN为0，其他为1

        # 创建数据字典，包含MRI、PET路径及标签
        subject_list = self.labels['Subject'].tolist()  # 获取所有患者的ID
        label_list = self.labels['Group'].tolist()  # 获取所有患者的组别标签
        age_list = self.labels['Age'].tolist()  # 获取所有患者的年龄
        self.data_dict = [{'MRI': os.path.join(mri_dir, subject_name + '.nii.gz'),
                           'PET': os.path.join(pet_dir, subject_name + '.nii.gz'),
                           'label': self.label_dict[subject_label],  # 通过标签字典获取标签
                           'age': subject_age,
                           'Subject': subject_name}
                          for subject_name, subject_label, subject_age in zip(subject_list, label_list, age_list)]  # 创建数据字典

    def __len__(self):
        return len(self.labels)  # 返回数据集的大小（样本数量）

    def get_weights(self):
        label_list = []
        # 遍历数据字典，获取所有标签
        for item in self.data_dict:
            label_list.append(item['label'])
        # 返回标签为0和1的数量
        return float(label_list.count(0)), float(label_list.count(1))


# 定义ADNI数据增强转换函数
def ADNI_transform(aug='True'):
    if aug == 'True':  # 如果启用数据增强
        train_transform = Compose([  # 定义训练数据的预处理和增强操作
                    LoadImaged(keys=['MRI', 'PET']),  # 加载MRI和PET图像
                    EnsureChannelFirstd(keys=['MRI', 'PET']),  # 确保图像通道维度在第一位
                    ScaleIntensityd(keys=['MRI', 'PET']),  # 进行强度标准化
                    # 数据增强操作
                    RandFlipd(keys=['MRI', 'PET'], prob=0.3, spatial_axis=0),  # 随机翻转
                    RandRotated(keys=['MRI', 'PET'], prob=0.3, range_x=0.05),  # 随机旋转
                    RandZoomd(keys=['MRI', 'PET'], prob=0.3, min_zoom=0.95, max_zoom=1),  # 随机缩放
                    EnsureTyped(keys=['MRI', 'PET'])  # 确保数据类型
                ])
    else:
        train_transform = Compose([  # 如果不启用数据增强，仅进行基本预处理
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    # 定义测试数据的预处理操作
    test_transform = Compose([
                LoadImaged(keys=['MRI', 'PET']),
                EnsureChannelFirstd(keys=['MRI', 'PET']),
                ScaleIntensityd(keys=['MRI', 'PET']),
                EnsureTyped(keys=['MRI', 'PET'])
            ])
    return train_transform, test_transform  # 返回训练和测试的转换操作


# 定义ADNI数据增强转换函数（Mnet版本）
def ADNI_transform_Mnet(aug='True'):
    if aug == 'True':  # 如果启用数据增强
        train_transform = Compose([  # 定义训练数据的预处理和增强操作
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    SpatialPadd(keys=['MRI', 'PET'], spatial_size=(91, 109, 91)),  # 空间填充到指定大小
                    # 数据增强操作
                    RandFlipd(keys=['MRI', 'PET'], prob=0.3, spatial_axis=0),
                    RandRotated(keys=['MRI', 'PET'], prob=0.3, range_x=0.05),
                    RandZoomd(keys=['MRI', 'PET'], prob=0.3, min_zoom=0.95, max_zoom=1),
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    else:
        train_transform = Compose([  # 如果不启用数据增强，仅进行基本预处理
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    SpatialPadd(keys=['MRI', 'PET'], spatial_size=(91, 109, 91)),
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    # 定义测试数据的预处理操作
    test_transform = Compose([
                LoadImaged(keys=['MRI', 'PET']),
                EnsureChannelFirstd(keys=['MRI', 'PET']),
                ScaleIntensityd(keys=['MRI', 'PET']),
                SpatialPadd(keys=['MRI', 'PET'], spatial_size=(91, 109, 91)),
                EnsureTyped(keys=['MRI', 'PET'])
            ])
    return train_transform, test_transform  # 返回训练和测试的转换操作


# 定义ADNI数据增强转换函数（ADVIT版本）
def ADNI_transform_ADVIT(aug='True'):
    # 定义训练数据的预处理操作
    train_transform = Compose([
                    LoadImaged(keys=['MRI', 'PET']),
                    EnsureChannelFirstd(keys=['MRI', 'PET']),
                    ScaleIntensityd(keys=['MRI', 'PET']),
                    SpatialPadd(keys=['MRI', 'PET'], spatial_size=(128, 128, 79)),  # 空间填充到指定大小
                    EnsureTyped(keys=['MRI', 'PET'])
                ])
    # 定义测试数据的预处理操作
    test_transform = Compose([
                LoadImaged(keys=['MRI', 'PET']),
                EnsureChannelFirstd(keys=['MRI', 'PET']),
                ScaleIntensityd(keys=['MRI', 'PET']),
                SpatialPadd(keys=['MRI', 'PET'], spatial_size=(128, 128, 79)),
                EnsureTyped(keys=['MRI', 'PET'])
            ])
    return train_transform, test_transform  # 返回训练和测试的转换操作
