import torch
import torch.utils.data

import os

import random

import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F

# 定义ADNI数据集类
class ADNIDataset:
    # 利用csv文件构造标签等数据结构
    def __init__(self, csv_file, args):
        self.args = args
        print('正在构建数据集...')
        self.getPatientsInfo(csv_file)

    # 读取患者信息
    def getPatientsInfo(self, csv_path):
        if not os.path.exists(csv_path):
            raise ValueError("{} 路径未找到".format(csv_path))

        self._paths = []  # 存储图像路径
        self._months = []  # 存储每个图像对应的月份
        self._labels = []  # 存储标签（如：AD，CN等）
        self._ptids = []  # 存储患者ID
        self.long_info = {}  # 存储患者的长期数据（按月份存储）

        with open(csv_path) as f:
            for line in f:
                if 'ADNI' in csv_path:  # 如果是ADNI数据集
                    img_path, month, label = line.split('\n')[0].split(',')
                    if self.args.n_times == 2:
                        ptid = img_path.split('/')[-3]
                        scan_time = img_path.split('/')[-2]
                        opflow_path = '.../voxelmorph_out/' + ptid + '_' + scan_time + '_flow.nii.gz'
                        if not os.path.exists(opflow_path):
                            continue

                else:
                    img_path, label = line.split('\n')[0].split(',')
                    month = 'bl'  # 如果没有指定月份，默认为'bl'（基线）
                
                patient_id = img_path.split('/')[-3]  # 提取患者ID
                label_map = {
                    'Nondemented': 0,
                    'Demented': 1,
                    'CN': 0,
                    'AD': 1,
                    'MCI': 2
                }
                label = label_map[label]  # 将标签转化为数字
                if label not in [0, 1]:  # 只处理AD和CN两种标签
                    continue
                if month == 'bl':
                    month = 0  # 将基线（bl）月份设为0
                else:
                    month = int(month[1:])  # 其他月份转为整数
                self._ptids.append(patient_id)  # 添加患者ID
                self._paths.append(img_path)  # 添加图像路径
                self._months.append(month)  # 添加月份
                self._labels.append(int(label))  # 添加标签
                
                if patient_id not in self.long_info.keys():
                    self.long_info[patient_id] = {}
                self.long_info[patient_id][month] = img_path  # 存储患者在不同月份的图像路径

        print("已从{}构建ADNI患者数据集（大小：{}）".format(csv_path, len(self._paths)))

    def __len__(self):
        return len(self._paths)  # 返回数据集的大小
    
    def pad_img(self, img, size=224):
        '''将图像填充为指定大小的正方体。'''
        x, y, z = img.shape
        img = img.unsqueeze(0).unsqueeze(0)  # BCHWD格式（批次，通道，高，宽，深度）
        max_size = max(x, y, z)  # 取图像的最大维度作为目标大小
        new_size = (int(size * x / max_size), int(size * y / max_size), int(size * z / max_size))
        img = F.interpolate(img, size=new_size, mode='trilinear', align_corners=True)  # 重采样图像

        x, y, z = new_size
        new_im = torch.zeros((1, 1, size, size, size))  # 创建一个大小为(size, size, size)的零张量
        x_min = int((size - x) / 2)
        x_max = x_min + x
        y_min = int((size - y) / 2)
        y_max = y_min + y
        z_min = int((size - z) / 2)
        z_max = z_min + z
        new_im[:, :, x_min:x_max, y_min:y_max, z_min:z_max] = img  # 将图像放置到新的张量中

        return new_im.squeeze(0)  # 去掉批次维度，返回图像

    def pad_img_3d(self, img, size=224):
        '''将3D图像填充为指定大小的正方体。'''
        img = img.permute(3, 0, 1, 2)  # 重新排列图像维度
        _, x, y, z = img.shape
        img = img.unsqueeze(0)  # BCHWD格式（批次，通道，高，宽，深度）
        max_size = max(x, y, z)  # 取图像的最大维度作为目标大小
        new_size = (int(size * x / max_size), int(size * y / max_size), int(size * z / max_size))
        img = F.interpolate(img, size=new_size, mode='trilinear', align_corners=True)  # 重采样图像

        x, y, z = new_size
        new_im = torch.zeros((1, 3, size, size, size))  # 创建一个大小为(size, size, size)的零张量
        x_min = int((size - x) / 2)
        x_max = x_min + x
        y_min = int((size - y) / 2)
        y_max = y_min + y
        z_min = int((size - z) / 2)
        z_max = z_min + z
        new_im[:, :, x_min:x_max, y_min:y_max, z_min:z_max] = img  # 将图像放置到新的张量中

        return new_im.squeeze(0)  # 去掉批次维度，返回图像

    def norm_img(self, img):
        '''归一化图像到[0, 1]范围'''
        return (img - img.min()) / (img.max() - img.min())  # 归一化

    def preprocess(self, path):
        '''处理图像：读取，归一化和填充'''
        img = torch.FloatTensor(sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(float))  # 读取图像
        img = self.norm_img(img)  # 归一化
        img = self.pad_img(img)  # 填充图像
        return img
    
    def preprocess_3dim(self, path):
        '''处理3D图像：读取，归一化和填充'''
        img = torch.FloatTensor(sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(float))  # 读取图像
        img = self.norm_img(img)  # 归一化
        img = self.pad_img_3d(img)  # 填充3D图像
        return img

    def __getitem__(self, idx):
        '''根据索引获取一个样本：图像，标签，光流等'''
        label = self._labels[idx]  # 获取标签
        path = self._paths[idx].replace('data2', 'data')  # 获取图像路径
        month = self._months[idx]  # 获取月份
        ptid = self._ptids[idx]  # 获取患者ID
        img = self.preprocess(path)  # 处理图像
        img_indicators = [1]  # 初始化图像指示器

        if self.args.n_times == 2:  # 如果需要处理时间序列
            scan_time = path.split('/')[-2]  # 获取扫描时间
            opflow_path = '.../voxelmorph_out/' + ptid + '_' + scan_time + '_flow.nii.gz'  # 光流路径
            if os.path.exists(opflow_path):  # 如果光流文件存在
                img_opflow = self.preprocess_3dim(opflow_path)  # 处理光流图像
                img_indicators.append(1)  # 光流存在，设置为1
            else:
                img_opflow = torch.zeros((3, img.shape[-3], img.shape[-2], img.shape[-1])).to(img.dtype)  # 光流不存在，返回全零图像
                img_indicators.append(0)  # 光流不存在，设置为0
            
        img_indicators = torch.Tensor(img_indicators)  # 转换为Tensor
        return img, img_opflow, label, img_indicators, path  # 返回图像，光流，标签，指示器和路径
