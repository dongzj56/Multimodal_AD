from sklearn.model_selection import train_test_split
import os
from ADNI import ADNI

def create_train_test_split(dataset, test_size=0.2):
    """
    创建训练集和测试集的拆分。
    """
    # 将数据集拆分为训练集和测试集的索引
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

    # 根据索引创建训练集和测试集
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    return train_dataset, test_dataset

def main():
    """
    主函数，展示如何进行训练集和测试集拆分，并输出样本信息。
    """
    dataroot = rf'C:\Users\dongz\Desktop\adni_dataset'
    label_filename = rf'C:\Users\dongz\Desktop\adni_dataset\ADNI.csv'
    mri_dir = os.path.join(dataroot, 'MRI')
    pet_dir = os.path.join(dataroot, 'PET')
    table_file = os.path.join(dataroot, r'TABLE\ADNIMERGE.xlsx')  # 表格数据路径
    task = 'EMCILMCI'  # 任务类型（例如：'ADCN'）
    augment = True  # 训练集进行数据增强
    model = 'Ours'  # 模型类型
    data_use = 'img'  # 数据使用类型（'img', 'all', 'mri', 'pet'）

    # 初始化训练集数据集
    adni_train_dataset = ADNI(csv_file=label_filename,
                              mri_dir=mri_dir,
                              pet_dir=pet_dir,
                              table_file=table_file,
                              task=task,
                              augment=augment,
                              data_use=data_use,
                              model=model)

    # 初始化测试集数据集，不进行数据增强
    adni_test_dataset = ADNI(csv_file=label_filename,
                             mri_dir=mri_dir,
                             pet_dir=pet_dir,
                             table_file=table_file,
                             task=task,
                             augment=False,  # 测试集不进行数据增强
                             data_use=data_use,
                             model=model)

    # 进行训练集和测试集拆分
    train_dataset, test_dataset = create_train_test_split(adni_train_dataset, test_size=0.2)

    print(f'训练集大小: {len(train_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')

    # 输出五个样本的 subject ID 和其他信息
    print("\n训练集五个样本的 Subject ID 和其他信息:")
    for i in range(5):
        sample_mri, sample_pet, sample_label = train_dataset[i]
        print(f"样本 {i+1} - Subject ID: {adni_train_dataset.data_dict[i]['Subject']}, MRI 形状: {sample_mri.shape}, PET 形状: {sample_pet.shape}, 标签: {sample_label}")

    print("\n测试集五个样本的 Subject ID 和其他信息:")
    for i in range(5):
        sample_mri, sample_pet, sample_label = test_dataset[i]
        print(f"样本 {i+1} - Subject ID: {adni_train_dataset.data_dict[i]['Subject']}, MRI 形状: {sample_mri.shape}, PET 形状: {sample_pet.shape}, 标签: {sample_label}")

if __name__ == '__main__':
    main()
