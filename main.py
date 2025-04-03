import argparse  # 导入argparse库，用于处理命令行参数
import datetime  # 导入datetime库，用于时间操作
import random  # 导入random库，用于生成随机数
import time  # 导入time库，用于处理时间
from pathlib import Path  # 从pathlib库导入Path类，用于路径操作

import numpy as np  # 导入numpy库，用于数值计算和数组操作
import torch  # 导入torch库，PyTorch深度学习框架
from torch import nn  # 从torch库导入nn模块，用于神经网络构建
from torch.utils.data import DataLoader  # 导入DataLoader，用于批量加载数据
import util.misc as utils  # 导入自定义的utils模块，提供一些辅助功能
from datasets.adni import ADNIDataset  # 导入自定义的ADNIDataset类，用于加载ADNI数据集
from models.criterion import Criterion  # 导入自定义的Criterion类，定义损失函数
from models.longformer import Longformer  # 导入自定义的Longformer类，定义模型

import torch.nn.functional as F  # 导入torch.nn.functional模块，提供常用的神经网络函数

from sklearn import metrics  # 导入sklearn.metrics模块，用于评估模型
from sklearn.preprocessing import label_binarize  # 导入label_binarize函数，用于标签二值化
from sklearn.metrics import roc_auc_score  # 导入roc_auc_score函数，用于计算AUC值


# 定义解析命令行参数的函数
def get_args_parser():
    parser = argparse.ArgumentParser('Longformer', add_help=False)  # 创建ArgumentParser对象，用于解析命令行参数
    parser.add_argument('--lr', default=2e-4, type=float)  # 添加学习率参数，默认值为2e-4
    parser.add_argument('--batch_size', default=1, type=int)  # 添加批次大小参数，默认值为1
    parser.add_argument('--epochs', default=500, type=int)  # 添加训练周期数参数，默认值为500
    parser.add_argument('--device', default='cpu', help='device to use for training / testing')  # 添加设备选择参数，默认为cpu
    parser.add_argument('--seed', default=42, type=int)  # 添加随机种子参数，默认值为42
    parser.add_argument('--lr_drop', default=40, type=int, nargs='+')  # 添加学习率下降周期参数，默认值为40

    # 模型参数
    parser.add_argument('--vision_encoder', default='vitg-adapter', type=str, help="vitg-adpater/vit/res50")  # 添加视觉编码器类型参数，默认为'vitg-adapter'
    parser.add_argument('--num_feature_scales', default=4, type=int, help='number of feature levels/scales')  # 添加特征尺度数参数，默认为4
    parser.add_argument('--n_times', default=1, type=int)  # 添加时间步数参数，默认为1

    # Transformer模型参数
    parser.add_argument('--enc_layers', default=4, type=int, help="Number of encoding layers in the transformer")  # 添加编码层数参数，默认为4
    parser.add_argument('--dec_layers', default=4, type=int, help="Number of decoding layers in the transformer")  # 添加解码层数参数，默认为4
    parser.add_argument('--hidden_dim', default=288, type=int, help="Size of the embeddings (dimension of the transformer)")  # 添加隐藏层维度参数，默认为288
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")  # 添加前馈层维度参数，默认为1024
    parser.add_argument('--nheads', default=6, type=int, help="Number of attention heads inside the transformer's attentions")  # 添加注意力头数参数，默认为6
    parser.add_argument('--num_queries', default=125, type=int, help="Number of query slots")  # 添加查询槽数参数，默认为125
    parser.add_argument('--num_classes', default=2, type=int, help="Number of classes")  # 添加类别数参数，默认为2
    parser.add_argument('--dec_n_points', default=4, type=int)  # 添加解码器点数参数，默认为4
    parser.add_argument('--enc_n_points', default=4, type=int)  # 添加编码器点数参数，默认为4

    # 匹配器参数
    parser.add_argument('--set_cost_loc', default=5, type=float, help="Localization coefficient in the matching cost")  # 添加定位系数参数，默认为5
    parser.add_argument('--set_cost_cls', default=2, type=float, help="Classification coefficient in the matching cost")  # 添加分类系数参数，默认为2

    # 损失函数系数
    parser.add_argument('--loc_loss_coef', default=5, type=float)  # 添加定位损失系数参数，默认为5
    parser.add_argument('--cls_loss_coef', default=2, type=float)  # 添加分类损失系数参数，默认为2
    parser.add_argument('--focal_alpha', default=0.25, type=float)  # 添加焦点损失系数参数，默认为0.25

    # 数据集相关参数
    parser.add_argument('--dataset_file', default='ADNI')  # 添加数据集文件名称参数，默认为'ADNI'
    parser.add_argument('--classification_type', default='NC/AD', help='NC/AD or sMCI/pMCI')  # 添加分类类型参数，默认为'NC/AD'
    parser.add_argument('--train_data_path', default='/data2/qiuhui/data/adni')  # 添加训练数据路径参数
    parser.add_argument('--test_data_path', default='/data2/qiuhui/data/adni')  # 添加测试数据路径参数
    parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')  # 添加输出目录参数

    parser.add_argument('--model', default=None, help='load from checkpoint')  # 添加是否加载预训练模型的参数
    parser.add_argument('--eval', action='store_true')  # 添加是否进行评估的标志参数
    parser.add_argument('--num_workers', default=0, type=int)  # 添加数据加载器的工作线程数参数
    parser.add_argument('--distributed', action='store_true')  # 添加是否进行分布式训练的标志参数
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')  # 添加分布式训练中的本地节点排名参数

    return parser  # 返回解析器

def main(args):  # 主函数，开始训练和评估过程

    device = torch.device(f'cpu')  # 设置训练设备为CPU

    # 设置随机种子，保证实验的可复现性
    seed = args.seed + utils.get_rank()  # 使用自定义的rank来设置种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    random.seed(seed)  # 设置Python的随机种子

    model = Longformer(args)  # 创建Longformer模型
    model.to(device)  # 将模型移动到指定设备（CPU）

    dataset_train = ADNIDataset(args.train_data_path, args=args)  # 加载训练数据集
    dataset_val = ADNIDataset(args.test_data_path, args=args)  # 加载验证数据集

    # 分布式训练时使用分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, sampler=val_sampler)

    # 设置优化器
    if isinstance(model, nn.Module):
        param_groups = model.parameters()  # 如果模型是nn.Module，获取其参数
    else:
        param_groups = model
    optimizer = torch.optim.Adam(param_groups, lr=5e-4, eps=1e-4, weight_decay=0.0)  # 使用Adam优化器

    output_dir = Path(args.output_dir)  # 设置输出目录

    for epoch in range(args.epochs):  # 循环遍历每个训练周期
        print("Start training")  # 输出训练开始的信息
        correct = 0
        num = 0
        model.train()  # 设置模型为训练模式
        start_time = time.time()  # 记录训练开始时间
        for batch_idx, (img, flow, label, img_indicators, img_idx) in enumerate(train_loader):  # 遍历训练数据
            img = img.cuda(non_blocking=True)  # 将输入图像加载到GPU
            flow = flow.cuda(non_blocking=True)  # 将流图像加载到GPU
            label = label.cuda(non_blocking=True)  # 将标签加载到GPU
            img_indicators = img_indicators.cuda(non_blocking=True)  # 将图像指示器加载到GPU

            outputs = model(img, flow, img_indicators, args)  # 使用模型进行前向传播
            bce_loss = nn.BCELoss()  # 定义二元交叉熵损失函数
            m = nn.Sigmoid()  # 定义Sigmoid激活函数
            loss = bce_loss(m(outputs), F.one_hot(label, num_classes=2).float())  # 计算损失
            pred = outputs.argmax(dim=-1)  # 获取预测的标签

            correct += ((outputs.argmax(dim=-1) == label) + 0).sum()  # 计算正确预测的数量
            num += len(label)  # 统计总样本数

            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新优化器参数

        print(f'epoch: {epoch}, Accuracy: {correct / num}, Correct: {correct}, Total: {num}, time: {time.asctime(time.localtime(time.time()))}')

        if args.output_dir:  # 如果指定了输出目录
            print("Saving ckpt")  # 保存模型的检查点
            checkpoint_paths = [output_dir / 'checkpoint.pth']  # 定义检查点保存路径
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')  # 定期保存检查点
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)  # 保存模型和优化器的状态

        print("Start validation")  # 开始验证过程
        correct = 0
        num = 0
        y_pred = []  # 用于存储预测值
        y_test = []  # 用于存储真实标签
        model.eval()  # 设置模型为评估模式
        for batch_idx, (img, flow, label, img_indicators, img_idx) in enumerate(val_loader):  # 遍历验证数据
            img = img.cuda(non_blocking=True)  # 将输入图像加载到GPU
            flow = flow.cuda(non_blocking=True)  # 将流图像加载到GPU
            label = label.cuda(non_blocking=True)  # 将标签加载到GPU
            img_indicators = img_indicators.cuda(non_blocking=True)  # 将图像指示器加载到GPU
            with torch.no_grad():  # 不计算梯度
                outputs = model(img, flow, img_indicators, args)  # 使用模型进行前向传播
            pred = outputs.argmax(dim=-1)  # 获取预测标签
            correct += ((pred == label) + 0).sum()  # 计算正确预测的数量

            y_pred += pred.tolist()  # 将预测结果添加到列表
            y_test += label.tolist()  # 将真实标签添加到列表

            num += len(label)  # 统计样本数

        # 计算准确率和AUC值
        acc = metrics.accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print(f'Accuracy: {correct / num}, Correct: {correct}, Total: {num}')
        with open(f'./{args.dataset_file}_eval_out.out', 'a+') as f:  # 保存评估结果到文件
            f.write(f'epoch: {epoch}, Accuracy: {acc}, AUC: {auc}, Correct: {correct}, Total: {num}, time: {time.asctime(time.localtime(time.time()))}\n')

        total_time = time.time() - start_time  # 计算每个epoch的训练时间
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # 转换为时间格式
        print(f'One epoch training time {total_time_str}')  # 输出训练时间

if __name__ == '__main__':
    # 解析命令行参数并启动训练
    parser = argparse.ArgumentParser('Longformer training script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.vision_encoder = 'densenet121'  # 设置视觉编码器为densenet121
    args.train_data_path = rf'LongFormer-main\datasets\label.csv'  # 设置训练数据路径
    if args.output_dir:  # 如果指定了输出目录
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 创建输出目录
    main(args)  # 启动主函数
