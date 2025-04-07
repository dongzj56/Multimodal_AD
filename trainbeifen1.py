# 导入必要的模块和包
import glob                # 用于文件路径匹配
import random              # 用于生成随机数

import numpy as np         # 数值计算库
import torch               # PyTorch深度学习框架

from datasets.ADNI import ADNI,ADNI_transform # 导入ADNI数据集及其数据预处理函数
from models.mymodel import model_ad, model_CNN_ad  # 导入两种模型定义：Transformer风格和CNN风格
from options.option import Option                # 导入选项解析类，用于读取配置参数
from sklearn.model_selection import KFold, train_test_split  # 导入交叉验证和数据集拆分函数
from torch.utils.data import DataLoader          # 导入数据加载器
from monai.data import Dataset                   # MONAI数据集类，用于医学影像数据处理
import ignite                                    # PyTorch Ignite框架，用于训练循环及指标计算
from ignite.metrics import Accuracy, Loss, Average, ConfusionMatrix  # 导入各类评估指标
from ignite.engine import Engine, Events         # 定义训练和评估引擎以及事件
from ignite.contrib.handlers import ProgressBar  # 导入进度条显示工具
from ignite.contrib.metrics import ROC_AUC       # 导入ROC_AUC指标计算
from ignite.handlers import Checkpoint, global_step_from_engine, DiskSaver, LRScheduler  # 导入模型保存、调度等工具
from utils.utils import getOptimizer, cal_confusion_metrics, mkdirs, get_dataset_weights  # 导入工具函数
from torch.nn.functional import softmax         # 导入softmax函数，用于概率输出
from utils.utils import Logger                    # 导入日志记录工具
import os                                         # 导入系统路径操作模块


# 程序主入口
if __name__ == '__main__':
    # 指定使用的设备为GPU 0
    # device = torch.device('cuda:{}'.format(0))
    device = torch.device('cpu')

    # 初始化选项参数，并创建输出目录
    opt = Option().parse()
    save_dir = os.path.join('./checkpoints', opt.name)  # 根据任务名称创建保存检查点的文件夹

    # 加载ADNI数据集
    # 从指定数据根目录加载数据，同时传入标签文件和任务类型
    mri_dir = os.path.join(opt.dataroot, 'MRI')  # 假设MRI数据存放在 'dataroot/MRI' 目录下
    pet_dir = os.path.join(opt.dataroot, 'PET')  # 假设PET数据存放在 'dataroot/PET' 目录下
    table_file = os.path.join(opt.dataroot, 'TABLE', 'ADNIMERGE.xlsx')  # 假设表格数据存放在 'dataroot/TABLE' 目录下

    # 创建ADNI数据集对象
    ADNI_data = ADNI(
        label_file=os.path.join(opt.dataroot, 'ADNI.csv'),  # 标签文件路径
        mri_dir=mri_dir,  # MRI数据目录
        pet_dir=pet_dir,  # PET数据目录
        table_file=table_file,  # 表格数据文件路径
        task=opt.task,  # 任务类型
        augment=(opt.aug == 'True'),  # 数据增强
        data_use='img',  # 使用图像数据（MRI + PET）
        model=opt.model  # 使用的模型类型
    ).data_dict

    # # 获取数据增强/预处理操作（训练和验证的不同变换）
    train_transforms, val_transforms = ADNI_transform2(opt.aug)

    # 初始化主日志记录器，用于记录整个实验过程
    logger_main = Logger(save_dir)

    # 准备K折交叉验证的划分
    num_fold = 5  # 使用5折交叉验证

    seed = 1  # 默认随机种子设为1
    # 根据不同任务设置不同的随机种子
    if opt.task == 'ADCN':
        seed = 42
    elif opt.task == 'pMCIsMCI':
        seed = 996
    elif opt.task == 'EMCILMCI':
        seed = 1234

    # 如果配置参数中要求随机种子随机化，则重新生成一个随机种子
    if opt.randint == 'True':
        seed = random.randint(1, 1000)
    print(f'The random seed is {seed}')

    # 创建KFold对象，设置划分数、是否打乱数据和随机种子
    kfold_splits = KFold(n_splits=num_fold, shuffle=True, random_state=seed)

    # 根据交叉验证划分构建数据加载器
    def setup_dataflow(train_idx, test_idx):
        # 对训练集进一步划分出验证集（20%作为验证集）
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=seed)

        # 根据索引从ADNI_data中取出对应的数据
        train_data = [ADNI_data[i] for i in train_idx.tolist()]
        val_data = [ADNI_data[i] for i in val_idx.tolist()]
        test_data = [ADNI_data[i] for i in test_idx.tolist()]

        # 针对特定任务，如果配置要求额外采样，则加载额外的ADCN数据并加入训练集
        if opt.task == 'pMCIsMCI' and opt.extra_sample == 'True':
            ADNI_ADCN_data = ADNI(dataroot=opt.dataroot, label_filename='ADNI.csv', task='ADCN').data_dict
            train_data += ADNI_ADCN_data

        # 创建数据集对象，并分别设置训练和验证阶段的数据预处理变换
        train_dataset = Dataset(data=train_data, transform=train_transforms)
        val_dataset = Dataset(data=val_data, transform=val_transforms)
        test_dataset = Dataset(data=test_data, transform=val_transforms)
        print(f'Train Datasets: {len(train_dataset)}')


        # 使用DataLoader构建批量数据加载器，训练集设置打乱和丢弃最后不完整的batch
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size)

        # 计算数据集的权重，通常用于类别不平衡处理
        weights = get_dataset_weights(train_dataset, train_idx)
        print(f'Val Datasets: {len(val_dataset)}')
        print(f'Test Datasets: {len(test_dataset)}')
        print('weights:',weights)
        # 返回训练、验证、测试的数据加载器及权重信息
        return train_loader, val_loader, test_loader, weights


    # 定义模型初始化函数，根据模型类型构建对应的网络结构
    def init_model(model):
        net_model = None
        if model == 'Transformer':
            # 构建Transformer风格的模型，参数从opt中获取
            net_model = model_ad(dim=opt.dim, depth=opt.trans_enc_depth, heads=4,
                                 dim_head=opt.dim // 4, mlp_dim=opt.dim * 4, dropout=opt.dropout).to(device)
            # 如需使用预训练模型，可解除下面代码注释（针对特定任务）
            # if opt.task == 'pMCIsMCI':
            #     checkpoint_all = torch.load('./pretrainAD.pt', map_location=device)
            #     Checkpoint.load_objects(to_load={'net_model': net_model}, checkpoint=checkpoint_all)
            #     print('Load pre-training model')
        elif model == 'CNN':
            # 构建CNN风格的模型
            net_model = model_CNN_ad(dim=opt.dim).to(device)
        else:
            raise ValueError(f"无效的模型类型: {model}")
        return net_model


    # 定义训练模型函数，包含训练、验证、测试以及模型保存等完整流程
    def train_model(train_dataloader, val_dataloader, test_dataloader, fold, weights):
        # 为当前fold创建保存检查点的目录
        save_path_fold = os.path.join(save_dir, str(fold))
        mkdirs(save_path_fold)
        # 为当前fold创建日志记录器
        logger = Logger(save_path_fold)
        # 初始化模型、优化器以及损失函数
        net_model = init_model(opt.model)
        optimizer, lr_schedualer = getOptimizer(net_model.parameters(), opt)
        criterion = torch.nn.CrossEntropyLoss()
        res_fold = []  # 用于保存当前fold的测试结果

        # 定义训练单步函数
        def train_step(engine, batch):
            output_dic = {}
            # 设置模型为训练模式
            net_model.train()
            # 从batch中获取MRI、PET影像数据和标签，并移动到设备上
            MRI = batch['MRI'].to(device)
            PET = batch['PET'].to(device)
            label = batch['label'].to(device)
            output_dic['label'] = label
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播，模型返回三个输出：主要预测和两个分支的输出
            output_logits, D_MRI_logits, D_PET_logits = net_model(MRI, PET)
            output_dic['logits'] = output_logits
            output_dic['D_MRI_logits'] = D_MRI_logits
            output_dic['D_PET_logits'] = D_PET_logits

            # 计算主要任务的交叉熵损失
            ce_loss = criterion(output_logits, label)
            # 构造MRI分支和PET分支的真实标签，MRI分支标签为1，PET分支标签为0
            mri_gt = torch.ones([D_MRI_logits.shape[0]], dtype=torch.int64).to(MRI.device)
            pet_gt = torch.zeros([D_PET_logits.shape[0]], dtype=torch.int64).to(PET.device)
            output_dic['D_MRI_label'] = mri_gt
            output_dic['D_PET_label'] = pet_gt
            # 计算辅助任务的损失，取两个分支损失的平均值
            ad_loss = (criterion(D_MRI_logits, mri_gt) + criterion(D_PET_logits, pet_gt)) / 2
            # 保存损失数值用于日志记录
            output_dic['ce_loss'] = ce_loss.item()
            output_dic['ad_loss'] = ad_loss.item()

            # 总损失为主要任务损失和辅助任务损失之和，进行反向传播
            all_loss = ad_loss + ce_loss
            all_loss.backward()
            # 优化器更新参数
            optimizer.step()
            # 返回本步的输出字典，供后续计算指标使用
            return output_dic

        # 使用ignite Engine创建训练引擎，并绑定训练单步函数
        trainer_label = Engine(train_step)
        # 为训练过程添加进度条显示
        ProgressBar().attach(trainer_label)
        # 添加学习率调度器处理器，在每个epoch开始时更新学习率
        lr_schedualer_handler = LRScheduler(lr_schedualer)
        trainer_label.add_event_handler(Events.EPOCH_STARTED, lr_schedualer_handler)

        # 定义验证单步函数
        def val_step(engine, batch):
            output_dic = {}
            # 设置模型为评估模式
            net_model.eval()
            # 不需要计算梯度，加快计算并节省内存
            with torch.no_grad():
                # 从batch中获取数据并移动到设备上
                MRI = batch['MRI'].to(device)
                PET = batch['PET'].to(device)
                label = batch['label'].to(device)
                output_dic['label'] = label

                # 前向传播（仅计算主要输出）
                output_logits, _, _ = net_model(MRI, PET)
                output_dic['logits'] = output_logits
                # 计算验证损失
                all_loss = criterion(output_logits, label)
                output_dic['loss'] = all_loss.item()
                return output_dic

        # 创建验证引擎
        evaluator = Engine(val_step)
        ProgressBar().attach(evaluator)

        # 定义一个自定义的one_hot_transform类，用于将预测结果转换为one-hot格式，以计算混淆矩阵
        class one_hot_transform:
            def __init__(self, target):
                self.target = target

            def __call__(self, output):
                y_pred, y = output[self.target], output['label']
                y_pred = torch.argmax(y_pred, dim=1).long()  # 取预测最大概率对应的类别
                y_pred = ignite.utils.to_onehot(y_pred, 2)  # 转换为one-hot编码，假设有2个类别
                y = y.long()
                return y_pred, y

        # 定义训练过程中需要计算的指标
        train_metrics = {
            "accuracy": Accuracy(output_transform=lambda x: [x['logits'], x['label']]),
            "MRI_accuracy": Accuracy(output_transform=lambda x: [x['D_MRI_logits'], x['D_MRI_label']]),
            "PET_accuracy": Accuracy(output_transform=lambda x: [x['D_PET_logits'], x['D_PET_label']]),
            "ce_loss": Average(output_transform=lambda x: x['ce_loss']),
            "ad_loss": Average(output_transform=lambda x: x['ad_loss'])
        }
        # 定义验证阶段的指标
        val_metrics = {
            "accuracy": Accuracy(output_transform=lambda x: [x['logits'], x['label']]),
            "confusion": ConfusionMatrix(num_classes=2, output_transform=one_hot_transform(target='logits')),
            "auc": ROC_AUC(output_transform=lambda x: [softmax(x['logits'], dim=1)[:, -1], x['label']]),
            "loss": Loss(criterion, output_transform=lambda x: [x['logits'], x['label']])
        }

        # 将训练指标附加到训练引擎上
        for name, metric in train_metrics.items():
            metric.attach(trainer_label, name)
        # 将验证指标附加到验证引擎上
        for name, metric in val_metrics.items():
            metric.attach(evaluator, name)

        # 定义每个epoch结束后记录训练指标的回调函数
        @trainer_label.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer_label):
            metrics = trainer_label.state.metrics
            logger.print_message('-------------------------------------------------')
            curr_lr = optimizer.param_groups[0]['lr']
            logger.print_message(f'Current learning rate: {curr_lr}')
            logger.print_message(f"Training Results - Epoch[{trainer_label.state.epoch}] ")
            logger.print_message(f"ce_loss: {metrics['ce_loss']:.4f} "
                                 f"ad_loss: {metrics['ad_loss']:.4f} "
                                 f"accuracy: {metrics['accuracy']:.4f} "
                                 f"MRIaccuracy: {metrics['MRI_accuracy']:.4f} "
                                 f"PETaccuracy: {metrics['PET_accuracy']:.4f} ")

        # 定义每个epoch结束后进行验证并记录验证结果的回调函数
        @trainer_label.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer_label):
            # 运行验证引擎，计算验证指标
            evaluator.run(val_dataloader)
            metrics = evaluator.state.metrics
            logger.print_message(f"Validation Results - Epoch[{trainer_label.state.epoch}] ")
            # 计算混淆矩阵相关的指标：敏感性、特异性和F1分数
            sen, spe, f1 = cal_confusion_metrics(metrics['confusion'])
            logger.print_message(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f} "
                                 f"sensitivity: {sen:.4f} specificity: {spe:.4f} "
                                 f"f1 score: {f1:.4f} AUC: {metrics['auc']:.4f} ")

        # 设置模型检查点保存，根据验证准确率（accuracy）来保存最佳模型
        checkpoint_saver = Checkpoint(
            {'net_model': net_model},
            save_handler=DiskSaver(save_path_fold, require_empty=False),
            n_saved=1, filename_prefix='best_label', score_name='accuracy',
            global_step_transform=global_step_from_engine(trainer_label),
            greater_or_equal=True
        )
        evaluator.add_event_handler(Events.COMPLETED, checkpoint_saver)

        # 在训练完成后，使用最佳模型在测试集上进行评估
        @trainer_label.on(Events.COMPLETED)
        def run_on_test(trainer_label):
            # 找到保存的最佳模型路径（根据文件命名规则）
            best_model_path = glob.glob(os.path.join(save_path_fold, 'best_label_net_model_*.pt'))[0]
            # 加载最佳模型的检查点
            checkpoint_all = torch.load(best_model_path, map_location=device)
            Checkpoint.load_objects(to_load={'net_model': net_model}, checkpoint=checkpoint_all)
            logger.print_message(f'Load best model {best_model_path}')
            # 移除检查点保存处理器，避免重复保存
            evaluator.remove_event_handler(checkpoint_saver, Events.COMPLETED)
            # 在测试集上运行验证引擎，计算测试指标
            evaluator.run(test_dataloader)
            metrics = evaluator.state.metrics
            logger.print_message('**************************************************************')
            logger.print_message(f"Test Results")
            sen, spe, f1 = cal_confusion_metrics(metrics['confusion'])
            logger.print_message(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f} "
                                 f"sensitivity: {sen:.4f} specificity: {spe:.4f} "
                                 f"f1 score: {f1:.4f} AUC: {metrics['auc']:.4f} ")
            logger_main.print_message_nocli(f"loss: {metrics['loss']:.4f} accuracy: {metrics['accuracy']:.4f} "
                                            f"sensitivity: {sen:.4f} specificity: {spe:.4f} "
                                            f"f1 score: {f1:.4f} AUC: {metrics['auc']:.4f} ")
            # 将当前fold的测试结果保存到状态中，供后续统计
            res_fold = [metrics['loss'], metrics['accuracy'], sen, spe, f1, metrics['auc']]
            evaluator.state.res_fold = res_fold

        # 开始运行训练引擎，训练总epoch数为stage1_epochs与stage2_epochs之和
        trainer_label.run(train_dataloader, opt.stage1_epochs + opt.stage2_epochs)

        # 返回当前fold在测试集上的评估结果
        return evaluator.state.res_fold


    # 记录每个fold的结果
    results = []
    # 使用KFold划分的索引，遍历每个fold
    for fold_idx, (train_idx, test_idx) in enumerate(kfold_splits.split(np.arange(len(ADNI_data)))):
        logger_main.print_message(f'************Fold {fold_idx}************')
        # 根据当前fold的训练和测试索引构建数据流
        train_dataloader, val_dataloader, test_dataloader, weights = setup_dataflow(train_idx, test_idx)
        # 训练模型并将结果添加到results列表中
        results.append(train_model(train_dataloader, val_dataloader, test_dataloader, fold_idx, weights))

    # 计算所有fold上各指标的均值和标准差
    results = np.array(results)
    res_mean = np.mean(results, axis=0)
    res_std = np.std(results, axis=0)
    logger_main.print_message(f'************Final Results************')
    logger_main.print_message(f'loss: {res_mean[0]:.4f} +- {res_std[0]:.4f}\n'
                              f'acc: {res_mean[1]:.4f} +- {res_std[1]:.4f}\n'
                              f'sen: {res_mean[2]:.4f} +- {res_std[2]:.4f}\n'
                              f'spe: {res_mean[3]:.4f} +- {res_std[3]:.4f}\n'
                              f'f1: {res_mean[4]:.4f} +- {res_std[4]:.4f}\n'
                              f'auc: {res_mean[5]:.4f} +- {res_std[5]:.4f}\n')
    # 输出最终使用的随机种子
    print(f'The random seed is {seed}')
