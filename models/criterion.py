import torch
import torch.nn.functional as F
from torch import nn

# 从 utils.misc 模块中导入 accuracy（计算准确率）和 get_world_size（获取进程数量，用于分布式训练等）
from utils.misc import (accuracy, get_world_size)


# ---------------------------------------------------------------
# 定义 sigmoid_focal_loss 函数：用于 RetinaNet 中的密集目标检测任务
# 该损失函数在计算二元交叉熵损失的基础上，加入了焦点机制，用于降低易分类样本的权重，
# 从而更加关注难分类样本。
# 参数说明：
#   inputs: 任意形状的浮点张量，代表每个样本的预测输出。
#   targets: 与 inputs 相同形状的浮点张量，存储每个元素的二分类标签（正类为1，负类为0）。
#   alpha: 平衡正负样本的权重因子，默认值为0.25。
#   gamma: 调制因子指数，用于平衡易分类样本和难分类样本，默认值为2。
# 返回：
#   返回计算后的损失张量。
# 输入样本形状说明：输入和目标的形状为 [bs, num_query, num_class]
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    # 通过 sigmoid 函数将 logits 转换为概率值
    prob = inputs.sigmoid()
    # 计算未加权的二元交叉熵损失，reduction="none" 表示不进行求和或均值操作
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 根据 targets 选择正负类对应的概率，计算 p_t:
    #   如果 target 为1，则 p_t = prob；如果 target 为0，则 p_t = 1 - prob
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 将交叉熵损失乘以调制因子 (1 - p_t) ** gamma，用于降低容易分类样本的权重
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 如果 alpha 非负，则为正负样本分别加权
    if alpha >= 0:
        # 为正负样本分别赋予权重：正类权重为 alpha，负类为 (1-alpha)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 对每个样本先求均值，再对所有样本求和返回最终损失
    return loss.mean(1).sum()


# ---------------------------------------------------------------
# 定义 focal_loss 类，继承自 nn.Module
# 该类实现了 focal loss 损失函数，主要用于目标检测任务中抑制背景类样本的影响
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss 损失函数实现:
            -α(1-yi)**γ * ce_loss(xi,yi)
        参数说明：
            :param alpha: 阿尔法α, 类别权重。
                          当alpha为列表时，为各类别权重，列表长度应等于 num_classes；
                          当alpha为常数时，默认将第一类（通常为背景类）的权重设为alpha，其余类别权重为1-alpha。
            :param gamma: 伽马γ, 用于调节易分类样本的权重。
            :param num_classes: 类别数量。
            :param size_average: 是否对损失取均值（True）或求和（False）。
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average

        # 如果 alpha 为列表，则对每一类赋予不同的权重，列表长度必须等于 num_classes
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # 检查列表长度是否与类别数相符
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            # 如果 alpha 为常数，则默认降低第一类（背景类）的影响
            assert alpha < 1  # alpha 应该小于1
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # 设置其他类别的权重为 1 - alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss 损失计算:
        :param preds: 预测输出，形状为 [B, N, C] 或 [B, C]，
                      分别对应检测任务（B: 批次, N: 检测框数, C: 类别数）或分类任务（B: 批次, C: 类别数）。
        :param labels: 真实标签，形状为 [B, N] 或 [B]。
        :return: 计算后的 focal loss 损失值。
        """
        # 将预测结果 reshape 成二维张量，形状为 [样本总数, 类别数]
        preds = preds.view(-1, preds.size(-1))
        # 将 alpha 转移到与 preds 相同的设备上（CPU或GPU）
        self.alpha = self.alpha.to(preds.device)
        # 对预测结果做 softmax，得到类别概率
        preds_softmax = F.softmax(preds, dim=1)
        # 计算 log softmax
        preds_logsoft = torch.log(preds_softmax)
        # 根据真实标签选择对应类别的概率和 log 概率，gather 用于从每一行中选取对应标签的位置
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        # 根据真实标签获取对应的类别权重
        self.alpha = self.alpha.gather(0, labels.view(-1))
        # 计算 focal loss 的基本公式： - (1 - p) ** gamma * log(p)
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        # 将损失乘以类别权重
        loss = torch.mul(self.alpha, loss.t())
        # 根据 size_average 决定取均值或求和
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# ---------------------------------------------------------------
# 定义 Criterion 类，用于构造最终的损失计算模块
# 该模块将 focal_loss 和交叉熵损失相结合，用于分类任务
class Criterion(nn.Module):

    def __init__(self, args):
        """
        构造损失准则（Criterion）。
        参数说明：
            num_classes: 目标类别数（不包括特殊的无目标类别）。
            matcher: 用于计算目标和候选框匹配关系的模块（此处未使用）。
            weight_dict: 字典，包含损失名称及其相对权重（此处未使用）。
            losses: 要应用的所有损失列表（参见 get_loss 获取可用损失列表）。
            focal_alpha: Focal Loss 中的 alpha 参数。
        """
        super().__init__()
        self.num_classes = args.num_classes
        # 实例化 focal_loss 损失函数，传入目标类别数
        self.loss = focal_loss(num_classes=args.num_classes)

    def forward(self, outputs, targets):
        """
        前向传播计算分类损失（NLL）。
        参数说明：
            outputs: 模型预测输出。
            targets: 真实标签。目标字典中必须包含 key "instance_labels"，其对应一个形状为 [nb_target_boxes] 的张量。
        """
        # 定义交叉熵损失函数
        ce = nn.CrossEntropyLoss()
        # 计算交叉熵损失
        ce_loss = ce(outputs, targets)
        # 总损失为 focal_loss 和交叉熵损失之和
        loss = self.loss(outputs, targets) + ce_loss
        print(targets)
        try:
            # 计算并打印准确率（accuracy 返回的是百分比，这里除以100）
            print('loss_ce: ', loss, 'accuracy: ', (accuracy(outputs, targets)[0]) / 100.0)
        except:
            import pdb;
            pdb.set_trace()
        return loss
