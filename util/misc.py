# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.
Mostly copy-paste from torchvision references.
"""

import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

# 导入 torchvision，并根据版本做兼容性处理
import torchvision

if float(torchvision.__version__[:3]) < 0.5:
    import math


    # 针对 torchvision 版本低于 0.5 的情况，定义辅助函数用于尺寸和缩放因子的检查与计算

    def _check_size_scale_factor(dim, size, scale_factor):
        # 检查 size 和 scale_factor 参数是否合法
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if not (scale_factor is not None and len(scale_factor) != dim):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )


    def _output_size(dim, input, size, scale_factor):
        # 根据 size 或 scale_factor 计算输出尺寸
        assert dim == 2
        _check_size_scale_factor(dim, size, scale_factor)
        if size is not None:
            return size
        # 如果未提供 size，则根据 scale_factor 计算输出尺寸
        assert scale_factor is not None and isinstance(scale_factor, (int, float))
        scale_factors = [scale_factor, scale_factor]
        # 使用 math.floor 计算每个维度的新尺寸
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]
elif float(torchvision.__version__[:3]) < 0.7:
    # 对于 torchvision 版本低于 0.7 的情况，导入 _new_empty_tensor 和 _output_size 函数
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


# =============================================================================
# 类 SmoothedValue：用于跟踪一系列数值，并计算滑动窗口内或全局的平均值、中位数等
# =============================================================================
class SmoothedValue(object):
    """
    跟踪一系列数值，并提供滑动窗口统计（中位数、平均值、全局平均值等）。
    """

    def __init__(self, window_size=20, fmt=None):
        # 默认格式化字符串，显示中位数和全局平均值
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # 用 deque 保存最近 window_size 个数值
        self.total = 0.0  # 全局累计和
        self.count = 0  # 数值计数
        self.fmt = fmt

    def update(self, value, n=1):
        # 更新数值，将 value 添加到 deque 中，并更新全局累计和和计数
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        在分布式训练中，同步不同进程间的累计值和计数
        注意：该方法不会同步 deque 中的数值
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()  # 同步所有进程
        dist.all_reduce(t)  # 聚合所有进程中的计数和总和
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        # 计算当前窗口内的中位数
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        # 计算当前窗口内的平均值
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        # 计算全局平均值（累计和除以计数）
        return self.total / self.count

    @property
    def max(self):
        # 返回当前窗口内的最大值
        return max(self.deque)

    @property
    def value(self):
        # 返回最新的数值
        return self.deque[-1]

    def __str__(self):
        # 使用格式化字符串输出统计信息
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


# =============================================================================
# 函数 all_gather：在分布式环境下收集所有进程的任意 picklable 数据
# =============================================================================
def all_gather(data):
    """
    对任意可 pickle 的数据进行 all_gather 操作。
    参数:
        datasets: 任意可 pickle 的对象
    返回:
        list[datasets]: 来自每个进程收集到的数据列表
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 将数据 pickle 序列化后转为 ByteTensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # 获取当前进程 tensor 的大小
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 为了支持不同形状的 tensor，将它们 padding 到相同大小
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    # 反序列化各进程收集到的数据
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


# =============================================================================
# 函数 reduce_dict：在分布式环境下对字典中的所有值进行归约（求和或平均）
# =============================================================================
def reduce_dict(input_dict, average=True):
    """
    参数:
        input_dict (dict): 要归约的字典，所有值都是 tensor
        average (bool): 是否取平均（True）或求和（False）
    将所有进程中的字典值归约后，返回与 input_dict 具有相同键的新字典。
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # 按照键名排序，确保所有进程顺序一致
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


# =============================================================================
# 类 MetricLogger：用于记录和输出训练过程中的各项指标
# =============================================================================
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        # 使用 defaultdict 保存各指标的 SmoothedValue 对象
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        # 更新指定指标的数值
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        # 如果属性存在于 meters 中则返回，否则检查 __dict__
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        # 将所有指标格式化为字符串进行输出
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        # 同步所有指标在各进程间的统计信息
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        # 添加自定义指标
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        遍历 iterable，每 print_freq 次输出一次日志信息。
        同时计算迭代时间、数据加载时间和估计剩余时间等。
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'datasets: {datasets}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'datasets: {datasets}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# =============================================================================
# 函数 get_sha：获取当前代码仓库的 git 版本信息（commit sha、分支和是否有未提交的修改）
# =============================================================================
def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


# =============================================================================
# 函数 collate_fn：用于数据加载时的样本整合（collate），例如在 DataLoader 中使用
# =============================================================================
def collate_fn(batch):
    # 将 batch 中的元素按位置解压（transpose），例如 [ (img1, label1), (img2, label2) ] -> ([img1, img2], [label1, label2] )
    batch = list(zip(*batch))

    # 对 batch[0]（例如图像）进行拼接操作
    # 注释掉的部分表示原本可能会使用 nested_tensor_from_tensor_list 来处理不同尺寸的情况
    batch[0] = torch.cat(batch[0])
    return tuple(batch)


# =============================================================================
# 辅助函数 _max_by_axis：对列表中每个子列表的对应位置取最大值
# =============================================================================
def _max_by_axis(the_list):
    # the_list: List[List[int]]，返回每一列的最大值
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# =============================================================================
# 函数 setup_for_distributed：在非主进程中禁用 print 输出，避免重复打印日志
# =============================================================================
def setup_for_distributed(is_master):
    """
    如果当前进程不是主进程，则重写内置 print 函数使其不输出内容（除非显式指定 force=True）
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


# =============================================================================
# 分布式环境判断及获取相关信息的辅助函数
# =============================================================================
def is_dist_avail_and_initialized():
    # 判断 torch.distributed 是否可用且已经初始化
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    # 获取所有进程的数量
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    # 获取当前进程的全局编号（rank）
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    # 获取本地（节点）进程数，从环境变量中获取
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    # 获取当前进程在本地节点中的编号，从环境变量中获取
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    # 判断当前是否为主进程（全局 rank 为 0）
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    # 仅在主进程上执行保存操作，避免重复保存
    if is_main_process():
        torch.save(*args, **kwargs)


# =============================================================================
# 函数 init_distributed_mode：根据环境变量初始化分布式训练
# =============================================================================
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 通过环境变量设置 rank、world_size、gpu 以及 dist_url
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        # 针对 SLURM 作业调度系统进行设置
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # 初始化进程组
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()  # 同步所有进程
    setup_for_distributed(args.rank == 0)


# =============================================================================
# 函数 accuracy：计算 top-k 准确率
# =============================================================================
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    计算指定 topk 的准确率。
    参数:
        output: 模型预测输出
        target: 真实标签
        topk: 元组，指定需要计算的 top-k 值（例如 (1,) 或 (1, 5)）
    返回:
        列表，包含各 top-k 的准确率（百分比形式）
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    # 取出每个样本预测分数最高的 maxk 个类别
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # 判断预测是否与真实标签匹配
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # 统计 top-k 中正确的个数，并计算准确率
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# =============================================================================
# 函数 interpolate：等价于 nn.functional.interpolate，但支持空 batch 的情况
# =============================================================================
def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    与 nn.functional.interpolate 功能相同，但在 batch 为空时能正确处理。
    在未来 PyTorch 原生支持后，此函数可被废弃。
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        # 针对旧版本，创建一个空 Tensor
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


# =============================================================================
# 函数 get_total_grad_norm：计算所有参数梯度的总范数
# =============================================================================
def get_total_grad_norm(parameters, norm_type=2):
    # 过滤掉没有梯度的参数
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    # 分别计算每个参数梯度的范数，然后再计算总范数
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


# =============================================================================
# 函数 inverse_sigmoid：计算 sigmoid 函数的逆变换（logit函数）
# =============================================================================
def inverse_sigmoid(x, eps=1e-5):
    # 对 x 限制在 [0, 1] 区间内，并加 eps 防止出现数值问题
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
