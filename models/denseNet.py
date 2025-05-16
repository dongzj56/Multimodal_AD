import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedDenseBottleneck(nn.Module):
    def __init__(self, inplanes, growthRate=32, dropRate=0,
                 kernel_size=3, dilation=1, expansion=4):
        super(DilatedDenseBottleneck, self).__init__()
        planes = expansion * growthRate

        # 1×1 变维
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        # 膨胀深度可分离卷积
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                                   padding=padding, groups=planes,
                                   bias=False, dilation=dilation)
        self.pointwise = nn.Conv2d(planes, growthRate, kernel_size=1, bias=False)

        self.dropRate = dropRate

    def forward(self, x):
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn1(out))  # 再次 BN+ReLU
        out = self.depthwise(out)
        out = self.pointwise(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        # 融合新旧特征
        return torch.cat([x, out], dim=1)


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn(x))
        x = self.conv(x)
        return self.pool(x)


class DilatedDenseNet(nn.Module):
    def __init__(self,
                 growthRate=32,
                 dropRate=0,
                 compression=2,
                 layers=(6, 12, 24, 16),
                 num_classes=4,
                 kernel_sizes=(3, 3, 3, 3),
                 dilations=(1, 1, 2, 3),
                 large_kernel_head=True):
        super(DilatedDenseNet, self).__init__()

        # 初始通道数
        self.inplanes = growthRate * 2
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Head
        if large_kernel_head:
            self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, growthRate, 3, 2, 1, bias=False),
                nn.BatchNorm2d(growthRate), nn.ReLU(inplace=True),
                nn.Conv2d(growthRate, growthRate, 3, 1, 1, bias=False),
                nn.BatchNorm2d(growthRate), nn.ReLU(inplace=True),
                nn.Conv2d(growthRate, self.inplanes, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True),
            )

        # Dense-Block + Transition
        self.blocks = nn.ModuleList()
        for idx, num in enumerate(layers):
            # Dense block
            block = self._make_block(num, kernel_sizes[idx], dilations[idx], dropRate)
            self.blocks.append(block)
            # Transition（最后一个 block 后不加）
            if idx != len(layers) - 1:
                trans_planes = self.inplanes // compression
                self.blocks.append(Transition(self.inplanes, trans_planes))
                self.inplanes = trans_planes

        # 分类头
        self.bn_last = nn.BatchNorm2d(self.inplanes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.0001); m.bias.data.zero_()

    def _make_block(self, num_layers, ksize, dilation, dropRate):
        layers = []
        for _ in range(num_layers):
            layers.append(DilatedDenseBottleneck(
                self.inplanes,
                growthRate=self.inplanes // 4,
                dropRate=dropRate,
                kernel_size=ksize,
                dilation=dilation
            ))
            self.inplanes += self.inplanes // 4
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))) if hasattr(self, 'bn1') else self.conv1(x)
        x = self.maxpool(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.relu(self.bn_last(x))
        x = self.avgpool(x).view(x.size(0), -1)
        return self.fc(x)
