import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet3D, self).__init__()
        # 初始卷积块 (对应Lasagne的Block 1)
        self.conv1a = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(32)
        self.conv1b = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(32)
        self.conv1c = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)

        # VoxRes模块
        self.voxres2 = self._make_voxres_block(64, 64)
        self.voxres3 = self._make_voxres_block(64, 64)
        self.bn4 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)

        self.voxres5 = self._make_voxres_block(64, 64)
        self.voxres6 = self._make_voxres_block(64, 64)

        # 最终分类层
        self.pool10 = nn.AdaptiveAvgPool3d(1)
        self.fc11 = nn.Linear(64, 128)
        self.prob = nn.Linear(128, num_classes)

        # 初始化参数
        self._initialize_weights()

    def _make_voxres_block(self, in_channels, out_channels):
        return nn.Sequential(
            VoxResModule(in_channels, out_channels)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.conv1c(x)

        # VoxRes模块
        x = self.voxres2(x)
        x = self.voxres3(x)
        x = F.relu(self.bn4(x))
        x = self.conv4(x)

        x = self.voxres5(x)
        x = self.voxres6(x)

        # 分类头
        x = self.pool10(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc11(x))
        x = self.prob(x)
        return x


class VoxResModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)  # 相加后应用ReLU
        return out