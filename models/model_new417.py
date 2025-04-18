import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch.nn import Parameter

class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet3D, self).__init__()
        self.conv1a = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(32)
        self.conv1b = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(32)
        self.conv1c = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.voxres2 = self._make_voxres_block(64, 64)
        self.voxres3 = self._make_voxres_block(64, 64)
        self.bn4 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, stride=2, padding=1)
        self.voxres5 = self._make_voxres_block(64, 64)
        self.voxres6 = self._make_voxres_block(64, 64)
        self.pool10 = nn.AdaptiveAvgPool3d(1)
        self.fc11 = nn.Linear(64, 128)
        self.prob = nn.Linear(128, num_classes)

    def _make_voxres_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.conv1c(x)
        x = self.voxres2(x)
        x = self.voxres3(x)
        x = F.relu(self.bn4(x))
        x = self.conv4(x)
        x = self.voxres5(x)
        x = self.voxres6(x)
        x = self.pool10(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc11(x))
        x = self.prob(x)
        return x


# 超图卷积实现
class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, use_attention=True, heads=1, dropout=0.1):
        super(HypergraphConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.dropout = dropout
        self.heads = heads

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.use_attention:
            nn.init.xavier_uniform_(self.att)

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def forward(self, x, hyperedge_index, adj, hyperedge_weight=None):
        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        B = 1.0 / degree(hyperedge_index[1], x.size(0), x.dtype)
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D)
        return out