import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch.nn import Parameter


# ResNet3D 实现
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch.nn import Parameter

# 超图注意力实现
class HypergraphAttention(MessagePassing):
    def __init__(self, in_channels, out_channels, use_attention=True, heads=1, dropout=0.1):
        super(HypergraphAttention, self).__init__(aggr='add')  # 使用加法聚合
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.dropout = dropout
        self.heads = heads

        # 定义卷积权重和注意力参数
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重和注意力参数
        nn.init.xavier_uniform_(self.weight)
        if self.use_attention:
            nn.init.xavier_uniform_(self.att)

    def message(self, x_j, edge_index_i, norm, alpha):
        # 计算每条边的消息，并加权
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def forward(self, x, hyperedge_index, adj, hyperedge_weight=None):
        # 计算节点度数
        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        B = 1.0 / degree(hyperedge_index[1], x.size(0), x.dtype)
        B[B == float("inf")] = 0

        # 注意力机制：为每一条超边计算重要性
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B)  # 从源节点到目标节点的传播
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D)  # 从目标节点到源节点的传播

        return out  # 返回聚合后的节点特征


# 多头注意力实现
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.fc = nn.Linear(d_v * n_head, d_model)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = q.view(sz_b, len_q, self.n_head, self.d_k)
        k = k.view(sz_b, len_k, self.n_head, self.d_k)
        v = v.view(sz_b, len_v, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # 转置为方便计算的维度
        output, attn = self.attention(q, k, v, mask=mask)  # 使用点积注意力机制
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # 重塑输出
        output = self.dropout(self.fc(output))  # 全连接层变换
        output += residual  # 残差连接
        output = self.layer_norm(output)  # 层归一化
        return output, attn  # 返回最终输出和注意力权重


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        # 计算缩放点积注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # 对mask为0的部分进行填充
        attn = torch.softmax(attn, dim=-1)  # 归一化为概率分布
        attn = self.attn_dropout(attn)  # dropout
        output = torch.matmul(attn, v)  # 加权求和
        return output, attn  # 返回加权后的输出和注意力矩阵



# 融合模型
class BrainModel(nn.Module):
    def __init__(self, configs):
        super(BrainModel, self).__init__()

        # ResNet3D特征提取
        self.resnet3d = ResNet3D(in_channels=configs.in_channels, num_classes=configs.num_classes)

        # 通道交换层（浅层特征融合）
        self.channel_swap = nn.ModuleList(
            [nn.Conv3d(configs.in_channels, configs.out_channels, kernel_size=3, padding=1) for _ in
             range(configs.num_channels)])

        # 超图卷积（深度特征融合）
        self.hypergraph_conv1 = HypergraphConv(configs.enc_in, configs.d_model, use_attention=True, heads=configs.heads,
                                               dropout=configs.dropout)
        self.hypergraph_conv2 = HypergraphConv(configs.d_model, configs.d_model, use_attention=True,
                                               heads=configs.heads, dropout=configs.dropout)

        # 超图注意力机制
        self.hypergraph_attention = MultiHeadAttention(n_head=configs.n_heads, d_model=configs.d_model, d_k=configs.d_k,
                                                       d_v=configs.d_v, dropout=configs.dropout)

        # 最终预测的全连接层
        self.fc = nn.Linear(configs.d_model, configs.num_classes)

    def forward(self, x, adj, mask):
        # Step 1: 使用ResNet3D提取特征
        resnet_features = self.resnet3d(x)  # [batch_size, channels, depth, height, width]

        # Step 2: 通过通道交换进行浅层特征融合
        fused_features = torch.zeros_like(resnet_features)
        for i in range(fused_features.shape[1]):  # 遍历所有通道
            fused_features[:, i, :, :, :] = self.channel_swap[i](resnet_features[:, i, :, :, :])

        # Step 3: 使用超图卷积进行深度特征融合（第一层）
        hypergraph_features1 = self.hypergraph_conv1(fused_features, adj)

        # Step 4: 使用超图卷积进行深度特征融合（第二层）
        hypergraph_features2 = self.hypergraph_conv2(hypergraph_features1, adj)

        # Step 5: 使用超图注意力机制进一步增强特征
        attention_output, attn_weights = self.hypergraph_attention(hypergraph_features2, hypergraph_features2,
                                                                   hypergraph_features2, mask)

        # Step 6: 最终的全连接层预测
        output = self.fc(attention_output.mean(dim=[2, 3, 4]))  # 对空间维度进行池化，输出最终预测结果

        return output, attn_weights
