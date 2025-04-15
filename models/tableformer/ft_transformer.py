import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# GEGLU Activation Function (Gated GELU)
class GEGLU(nn.Module):
    def forward(self, x):
        # 将输入切分成两部分，第一部分作为输入，第二部分作为门控值（gates）
        x, gates = x.chunk(2, dim=-1)
        # 对门控值应用 GELU 激活函数，且与输入值进行逐元素相乘
        return x * F.gelu(gates)

# FeedForward Neural Network Layer
def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.LayerNorm(dim),  # 对输入进行层归一化
        nn.Linear(dim, dim * mult * 2),  # 将输入映射到更高维度
        GEGLU(),  # 激活函数
        nn.Dropout(dropout),  # Dropout 防止过拟合
        nn.Linear(dim * mult, dim)  # 映射回原始维度
    )

# Multi-Head Self Attention Mechanism
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 内部维度
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子

        self.norm = nn.LayerNorm(dim)  # 对输入进行归一化

        # 将输入通过线性层映射到 Q, K, V
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)  # Dropout 防止过拟合

    def forward(self, x):
        h = self.heads

        x = self.norm(x)  # 对输入进行归一化

        # 将输入切分为查询（Q）、键（K）和值（V）
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # 重新排列 Q, K, V 为多头注意力格式
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale  # 缩放 Q

        # 计算 Q 和 K 的相似度（点积注意力）
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # 对相似度应用 softmax 归一化得到注意力权重
        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)  # Dropout

        # 使用注意力权重加权 V
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)  # 合并多个头的输出
        out = self.to_out(out)  # 映射回原始维度

        return out, attn  # 返回注意力输出和注意力权重

# Transformer 模块
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        # 堆叠多个 Transformer 层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
                FeedForward(dim, dropout=ff_dropout),
            ]))

    def forward(self, x, return_attn=False):
        post_softmax_attns = []

        # 逐层传递输入
        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x  # 残差连接
            x = ff(x) + x  # FeedForward 层加残差连接

        if not return_attn:
            return x  # 如果不需要返回注意力权重，只返回输出

        return x, torch.stack(post_softmax_attns)  # 如果需要返回注意力权重，返回输出和权重

# 数值型数据的嵌入器
class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))  # 权重参数
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))  # 偏置参数

    def forward(self, x):
        # 将输入的数值数据重新排列并与权重、偏置进行逐元素相乘加和
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# 主模型类 FTTransformer
class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,  # 类别数据的数量
        num_continuous,  # 连续型数值数据的数量
        dim,  # 特征维度
        depth,  # Transformer 层数
        heads,  # 注意力头数
        dim_head=16,  # 每个注意力头的维度
        dim_out=1,  # 输出维度
        num_special_tokens=2,  # 特殊 token 的数量
        attn_dropout=0.,  # 注意力层的 dropout 比例
        ff_dropout=0.  # FeedForward 层的 dropout 比例
    ):
        super().__init__()

        # 检查输入类别数量是否合法
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # 计算类别相关信息
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # 类别嵌入表
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        if self.num_unique_categories > 0:
            # 生成类别偏移量，使得每个类别的 ID 对应到嵌入表的正确位置
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # 类别嵌入层
            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # 连续型数值数据处理
        self.num_continuous = num_continuous
        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类 token

        # Transformer 层
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # 输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim_out)
        )

    def forward(self, x_categ, x_numer, return_attn=False):
        # 检查类别数据的维度
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        if self.num_unique_categories > 0:
            # 对类别数据进行偏移和嵌入
            x_categ = x_categ + self.categories_offset
            x_categ = self.categorical_embeds(x_categ)
            xs.append(x_categ)

        # 对连续数据进行嵌入
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)
            xs.append(x_numer)

        # 将类别和数值数据拼接在一起
        x = torch.cat(xs, dim=1)

        # 添加 cls token 到数据开头
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 通过 Transformer 层处理数据
        x, attns = self.transformer(x, return_attn=True)

        # 获取 cls token 输出作为最终表示
        x = x[:, 0]

        # 通过线性层进行映射得到最终输出
        logits = self.to_logits(x)

        if not return_attn:
            return logits

        return logits, attns  # 返回输出和注意力权重
