'''FT_Transformer模块实现'''

import torch  # 导入 PyTorch 库，提供深度学习框架的核心功能
import torch.nn.functional as F  # 导入 PyTorch 中的功能性函数，如激活函数、损失函数等
from torch import nn, einsum  # 导入 PyTorch 的 nn 模块和 einsum 函数（用于高效的张量操作）

from einops import rearrange, repeat  # 导入 einops 库，用于处理张量的重排和重复操作

# feedforward and attention

class GEGLU(nn.Module):  # 定义 GEGLU 激活函数类
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)  # 将输入 x 按最后一个维度拆分成两个部分，一个为 x，一个为 gates
        return x * F.gelu(gates)  # 对 gates 应用 GELU 激活函数并与 x 相乘

def FeedForward(dim, mult = 4, dropout = 0.):  # 定义前馈神经网络，参数为输入维度 dim、扩展因子 mult 和 dropout 比例
    return nn.Sequential(
        nn.LayerNorm(dim),  # 对输入进行层归一化
        nn.Linear(dim, dim * mult * 2),  # 将输入映射到一个更大的空间
        GEGLU(),  # 使用 GEGLU 激活函数
        nn.Dropout(dropout),  # 添加 dropout 以防止过拟合
        nn.Linear(dim * mult, dim)  # 将映射后的空间再次映射回原始维度
    )

class Attention(nn.Module):  # 定义注意力机制类
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads  # 内部维度等于头数 * 每个头的维度
        self.heads = heads  # 设置头数
        self.scale = dim_head ** -0.5  # 缩放因子，通常是每个头的维度的倒数的平方根

        self.norm = nn.LayerNorm(dim)  # 层归一化

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 输入到 Q、K、V 的线性变换
        self.to_out = nn.Linear(inner_dim, dim, bias = False)  # 输出的线性变换

        self.dropout = nn.Dropout(dropout)  # 添加 dropout

    def forward(self, x):  # 定义前向传播
        h = self.heads  # 头数

        x = self.norm(x)  # 对输入进行归一化

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)  # 将 Q、K、V 从输入中分割出来
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))  # 重排张量，使得每个头的数据分开
        q = q * self.scale  # 对 Q 进行缩放

        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # 计算 Q 和 K 的相似度（点积）

        attn = sim.softmax(dim = -1)  # 计算注意力权重
        dropped_attn = self.dropout(attn)  # 应用 dropout

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)  # 将注意力权重与 V 相乘
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)  # 重排输出张量
        out = self.to_out(out)  # 通过输出层进行线性变换

        return out, attn  # 返回输出和注意力权重

# transformer

class Transformer(nn.Module):  # 定义 Transformer 模型
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])  # 创建一个空的层列表

        for _ in range(depth):  # 遍历深度，添加每一层的注意力和前馈网络
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
            ]))

    def forward(self, x, return_attn = False):  # 前向传播
        post_softmax_attns = []  # 用于保存每一层的注意力权重

        for attn, ff in self.layers:  # 遍历每一层
            attn_out, post_softmax_attn = attn(x)  # 计算注意力输出和注意力权重
            post_softmax_attns.append(post_softmax_attn)  # 保存注意力权重

            x = attn_out + x  # 残差连接
            x = ff(x) + x  # 前馈网络的残差连接

        if not return_attn:
            return x  # 返回最终输出

        return x, torch.stack(post_softmax_attns)  # 返回输出和所有注意力权重

# numerical embedder

class NumericalEmbedder(nn.Module):  # 定义数值嵌入器，用于处理连续的数值特征
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))  # 初始化权重参数
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))  # 初始化偏置参数

    def forward(self, x):  # 前向传播
        x = rearrange(x, 'b n -> b n 1')  # 将输入张量重排
        return x * self.weights + self.biases  # 计算嵌入结果

# main class

class FTTransformer(nn.Module):  # 定义最终的 FTTransformer 类
    def __init__(self, *, categories, num_continuous, dim, depth, heads, dim_head = 16, dim_out = 1, num_special_tokens = 2, attn_dropout = 0., ff_dropout = 0.):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'  # 确保每个类别的数量大于 0
        assert len(categories) + num_continuous > 0, 'input shape must not be null'  # 确保输入的类别和连续特征的数量大于 0

        # categories related calculations
        self.num_categories = len(categories)  # 类别的数量
        self.num_unique_categories = sum(categories)  # 所有类别的总数

        self.num_special_tokens = num_special_tokens  # 特殊 token 的数量
        total_tokens = self.num_unique_categories + num_special_tokens  # 总 token 数量

        # 创建类别嵌入表
        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)  # 为类别索引添加偏移量
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]  # 计算累积偏移量
            self.register_buffer('categories_offset', categories_offset)  # 将偏移量注册为缓冲区，避免作为参数更新

            self.categorical_embeds = nn.Embedding(total_tokens, dim)  # 创建类别嵌入层

        # continuous
        self.num_continuous = num_continuous  # 连续特征的数量

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)  # 创建数值嵌入器

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 定义 CLS token

        # transformer
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )  # 创建 Transformer 层

        # to logits
        self.to_logits = nn.Sequential(  # 最终的线性层，用于生成输出
            nn.LayerNorm(dim),  # 对输出进行归一化
            nn.ReLU(),  # 激活函数
            nn.Linear(dim, dim_out)  # 输出层，维度为 dim_out
        )

    def forward(self, x_categ, x_numer, return_attn = False):
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'  # 确保类别输入的形状正确

        xs = []
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset  # 对类别输入应用偏移量
            x_categ = self.categorical_embeds(x_categ)  # 获取类别嵌入
            xs.append(x_categ)  # 将类别嵌入添加到输入数据中

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)  # 获取数值特征的嵌入
            xs.append(x_numer)  # 将数值嵌入添加到输入数据中

        # concat categorical and numerical
        x = torch.cat(xs, dim = 1)  # 拼接类别嵌入和数值嵌入

        # append cls tokens
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  # 扩展 CLS token 使其匹配批量大小
        x = torch.cat((cls_tokens, x), dim = 1)  # 将 CLS token 添加到输入数据的开头

        # attend
        x, attns = self.transformer(x, return_attn = True)  # 通过 Transformer 进行处理

        # get cls token
        x = x[:, 0]  # 获取 CLS token 的输出

        # out in the paper is linear(relu(ln(cls)))
        logits = self.to_logits(x)  # 通过线性层得到最终输出

        if not return_attn:
            return logits  # 如果不需要注意力权重，返回 logits

        return logits, attns  # 如果需要注意力权重，返回 logits 和注意力权重
