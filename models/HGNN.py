import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# 1) 多尺度 k-NN 超图构建（稠密 H）
# ---------------------------------------------------------
class HyperGraphIncidenceBuilder:
    """
    ks = {0: (k_loc, k_glob), 1: (k_loc, k_glob)}
    """
    def __init__(self, ks):
        self.ks = ks            # dict[int, tuple]

    @torch.no_grad()
    def _build_modal_edges(self, feat, k_set):
        """
        根据余弦相似度为单模态生成多尺度超边
        feat  : [N, C]
        k_set : (k_loc, k_glob)
        return: H_modal [N, E_modal]
        """
        N = feat.size(0)
        sim = F.normalize(feat, dim=1) @ F.normalize(feat, dim=1).T  # [N,N]
        H_list = []

        for k in k_set:                          # 对每档 k 建一批超边
            _, knn = sim.topk(k + 1, dim=-1)     # 含自身
            H = torch.zeros(N, N, device=feat.device)
            for e_idx, nbrs in enumerate(knn):   # N 条超边
                H[nbrs, e_idx] = 1.
            H_list.append(H)

        return torch.cat(H_list, dim=1)          # [N, N*len(k_set)]

    @torch.no_grad()
    def __call__(self, feats):
        """
        feats : list[Tensor]  (len=2) 各为 [N,C]
        return: H_dense [2N, E_total]
        """
        H_modal = []
        for midx, feat in enumerate(feats):
            H_modal.append(self._build_modal_edges(feat, self.ks[midx]))

        # —— 模态间二元超边 —— #
        N = feats[0].size(0)
        H_inter = torch.zeros(2 * N, N, device=feats[0].device)
        for r in range(N):
            H_inter[r, r]       = 1.   # mod1-节点 r
            H_inter[N + r, r]   = 1.   # mod2-节点 r

        # 拼接：H = [H_mod1 | H_mod2 | H_inter]
        H = torch.cat([
            torch.block_diag(*H_modal),   # block diag 拼出 [2N, ΣE_m]
            H_inter                       # [2N, N]
        ], dim=1)
        return H                         # 稠密矩阵


# ---------------------------------------------------------
# 2) 稠密版 HyperGraph Convolution
# ---------------------------------------------------------
class HyperGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, H):
        """
        x : [B, 2N, F]   H : [2N, E]
        """
        x = self.lin(x)                          # 线性映射
        DV = H.sum(dim=1).clamp(min=1.)          # 节点度 [2N]
        DE = H.sum(dim=0).clamp(min=1.)          # 超边度 [E]

        H_norm = (H / DV.sqrt().unsqueeze(1)) / DE.unsqueeze(0)
        A = H_norm @ H_norm.t()                  # [2N,2N]
        x = torch.bmm(A.expand(x.size(0), -1, -1), x)
        return F.relu(x)


# ---------------------------------------------------------
# 3) 通用双模态超图融合模块
# ---------------------------------------------------------
class DualModalHyperGraph(nn.Module):
    """
    参数
    ----
    in_dim      输入特征维度  C
    hidden_dim  超图卷积输出维度
    num_layers  HGNNConv 层数
    ks          {0:(k_loc,k_glob), 1:(k_loc,k_glob)}
    """
    def __init__(self,
                 in_dim=64,
                 hidden_dim=128,
                 num_layers=2,
                 ks={0: (6, 18), 1: (4, 12)}):
        super().__init__()
        self.builder = HyperGraphIncidenceBuilder(ks)
        self.layers = nn.ModuleList([
            HyperGraphConv(in_dim if l == 0 else hidden_dim, hidden_dim)
            for l in range(num_layers)
        ])

    def forward(self, feat_mod1, feat_mod2):
        """
        feat_mod1 / feat_mod2 : [B, N, C]
        """
        B, N, _ = feat_mod1.shape
        H = self.builder([feat_mod1.mean(0), feat_mod2.mean(0)])   # [2N,E]

        x = torch.cat([feat_mod1, feat_mod2], dim=1)               # [B,2N,C]
        for layer in self.layers:
            x = layer(x, H)

        out_mod1, out_mod2 = x[:, :N, :], x[:, N:, :]
        return out_mod1, out_mod2            # [B,N,hidden_dim] × 2

class CrossModalEdgeAttention(nn.Module):
    """
    在 1-对-1 二元超边 (mod1_r, mod2_r) 内做 QK^T/√d → softmax → V
    输入:
        x1, x2 : [B, N, H] (两模态结点嵌入)
    输出:
        z1, z2 : [B, N, H] (attention 更新后结点),  α : [B, N] (权重)
    """
    def __init__(self, dim, dk=64):
        super().__init__()
        self.Wq = nn.Linear(dim, dk, bias=False)
        self.Wk = nn.Linear(dim, dk, bias=False)
        self.Wv = nn.Linear(dim, dk, bias=False)   # 可设 dk=dim
        self.scale = dk ** -0.5

    def forward(self, x1, x2):
        # Q 来自模态-1, K/V 来自模态-2
        Q = self.Wq(x1)                     # [B,N,dk]
        K = self.Wk(x2)
        V = self.Wv(x2)

        # 1-对-1 attention：实质是逐脑区 scalar
        score = (Q * K).sum(dim=-1) * self.scale   # [B,N]
        α12   = torch.softmax(score, dim=-1)       # [B,N]

        # 更新结点
        z1 = α12.unsqueeze(-1) * V                 # [B,N,dk]

        # 交换顺序再算一次得到 z2
        score21 = (self.Wq(x2) * self.Wk(x1)).sum(dim=-1) * self.scale
        α21     = torch.softmax(score21, dim=-1)
        z2 = α21.unsqueeze(-1) * self.Wv(x1)

        return z1, z2, α12, α21

class DualModalHyperGraphWithAttn(nn.Module):
    def __init__(self,
                 in_dim=64,
                 hidden_dim=128,
                 num_layers=2,
                 ks={0:(6,18),1:(4,12)},
                 dk=64):
        super().__init__()
        self.hg = DualModalHyperGraph(in_dim, hidden_dim, num_layers, ks)
        self.attn = CrossModalEdgeAttention(hidden_dim, dk)

        # 聚合为全局向量
        self.pool = nn.Linear(dk, dk)      # 简单线性 + tanh
        self.act  = nn.Tanh()

    def forward(self, feat1, feat2):
        o1, o2 = self.hg(feat1, feat2)     # [B,N,H]

        z1, z2, α12, α21 = self.attn(o1, o2)   # [B,N,dk], [B,N]

        # 全局聚合：∑ α_i · z_i
        g1 = (α12.unsqueeze(-1) * z1).sum(dim=1)   # [B,dk]
        g2 = (α21.unsqueeze(-1) * z2).sum(dim=1)   # [B,dk]

        g1 = self.act(self.pool(g1))               # 非线性
        g2 = self.act(self.pool(g2))

        return {
            "node_mod1": z1, "node_mod2": z2,      # 更新后结点特征
            "global_mod1": g1, "global_mod2": g2,  # 融合后全局向量
            "alpha_m1": α12, "alpha_m2": α21       # 注意力权重
        }
