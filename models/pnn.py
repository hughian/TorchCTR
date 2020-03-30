import torch
import torch.nn as nn
from CTR.core.core import MLP, GroupEmbedding, Prediction, LinearModel

__all__ = ['PNN']


class InnerProduct(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, require_logits=True):
        super(InnerProduct, self).__init__()
        self.feature_metas = feature_metas
        self.require_logits = require_logits
        self.inner_weights = nn.ModuleList(
            [nn.ParameterList([torch.nn.Parameter(torch.randn(embedding_dim), requires_grad=True)
                               for j in range(len(feature_metas))])
             for i in range(len(feature_metas))])

    def forward(self, x):
        # x : size -> (batch_size, len(feature_metas), embedding_dim)
        # 所有向量两两点乘，得到长度为 field_num * (field_num - 1) // 2 的向量 p,
        # 然后将这个向量 p 与前面的线性部分 z 拼接起来送入接下来的 dnn
        z = torch.sum(x, dim=-1, keepdim=False)
        row, col = [], []
        for i in range(len(self.feature_metas) - 1):
            for j in range(i + 1, len(self.feature_metas)):
                row.append(i)
                col.append(j)

        p = torch.index_select(x, dim=1, index=torch.LongTensor(row))
        q = torch.index_select(x, dim=1, index=torch.LongTensor(col))
        # p.shape = (batch_size, 231, 8)
        if self.require_logits:
            inner = torch.sum(p * q, dim=-1, keepdim=False)
        else:
            inner = torch.flatten(p * q, start_dim=1)

        out = torch.cat([z, inner], dim=1)
        return out


class OuterProduct(nn.Module):
    def __init__(self, feature_metas, embedding_dim):
        super(OuterProduct, self).__init__()
        self.feature_metas = feature_metas
        self.kernels = nn.ParameterDict()
        for i in range(len(feature_metas) - 1):
            for j in range(i + 1, len(feature_metas)):
                kernel = nn.Parameter(torch.randn(size=(embedding_dim, embedding_dim)), requires_grad=True)
                nn.init.normal_(kernel, mean=0., std=1e-3)  # TODO init_func
                self.kernels[str(i)+'_'+str(j)] = kernel

    def forward(self, x):
        z = torch.sum(x, dim=-1, keepdim=False)
        outer = []
        for i in range(len(self.feature_metas)-1):
            for j in range(i+1, len(self.feature_metas)):
                inp_i = torch.unsqueeze(x[:, i, :], dim=1)
                inp_j = torch.unsqueeze(x[:, j, :], dim=-1)
                # print(inp_i.shape, inp_j.shape)
                kernel = self.kernels[str(i)+'_'+str(j)]
                prod = torch.sum(torch.matmul(torch.matmul(inp_i, kernel), inp_j), dim=-1, keepdim=False)
                outer.append(prod)
        outer = torch.cat(outer, dim=1)

        out = torch.cat([z, outer], dim=1)
        return out


class PNN(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(128, 64), product_mode='outer', linear_bias=False):
        super(PNN, self).__init__()
        # mode : 'inner' or 'outer'
        self.feature_metas = feature_metas
        self.linear_model = LinearModel(feature_metas, linear_bias)
        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        if product_mode == 'inner':
            self.product = InnerProduct(feature_metas, embedding_dim)
        else:
            self.product = OuterProduct(feature_metas, embedding_dim)
        m = (len(feature_metas) + 1) * len(feature_metas) // 2  # TODO, require_logits affect this shape
        self.dnn = MLP([m] + list(hidden_units), use_bn=True)
        self.pred = Prediction()

    def forward(self, *inputs):
        # inputs = (x_sparse, x_varlen, x_dense)
        logits = self.linear_model(*inputs)
        embed = self.embeddings(*inputs)  # (batch_size, len(feature_metas), embedding_dim)

        prod = self.product(embed)
        logits += self.dnn(prod)
        # pred
        out = self.pred(logits)
        return out
