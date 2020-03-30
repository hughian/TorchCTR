import torch
import torch.nn as nn
from CTR.core.core import MLP, Prediction, GroupEmbedding, LinearModel

__all__ = ['DCN']


class CrossNet(nn.Module):
    # TODO：CrossNet 公式对照,尤其是初始时的 shape
    def __init__(self, field_num, embedding_dim, layer_num=3):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        # input 是一个 list, 每个元素是一个 Tensor, shape 为 (?, embedding_dim)
        self.Ws = nn.ParameterList()
        m = field_num * embedding_dim
        print('####', m)
        for i in range(layer_num):
            kernel = nn.Parameter(torch.zeros((m, 1)))
            bias = nn.Parameter(torch.zeros(m, 1))
            nn.init.xavier_uniform_(kernel)
            nn.init.xavier_uniform_(bias)

            self.Ws.append(kernel)
            self.Ws.append(bias)
        self.kernel = nn.Parameter(data=torch.zeros(m, 1))
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, x, require_logits=True):
        # x : size -> (batch_size, len(features), embedding_dim)
        x0 = torch.unsqueeze(torch.flatten(x, start_dim=1), dim=-1)
        x = x0.permute([0, 2, 1])

        for i in range(self.layer_num):
            kernel = self.Ws[2 * i]
            bias = self.Ws[2 * i + 1]
            x = torch.matmul(x0, torch.matmul(x, kernel)) + bias + x.permute([0, 2, 1])
            x = x.permute([0, 2, 1])

        x = x.squeeze(dim=1)
        if require_logits:
            x = torch.matmul(x, self.kernel)
        return x


class DCN(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(128, 64), linear_bias=False):
        super(DCN, self).__init__()
        self.feature_metas = feature_metas
        self.linear_model = LinearModel(feature_metas, linear_bias=linear_bias)

        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        self.cross = CrossNet(len(feature_metas), embedding_dim, layer_num=3)
        self.dnn = MLP([len(feature_metas) * embedding_dim] + list(hidden_units), activation='relu', use_bn=True)

        self.pred = Prediction()

    def forward(self, *x):
        # =========================== Linear =================================
        logits = self.linear_model(*x)

        embed = self.embeddings(*x)  # (batch_size, len(features), embedding_dim)
        # ============================= Cross ================================
        logits += self.cross(embed)
        # =============================  DNN  ================================
        dnn_x = torch.flatten(embed, start_dim=1)  # (batch_size, embedding_dim * len(features))
        logits += self.dnn(dnn_x)
        # pred
        out = self.pred(logits)
        return out
