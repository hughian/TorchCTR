import torch
import torch.nn as nn
import torch.nn.functional as F
from CTR.core.core import MLP, Prediction, GroupEmbedding, LinearModel

__all__ = ['xDeepFM']


class CIN(nn.Module):
    def __init__(self, input_dims, hidden_width=(64, 42), conv_bias=False,
                 kernel_initializer='glorot_uniform', l2_reg=1e-5):
        super(CIN, self).__init__()
        self.input_dims = input_dims
        self.hidden_width = hidden_width
        self.kernel_initializer = kernel_initializer
        self.l2_reg = l2_reg
        self.convs = nn.ModuleList()  # use module list or the params can't be traced
        m, D = input_dims  # m field size,  D embedding size
        field_nums = [m] + list(hidden_width)
        for idx, layer_size in enumerate(hidden_width):
            conv1d = nn.Conv1d(m * field_nums[idx], layer_size, kernel_size=(1,), stride=1, bias=conv_bias)
            self.convs.append(conv1d)
        self.dense = nn.Linear(sum(field_nums), 1, bias=False)

    def forward(self, x):
        m, D = self.input_dims
        finals = [x]
        x0 = torch.split(finals[-1], D * [1], dim=2)  # D * (batch_size, m, 1) tuple
        x0 = torch.stack(x0, dim=0)
        for idx, layer_size in enumerate(self.hidden_width):
            x = torch.split(finals[-1], D * [1], dim=2)  # D * (batch_size, field_num, 1)
            x = torch.stack(x, dim=0)
            dot = torch.matmul(x0, torch.transpose(x, dim0=2, dim1=3))
            dot = dot.view([D, -1, m * dot.size(-1)])  # (D, batch_size, m * field_num)
            dot = dot.permute([1, 2, 0])  # (batch_size, m * field_num, D), channel_first
            conv = self.convs[idx](dot)
            out = F.relu(conv)
            finals.append(out)

        finals = torch.cat(finals, dim=1)  # (batch_size, m + sum(hidden_width), D)
        finals = torch.sum(finals, dim=-1)  # (batch_size, m + sum(hidden_width))
        logits = self.dense(finals)
        return logits


class xDeepFM(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(256, 64), linear_bias=False, dense_bias=False):
        super(xDeepFM, self).__init__()
        self.feature_metas = feature_metas
        self.linear_model = LinearModel(feature_metas, linear_bias=linear_bias)
        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        self.cin = CIN(input_dims=(len(feature_metas), embedding_dim))
        self.dnn = MLP(hidden_units=[len(feature_metas) * embedding_dim] + list(hidden_units), use_bn=True)
        self.pred = Prediction()

    def forward(self, *inputs):
        # inputs = x_sparse, x_varlen, x_dense
        # =========================== Linear =================================
        logits = self.linear_model(*inputs)
        # ===========================  CIN  ==================================
        embed = self.embeddings(*inputs)
        logits += self.cin(embed)
        # ===========================  DNN  ==================================
        dnn_x = embed.view(-1, embed.size(1) * embed.size(2))
        logits += self.dnn(dnn_x)
        # pred
        out = self.pred(logits)
        return out
