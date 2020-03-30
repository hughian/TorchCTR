import torch
import torch.nn as nn
from CTR.core.core import MLP, GroupEmbedding, Prediction, LinearModel

__all__ = ['DeepFM']


class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()

    def forward(self, x):
        # input size (batch_size, 20, 8)
        square_of_sum = x.sum(dim=1, keepdim=True).pow(2)  # (batch_size, 1, 8)
        sum_of_square = torch.sum(x * x, dim=1, keepdim=True)  # (batch_size, 1, 8)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * cross_term.sum(dim=2)  # (batch_size, 1)
        return cross_term


class DeepFM(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(128, 64), linear_bias=False):
        super(DeepFM, self).__init__()
        self.feature_metas = feature_metas

        self.linear_model = LinearModel(feature_metas, linear_bias=linear_bias)

        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        self.fm = FM()
        self.dnn = MLP([len(feature_metas) * embedding_dim] + list(hidden_units), activation='relu', use_bn=True)
        self.pred = Prediction()

    def forward(self, *x):
        # x  :  (batch_size, len(features))
        # =========================== Linear =================================
        linear = self.linear_model(*x)
        # ===========================   FM   =================================
        embed = self.embeddings(*x)
        fm = self.fm(embed)  # (batch_size, 1)
        # ===========================   DNN  =================================
        dnn_x = torch.flatten(embed, start_dim=1)  # (batch_size, embedding_dim * len(features))
        dnn = self.dnn(dnn_x)
        # pred
        out = linear + fm + dnn  # (batch_size, 1)
        out = self.pred(out)
        return out
