import torch
import torch.nn as nn
from CTR.core.core import MLP, Prediction, GroupEmbedding, LinearModel

# TODO: BiInteraction 的过程。


class BiInteraction(nn.Module):
    def __init__(self):
        super(BiInteraction, self).__init__()
        pass

    def forward(self, x):
        pass


class NFFM(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(256, 64), linear_bias=False, dense_bias=False):
        super(NFFM, self).__init__()
        self.feature_metas = feature_metas
        self.linear_model = LinearModel(feature_metas, linear_bias=linear_bias)

        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        # self.fm = FM()
        self.dnn = MLP(hidden_units=[len(feature_metas) * embedding_dim] + list(hidden_units), bias=dense_bias)
        self.pred = Prediction()

    def forward(self, *inputs):
        # x = x_sparse, x_varlen, x_dense
        # =========================== Linear =================================
        logits = self.linear_model(*inputs)
        # ========================  Embedding  ===============================
        embed = self.embeddings(*inputs)
        # TODO?
        # ===========================  DNN  ==================================
        dnn_x = embed.view(-1, embed.size(1) * embed.size(2))
        logits += self.dense(dnn_x)
        # pred
        out = self.pred(logits)
        return out
