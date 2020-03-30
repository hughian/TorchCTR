import torch
import torch.nn as nn
from CTR.core.core import MLP, Prediction, GroupEmbedding, LinearModel

__all__ = ['WideDeep']


class WideDeep(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(128, 64), linear_bias=False):
        super(WideDeep, self).__init__()
        self.feature_metas = feature_metas
        self.linear_model = LinearModel(feature_metas, linear_bias=linear_bias)

        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        self.dnn = MLP([len(feature_metas) * embedding_dim] + list(hidden_units),
                       activation='relu', dropout_prob=0.5, use_bn=True)
        self.pred = Prediction()

    def forward(self, *inputs):
        # x  : (batch_size, len(features))
        # =========================== Linear =================================
        logits = self.linear_model(*inputs)

        # ===========================  DNN  ==================================
        embed = self.embeddings(*inputs)
        dnn_x = torch.flatten(embed, start_dim=1)  # (batch_size, embedding_dim * len(features))
        logits += self.dnn(dnn_x)
        # pred
        out = self.pred(logits)
        return out
