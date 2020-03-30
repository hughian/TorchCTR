import torch
import torch.nn as nn
from CTR.core.core import MLP, Prediction, GroupEmbedding, LinearModel

__all__ = ['NFM']


class BiInteractionPooling(nn.Module):
    def __init__(self, drop_prob=0.):
        super(BiInteractionPooling, self).__init__()
        self.dropout = None
        if drop_prob > 0:
            self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        # x  : size -> (batch_size, field_num, embedding_dim)
        square_of_sum = x.sum(dim=1, keepdim=True).pow(2)  # (batch_size, 1, embedding_dim)
        sum_of_square = torch.sum(x * x, dim=1, keepdim=True)  # (batch_size, 1, embedding_dim)
        cross_term = 0.5 * (square_of_sum - sum_of_square)  # (batch_size, 1, embedding_dim)
        # 对比 FM 就少了一个 sum(dim=2) 的过程
        if self.dropout:
            cross_term = self.dropout(cross_term)
        return cross_term


class NFM(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(128, 64), bi_dropout_prob=0., linear_bias=False):
        super(NFM, self).__init__()
        self.feature_metas = feature_metas
        self.linear_model = LinearModel(feature_metas, linear_bias=linear_bias)

        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        self.bi_interaction = BiInteractionPooling(bi_dropout_prob)
        self.dnn = MLP([embedding_dim] + list(hidden_units), activation='relu', use_bn=True)

        self.pred = Prediction(task='binary')

    def forward(self, *inputs):
        # =========================== Linear =================================
        logits = self.linear_model(*inputs)

        embed = self.embeddings(*inputs)
        # ======================== BiInteraction =============================
        bi = self.bi_interaction(embed)  # (batch_size, 1, embedding_dim)
        # ===========================  DNN  ==================================
        dnn_x = torch.flatten(bi, start_dim=1)  # (batch_size, embedding_dim * len(features))
        logits += self.dnn(dnn_x)

        out = self.pred(logits)
        return out
