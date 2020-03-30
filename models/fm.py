import torch
import torch.nn as nn
from CTR.core.core import GroupEmbedding, Prediction, LinearModel


class FM(nn.Module):
    """
    Factorization Machine
    """

    def __init__(self, feature_metas, embedding_dim=8, linear_bias=False):
        super(FM, self).__init__()
        self.linear_model = LinearModel(feature_metas, linear_bias=linear_bias)

        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        self.pred = Prediction()

    def forward(self, *x):
        # x  : (batch_size, len(features))
        # =========================== Linear =================================
        logits = self.linear_model(*x)

        embed = self.embeddings(*x)  # (batch_size, len(features), embedding_dim)

        square_of_sum = embed.sum(dim=1, keepdim=True).pow(2)          # (batch_size, 1, 8)
        sum_of_square = torch.sum(embed * embed, dim=1, keepdim=True)  # (batch_size, 1, 8)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * cross_term.sum(dim=2)  # (batch_size, 1)

        logits += cross_term
        return self.pred(logits)
