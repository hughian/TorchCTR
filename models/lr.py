import torch
import torch.nn as nn
from CTR.core.core import LinearModel


class LR(nn.Module):
    def __init__(self, feature_metas, use_bias=True):
        super(LR, self).__init__()
        self.feature_metas = feature_metas
        self.linear_model = LinearModel(feature_metas, linear_bias=use_bias)

    def forward(self, *x):
        # x_sparse, x_varlen, x_dense = x
        logits = self.linear_model(*x)
        pred = torch.sigmoid(logits)
        return pred
