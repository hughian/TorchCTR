import torch
import torch.nn as nn
from CTR.core.core import GroupEmbedding, MLP, Prediction, LinearModel

__all__ = ['AFM']


class AttentionPooling(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, attention_factor=4):
        super(AttentionPooling, self).__init__()
        self.feature_metas = feature_metas
        self.att = nn.Linear(embedding_dim, attention_factor)
        self.att_proj = nn.Linear(attention_factor, 1, bias=False)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.zeros_(self.att.bias)
        nn.init.xavier_uniform_(self.att.weight)

    def forward(self, x):
        # x  :  size -> (batch_size, len(features), embedding_dim)
        row, col = [], []
        for i in range(len(self.feature_metas) - 1):
            for j in range(i + 1, len(self.feature_metas)):
                row.append(i), col.append(j)

        p = torch.index_select(x, dim=1, index=torch.LongTensor(row))
        q = torch.index_select(x, dim=1, index=torch.LongTensor(col))
        interaction = p * q  # (batch_size, len(feature) * (len(feature)-1)//2, embedding_dim)
        att_weight = torch.relu(self.att(interaction))
        att_weight = self.att_proj(att_weight)

        att_score = torch.softmax(att_weight, dim=1)
        out = torch.sum(interaction * att_score, dim=1)
        return out


class AFM(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(128, 64), use_dnn=True, linear_bias=False):
        super(AFM, self).__init__()
        self.feature_metas = feature_metas
        self.linear_model = LinearModel(feature_metas, linear_bias=linear_bias)

        self.embeddings = GroupEmbedding(feature_metas, embedding_dim)
        self.att_pool = AttentionPooling(feature_metas, embedding_dim=embedding_dim, attention_factor=4)
        self.dnn = MLP([embedding_dim] + list(hidden_units), dropout_prob=0.5, use_bn=True) if use_dnn else None
        self.pred = Prediction()

    def forward(self, *x):
        # x = x_sparse, x_varlen, x_dense
        logits = self.linear_model(*x)

        embed = self.embeddings(*x)
        att = self.att_pool(embed)
        logits += self.dnn(att)  # what should we do if we don't use dnn

        out = self.pred(logits)
        return out
