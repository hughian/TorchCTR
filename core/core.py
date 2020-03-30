import torch
import torch.nn as nn
from .features import FeatureType, FeatureMetas


def act_func(activation):
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    else:
        raise


def loss_func(loss, reduction='sum'):
    if loss.lower() == 'bce_loss':
        return nn.BCELoss(reduction=reduction)
    elif loss.lower == 'mse':
        return nn.MSELoss(reduction=reduction)
    else:
        raise


def init_func(weight, initializer):
    if initializer == 'glorot_uniform':
        nn.init.xavier_normal_(weight)
    elif initializer == 'normal':
        nn.init.normal_(weight)
    elif initializer == 'uniform':
        nn.init.uniform_(weight)
    elif initializer == 'kaiming_unifom':
        nn.init.kaiming_uniform_(weight, a=0.2)
    elif initializer == 'zeros':
        nn.init.zeros_(weight)
    else:
        raise  # TODO 更多的初始化


class GroupEmbedding(nn.Module):
    def __init__(self, feature_metas: FeatureMetas, embedding_dim, initializer='normal', dense_embedding=True):
        # TODO:
        #  for sparse : nn.Embedding
        #  for varlen : nn.EmbeddingBag, with reduction_mode ='sum'
        #  for dense  : nn.Linear, linear embedding do not need this
        #   hashing: if hashing: torch.fmod(feat, dim_hash) -> nn.Embedding(dim_hash, embedding_dim)

        super(GroupEmbedding, self).__init__()
        self.feature_metas = feature_metas
        self.dense_embedding = dense_embedding
        self.embedding_dict = nn.ModuleDict()
        for feat in self.feature_metas.values():
            feat_type = feature_metas.get_feature_type(feat.name)
            if feat_type == FeatureType.SparseFeature:
                self.embedding_dict[feat.name] = nn.Embedding(feat.dim, embedding_dim)
            elif feat_type == FeatureType.DenseFeature:
                if dense_embedding:
                    self.embedding_dict[feat.name] = nn.Linear(feat.dim, embedding_dim, bias=False)
            elif feat_type == FeatureType.ListSparseFeature:
                self.embedding_dict[feat.name] = nn.EmbeddingBag(feat.dim, embedding_dim, mode='sum')
            else:
                raise TypeError(FeatureType.UnknownFeature)

        # self.embedding_dict = nn.ModuleDict({feat.name: nn.Embedding(feat.dim, embedding_dim)
        #                                      for feat in feature_metas.meta_dict.values()})

        for embed in self.embedding_dict.values():
            if initializer == 'uniform':
                nn.init.uniform_(embed.weight)
            else:
                nn.init.normal_(embed.weight, mean=0., std=1e-3)

    def forward(self, *inputs):
        x_sparse, x_varlen, x_dense = inputs
        sparse_embed_list = [self.embedding_dict[name](x_sparse[:, feat.index: feat.index + 1])
                             for name, feat in self.feature_metas.sparse_items()]
        varlen_embed_list = [self.embedding_dict[name](x_varlen[:, feat.index: feat.index + 1])
                             for name, feat in self.feature_metas.varlen_items()]
        embed_list = sparse_embed_list + varlen_embed_list
        if self.dense_embedding:
            dense_embed_list = [torch.unsqueeze(self.embedding_dict[name](x_dense[:, feat.index:feat.index + 1]), dim=1)
                                for name, feat in self.feature_metas.dense_items()]
            embed_list += dense_embed_list
        # [(batch_size, 1, embedding_dim)]
        # TODO: should we squeeze(dim=1) -> [batch_size, embedding_dim]?
        embed = torch.cat(sparse_embed_list + varlen_embed_list, dim=1)
        return embed


class LinearModel(nn.Module):
    def __init__(self, feature_metas, linear_bias=False):
        super(LinearModel, self).__init__()
        self.embeddings = GroupEmbedding(feature_metas, embedding_dim=1, initializer='uniform', dense_embedding=False)
        self.linear = nn.Linear(len(feature_metas), 1, bias=linear_bias)

    def forward(self, *inputs):
        x_sparse, x_varlen, x_dense = inputs
        embed = self.embeddings(x_sparse, x_varlen, x_dense)
        concat = torch.cat([torch.flatten(embed, start_dim=1), x_dense], dim=-1)
        logits = self.linear(concat)
        return logits


class MLP(nn.Module):
    def __init__(self, hidden_units, activation='relu', dropout_prob=0., use_bn=True,
                 init_std=1e-3, seed=1234, bias=False):
        super(MLP, self).__init__()
        if not isinstance(hidden_units, (list, tuple)):
            raise ValueError('Expect list/tuple as hidden_units')
        self.dnn = nn.Sequential()
        for i in range(len(hidden_units) - 1):
            self.dnn.add_module(f'linear_{i + 1}', nn.Linear(hidden_units[i], hidden_units[i + 1]))
            if use_bn:
                self.dnn.add_module(f'bn_{i + 1}', nn.BatchNorm1d(hidden_units[i + 1]))
            self.dnn.add_module(f'{activation}_{i + 1}', act_func(activation))
        if dropout_prob > 0.:
            self.dnn.add_module('dropout', nn.Dropout(dropout_prob))
        self.dnn.add_module(f'dense', nn.Linear(hidden_units[-1], 1, bias=bias))

        torch.random.manual_seed(seed)
        for name, tensor in self.dnn.named_parameters():
            if 'weight' in name and 'bn' not in name:  # only apply to linear.weight
                nn.init.xavier_normal_(tensor)

    def forward(self, x):
        return self.dnn(x)


class Prediction(nn.Module):
    def __init__(self, task='binary', use_bias=False):
        super(Prediction, self).__init__()
        self.bias = None
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, ))

        if task == 'binary':
            self.fn = nn.Sigmoid()
        elif task == 'multiclass':
            self.fn = nn.Softmax(dim=1)
        elif task == 'regression':
            self.fn = nn.Identity()
        else:
            raise ValueError('Unexpected Task')

    def forward(self, x):
        out = x
        if self.bias:
            out += self.bias
        out = self.fn(out)
        return out
