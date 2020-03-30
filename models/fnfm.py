# -*- coding:utf-8 -*-
from typing import List
import itertools

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal, glorot_normal, Zeros
from tensorflow.keras.layers import (Dense, Embedding, Lambda, add,
                                     multiply, Layer)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import torch
import torch.nn as nn
import torch.nn.functional as F
from CTR.core import MLP, GroupEmbedding, Prediction


def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, axis)


def activation_func(activation='relu'):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('undefined activation')


class MLP(nn.Module):
    def __init__(self, hidden_size, input_shape, activation='relu', l2_reg=0, keep_prob=1, use_bn=False, seed=1024):
        super(MLP, self).__init__()
        self.l2_reg = l2_reg
        self.seed = seed
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(hidden_size)

        layers = []
        for i in range(len(hidden_size)):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            if use_bn:
                layers.append(nn.BatchNorm1d())
            layers.append(activation_func(activation))
            layers.append(nn.Dropout(1 - keep_prob))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class BiInteractionPooling(nn.Module):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    def forward(self, x):
        if len(x.size()) != 3:
            raise ValueError(f'Unexpected inputs dims {len(x.size())}, expected to be 3')
        square_of_sum = torch.pow(torch.sum(x, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(torch.pow(x, 2), dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)  # (N, 1, embedding_size)
        return cross_term


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, attention_factor=4, l2_reg_w=0, keep_prob=1.0, seed=1024):
        super(AttentionLayer, self).__init__()
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.keep_prob = keep_prob
        self.seed = seed
        self.W = nn.Parameter(data=torch.zeros((embedding_dim, self.attention_factor), dtype=torch.float32))
        self.H = nn.Parameter(data=torch.zeros((self.attention_factor, 1), dtype=torch.float32))
        self.b = nn.Parameter(data=torch.zeros((self.attention_factor,), dtype=torch.float32))

        nn.init.xavier_normal_(self.W, gain=1.0, seed=seed)
        nn.init.xavier_normal_(self.H, gain=1.0, seed=seed)

    def forward(self, x):
        bi_interaction = x
        # 来自 AFM
        # a_{i,j}' = h · ReLU(W <V_i, V_j> x_i x_j + b)
        # a_{i,j} = softmax([a_{i,j}'])
        tensordot = torch.tensordot(bi_interaction, self.W, dims=(-1, 0))
        attention_temp = F.relu(tensordot + self.b)
        tensordot = torch.tensordot(attention_temp, self.H, dims=(-1, 0))
        normed_score = F.softmax(tensordot, dim=1)
        out = torch.sum(normed_score * bi_interaction, dim=1)
        return out


class INT(nn.Module):
    def __init__(self, field_num, attention_factor):
        super(INT, self).__init__()
        self.attention_factor = attention_factor
        self.W = nn.Parameter(torch.zeros((field_num, self.attention_factor), dtype=torch.float32))
        nn.init.xavier_normal_(self.W)

    def forward(self, embeds_vec_list: List):
        bi_interaction = []
        for r, c in itertools.combinations(range(len(embeds_vec_list)), 2):
            field_score = torch.sum(self.W[r] * self.W[c])
            embed = embeds_vec_list[r] * embeds_vec_list[c]
            bi_interaction.append(field_score * embed)
        out = torch.cat(bi_interaction, dim=1)
        return out


def get_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w, space_optimized=True):
    sparse_embedding, dense_embedding, linear_embedding = {}, {}, {}
    if space_optimized:
        for feat in feature_dim_dict["sparse"]:
            sparse_embedding[feat.name] = Embedding(feat.dimension, embedding_size,
                                                    embeddings_initializer=RandomNormal(stddev=0.0001, seed=seed),
                                                    embeddings_regularizer=l2(l2_rev_V),
                                                    name=f'sparse_emb_{feat.name}')
        for feat in feature_dim_dict["dense"]:
            dense_embedding[feat.name] = Dense(embedding_size,
                                               kernel_initializer=RandomNormal(stddev=0.0001, seed=seed),
                                               kernel_regularizer=l2(l2_rev_V),
                                               use_bias=False,
                                               name=f'dense_emb_{feat.name}')
    else:
        for j_feat in feature_dim_dict["sparse"]:
            for i, i_feat in enumerate(feature_dim_dict['sparse'] + feature_dim_dict['dense']):
                sparse_embedding[j_feat.name] = {
                    i_feat.name: Embedding(j_feat.dimension, embedding_size,
                                           embeddings_initializer=RandomNormal(stddev=0.0001, seed=seed),
                                           embeddings_regularizer=l2(l2_rev_V),
                                           name=f'sparse_emb_{j_feat.name}_{i}_{i_feat.name}')}
        for j_feat in feature_dim_dict["dense"]:
            for i, i_feat in enumerate(feature_dim_dict['sparse'] + feature_dim_dict['dense']):
                dense_embedding[j_feat.name] = {
                    i_feat.name: Dense(embedding_size,
                                       kernel_initializer=RandomNormal(stddev=0.0001, seed=seed),
                                       kernel_regularizer=l2(l2_rev_V),
                                       use_bias=False,
                                       name=f'dense_emb_{j_feat.name}_{i}_{i_feat.name}')}
    for i, feat in enumerate(feature_dim_dict["sparse"]):
        linear_embedding[feat.name] = Embedding(feat.dimension, 1,
                                                embeddings_initializer=RandomNormal(stddev=init_std, seed=seed),
                                                embeddings_regularizer=l2(l2_reg_w),
                                                name=f"linear_emb_{i}-{feat.name}")

    return sparse_embedding, dense_embedding, linear_embedding


class FFMEmbedding(nn.Module):
    def __init__(self, feature_metas, embedding_dim, space_optimized=True):
        super(FFMEmbedding, self).__init__()
        self.feature_metas = feature_metas
        self.space_optimized = space_optimized
        self.embedding_dict = nn.ModuleDict()
        if space_optimized:
            for feat in feature_metas.meta_dict.values():
                self.embedding_dict[feat.name] = nn.Embedding(feat.dim, embedding_dim)
            self.int = INT(len(feature_metas), attention_factor=4)
        else:
            for j_feat, i_feat in itertools.combinations(feature_metas.meta_dict.values(), 2):
                self.embedding_dict[j_feat.name + '_' + i_feat.name] = nn.Embedding(j_feat.dim, embedding_dim)
        for embed in self.embedding_dict.values():
            nn.init.normal_(embed.weight, mean=0., std=1e-3)

    def forward(self, x):
        if self.space_optimized:
            embed_list = [self.embedding_dict[name](x[:, feat.index: feat.index + 1])
                          for name, feat in self.feature_metas.meta_dict.items()]
            embed = torch.cat(embed_list, dim=1)
            embed = self.int(embed)
        else:
            embed_list = [self.embedding_dict[j_feat.name + '_' + i_feat.name](x[:, j_feat.index: j_feat.index + 1])
                          for j_feat, i_feat in itertools.combinations(self.feature_metas.meta_dict.values(), 2)]
            embed = torch.cat(embed_list, dim=1)
        return embed


class FNFM(nn.Module):
    def __init__(self, feature_metas, embedding_dim=8, hidden_units=(128, 64), reduction='att',
                 use_bn=True, space_optimized=True, linear_bias=False):
        super(FNFM, self).__init__()
        assert reduction in {'att', 'bi', 'sum'}

        self.feature_metas = feature_metas
        self.reduction = reduction
        self.linear_embeddings = GroupEmbedding(feature_metas, embedding_dim=1, initializer='uniform')
        self.linear = nn.Linear(len(feature_metas), 1, bias=linear_bias)

        self.embeddings_ffm = FFMEmbedding(feature_metas, embedding_dim, space_optimized)
        self.dnn = MLP(hidden_units=[len(feature_metas) * embedding_dim] + list(hidden_units))

        self.int = None
        self.att = None
        self.bi = None
        if reduction == 'att':
            self.att = AttentionLayer(embedding_dim, attention_factor=4)
        elif reduction == 'bi':
            self.bi = BiInteractionPooling()
        self.bn = None
        if use_bn:
            self.bn = nn.BatchNorm1d()
        self.pred = Prediction()

    def forward(self, x):
        # ======================== Linear ==========================
        linear_embed = self.linear_embeddings(x)
        logits = self.linear(torch.flatten(linear_embed, start_dim=1))
        # ====================== Embedding ==========================
        embed = self.embeddings_ffm(x)

        if self.reduction == 'att':
            ffm_out = self.att(embed)
        elif self.reduction == 'bi':
            ffm_out = self.bi(embed)
        else:
            ffm_out = torch.flatten(embed)

        if self.bn:
            ffm_out = self.bn(ffm_out)
        # ======================== DNN =============================
        logits += self.dnn(ffm_out)
        # pred
        out = self.pred(logits)
        return out


def FNFM(feature_dim_dict,
         embedding_size=4,
         hidden_size=(128, 128),
         l2_reg_embedding=1e-5,
         l2_reg_linear=1e-5,
         l2_reg_deep=0,
         init_std=0.0001, seed=1024, final_activation='sigmoid',
         include_linear=True, use_bn=True, reduce_sum=False,
         pooling_method=True, att_factor=4, space_optimized=True):
    """Instantiates the Field-aware Neural Factorization Machine architecture.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param hidden_size: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param keep_prob: float in (0,1]. keep_prob used in deep net
    :param final_activation: str,output activation,usually ``'sigmoid'`` or ``'linear'``
    :param include_linear: bool,whether include linear term or not
    :param use_bn: bool,whether use bn after ffm out or not
    :param reduce_sum: bool,whether apply reduce_sum on cross vector
    :param pooling_method: method for pooling embedding vector, one of [concat, bi, attention]
    :return: A Keras model instance.
    """
    if pooling_method not in ['concat', 'bi', 'att']:
        raise ValueError('pooling method error')

    check_feature_config_dict(feature_dim_dict)
    if 'sequence' in feature_dim_dict and len(feature_dim_dict['sequence']) > 0:
        raise ValueError("now sequence input is not supported in NFFM")

    # input layer
    sparse_input_dict, dense_input_dict = create_singlefeat_inputdict(feature_dim_dict)

    # embedding layer
    sparse_embedding, dense_embedding, linear_embedding = get_embeddings(
        feature_dim_dict, embedding_size, init_std, seed, l2_reg_embedding, l2_reg_linear,
        space_optimized=space_optimized)

    if space_optimized:
        # get_embedding_vec_list = [sparse_embedding[feat](v) for feat, v in sparse_input_dict.item()]
        deep_emb_list = get_embedding_vec_list(sparse_embedding, sparse_input_dict)
        embed_list = INT(att_factor, l2_reg_w=l2_reg_embedding)(deep_emb_list)
    else:
        embed_list = []

        for i, j in itertools.combinations(feature_dim_dict['sparse'], 2):
            element_wise_prod = multiply([sparse_embedding[i.name][j.name](sparse_input_dict[i.name]),
                                          sparse_embedding[j.name][i.name](sparse_input_dict[j.name])])
            if reduce_sum:
                element_wise_prod = Lambda(lambda vec: K.sum(vec, axis=-1))(element_wise_prod)
            embed_list.append(element_wise_prod)

        for i, j in itertools.combinations(feature_dim_dict['dense'], 2):
            element_wise_prod = multiply([dense_embedding[i.name][j.name](dense_input_dict[i.name]),
                                          dense_embedding[j.name][i.name](dense_input_dict[j.name])])
            if reduce_sum:
                element_wise_prod = Lambda(lambda vec: K.sum(vec, axis=-1))(element_wise_prod)
            embed_list.append(Lambda(lambda x: K.expand_dims(x, axis=1))(element_wise_prod))

        for i in feature_dim_dict['sparse']:
            for j in feature_dim_dict['dense']:
                element_wise_prod = multiply([sparse_embedding[i.name][j.name](sparse_input_dict[i.name]),
                                              dense_embedding[j.name][i.name](dense_input_dict[j.name])])
                if reduce_sum:
                    element_wise_prod = Lambda(lambda vec: K.sum(vec, axis=-1))(element_wise_prod)
                embed_list.append(element_wise_prod)
        embed_list = concat_func(embed_list, axis=1)

    # BiInteraction - Concat (cross then concat)
    # BiInteraction - Attention (attentional pooling)
    # pooling layer (attention, bi-interactive,  pooling)
    if pooling_method == 'att':
        ffm_out = AttentionLayer(att_factor)(embed_list)
    elif pooling_method == 'bi':
        ffm_out = BiInteractionPooling()(embed_list)  # Sum Pooling
    else:  # sum pooling
        ffm_out = tf.keras.layers.Flatten()(embed_list)

    # batch_norm
    if use_bn:
        ffm_out = tf.keras.layers.BatchNormalization()(ffm_out)

    ffm_out = MLP(hidden_size, l2_reg=l2_reg_deep)(ffm_out)
    final_logit = Dense(1, use_bias=False)(ffm_out)

    # [linear_embedding[feat](input) for feat, input in sparse_input_dict.items()]
    linear_emb_list = get_embedding_vec_list(linear_embedding, sparse_input_dict)

    linear_logit = get_linear_logit(linear_emb_list, dense_input_dict, l2_reg_linear)

    if include_linear:
        final_logit = add([final_logit, linear_logit])

    # output layer
    output = PredictionLayer(final_activation)(final_logit)

    # model
    inputs_list = get_inputs_list([sparse_input_dict, dense_input_dict])
    model = Model(inputs=inputs_list, outputs=output)
    return model
