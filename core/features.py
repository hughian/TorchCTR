from typing import NamedTuple
from collections import OrderedDict


# >= python 3.6 required
class SparseFeature(NamedTuple):
    """
    name:           Feature name
    dim:            Dimension of the feature in its one-hot encoded form
    embedding_dim:  Dimension to which you want the feature to be embedded, 8 as default, currently,
                    this was not used
    hashing:        false as default
    dtype:          Data type, 'int32' as default
    """
    name: str
    index: int
    dim: int
    embedding_dim: int = 8
    hashing: bool = False
    dtype: str = 'int32'


class DenseFeature(NamedTuple):
    """
    name:           Feature name
    dim:            Dimension of the feature
    embedding_dim:  Dimension to which you want the feature to be embedded, 8 as default
    dtype:          Data type, 'float32' as default
    """
    name: str
    index: int
    dim: int
    embedding_dim: int = 8
    dtype: str = 'float32'


# TODO : rename to VarlenFeature
class ListSparseFeature(NamedTuple):
    """
    hashing:
    max_length:     Integer. Max length of the feature list
    name:           Feature name
    dim:            Dimension of the feature in its one-hot encoded form
    embedding_dim:  Dimension to which you want the feature to be embedded
    dtype:          Data type, 'int32' as default
    """
    name: str
    index: int
    max_len: int
    dim: int
    embedding_dim: int = 8
    hashing: bool = False
    dtype: str = 'int32'


class FeatureType(object):
    DenseFeature = 1
    SparseFeature = 2
    ListSparseFeature = 3
    UnknownFeature = -1


class FeatureMetas(object):
    """
        Meta information of all features.
    """
    def __init__(self):
        self.meta_dict = OrderedDict()
        self.sparse_names = list()
        self.dense_names = list()
        self.varlen_names = list()

    def __getitem__(self, key):
        return self.meta_dict[key]

    def __len__(self):
        return self.meta_dict.__len__()

    def items(self):
        return self.meta_dict.items()

    def keys(self):
        return self.meta_dict.keys()

    def values(self):
        return self.meta_dict.values()

    def sparse_items(self):
        for name in self.sparse_names:
            yield name, self.meta_dict[name]

    def varlen_items(self):
        for name in self.varlen_names:
            yield name, self.meta_dict[name]

    def dense_items(self):
        for name in self.dense_names:
            yield name, self.meta_dict[name]

    def get_feature_type(self, name):
        if name not in self.meta_dict:
            return FeatureType.UnknownFeature
        elif isinstance(self.meta_dict[name], SparseFeature):
            return FeatureType.SparseFeature
        elif isinstance(self.meta_dict[name], DenseFeature):
            return FeatureType.DenseFeature
        elif isinstance(self.meta_dict[name], ListSparseFeature):
            return FeatureType.ListSparseFeature
        else:
            return FeatureType.UnknownFeature

    def add_sparse_feature(self, name, index, dim, embedding_dim=8, hashing=False, dtype='int32'):
        feat = SparseFeature(name=name, index=index, dim=dim,
                             embedding_dim=embedding_dim, hashing=hashing, dtype=dtype)
        self.meta_dict[name] = feat
        self.sparse_names.append(name)

    def add_dense_feature(self, name, index, dim, embedding_dim=8, dtype='float32'):
        feat = DenseFeature(name=name, index=index, dim=dim, embedding_dim=embedding_dim, dtype=dtype)
        self.meta_dict[name] = feat
        self.dense_names.append(name)

    def add_varlen_feature(self, name, index, max_len, dim, embedding_dim=8, hashing=False, dtype='int32'):
        """Add a list sparse feature whose length is not fixed"""
        feat = ListSparseFeature(name=name, index=index, max_len=max_len, dim=dim,
                                 embedding_dim=embedding_dim, hashing=hashing, dtype=dtype)
        self.meta_dict[name] = feat
        self.varlen_names.append(name)
