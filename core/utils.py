from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .core import loss_func
import pandas as pd
from .features import FeatureMetas
from tqdm import tqdm


class myDataSet(torch.utils.data.Dataset):
    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame, feature_metas: FeatureMetas):
        super(myDataSet, self).__init__()
        self.length = len(x_df)
        self.x_dense = torch.from_numpy(x_df[feature_metas.dense_names].values.astype('float32'))
        self.x_sparse = torch.from_numpy(x_df[feature_metas.sparse_names].values).long()
        self.x_varlen = torch.from_numpy(x_df[feature_metas.varlen_names].values).long()
        self.y = torch.from_numpy(y_df.values.astype('float32'))

    def __getitem__(self, index):
        return self.x_sparse[index], self.x_varlen[index], self.x_dense[index], self.y[index]

    def __len__(self):
        return self.length


# TODO we have two choices:
#  1) make this a base module, all subclass this module
#  2) create a wrapper module, Wrapper(model, optim, loss, metrics), seems, second 2 is better
class Base(nn.Module, ABC):
    def __init__(self):
        super(Base, self).__init__()
        self.optimizer = None
        self.criterion = None
        self.metrics = None

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.criterion = loss_func(loss)
        self.metrics = metrics  # list

    def fit(self, x, y, val_data, epochs, batch_size):
        # build dataset and data loader
        train_dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)

        # optimizer & loss func
        optimizer = self.optimizer
        criterion = self.criterion
        metrics = self.metrics

        model = self.train()
        for epoch in range(epochs):
            print(f'epoch: {epoch}')
            epoch_loss = 0.
            with tqdm(enumerate(train_loader), disable=False) as t:
                for i, (x_train, y_train) in t:
                    x_batch = x_train.long()  # to device
                    y_batch = y_train.float()  # to device

                    pred = model(x_batch).squeeze()
                    optimizer.zero_grad()
                    loss = criterion(pred, y_batch.squeeze())
                    # loss += reg_loss, l1, l2
                    epoch_loss += loss.item()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                for metrics_func in metrics:
                    x_val, y_val = val_data
                    pred = model(torch.from_numpy(x_val).long())
                    roc_pred = pred.detach().numpy()
                    roc_true = y_val
                    metric = metrics_func(roc_true, roc_pred)
                    print(metric)
        return model

    def save(self, path):
        # also can save entire model by,
        #       `torch.save(model, path)`
        # then load would be:
        #       `model = torch.load(path)`
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()


class Wrapper:
    def __init__(self, model, optim, loss, metrics):
        self.model = model
        self.optim = optim
        self.loss = loss
        self.metrics = metrics

    def fit(self, x, y, val_data, epochs, batch_size):
        pass

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)


class Regularization(nn.Module):
    """
    Regularization wrapper for module
    """

    def __init__(self, model, weight_decay, p=2):
        """
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求 2 范数, 当 p=0 为 L2 正则化, p=1 为 L1 正则化
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            raise ValueError("param weight_decay can not be <=0")
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.info(self.weight_list)

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss()
        return reg_loss

    def get_weight(self, model):
        """
        获得模型的权重列表
        :param model:
        :return:
        """
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self):
        """
        calculate norms
        """
        weight_list = self.weight_list
        weight_decay = self.weight_decay
        p = self.p
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def info(self, weight_list):
        """
        print weight info
        :param weight_list:
        :return:
        """
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
