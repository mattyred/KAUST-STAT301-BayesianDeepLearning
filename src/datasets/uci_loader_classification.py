import numpy as np
import torch
import torchvision.datasets as datasets
import logging
import os
from os import path, system
from sklearn.model_selection import KFold
import pandas as pd
import zipfile
import urllib.request


class UCIDatasets():
    def __init__(self,  name,  data_path="", n_splits = 10):
        self.datasets = {
            "banana": "https://www.openml.org/data/download/1586217/phpwRjVjk",
        }
        self.data_path = data_path
        self.name = name
        self.n_splits = n_splits
        self._load_dataset()

    
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not path.exists(self.data_path+"UCI"):
            os.mkdir(self.data_path+"UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        if not path.exists(self.data_path+"UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path+"UCI/" + file_name)

            if self.name == "banana":
                os.system("tail -n -5300 " + self.data_path+"UCI/" + file_name + ">" + self.data_path+"UCI/TMP")
                os.system("mv " + self.data_path+"UCI/TMP " + self.data_path+"UCI/" + file_name)
        data = None


        if self.name == "banana":
            tmp = pd.read_csv(self.data_path+'UCI/phpwRjVjk', header=0, delimiter=",").values
            tmp[:,-1] = tmp[:,-1] - 1 ## Set labels to {0,1} from {1,2}
            np.random.seed(847984)
            np.random.shuffle(tmp)
            self.data = tmp

            self.out_dim = 2


        kf = KFold(n_splits=self.n_splits)
        self.in_dim = self.data.shape[1] - 1 # self.out_dim
        self.data_splits = kf.split(self.data)
        self.data_splits = [(idx[0], idx[1]) for idx in self.data_splits]

        
    def get_split(self, split=-1, train=True):
        if split == -1:
            split = 0
        if 0 <= split and split < self.n_splits: 
            train_index, test_index = self.data_splits[split]
            x_train, y_train = self.data[train_index, :self.in_dim], self.data[train_index, self.in_dim:]
            x_test, y_test = self.data[test_index, :self.in_dim], self.data[test_index, self.in_dim:]
            x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0)**0.5
            bad_indices = np.argwhere(x_stds == 0)
            x_stds[bad_indices] = 1.0
            x_train = (x_train - x_means)/x_stds
            x_test = (x_test - x_means)/x_stds
            if train:
                inps = torch.from_numpy(x_train).float()
                tgts = torch.from_numpy(y_train).to(torch.int64).flatten()
                train_data = torch.utils.data.TensorDataset(inps, tgts)
                return train_data
            else:
                inps = torch.from_numpy(x_test).float()
                tgts = torch.from_numpy(y_test).to(torch.int64).flatten()
                test_data = torch.utils.data.TensorDataset(inps, tgts)

                return test_data
