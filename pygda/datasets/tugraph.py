import os.path as osp
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
import torch.nn.functional as F
import random

import scipy
import pickle as pkl
from sklearn.preprocessing import label_binarize
import csv
import json

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class GraphTUDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        super(GraphTUDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [".pkl"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        path = osp.join(self.raw_dir, '{}.pkl'.format(self.name))
        data_list = pkl.load(open(path, 'rb'))
        random.shuffle(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])
