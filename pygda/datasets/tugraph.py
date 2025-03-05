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
    """
    TUGraph Dataset loader for graph-based analysis.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved
    name : str
        Name of the TU dataset
    transform : callable, optional
        Function/transform that takes in a Data object and returns a transformed
        version. Default: None
    pre_transform : callable, optional
        Function/transform to be applied to the data object before saving.
        Default: None
    pre_filter : callable, optional
        Function that takes in a Data object and returns a boolean value,
        indicating whether the data object should be included. Default: None

    Notes
    -----
    Dataset Structure:

    - Collection of graphs
    - Each graph has its own structure and features
    - Supports various graph classification tasks
    - Random shuffling for better training
    """
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
        """
        Names of required raw files.

        Returns
        -------
        list[str]
            List of required raw file names

        Notes
        -----
        Required files:

        - .pkl: Pickle file containing list of graph data objects
        """
        return [".pkl"]

    @property
    def processed_file_names(self):
        """
        Names of processed data files.

        Returns
        -------
        list[str]
            List of processed file names

        Notes
        -----
        Processed files:

        - data.pt: Contains processed PyTorch Geometric data objects
        """
        return ['data.pt']

    def download(self):
        """
        Download raw data files.

        Notes
        -----
        Empty implementation - data should be manually placed in raw directory
        """
        pass

    def process(self):
        """
        Process raw data into PyTorch Geometric Data format.

        Notes
        -----
        Processing Steps:
        
        - Load pickle data:

            * List of graph data objects

        - Random shuffling:

            * Shuffle graphs for better training

        - Apply pre-transform:

            * Transform each graph if specified

        - Collate graphs:

            * Combine into single data object

        - Save processed data

        Features:

        - Multiple graph handling
        - Random shuffling
        - Optional pre-transform support
        - Batch processing support
        """
        path = osp.join(self.raw_dir, '{}.pkl'.format(self.name))
        data_list = pkl.load(open(path, 'rb'))
        random.shuffle(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])
