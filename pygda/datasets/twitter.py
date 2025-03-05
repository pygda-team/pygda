import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array

import csv
import json
import pickle as pkl
import scipy
import scipy.io as sio

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class TwitterDataset(InMemoryDataset):
    """
    Twitter social network dataset loader for graph-based analysis.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved
    name : str
        Name of the Twitter dataset
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

    - Nodes represent Twitter users
    - Edges represent user interactions/connections
    - Node features from user attributes
    - Labels indicate user categories (0-based indexing)
    - Includes train/val/test splits (80/10/10)
    """

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        self.root = root
        super(TwitterDataset, self).__init__(root, transform, pre_transform, pre_filter)

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

        - data.pt: PyTorch file containing graph data
        """
        return ["data.pt"]

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

        - data.pt: Contains processed PyTorch Geometric data object
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
        
        - Load dataset file:

            * Node features (x)
            * Edge indices
            * Node labels

        - Adjust labels:

            * Convert to 0-based indexing
            * Handle label shifts

        - Create Data object with:

            * Edge indices
            * Node features
            * Node labels
            * Train/val/test masks

        - Generate splits:

            * Random permutation
            * Training (80%)
            * Validation (10%)
            * Testing (10%)

        - Create masks:

            * Boolean tensors
            * Node-wise split indicators

        - Optional processing:

            * Apply pre-transform if specified
            * Handle eigenvalue computation

        - Data organization:

            * Collate into list
            * Save processed format

        Features:
        
        - Label adjustment
        - Random split generation
        - Mask creation
        - Optional pre-transform support
        """
        path = osp.join(self.raw_dir, 'data.pt')
        graph = torch.load(path)
        x, edge_index, y = graph.x, graph.edge_index, graph.y   
        y = y - 1
        
        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.8)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            if not os.path.exists(self.processed_paths[0] + 'eival.pt'):
                data = self.pre_transform(data, self.processed_paths[0])

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])