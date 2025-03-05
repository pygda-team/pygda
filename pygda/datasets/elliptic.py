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


class EllipticDataset(InMemoryDataset):
    """
    Elliptic Bitcoin transaction network dataset loader for graph-based analysis.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved
    name : str
        Name of the elliptic dataset (must be in format 'tx{N}' where N is 0-48)
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

    - Nodes represent Bitcoin transactions
    - Edges represent transaction flows
    - Node features from transaction data
    - Labels indicate transaction categories
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
        super(EllipticDataset, self).__init__(root, transform, pre_transform, pre_filter)

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

        - data.pkl: Pickle file containing adjacency matrix, labels, and features
        """
        return ["data.pkl"]

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
    
    def load_data(self, data_dir, filename):
        """
        Load raw pickle dataset file.

        Parameters
        ----------
        data_dir : str
            Path to the pickle file
        filename : str
            Name of the dataset file (must be in format 'tx{N}' where N is 0-48)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contains:

            - x: Node feature matrix [num_nodes, num_features]
            - edge_index: Graph connectivity [2, num_edges]
            - y: Node labels [num_nodes]

        Notes
        -----
        - Validates dataset number
        - Loads pickle file containing network structure
        - Converts data to PyTorch tensors
        """
        assert int(filename[2:]) in range(0, 49), 'Invalid dataset'
        
        data = pkl.load(open(data_dir, 'rb'))
        A, y, x = data

        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        return x, edge_index, y
    
    def process(self, mask=True):
        """
        Process raw data into PyTorch Geometric Data format.

        Parameters
        ----------
        mask : bool, optional
            Whether to generate train/val/test masks. Default: True

        Notes
        -----
        Processing Steps:

        - Load pickle data:
        
            * Node features
            * Adjacency matrix
            * Node labels

        - Convert to PyTorch format:

            * Edge indices from sparse adjacency
            * Float features
            * Integer labels

        - Create Data object with:

            * Edge indices
            * Node features
            * Node labels
            * Train/val/test masks

        - Apply pre-transform if specified
        - Save processed data

        Data Split:

        - Training: 80%
        - Validation: 10%
        - Testing: 10%

        Features:

        - Sparse matrix conversion
        - Type casting
        - Random split generation
        - Optional pre-transform support
        """
        path = osp.join(self.raw_dir, '{}.pkl'.format(self.name[2:]))
        x, edge_index, y = self.load_data(path, self.name)
        
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