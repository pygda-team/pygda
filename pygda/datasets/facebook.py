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


class FacebookDataset(InMemoryDataset):
    """
    Facebook100 social network dataset loader for graph-based analysis.

    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved
    name : str
        Name of the Facebook dataset (must be one of the FB100 universities)
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

    - Nodes represent Facebook users
    - Edges represent friendships
    - Node features from user metadata
    - Labels indicate user attributes
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
        super(FacebookDataset, self).__init__(root, transform, pre_transform, pre_filter)

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

        - data.mat: MATLAB file containing network data and user metadata
        """
        return ["data.mat"]

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
    
    def load_fb100(self, data_dir, filename, one_hot = True):
        """
        Load raw Facebook100 dataset file.

        Parameters
        ----------
        data_dir : str
            Path to the MATLAB file
        filename : str
            Name of the dataset (must be one of the FB100 universities)
        one_hot : bool, optional
            Whether to convert categorical features to one-hot encoding. Default: True

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Contains:

            - x: Node feature matrix [num_nodes, num_features]
            - edge_index: Graph connectivity [2, num_edges]
            - y: Node labels [num_nodes]

        Notes
        -----
        - Validates university name
        - Loads MATLAB file containing network structure
        - Processes metadata into features
        - Optionally converts features to one-hot encoding
        - Aggregates data from all universities for consistent encoding
        """
        assert filename in ('FB_Penn94', 'FB_Amherst41', 'FB_Cornell5', 'FB_Johns Hopkins55',
                            'FB_Caltech36', 'FB_Brown11', 'FB_Yale4', 'FB_Texas80',
                            'FB_Bingham82', 'FB_Duke14', 'FB_Princeton12', 'FB_WashU32',
                            'FB_Brandeis99', 'FB_Carnegie49')
        mat = sio.loadmat(data_dir)
        
        A = mat['A']
        metadata = mat['local_info'].astype(np.int64)
        x = np.hstack(
                (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        y = metadata[:, 1] - 1 # -1 means unlabeled
        
        # make features into one-hot encodings
        if one_hot is True:
            from sklearn.preprocessing import label_binarize
            
            x_all = np.empty((0, 6))
            for f in ('FB_Penn94', 'FB_Amherst41', 'FB_Cornell5', 'FB_Johns Hopkins55',
                                'FB_Caltech36', 'FB_Brown11', 'FB_Yale4', 'FB_Texas80',
                                'FB_Bingham82', 'FB_Duke14', 'FB_Princeton12', 'FB_WashU32',
                                'FB_Brandeis99', 'FB_Carnegie49'):
                path_f = osp.join(os.getcwd()+'/data/{}/raw'.format(f), '{}.mat'.format(f[3:]))
                mat_f = sio.loadmat(path_f)
                metadata_f = mat_f['local_info'].astype(np.int64)
                x_f = np.hstack(
                        (np.expand_dims(metadata_f[:, 0], 1), metadata_f[:, 2:])
                )
                x_all = np.vstack(
                    (x_all, x_f)
                )

            x_t = np.empty((x.shape[0], 0))
            for col in range(x.shape[1]):
                feat_col = x[:, col]
                feat_onehot = label_binarize(feat_col, classes=np.unique(x_all[:, col]))
                x_t = np.hstack((x_t, feat_onehot))
            x = x_t
            print(x.shape)
            
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, edge_index, y
    
    def process(self):
        """
        Process raw data into PyTorch Geometric Data format.

        Notes
        -----
        Processing Steps:
        
        - Load MATLAB data:
        
            * Network structure
            * User metadata
            * User attributes

        - Convert to PyTorch format:

            * Edge indices from adjacency
            * One-hot features from metadata
            * Integer labels from attributes

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
        
        - One-hot encoding
        - Metadata processing
        - Random split generation
        - Optional pre-transform support
        """
        path = osp.join(self.raw_dir, '{}.mat'.format(self.name[3:]))
        x, edge_index, y = self.load_fb100(path, self.name)

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
