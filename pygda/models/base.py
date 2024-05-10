import time
from abc import ABC, abstractmethod

import torch
import numpy as np
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader

from ..utils import logger


class BaseGDA(ABC):
    """
    Abstract Class for Graph Domain Adaptation.

    Parameters
    ----------
    in_dim  :  int
        Input feature dimension.
    hid_dim :  int
        Hidden dimension of model.
    num_classes: int
        Total number of classes.
    num_layers : int, optional
        Total number of layers in model.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Learning rate. Default: ``0.001``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    device : str, optional
        GPU or CPU. Default: ``cuda:0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=2,
        dropout=0.,
        weight_decay=0.,
        act=F.relu,
        lr=4e-3,
        epoch=100,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):

        super(BaseGDA, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.verbose = verbose
        self.kwargs = kwargs

        self.lr = lr
        self.epoch = epoch
        self.device = device
        self.batch_size = batch_size

        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the '
                                 'same length as hidden layers dimension or'
                                 'the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')

        self.model = None

    def fit(self, data, **kwargs):
        """
        Training the graph neural network.

        Parameters
        ----------
        data : torch_geometric.data.Data, optional
            The input graph.
        """


    def predict(self, data, **kwargs):
        """Prediction for testing graph using the fitted graph domain adaptation model.
        Return predicted labels and probabilities by default.

        Parameters
        ----------
        data : torch_geometric.data.Data, optional
            The testing graph.

        Returns
        -------
        pred : torch.Tensor
            The predicted labels of shape :math:`N`.
        prob : torch.Tensor
            The output probabilities of shape :math:`N`.
        """

    @abstractmethod
    def init_model(self, **kwargs):
        """
        Initialize the graph neural network.

        Returns
        -------
        model : torch.nn.Module
            The initialized graph neural network.
        """
    
    @abstractmethod
    def process_graph(self, data,  **kwargs):
        """
        Data preprocessing for the input graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph.
        """

    @abstractmethod
    def forward_model(self, data,  **kwargs):
        """
        Forward pass of the graph neural network.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph.

        Returns
        -------
        loss : torch.Tensor
            The loss of the current batch.
        """