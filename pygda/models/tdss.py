import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from . import BaseGDA
from ..nn import A2GNNBase
from ..utils import logger, MMD
from ..metrics import eval_macro_f1, eval_micro_f1

from torch_cluster import random_walk
from torch_geometric.utils import dense_to_sparse, add_remaining_self_loops, remove_self_loops, coalesce
from torch_sparse import spspmm
from torch_geometric.data import Data


class TwoHopNeighbor(object):
    """
    A graph transformation that adds two-hop neighbors to the input graph.
    
    This class computes the two-hop neighborhood of each node by performing
    sparse matrix multiplication of the adjacency matrix with itself, then
    concatenating the original edges with the new two-hop edges.
    
    Parameters
    ----------
    None
        This class takes no parameters.
    
    Returns
    -------
    torch_geometric.data.Data
        The input data object with modified edge_index and edge_attr.
        The edge_index will contain both original edges and two-hop edges.
        If edge_attr was None, the new edges will have no attributes.
        If edge_attr existed, new edges will have zero-valued attributes.
    
    Notes
    -----
    The transformation works by:

    - Computing the two-hop adjacency matrix using sparse matrix multiplication

    - Removing self-loops from the two-hop connections

    - Concatenating original edges with two-hop edges

    - Coalescing duplicate edges and their attributes

    Examples
    --------
    >>> from torch_geometric.data import Data
    >>> import torch
    >>> 
    >>> # Create a simple graph with 3 nodes
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    >>> data = Data(edge_index=edge_index, num_nodes=3)
    >>> 
    >>> # Apply two-hop neighbor transformation
    >>> transformer = TwoHopNeighbor()
    >>> result = transformer(data)
    >>> print(result.edge_index)
    tensor([[0, 1, 1, 2, 0, 2],
            [1, 0, 2, 1, 2, 0]])
    """

    def __call__(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        N = data.num_nodes

        value = edge_index.new_ones((edge_index.size(1), ), dtype=torch.float)

        index, value = spspmm(edge_index, value, edge_index, value, N, N, N, True)
        value.fill_(0)
        index, value = remove_self_loops(index, value)

        edge_index = torch.cat([edge_index, index], dim=1)
        if edge_attr is None:
            data.edge_index, _ = coalesce(edge_index, None, N, N)
        else:
            value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
            value = value.expand(-1, *list(edge_attr.size())[1:])
            edge_attr = torch.cat([edge_attr, value], dim=0)
            data.edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
            data.edge_attr = edge_attr

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class TDSS(BaseGDA):
    """
    Smoothness Really Matters: A Simple Yet Effective Approach for Unsupervised Graph Domain Adaptation (AAAI-25).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    mode : str, optional
        Mode for node or graph level tasks. Default: ``node``.
    smooth_mode : str, optional
        Mode for smoothness construction (RW or K-hop). Default: ``RW``.
    num_layers : int, optional
        Total number of layers in model. Default: ``3``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    s_pnums : int, optional
        Number of propagations for source models. Default: ``0``.
    t_pnums : int, optional
        Number of propagations for target models. Default: ``30``.
    k : int, optional
        K-hop neighbors. Default: ``2``.
    rw_len : int, optional
        Random walk length. Default: ``4``.
    alpha : float, optional
        MMD loss trade-off parameter. Default: ``7``.
    beta : float, optional
        Smoothness loss trade-off parameter. Default: ``2e-4``.
    adv : bool, optional
        Adversarial training or not. Default: ``False``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``200``.
    device : str, optional
        GPU or CPU. Default: ``cuda:0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``2``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        mode='node',
        smooth_mode='RW',
        num_layers=2,
        dropout=0.,
        act=F.relu,
        s_pnums=0,
        t_pnums=30,
        k=2,
        rw_len=4,
        alpha=0.001,
        beta=1e-4,
        weight_decay=0.005,
        adv=False,
        lr=0.01,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(TDSS, self).__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            weight_decay=weight_decay,
            lr=lr,
            epoch=epoch,
            device=device,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            **kwargs)
        
        assert mode == 'node', 'TDSS only supports node-level tasks'
        assert adv == False, 'TDSS does not support adversarial training'

        self.mode = mode
        self.smooth_mode = smooth_mode
        self.s_pnums=s_pnums
        self.t_pnums=t_pnums
        self.k=k
        self.rw_len=rw_len
        self.alpha=alpha
        self.beta=beta  
        self.adv=adv

    def init_model(self, **kwargs):
        """
        Initialize the A2GNN base model.

        Parameters
        ----------
        **kwargs
            Additional parameters for model initialization.

        Returns
        -------
        A2GNNBase
            Initialized model with specified architecture parameters.

        Notes
        -----
        Configures model with:

        - Asymmetric propagation settings
        - Domain adaptation components
        - Task-specific architecture (node/graph)
        - Optional adversarial training module
        """

        return A2GNNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            adv=self.adv,
            dropout=self.dropout,
            act=self.act,
            mode=self.mode,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data, alpha):
        """
        Forward pass of the A2GNN model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.
        alpha : float
            Gradient reversal scaling parameter.

        Returns
        -------
        tuple
            Contains:
            - loss : torch.Tensor
                Combined loss from multiple components.
            - source_logits : torch.Tensor
                Source domain predictions.
            - target_logits : torch.Tensor
                Target domain predictions.

        Notes
        -----
        Implements multiple components:

        - Asymmetric propagation (s_pnums vs t_pnums)
        - Classification loss on source domain
        - Domain adaptation (adversarial or MMD)
        - Feature bottleneck processing
        """

        # source domain cross entropy loss
        source_logits = self.a2gnn(source_data, self.s_pnums)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        if self.mode == 'node':
            source_batch = None
            target_batch = None
        else:
            source_batch = source_data.batch
            target_batch = target_data.batch

        source_features = self.a2gnn.feat_bottleneck(source_data.x, source_data.edge_index, source_batch, self.s_pnums)
        target_features = self.a2gnn.feat_bottleneck(target_data.x, target_data.edge_index, target_batch, self.t_pnums)

        # Adv loss
        if self.adv:
            source_dlogits = self.a2gnn.domain_classifier(source_features, alpha)
            target_dlogits = self.a2gnn.domain_classifier(target_features, alpha)
            
            domain_label = torch.tensor(
                [0] * source_data.x.shape[0] + [1] * target_data.x.shape[0]
                ).to(self.device)
            
            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)
            loss = loss + self.alpha * domain_loss
        else:
            # MMD loss
            mmd_loss = MMD(source_features, target_features)
            loss = loss + self.alpha * mmd_loss

        # Laplacian loss
        laplacian_loss = self.compute_laplacian_loss(target_features, target_data.edge_index_smooth)
        loss = loss + self.beta * laplacian_loss

        target_logits = self.a2gnn(target_data, self.t_pnums)

        return loss, source_logits, target_logits

    def smoothness(self, edge_index, edge_attr, num_nodes):
        """
        Compute smoothness-based edge connections using random walk or k-hop neighbors.
    
        This method constructs smoothness-based adjacency matrices using either
        random walk sampling or k-hop neighbor expansion, depending on the
        smooth_mode parameter.
    
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor of shape (2, num_edges) representing the graph structure.
        edge_attr : torch.Tensor or None
            Edge attribute tensor of shape (num_edges, feature_dim) or None.
        num_nodes : int
            Number of nodes in the graph.
    
        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:

            - edge_index : torch.Tensor
            Modified edge index tensor with smoothness-based connections.

            - edge_attr : torch.Tensor or None
            Modified edge attribute tensor or None.
    
        Notes
        -----
        If smooth_mode is 'RW':
        
            - Performs random walk sampling with length rw_len
            
            - Creates dense adjacency matrix from walk paths
            
            - Converts back to sparse format
        
        If smooth_mode is 'K-hop':
            
            - For k=1: adds self-loops to existing edges
            
            - For k>1: applies TwoHopNeighbor transformation (k-1) times
            
            - Adds remaining self-loops to the expanded graph
    
        Examples
        --------
        >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        >>> edge_attr = torch.randn(4, 10)
        >>> num_nodes = 3
        >>> smooth_edge_index, smooth_edge_attr = self.smoothness(edge_index, edge_attr, num_nodes)
        """

        if self.smooth_mode == 'RW':
            row, col = edge_index
            start = torch.arange(num_nodes, device=edge_index.device)
            walk = random_walk(row, col, start, walk_length=self.rw_len)
            adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=edge_index.device)
            adj[walk[start], start.unsqueeze(1)] = 1.0
            edge_index, edge_attr = dense_to_sparse(adj)
        else:
            if self.k == 1:
                edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr)
            else:
                neighbor_augment = TwoHopNeighbor()
                hop_data = Data(edge_index=edge_index, edge_attr=edge_attr)
                hop_data.num_nodes = num_nodes
                for _ in range(self.k-1):
                    hop_data = neighbor_augment(hop_data)
                hop_edge_index = hop_data.edge_index
                hop_edge_attr = hop_data.edge_attr
                edge_index, edge_attr = add_remaining_self_loops(hop_edge_index, hop_edge_attr, num_nodes=num_nodes)
    
        return edge_index, edge_attr
    
    def compute_laplacian_loss(self, features, edge_index):
        """
        Compute the Laplacian regularization loss for graph smoothness.
    
        This method computes the normalized Laplacian loss which encourages
        smoothness in node features across connected nodes in the graph.
        The loss is computed as the sum of squared differences between
        normalized features of connected nodes.
    
        Parameters
        ----------
        features : torch.Tensor
            Node feature tensor of shape (num_nodes, feature_dim).
        edge_index : torch.Tensor
            Edge index tensor of shape (2, num_edges) representing the graph structure.
    
        Returns
        -------
        torch.Tensor
            Scalar tensor representing the Laplacian regularization loss.
            The loss is normalized by dividing by 2 to account for
            double counting of edges.
    
        Notes
        -----
        The computation follows these steps:
        
        - Compute node degrees from edge_index
        
        - Calculate degree inverse square root for normalization
        
        - Normalize features using degree normalization
        
        - Compute squared differences between connected nodes
        
        - Weight by edge weights (all ones in this implementation)
        
        - Sum and divide by 2 to avoid double counting
    
        The loss encourages connected nodes to have similar features,
        promoting graph smoothness and regularization.
    
        Examples
        --------
        >>> features = torch.randn(10, 64)
        >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        >>> loss = self.compute_laplacian_loss(features, edge_index)
        >>> print(loss.item())  # Scalar loss value
        """
        
        edge_weight = torch.ones(edge_index.size(1), device=features.device)
        row, col = edge_index.to(features.device)

        deg = torch.zeros(features.size(0), device=features.device)
        deg = deg.scatter_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

        normalized_features_row = features[row] * deg_inv_sqrt[row].view(-1, 1)
        normalized_features_col = features[col] * deg_inv_sqrt[col].view(-1, 1)
        feature_diff = normalized_features_row - normalized_features_col

        laplacian_loss = (feature_diff).pow(2).sum(dim=1) * edge_weight
        
        return laplacian_loss.sum() / 2.


    def fit(self, source_data, target_data):
        """
        Train the A2GNN model on source and target domain data.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process includes:

        Data Preparation

        - Configures loaders for node/graph level tasks
        - Handles both full-batch and mini-batch scenarios
        - Sets up appropriate batch processing

        Training Loop

        - Dynamic adaptation parameter scaling
        - Asymmetric message propagation
        - Domain adaptation through either:
            
            * Adversarial training
            * MMD minimization

        - Comprehensive progress monitoring

        Implementation Features

        - Flexible task handling (node/graph)
        - Efficient batch processing
        - Adaptive learning mechanisms
        """

        if self.mode == 'node':
            self.num_source_nodes, _ = source_data.x.shape
            self.num_target_nodes, _ = target_data.x.shape

            print('before smoothness')
            print(target_data.edge_index.shape)

            target_data.edge_index_smooth, target_data.edge_attr_smooth = self.smoothness(target_data.edge_index, target_data.edge_attr, self.num_target_nodes)

            print('after smoothness')
            print(target_data.edge_index_smooth.shape)

            if self.batch_size == 0:
                self.source_batch_size = source_data.x.shape[0]
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.source_batch_size)
                self.target_batch_size = target_data.x.shape[0]
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.target_batch_size)
            else:
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
        elif self.mode == 'graph':
            if self.batch_size == 0:
                num_source_graphs = len(source_data)
                num_target_graphs = len(target_data)
                self.source_loader = DataLoader(source_data, batch_size=num_source_graphs, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=num_target_graphs, shuffle=True)
            else:
                self.source_loader = DataLoader(source_data, batch_size=self.batch_size, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=self.batch_size, shuffle=True)
        else:
            assert self.mode in ('graph', 'node'), 'Invalid train mode'

        self.a2gnn = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.a2gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            p = float(epoch) / self.epoch
            alpha = 2. / (1. + np.exp(-10. * p)) - 1

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.a2gnn.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, alpha)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = source_logits, sampled_source_data.y
                else:
                    source_logits, source_labels = source_logits, sampled_source_data.y
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
            
            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(epoch_source_labels, epoch_source_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)
    
    def process_graph(self, data):
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data to be processed.

        Notes
        -----
        Placeholder method as preprocessing is handled through:
        
        - Asymmetric propagation mechanisms
        - Domain-specific feature processing
        - Batch-wise data handling
        """

    def predict(self, data, source=False):
        """
        Make predictions on input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        source : bool, optional
            Whether predicting on source domain. Default: ``False``.

        Returns
        -------
        tuple
            Contains:
            - logits : torch.Tensor
                Model predictions.
            - labels : torch.Tensor
                True labels.

        Notes
        -----
        - Uses different propagation steps for source/target
        - Handles batch processing efficiently
        - Concatenates results for full predictions
        - Maintains evaluation mode consistency
        """

        self.a2gnn.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.a2gnn(sampled_data, self.s_pnums)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))
        else:
            for idx, sampled_data in enumerate(self.target_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.a2gnn(sampled_data, self.t_pnums)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
