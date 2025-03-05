import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch.nn import Sequential, Linear
from .node_centric_conv import NodeCentricConv, NodeCentricMLP

from torch_geometric.nn import global_mean_pool


class GraphATABase(nn.Module):
    """
    Base model for GraphATA.

    This is the base implementation of a graph neural network that uses
    node-centric attention mechanisms for feature transformation and classification.
    It supports both node-level and graph-level tasks through a flexible architecture.

    Parameters
    ----------
    in_dim : int
        Input dimension of model (number of node features).
    hid_dim : int
        Hidden dimension of model for intermediate representations.
    num_classes : int
        Number of output classes for classification.
    model_weights : tuple
        Collection of model weights for feature transformation layers.
    model_list : list
        List of models for ensemble classification.
    num_layers : int, optional
        Total number of GNN layers in model. Default: ``1``.
    dropout : float, optional
        Dropout rate for regularization. Default: ``0.1``.
    act : callable activation function or None, optional
        Activation function between GNN layers.
        Default: ``torch.nn.functional.relu``.
    gnn : string, optional
        The backbone GNN model type.
        Default: ``gcn``.
    mode : str, optional
        Operation mode, either 'node' for node-level or 'graph' for graph-level tasks.
        Default: ``node``.
    **kwargs : optional
        Additional parameters for the GNN backbone.

    Attributes
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden layer dimension.
    num_classes : int
        Number of output classes.
    num_layers : int
        Number of GNN layers.
    dropout : float
        Dropout probability.
    act : callable
        Activation function.
    mode : str
        Task mode ('node' or 'graph').
    convs : torch.nn.ModuleList
        List of NodeCentricConv layers.
    cls : NodeCentricMLP
        Classification layer with attention-based ensemble.

    Notes
    -----
    Architecture components:

    - Feature Transformation:
        
        * Multiple NodeCentricConv layers
        * Activation and dropout between layers
        * Attention-based feature aggregation

    - Task-Specific Processing:
        
        * Node-level: Direct classification
        * Graph-level: Global pooling before classification

    - Classification:
        
        * Ensemble of models with attention
        * Task-specific output handling
        * Log-softmax normalization

    Examples
    --------
    >>> model = GraphATABase(
    ...     in_dim=64,
    ...     hid_dim=32,
    ...     num_classes=7,
    ...     model_weights=weights,
    ...     model_list=models,
    ...     num_layers=2,
    ...     mode='node'
    ... )
    >>> x = torch.randn(100, 64)  # 100 nodes, 64 features
    >>> edge_index = torch.randint(0, 100, (2, 400))  # 400 edges
    >>> out = model(x, edge_index)  # Shape: (100, 7)
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 model_weights,
                 model_list,
                 num_layers=1,
                 dropout=0.1,
                 act=F.relu,
                 gnn='gcn',
                 mode='node',
                 **kwargs):
        super(GraphATABase, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        self.mode = mode

        self.convs = nn.ModuleList()

        self.convs.append(NodeCentricConv(self.in_dim, self.hid_dim, model_weights[0]))

        for i in range(self.num_layers - 1):
            self.convs.append(NodeCentricConv(self.hid_dim, self.hid_dim, model_weights[i+1]))

        self.cls = NodeCentricMLP(self.num_classes, model_list)
            
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Forward pass of the GNN model.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape (num_nodes, in_dim).
        edge_index : torch.Tensor
            Edge indices, shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights for weighted graph operations.
            Default: ``None``.
        batch : torch.Tensor, optional
            Batch vector for graph-level tasks, indicating node-to-graph assignment.
            Shape: (num_nodes,). Default: ``None``.

        Returns
        -------
        torch.Tensor
            Log-softmax probabilities:

            - For node mode: shape (num_nodes, num_classes)
            - For graph mode: shape (num_graphs, num_classes)

        Notes
        -----
        Process:

        - Feature transformation through GNN layers:
            
            * Sequential application of NodeCentricConv
            * Attention-based feature aggregation
            * Activation and dropout between layers

        - Task-specific processing:
            
            * Node mode: Direct use of node features
            * Graph mode: Global mean pooling over nodes

        - Classification:
            
            * Attention-based ensemble classification
            * Log-softmax normalization
        """
        x = self.feat_bottleneck(x, edge_index, edge_weight, batch)
        x = self.feat_classifier(x, edge_index, edge_weight) 

        x = F.log_softmax(x, dim=1)

        return x
    
    def feat_bottleneck(self, x, edge_index, edge_weight=None, batch=None):
        """
        Feature extraction through GNN layers.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape (num_nodes, in_dim).
        edge_index : torch.Tensor
            Edge indices, shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights for weighted graph operations.
            Default: ``None``.
        batch : torch.Tensor, optional
            Batch vector for graph-level tasks, indicating node-to-graph assignment.
            Shape: (num_nodes,). Default: ``None``.

        Returns
        -------
        torch.Tensor
            Transformed node features, shape (num_nodes, hid_dim).

        Notes
        -----
        Process:

        - Sequential GNN layer application:
            
            * NodeCentricConv transformation
            * Feature aggregation with attention

        - Regularization:
            
            * Activation between layers (except last)
            * Dropout for training stability
            * Residual connections through attention
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.mode == 'graph':
            x = global_mean_pool(x, batch)

        return x
    
    def feat_classifier(self, x, edge_index, edge_weight=None, batch=None):
        """
        Final classification layer.

        Parameters
        ----------
        x : torch.Tensor
            Node features from bottleneck, shape (num_nodes, hid_dim).
        edge_index : torch.Tensor
            Edge indices, shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights for weighted graph operations.
            Default: ``None``.
        batch : torch.Tensor, optional
            Batch vector for graph-level tasks, indicating node-to-graph assignment.
            Shape: (num_nodes,). Default: ``None``.

        Returns
        -------
        torch.Tensor
            Classification logits:

            - Node mode: shape (num_nodes, num_classes)
            - Graph mode: shape (num_graphs, num_classes)

        Notes
        -----
        Operation modes:

        - Node mode:

          * Uses GNN classifier with edge information
          * Maintains graph structure in classification

        - Graph mode:

          * Uses pooled features
          * Global classification without edge information
        """
        if self.mode == 'node':
            x = self.cls(x, edge_index, edge_weight)
        else:
            x = self.cls(x, edge_index, edge_weight)
        
        return x
