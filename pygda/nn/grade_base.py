import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GRADEBase(nn.Module):
    """
    Base class for GRADE.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Number of classes.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    disc : str, optional
        Discriminator. Default: ``JS``.
    mode : str, optional
        Mode for node or graph level tasks. Default: ``node``.
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 num_layers=1,
                 dropout=0.1,
                 act=F.relu,
                 disc='JS',
                 mode='node',
                 **kwargs):
        super(GRADEBase, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        self.mode = mode

        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(self.in_dim, self.hid_dim))

        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hid_dim, self.hid_dim))

        self.cls = nn.Linear(self.hid_dim, self.num_classes)

        if disc == "JS":
            self.discriminator = nn.Sequential(
                nn.Linear(
                    self.hid_dim * self.num_layers + self.num_classes, 2)
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(
                    self.hid_dim * self.num_layers + self.num_classes * 2, 2)
            )

        self.criterion = nn.CrossEntropyLoss()
            
    def forward(self, data):
        """
        Forward pass of the GRADE model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object containing:
            - x: Node features
            - edge_index: Edge indices
            - batch: Batch indices (for graph-level tasks)

        Returns
        -------
        tuple
            Contains:
            - torch.Tensor: Classification logits
            - torch.Tensor: Concatenated features from all layers

        Notes
        -----
        Process:

        1. Extract features through GNN layers
        2. Apply classification layer
        3. Concatenate features for discrimination
        """
        if self.mode == 'node':
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        x, feat_list = self.feat_bottleneck(x, edge_index, batch)
        x = self.feat_classifier(x)

        feat_list.append(x)
        feat_list = torch.cat(feat_list, dim=1)

        return x, feat_list
    
    def feat_bottleneck(self, x, edge_index, batch):
        """
        Feature extraction through GNN layers.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix.
        edge_index : torch.Tensor
            Edge indices.
        batch : torch.Tensor or None
            Batch indices for graph-level tasks.

        Returns
        -------
        tuple
            Contains:
            - torch.Tensor: Final layer features
            - list: Features from each layer

        Notes
        -----
        Process:

        1. Sequential GCN layer application
        2. Activation and dropout
        3. Feature collection per layer
        4. Graph pooling (if graph-level task)
        """
        feat_list = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.mode == 'node':
                feat_list.append(x)
            else:
                feat_list.append(global_mean_pool(x, batch))
        
        if self.mode == 'graph':
            x = global_mean_pool(x, batch)

        return x, feat_list
    
    def feat_classifier(self, x):
        """
        Classification layer.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Classification logits.

        Notes
        -----
        Simple linear transformation for classification.
        """
        x = self.cls(x)
        
        return x
    
    def one_hot_embedding(self, labels):
        """
        Convert labels to one-hot encoding.

        Parameters
        ----------
        labels : torch.Tensor
            Input labels as indices.

        Returns
        -------
        torch.Tensor
            One-hot encoded labels.

        Notes
        -----
        Creates a one-hot encoding matrix of shape (num_samples, num_classes).
        """
        y = torch.eye(self.num_classes, device=labels.device)
        return y[labels]
