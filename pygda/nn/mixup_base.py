import torch
from torch import nn
import torch.nn.functional as F
import copy

from ..nn import MixUpGCNConv
from torch.nn import Linear


class MixupBase(nn.Module):
    """
    GNN mixup base model for graph data augmentation.

    Parameters
    ----------
    in_dim : int
        Input dimension of model
    hid_dim : int
        Hidden dimension of model
    num_classes : int
        Number of classes
    num_layers : int, optional
        Total number of layers in model. Default: 1
    dropout : float, optional
        Dropout rate. Default: 0.1
    act : callable, optional
        Activation function. Default: torch.nn.functional.relu
    rw_lmda : float, optional
        Edge reweight hyperparameter. Default: 0.8
    **kwargs : optional
        Additional parameters for the backbone

    Notes
    -----
    Architecture:

    - Multiple MixUpGCNConv layers
    - Intermediate feature mixing
    - Final classification layer
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 num_layers=1,
                 dropout=0.1,
                 act=F.relu,
                 rw_lmda=0.8,
                 **kwargs):
        super(MixupBase, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        self.rw_lmda = rw_lmda

        self.convs = nn.ModuleList()

        self.convs.append(MixUpGCNConv(self.in_dim, self.hid_dim))
            
        for _ in range(self.num_layers - 1):
            self.convs.append(MixUpGCNConv(self.hid_dim, self.hid_dim))

        self.cls = Linear(self.hid_dim, self.num_classes)
            
    def forward(self, x, edge_index, edge_index_b, lam, id_new_value_old, edge_weight):
        """
        Forward pass of the MixupBase model.

        Parameters
        ----------
        x : torch.Tensor
            Input node features
        edge_index : torch.Tensor
            Edge indices of original graph
        edge_index_b : torch.Tensor
            Edge indices of mixed graph
        lam : float
            Mixup interpolation coefficient
        id_new_value_old : torch.Tensor
            Mapping between original and mixed node indices
        edge_weight : torch.Tensor
            Edge weights for graph convolution

        Returns
        -------
        torch.Tensor
            Classification logits for each node
        """
        x = self.feat_bottleneck(x, edge_index, edge_index_b, lam, id_new_value_old, edge_weight)
        x = self.feat_classifier(x)

        return x
    
    def feat_bottleneck(self, x, edge_index, edge_index_b, lam, id_new_value_old, edge_weight):
        """
        Feature extraction and mixing through GNN layers.

        Parameters
        ----------
        x : torch.Tensor
            Input node features
        edge_index : torch.Tensor
            Edge indices of original graph
        edge_index_b : torch.Tensor
            Edge indices of mixed graph
        lam : float
            Mixup interpolation coefficient
        id_new_value_old : torch.Tensor
            Mapping between original and mixed node indices
        edge_weight : torch.Tensor
            Edge weights for graph convolution

        Returns
        -------
        torch.Tensor
            Mixed node features after GNN processing

        Notes
        -----
        - Initial feature propagation (layers 0-1)
        - Feature mixing with interpolation
        - Additional layer processing (if num_layers > 2)
        - Dropout and activation at each step
        """
        x1 = self.convs[0](x, x, edge_index, edge_weight, self.rw_lmda)
        x1 = self.act(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.convs[1](x1, x1, edge_index, edge_weight, self.rw_lmda)
        x2 = self.act(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x0_b = x[id_new_value_old]
        x1_b = x1[id_new_value_old]

        x_mix = x * lam + x0_b * (1 - lam)

        new_x1 = self.convs[0](x, x_mix, edge_index, edge_weight, self.rw_lmda)
        new_x1_b = self.convs[0](x0_b, x_mix, edge_index_b, edge_weight, self.rw_lmda)
        new_x1 = self.act(new_x1)
        new_x1_b = self.act(new_x1_b)

        x1_mix = new_x1 * lam + new_x1_b * (1 - lam)
        x1_mix = F.dropout(x1_mix, p=self.dropout, training=self.training)

        new_x2 = self.convs[1](x1, x1_mix, edge_index, edge_weight, self.rw_lmda)
        new_x2_b = self.convs[1](x1_b, x1_mix, edge_index_b, edge_weight, self.rw_lmda)
        new_x2 = self.act(new_x2)
        new_x2_b = self.act(new_x2_b)

        x2_mix = new_x2 * lam + new_x2_b * (1 - lam)
        x2_mix = F.dropout(x2_mix, p=self.dropout, training=self.training)

        x = x2
        x_mix = x2_mix

        for i in range(2, len(self.convs)):
            x_t = self.convs[i](x, x, edge_index, edge_weight, self.rw_lmda)
            x_t = self.act(x_t)
            x_t = F.dropout(x_t, p=self.dropout, training=self.training)
            x_b = x[id_new_value_old]

            new_x = self.convs[i](x, x_mix, edge_index, edge_weight, self.rw_lmda)
            new_x_b = self.convs[i](x_b, x_mix, edge_index_b, edge_weight, self.rw_lmda)
            new_x = self.act(new_x)
            new_x_b = self.act(new_x_b)

            x_mix = new_x * lam + new_x_b * (1 - lam)
            x_mix = F.dropout(x_mix, p=self.dropout, training=self.training)

            x = x_t

        return x_mix
    
    def feat_classifier(self, x):
        """
        Final classification layer.

        Parameters
        ----------
        x : torch.Tensor
            Input features from bottleneck

        Returns
        -------
        torch.Tensor
            Classification logits for each node

        Notes
        -----
        Applies linear transformation to get class predictions
        """
        x = self.cls(x)
        
        return x
