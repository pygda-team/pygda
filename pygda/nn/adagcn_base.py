import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from .ppmi_conv import PPMIConv
from torch_geometric.nn import global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, gnn_type='gcn', num_layers=3, act=F.relu, dropout=0.1, **kwargs):
        super(GNN, self).__init__()

        self.gnn_type = gnn_type
        self.act = act
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()

        if self.gnn_type == 'gcn':
            self.conv_layers.append(GCNConv(in_dim, hid_dim))

            for i in range(1, self.num_layers):
                self.conv_layers.append(GCNConv(hid_dim, hid_dim))
        else:
            self.conv_layers.append(PPMIConv(in_dim, hid_dim))

            for i in range(1, self.num_layers):
                self.conv_layers.append(PPMIConv(hid_dim, hid_dim))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, mode='node'):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = self.act(x)
                x = self.dropout(x)
        
        if mode == 'graph':
            x = global_mean_pool(x, batch)
        
        return x


class AdaGCNBase(nn.Module):
    """
    Graph Transfer Learning via Adversarial Domain Adaptation with Graph Convolution (TKDE-22).

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
    gnn_type: string, optional
        Use GCN or PPMIConv. Default: ``gcn``.
    mode : str, optional
        Mode for node or graph level tasks. Default: ``node``.
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 num_layers=3,
                 dropout=0.1,
                 act=F.relu,
                 gnn_type='gcn',
                 mode='node',
                 **kwargs):
        super(AdaGCNBase, self).__init__()

        self.encoder = GNN(in_dim=in_dim, hid_dim=hid_dim, gnn_type=gnn_type, act=act, num_layers=num_layers)
        
        self.cls_model = nn.Sequential(nn.Linear(hid_dim, num_classes))

        self.mode = mode
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, data):
        if self.mode == 'node':
            x, edge_index, batch = data.x, data.edge_index, None
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.encoder(x, edge_index, batch, mode=self.mode)

        return x
