import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GRADEBase(nn.Module):
    """
    Non-IID Transfer Learning on Graphs (AAAI-23)

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
                 **kwargs):
        super(GRADEBase, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

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
        x, edge_index = data.x, data.edge_index
        x, feat_list = self.feat_bottleneck(x, edge_index)
        x = self.feat_classifier(x)

        feat_list.append(x)
        feat_list = torch.cat(feat_list, dim=1)

        return x, feat_list
    
    def feat_bottleneck(self, x, edge_index):
        feat_list = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            feat_list.append(x)

        return x, feat_list
    
    def feat_classifier(self, x):
        x = self.cls(x)
        
        return x
    
    def one_hot_embedding(self, labels):
        y = torch.eye(self.num_classes, device=labels.device)
        return y[labels]
