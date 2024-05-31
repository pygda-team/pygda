import torch
from torch import nn
import torch.nn.functional as F

from .prop_gcn_conv import PropGCNConv
from .reverse_layer import GradReverse


class A2GNNBase(nn.Module):
    """
    Rethinking Propagation for Unsupervised Graph Domain Adaptation (AAAI-24).

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
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 num_layers=1,
                 adv=False,
                 dropout=0.1,
                 act=F.relu,
                 **kwargs):
        super(A2GNNBase, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.adv = adv
        self.dropout = dropout
        self.act = act

        self.convs = nn.ModuleList()

        self.convs.append(PropGCNConv(self.in_dim, self.hid_dim))

        for _ in range(self.num_layers - 1):
            self.convs.append(PropGCNConv(self.hid_dim, self.hid_dim))

        self.cls = PropGCNConv(self.hid_dim, self.num_classes)

        if self.adv:
            self.domain_discriminator = nn.Linear(self.hid_dim, 2)
            
    def forward(self, data, prop_nums):
        x, edge_index = data.x, data.edge_index
        x = self.feat_bottleneck(x, edge_index, prop_nums=prop_nums)
        x = self.feat_classifier(x, edge_index, prop_nums=1)

        return x
    
    def feat_bottleneck(self, x, edge_index, prop_nums=30):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, prop_nums)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    
    def feat_classifier(self, x, edge_index, prop_nums=1):
        x = self.cls(x, edge_index, prop_nums)
        
        return x
    
    def domain_classifier(self, x, alpha):
        d_logit = self.domain_discriminator(GradReverse.apply(x, alpha))
        
        return d_logit
