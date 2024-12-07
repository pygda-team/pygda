import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch.nn import Sequential, Linear

from torch_geometric.utils.convert import to_networkx

from ..utils import RandomWalker, Negative_Sampler


class SOGABase(nn.Module):
    """
    Source Free Graph Unsupervised Domain Adaptation (WSDM-24).

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
    gnn : string, optional
        The backbone of GNN model.
        Default: ``gcn``.
    num_negative_samples : int, optional
        The number of negative samples in NCE loss.
        Default: ``5``.
    num_positive_samples : int, optional
        The number of positive samples in NCE loss.
        Default: ``2``.
    device : str, optional
        GPU or CPU. Default: ``cuda:0``.
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
                 gnn='gcn',
                 num_negative_samples=5,
                 num_positive_samples=2,
                 device='cuda:0',
                 **kwargs):
        super(SOGABase, self).__init__()
        
        assert gnn in ('gcn', 'sage', 'gat', 'gin'), 'Invalid gnn backbone'
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn = gnn
        self.act = act
        self.num_negative_samples = num_negative_samples
        self.num_positive_samples = num_positive_samples
        self.device = device

        self.cce = CustomizedCrossEntropy(num_classes, device)

        self.convs = nn.ModuleList()

        if self.gnn == 'gcn':
            self.convs.append(GCNConv(self.in_dim, self.hid_dim))
            
            for _ in range(self.num_layers - 1):
                self.convs.append(GCNConv(self.hid_dim, self.hid_dim))
            
        elif self.gnn == 'sage':
            self.convs.append(SAGEConv(self.in_dim, self.hid_dim))
            
            for _ in range(self.num_layers - 1):
                self.convs.append(SAGEConv(self.hid_dim, self.hid_dim))

        elif self.gnn == 'gat':
            self.convs.append(GATConv(self.in_dim, self.hid_dim, heads=1, concat=False))
            
            for _ in range(self.num_layers - 1):
                self.convs.append(GATConv(self.hid_dim, self.hid_dim, heads=1, concat=False))

        elif self.gnn == 'gin':
            self.convs.append(GINConv(Sequential(Linear(self.in_dim, self.hid_dim)), train_eps=True))

            for _ in range(self.num_layers - 1):
                self.convs.append(GINConv(Sequential(Linear(self.hid_dim, self.hid_dim)), train_eps=True))
        
        self.cls = Linear(self.hid_dim, self.num_classes)

            
    def forward(self, x, edge_index, edge_weight=None):
        x = self.feat_bottleneck(x, edge_index, edge_weight)
        x = self.feat_classifier(x) 

        return x
    
    def feat_bottleneck(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    
    def feat_classifier(self, x):
        x = self.cls(x)
        
        return x
    
    def init_target(self, graph_struct, graph_neigh):
        self.target_G_struct = to_networkx(graph_struct)
        self.target_G_neigh = to_networkx(graph_neigh)
        
        self.Positive_Sampler = RandomWalker(self.target_G_struct, p=0.25, q=2, use_rejection_sampling=1)
        self.Negative_Sampler = Negative_Sampler(self.target_G_struct)
        self.center_nodes_struct, self.positive_samples_struct = self.generate_positive_samples()
        self.negative_samples_struct = self.generate_negative_samples(graph_struct.x.size(0))

        self.Positive_Sampler = RandomWalker(self.target_G_neigh, p=0.25, q=2, use_rejection_sampling=1)
        self.Negative_Sampler = Negative_Sampler(self.target_G_struct)
        self.center_nodes_neigh, self.positive_samples_neigh = self.generate_positive_samples()
        self.negative_samples_neigh = self.generate_negative_samples(graph_neigh.x.size(0))
    
    def generate_positive_samples(self):
        self.Positive_Sampler.preprocess_transition_probs()        
        self.positive_samples = self.Positive_Sampler.simulate_walks(num_walks=1, walk_length=self.num_positive_samples, workers=1, verbose=1)
        for i in range(len(self.positive_samples)):
            if len(self.positive_samples[i]) != 2:
                self.positive_samples[i].append(self.positive_samples[i][0])

        samples = torch.tensor(self.positive_samples).to(self.device)

        center_nodes = torch.unsqueeze(samples[:, 0], dim=-1)
        positive_samples = torch.unsqueeze(samples[:, 1], dim=-1)

        return center_nodes, positive_samples

    def generate_negative_samples(self, num_target_nodes):
        negative_samples = torch.tensor([self.Negative_Sampler.sample() for _ in range(self.num_negative_samples * num_target_nodes)]).view([num_target_nodes, self.num_negative_samples]).to(self.device)

        return negative_samples


class CustomizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super(CustomizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.device = device
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        batch_size, feature_dim = outputs.size()
        log_probs = self.logsoftmax(outputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        targets = targets.to(self.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
