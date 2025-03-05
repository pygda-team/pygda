import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch.nn import Sequential, Linear

from torch_geometric.utils.convert import to_networkx

from ..utils import RandomWalker, Negative_Sampler


class SOGABase(nn.Module):
    """
    Base class for SOGA.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality
    hid_dim : int
        Hidden feature dimensionality
    num_classes : int
        Number of target classes
    num_layers : int, optional
        Number of GNN layers. Default: 1
    dropout : float, optional
        Dropout rate. Default: 0.1
    act : callable, optional
        Activation function. Default: F.relu
    gnn : str, optional
        GNN backbone type ('gcn', 'sage', 'gat', 'gin'). Default: 'gcn'
    num_negative_samples : int, optional
        Number of negative samples for NCE loss. Default: 5
    num_positive_samples : int, optional
        Number of positive samples for NCE loss. Default: 2
    device : str, optional
        Computing device. Default: 'cuda:0'

    Notes
    -----
    - Flexible GNN backbone selection
    - Multi-layer design
    - NCE-based learning
    - Customized cross-entropy
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
        """
        Forward pass through the SOGA model.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_dim]
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges]
        edge_weight : torch.Tensor, optional
            Edge weights [num_edges]. Default: None

        Returns
        -------
        torch.Tensor
            Node classification logits [num_nodes, num_classes]

        Notes
        -----
        Two-stage process:
        
        - Feature transformation (bottleneck)
        - Classification
        """
        x = self.feat_bottleneck(x, edge_index, edge_weight)
        x = self.feat_classifier(x) 

        return x
    
    def feat_bottleneck(self, x, edge_index, edge_weight=None):
        """
        Feature transformation through GNN layers.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_dim]
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges]
        edge_weight : torch.Tensor, optional
            Edge weights [num_edges]. Default: None

        Returns
        -------
        torch.Tensor
            Transformed node features [num_nodes, hid_dim]

        Notes
        -----
        - Sequential GNN layers
        - Intermediate activation
        - Dropout regularization
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    
    def feat_classifier(self, x):
        """
        Classification layer for node features.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, hid_dim]

        Returns
        -------
        torch.Tensor
            Classification logits [num_nodes, num_classes]

        Notes
        -----
        Simple linear transformation from hidden to output dimension
        """
        x = self.cls(x)
        return x
    
    def init_target(self, graph_struct, graph_neigh):
        """
        Initialize target domain samplers and structures.

        Parameters
        ----------
        graph_struct : Data
            Structural graph data
        graph_neigh : Data
            Neighborhood graph data

        Notes
        -----
        - NetworkX graph conversions
        - Random walk samplers
        - Positive/negative samples
        - Both structural and neighborhood views
        """
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
        """
        Generate positive samples using random walks.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Contains:

            - Center nodes [num_samples, 1]
            - Positive samples [num_samples, 1]

        Notes
        -----
        - Biased random walks (p=0.25, q=2)
        - Walk length based on num_positive_samples
        - Single walk per node
        """
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
        """
        Generate negative samples for contrastive learning.

        Parameters
        ----------
        num_target_nodes : int
            Number of nodes in target graph

        Returns
        -------
        torch.Tensor
            Negative samples [num_nodes, num_negative_samples]

        Notes
        -----
        Generates fixed number of negative samples per node
        """
        negative_samples = torch.tensor([self.Negative_Sampler.sample() for _ in range(self.num_negative_samples * num_target_nodes)]).view([num_target_nodes, self.num_negative_samples]).to(self.device)

        return negative_samples


class CustomizedCrossEntropy(nn.Module):
    """
    Label-smoothed cross entropy loss with optional reduction.

    Parameters
    ----------
    num_classes : int
        Number of target classes
    device : str
        Computing device for tensors
    epsilon : float, optional
        Label smoothing factor. Default: 0.1
    reduction : bool, optional
        Whether to return mean loss. Default: True
    """

    def __init__(self, num_classes, device, epsilon=0.1, reduction=True):
        super(CustomizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.device = device
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets):
        """
        Compute label-smoothed cross entropy loss.

        Parameters
        ----------
        outputs : torch.Tensor
            Model predictions [batch_size, num_classes]
        targets : torch.Tensor
            Target class indices [batch_size]

        Returns
        -------
        torch.Tensor
            Loss value (scalar if reduction=True, else [batch_size])

        Notes
        -----
        - Compute log probabilities
        - Create one-hot targets
        - Apply label smoothing
        - Compute loss
        - Optional reduction
        """
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
