import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import uniform

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul


class GraphConv(MessagePassing):
    """
    Basic graph convolution layer with center node handling.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    aggr : str, optional
        Aggregation method ('mean', 'add', etc.). Default: 'mean'
    bias : bool, optional
        Whether to include bias. Default: True
    **kwargs : optional
        Additional arguments for MessagePassing

    Notes
    -----
    Performs message passing with:
    
    - Linear transformation of source nodes
    - Separate transformation for center nodes
    - Message aggregation
    """

    def __init__(self, in_channels, out_channels, aggr='mean', bias=True,
                 **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.lin.reset_parameters()

    def forward(self, x, edge_index, x_cen):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input node features
        edge_index : torch.Tensor
            Edge indices
        x_cen : torch.Tensor
            Center node features

        Returns
        -------
        torch.Tensor
            Updated node features
        """
        h = torch.matmul(x, self.weight)
        aggr_out = self.propagate(edge_index, size=None, h=h, edge_weight=None)
        return aggr_out + self.lin(x_cen)

    def message(self, h_j):
        return h_j

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MixUpGCNConv(gnn.MessagePassing):
    """
    Graph convolutional operator with mixup capabilities.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    improved : bool, optional
        If True, uses A + 2I for self-loops. Default: False
    cached : bool, optional
        Whether to cache normalized adjacency matrix. Default: False
    add_self_loops : bool, optional
        Whether to add self-loops to input graph. Default: False
    normalize : bool, optional
        Whether to apply symmetric normalization. Default: True
    bias : bool, optional
        Whether to include bias. Default: True
    **kwargs : optional
        Additional arguments for MessagePassing
    """

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = False, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        self.lin_cen = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_cen: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, lmda = 1) -> Tensor:
        """
        Forward pass with mixup interpolation.

        Parameters
        ----------
        x : torch.Tensor
            Input node features
        x_cen : torch.Tensor
            Center node features
        edge_index : torch.Tensor or SparseTensor
            Edge indices
        edge_weight : torch.Tensor, optional
            Edge weights. Default: None
        lmda : float, optional
            Mixup interpolation coefficient. Default: 1

        Returns
        -------
        torch.Tensor
            Updated node features with mixup

        Notes
        -----
        - Graph normalization (if enabled)
        - Linear transformation
        - Message passing with mixup
        - Center node transformation
        - Optional bias addition
        """
        edge_rw = edge_weight
        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)
        
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight, lmda=lmda, edge_rw=edge_rw, size=None) + self.lin_cen(x_cen)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor, lmda, edge_rw) -> Tensor:
        """
        Define message computation.

        Parameters
        ----------
        x_j : torch.Tensor
            Source node features
        edge_weight : torch.Tensor
            Normalized edge weights
        lmda : float
            Mixup interpolation coefficient
        edge_rw : torch.Tensor
            Raw edge weights for reweighting

        Returns
        -------
        torch.Tensor
            Computed messages with mixup interpolation

        Notes
        -----
        Interpolates between normalized and reweighted messages:
        message = (1-lambda) * normalized + lambda * reweighted
        """
        x_j = (edge_weight.view(-1, 1) * x_j)
        x_j = (1-lmda) * x_j + (lmda) * (edge_rw.view(-1, 1) * x_j)
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)
