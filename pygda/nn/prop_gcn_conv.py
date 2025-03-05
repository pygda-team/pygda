from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, 
    #       Optional[int]) -> PairTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    """
    Compute symmetric normalization for graph convolution.

    Parameters
    ----------
    edge_index : torch.Tensor or SparseTensor
        Edge indices or sparse adjacency matrix
    edge_weight : torch.Tensor, optional
        Edge weights. Default: None
    num_nodes : int, optional
        Number of nodes. Default: None
    improved : bool, optional
        If True, use A + 2I instead of A + I. Default: False
    add_self_loops : bool, optional
        Whether to add self-loops. Default: True
    dtype : torch.dtype, optional
        Data type for computations. Default: None

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] or SparseTensor
        Normalized edge indices and weights, or normalized sparse tensor
    """
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class PropGCNConv(MessagePassing):
    """
    Propagation Graph Convolutional Network layer with multiple propagation steps.

    Parameters
    ----------
    in_channels : int
        Size of input features
    out_channels : int
        Size of output features
    improved : bool, optional
        If True, use A + 2I instead of A + I. Default: False
    cached : bool, optional
        Whether to cache normalized adjacency matrix. Default: False
    add_self_loops : bool, optional
        Whether to add self-loops. Default: True
    normalize : bool, optional
        Whether to apply symmetric normalization. Default: True
    bias : bool, optional
        Whether to include bias. Default: True
    **kwargs : optional
        Additional MessagePassing arguments

    Notes
    -----
    - Multiple propagation steps
    - Cached normalization
    - Sparse tensor support
    - Configurable self-loops
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(PropGCNConv, self).__init__(**kwargs)

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


    def forward(self, x: Tensor, edge_index: Adj,
                prop_nums = 1, edge_weight: OptTensor = None) -> Tensor:
        """
        Forward pass with multiple propagation steps.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix
        edge_index : torch.Tensor or SparseTensor
            Edge indices or sparse adjacency matrix
        prop_nums : int, optional
            Number of propagation steps. Default: 1
        edge_weight : torch.Tensor, optional
            Edge weights. Default: None

        Returns
        -------
        torch.Tensor
            Output node features

        Notes
        -----
        - Graph normalization (if enabled)
        - Linear transformation
        - Multiple propagation steps
        - Optional bias addition
        """

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

        out = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        for i in range(prop_nums):
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight, 
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        """
        Define message computation.

        Parameters
        ----------
        x_j : torch.Tensor
            Source node features
        edge_weight : torch.Tensor, optional
            Edge weights

        Returns
        -------
        torch.Tensor
            Computed messages

        Notes
        -----
        Applies edge weights if provided, otherwise passes features directly
        """
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        """
        Fused message and aggregation computation.

        Parameters
        ----------
        adj_t : SparseTensor
            Sparse adjacency matrix
        x : torch.Tensor
            Node features

        Returns
        -------
        torch.Tensor
            Aggregated messages

        Notes
        -----
        Optimized sparse matrix multiplication for efficiency
        """
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)