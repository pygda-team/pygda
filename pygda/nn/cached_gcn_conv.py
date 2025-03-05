import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.inits import glorot, zeros


class CachedGCNConv(MessagePassing):
    """

    Implementation of the GCN layer from "Semi-supervised Classification with Graph 
    Convolutional Networks" (Kipf & Welling, 2017) with caching capabilities.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    weight : torch.Tensor, optional
        Pre-initialized weight matrix. If None, weights are initialized using Glorot.
    bias : torch.Tensor, optional
        Pre-initialized bias vector. If None, bias is initialized to zeros.
    improved : bool, optional
        If True, uses A + 2I instead of A + I for self-loops. Default is False.
    use_bias : bool, optional
        Whether to use bias term. Default is True.
    **kwargs : dict
        Additional arguments for MessagePassing base class.

    """

    def __init__(self, in_channels, out_channels,
                 weight=None,
                 bias=None,
                 improved=False,
                 use_bias=True,
                 **kwargs):
        super().__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cache_dict = {}

        if weight is None:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels).to(torch.float32))
            glorot(self.weight)
        else:
            self.weight = weight

        if bias is None:
            if use_bias:
                self.bias = Parameter(torch.Tensor(out_channels).to(torch.float32))
            else:
                self.register_parameter('bias', None)
            zeros(self.bias)
        else:
            self.bias = bias

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        """
        Compute normalized adjacency matrix.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices (2, num_edges).
        num_nodes : int
            Number of nodes in the graph.
        edge_weight : torch.Tensor, optional
            Edge weights. Default is None (all ones).
        improved : bool, optional
            Whether to use improved normalization. Default is False.
        dtype : torch.dtype, optional
            Data type for computations.

        Returns
        -------
        tuple
            (edge_index, normalized_weights) where normalized_weights contains the 
            symmetric normalization coefficients.

        Notes
        -----
        Implements D^(-1/2) * A * D^(-1/2) normalization with self-loops.
        """
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, cache_name="default_cache", edge_weight=None):
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix (num_nodes, in_channels).
        edge_index : torch.Tensor
            Edge indices (2, num_edges).
        cache_name : str, optional
            Identifier for cached normalization. Default is "default_cache".
        edge_weight : torch.Tensor, optional
            Edge weights. Default is None.

        Returns
        -------
        torch.Tensor
            Output feature matrix (num_nodes, out_channels).

        Notes
        -----
        Caches the normalized adjacency computation for efficiency in 
        transductive settings.
        """
        x = torch.matmul(x, self.weight)

        if not cache_name in self.cache_dict:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            self.cache_dict[cache_name] = edge_index, norm
        else:
            edge_index, norm = self.cache_dict[cache_name]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """
        Message computation for message passing.

        Parameters
        ----------
        x_j : torch.Tensor
            Features of neighboring nodes.
        norm : torch.Tensor
            Normalization coefficients.

        Returns
        -------
        torch.Tensor
            Normalized messages.
        """
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        """
        Update node embeddings after message aggregation.

        Parameters
        ----------
        aggr_out : torch.Tensor
            Aggregated messages.

        Returns
        -------
        torch.Tensor
            Updated node features with optional bias.
        """
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
