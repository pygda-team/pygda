from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_scatter import scatter

from torch_geometric.utils import add_remaining_self_loops

from torch_geometric.nn.conv import MessagePassing, SimpleConv
from torch_geometric.typing import Adj, OptTensor, PairTensor, OptPairTensor, Size

from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd


def _make_ix_like(input, dim=0):
    """
    Create an index tensor for sparsemax computation.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of any dimension
    dim : int, optional
        Dimension along which to create indices (default is 0)

    Returns
    -------
    torch.Tensor
        Index tensor with same device and dtype as input
    """
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """
    Compute the threshold and support for sparsemax operation.

    This is a building block for the sparsemax function that computes
    the threshold value and support size.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of any dimension
    dim : int, optional
        Dimension along which to apply the sparsemax (default is 0)

    Returns
    -------
    tau : torch.Tensor
        Threshold value
    support_size : torch.Tensor
        Size of the support
    """
    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


class SparsemaxFunction(Function):
    """
    Custom autograd function for sparsemax operation.
    
    Implements the forward and backward passes for the sparsemax
    normalization, which is a sparse alternative to softmax.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, dim=0):
        """
        Forward pass of sparsemax.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object to save variables for backward
        input : torch.Tensor
            Input tensor of any shape
        dim : int, optional
            Dimension along which to apply sparsemax (default is 0)

        Returns
        -------
        torch.Tensor
            Output tensor with same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        Backward pass of sparsemax.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object with saved variables
        grad_output : torch.Tensor
            Gradient of the loss with respect to sparsemax output

        Returns
        -------
        tuple
            (gradient with respect to input, None)
        """
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    """
    Sparsemax activation function module.

    A sparse alternative to softmax that produces sparse probability
    distributions.

    Parameters
    ----------
    dim : int, optional
        Dimension along which to apply sparsemax (default is 0)
    """
    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class NodeCentricConv(MessagePassing):
    """
    Node-centric graph convolution layer.

    This layer implements a node-centric message passing mechanism with
    attention and weight-based feature transformation. It combines neighbor
    aggregation with learnable attention mechanisms and feature transformations.

    Parameters
    ----------
    in_channels : Union[int, Tuple[int, int]]
        Size of input features. If tuple, specifies different sizes for
        source and target nodes. If int, same size is used for both.
    out_channels : int
        Size of output features
    model_weights : tuple, optional
        Tuple of model weights for feature transformation (default is ())
    lamb : float, optional
        Scaling factor for the base transformation (default is 0.2)
    aggr : str, optional
        Aggregation scheme to use ('mean', 'sum', 'max') (default is 'mean')
    **kwargs : dict
        Additional arguments for MessagePassing base class

    Attributes
    ----------
    in_channels : Union[int, Tuple[int, int]]
        Input feature dimensions
    out_channels : int
        Output feature dimension
    weights_list : list
        List of transposed weight matrices for feature transformation
    lamb : float
        Scaling factor for base transformation
    att : torch.nn.Parameter
        Attention weight matrix of shape (out_channels, 1)
    weight : torch.nn.Parameter
        Transform weight matrix of shape (in_channels, out_channels)
    neigh_aggr : SimpleConv
        Neighbor aggregation layer with mean aggregation
    sparse_attention : Sparsemax
        Sparsemax attention mechanism for dimension 1

    Notes
    -----
    The forward pass consists of several steps:

    1. Adds self-loops to the input graph
    2. Aggregates neighbor features
    3. Applies feature transformation with attention mechanism
    4. Combines transformed features with base transformation
    """
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            model_weights: tuple = (),
            lamb: float = 0.2,
            aggr: str = 'mean',
            **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.weights_list = list()
        for weight in model_weights:
            self.weights_list.append(weight.t())

        self.lamb = lamb

        self.att = Parameter(torch.Tensor(self.out_channels, 1))
        nn.init.xavier_normal_(self.att)
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        nn.init.xavier_normal_(self.weight)

        self.neigh_aggr = SimpleConv(aggr='mean')

        self.sparse_attention = Sparsemax(dim=1)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_type: OptTensor = None) -> Tensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        x : Union[torch.Tensor, PairTensor]
            Node feature matrix of shape (num_nodes, in_channels) or 
            pair of node feature matrices
        edge_index : Adj
            Graph connectivity in COO format of shape (2, num_edges)
        edge_type : OptTensor, optional
            Edge type information (default is None)

        Returns
        -------
        torch.Tensor
            Updated node features of shape (num_nodes, out_channels)

        Notes
        -----
        The forward pass performs these operations:

        1. Adds self-loops to edge_index
        2. Aggregates neighbor features using SimpleConv
        3. Applies multiple feature transformations with attention
        4. Combines transformed features with base transformation
        """
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        neigh_rep = self.neigh_aggr(x, edge_index)
        
        atts = []
        reps = []

        out = self.propagate(edge_index, x=x, gamma=torch.sigmoid(neigh_rep), edge_weight=None, size=None)
        
        for i, weight in enumerate(self.weights_list):
            rep = torch.matmul(neigh_rep, weight)
            res = torch.matmul(rep, self.att)
            atts.append(res)
            rep = torch.matmul(out, weight)
            reps.append(rep)
        atts = torch.cat(atts, dim=1)
        w = self.sparse_attention(atts)
        gamma = torch.stack(reps)
        w = w.t().unsqueeze(-1)

        wg = torch.matmul(neigh_rep, self.weight)
        gamma = torch.sum(w * gamma, dim=0)

        out = gamma + wg * self.lamb

        return out
    
    def message(self, x_j: Tensor, gamma_i: Tensor, edge_weight: OptTensor) -> Tensor:
        """
        Compute messages between nodes.

        Parameters
        ----------
        x_j : torch.Tensor
            Features of source nodes of shape (num_edges, in_channels)
        gamma_i : torch.Tensor
            Transformation coefficient for target nodes of shape (num_edges, in_channels)
        edge_weight : OptTensor
            Optional edge weights of shape (num_edges,)

        Returns
        -------
        torch.Tensor
            Computed messages of shape (num_edges, in_channels)

        Notes
        -----
        The message function performs element-wise multiplication between
        source node features and transformation coefficients.
        """
        out = gamma_i * x_j
        
        return out


class NodeCentricMLP(torch.nn.Module):
    """
    Node-centric multi-layer perceptron.

    This module implements a node-centric MLP with attention-based
    model ensemble.

    Parameters
    ----------
    num_classes : int
        Number of classes
    model_list : list
        List of models to ensemble

    Attributes
    ----------
    att : torch.nn.Parameter
        Attention weight matrix
    sparse_attention : Sparsemax
        Sparsemax attention mechanism
    """
    def __init__(self, num_classes, model_list):
        super(NodeCentricMLP, self).__init__()
        self.num_classes = num_classes
        self.model_list = model_list

        self.att = Parameter(torch.Tensor(self.num_classes, 1))
        nn.init.xavier_normal_(self.att)

        self.sparse_attention = Sparsemax(dim=1)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the module.

        Parameters
        ----------
        x : torch.Tensor
            Input node features

        Returns
        -------
        torch.Tensor
            Ensemble output after attention-weighted combination
        """
        outputs = []
        weights = []
        for i in range(len(self.model_list)):
            if self.model_list[i].mode == 'node':
                cls_output = self.model_list[i].cls(x, edge_index, edge_weight)
            else:
                cls_output = self.model_list[i].cls(x)
            att = torch.matmul(cls_output, self.att)
            outputs.append(cls_output)
            weights.append(att)
        weights = torch.cat(weights, dim=1)
        w = self.sparse_attention(weights)

        outputs = torch.stack(outputs)
        w = w.t().unsqueeze(-1)
        x = torch.sum(w * outputs, dim=0)

        return x
