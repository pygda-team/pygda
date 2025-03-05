import os.path as osp
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import Data
import torch.nn.functional as F

from typing import Any
from torch import Tensor

from torch_sparse import coalesce


def grad_with_checkpoint(outputs, inputs):
    """
    Compute gradients with checkpointing for memory efficiency.

    Parameters
    ----------
    outputs : torch.Tensor
        Output tensor requiring gradient computation
    inputs : torch.Tensor or tuple of torch.Tensor
        Input tensor(s) to compute gradients with respect to

    Returns
    -------
    list[torch.Tensor]
        List of gradient tensors for each input

    Notes
    -----
    Processing Steps:

    - Handle single/multiple inputs
    - Retain gradients for non-leaf tensors
    - Compute backward pass
    - Clone and clear gradients

    Features:
    
    - Memory efficient
    - Multiple input support
    - Gradient preservation
    - Clean gradient states
    """
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs


def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    """
    Convert linear indices to upper triangular matrix indices.

    Parameters
    ----------
    n : int
        Size of the square matrix
    lin_idx : torch.Tensor
        Linear indices to convert

    Returns
    -------
    torch.Tensor
        Stack of row and column indices (2, num_indices)

    Notes
    -----
    Processing Steps:

    - Calculate row indices
    - Calculate column indices
    - Stack indices together

    Features:
    
    - Matrix coordinate conversion
    - Efficient computation
    - Double precision handling
    """
    row_idx = (
        n
        - 2
        - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
    ).long()
    col_idx = (
        lin_idx
        + row_idx
        + 1 - n * (n - 1) // 2
        + (n - row_idx) * ((n - row_idx) - 1) // 2
    )
    return torch.stack((row_idx, col_idx))


def inner(t1, t2):
    """
    Compute normalized inner product distance between tensors.

    Parameters
    ----------
    t1 : torch.Tensor
        First tensor of shape (batch_size, feature_dim)
    t2 : torch.Tensor
        Second tensor of shape (batch_size, feature_dim)

    Returns
    -------
    torch.Tensor
        Mean normalized inner product distance

    Notes
    -----
    Processing Steps:

    - Normalize input tensors
    - Compute inner products
    - Calculate mean distance

    Features:
    
    - L2 normalization
    - Numerical stability
    - Batch processing
    """
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def inner_margin(t1, t2, margin):
    """
    Compute margin-based inner product distance between tensors.

    Parameters
    ----------
    t1 : torch.Tensor
        First tensor of shape (batch_size, feature_dim)
    t2 : torch.Tensor
        Second tensor of shape (batch_size, feature_dim)
    margin : float
        Margin threshold for distance computation

    Returns
    -------
    torch.Tensor
        Mean margin-based distance

    Notes
    -----
    Processing Steps:

    - Normalize input tensors
    - Compute inner products
    - Apply margin threshold
    - Calculate mean distance

    Features:
    
    - Margin-based learning
    - L2 normalization
    - ReLU activation
    """
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return F.relu(1-(t1 * t2).sum(1)-margin).mean()

def diff(t1, t2):
    """
    Compute normalized squared Euclidean distance between tensors.

    Parameters
    ----------
    t1 : torch.Tensor
        First tensor of shape (batch_size, feature_dim)
    t2 : torch.Tensor
        Second tensor of shape (batch_size, feature_dim)

    Returns
    -------
    torch.Tensor
        Mean normalized squared distance

    Notes
    -----
    Processing Steps:

    - Normalize input tensors
    - Compute squared differences
    - Calculate mean distance

    Features:
    
    - L2 normalization
    - Euclidean distance
    - Batch processing
    """
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return 0.5*((t1-t2)**2).sum(1).mean()


def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    """
    Find root using bisection method for edge weight adjustment.

    Parameters
    ----------
    edge_weights : torch.Tensor
        Edge weights to adjust
    a : float
        Lower bound
    b : float
        Upper bound
    n_perturbations : int
        Target number of perturbations
    epsilon : float, optional
        Convergence threshold. Default: 1e-5
    iter_max : int, optional
        Maximum iterations. Default: 1e5

    Returns
    -------
    float
        Root value for weight adjustment

    Notes
    -----
    Processing Steps:

    - Define target function
    - Iterative bisection
    - Check convergence
    - Update bounds

    Features:
    
    - Numerical optimization
    - Convergence control
    - Iteration limiting
    """
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu) * func(a) < 0):
            b = miu
        else:
            a = miu
        if ((b - a) <= epsilon):
            break
    return miu


def project(n_perturbations, values, eps, inplace=False):
    """
    Project values onto constrained space with perturbation budget.

    Parameters
    ----------
    n_perturbations : int
        Number of allowed perturbations
    values : torch.Tensor
        Values to project
    eps : float
        Small constant for numerical stability
    inplace : bool, optional
        Whether to modify values in-place. Default: False

    Returns
    -------
    torch.Tensor
        Projected values

    Notes
    -----
    Processing Steps:

    - Check perturbation budget
    - Find projection threshold
    - Apply clamping
    - Handle numerical bounds

    Features:
    
    - Constraint satisfaction
    - Memory efficiency
    - Numerical stability
    """
    if not inplace:
        values = values.clone()

    if torch.clamp(values, 0, 1).sum() > n_perturbations:
        left = (values - 1).min()
        right = values.max()
        miu = bisection(values, left, right, n_perturbations)
        values.data.copy_(torch.clamp(
            values - miu, min=eps, max=1 - eps
        ))
    else:
        values.data.copy_(torch.clamp(values, min=eps, max=1 - eps))
    
    return values


def get_modified_adj(modified_edge_index, perturbed_edge_weight, n, device, edge_index, edge_weight, make_undirected=False):
    """
    Create modified adjacency matrix with perturbed edges.

    Parameters
    ----------
    modified_edge_index : torch.Tensor
        Modified edge indices
    perturbed_edge_weight : torch.Tensor
        Perturbed edge weights
    n : int
        Number of nodes
    device : torch.device
        Device to store tensors
    edge_index : torch.Tensor
        Original edge indices
    edge_weight : torch.Tensor
        Original edge weights
    make_undirected : bool, optional
        Whether to make graph undirected. Default: False

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Modified edge indices and weights

    Notes
    -----
    Processing Steps:

    - Handle undirected option
    - Combine edges
    - Coalesce edges
    - Adjust weights

    Features:
    
    - Graph modification
    - Edge coalescing
    - Weight adjustment
    - Undirected support
    """
    if make_undirected:
        modified_edge_index, modified_edge_weight = to_symmetric(modified_edge_index, perturbed_edge_weight, n)
    else:
        modified_edge_index, modified_edge_weight = modified_edge_index, perturbed_edge_weight
    edge_index = torch.cat((edge_index.to(device), modified_edge_index), dim=-1)
    edge_weight = torch.cat((edge_weight.to(device), modified_edge_weight))
    edge_index, edge_weight = coalesce(edge_index, edge_weight, m=n, n=n, op='sum')

    # Allow removal of edges
    edge_weight[edge_weight > 1] = 2 - edge_weight[edge_weight > 1]
    return edge_index, edge_weight


def to_symmetric(edge_index, edge_weight, n, op='mean'):
    """
    Convert directed graph to undirected by symmetrization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices
    edge_weight : torch.Tensor
        Edge weights
    n : int
        Number of nodes
    op : str, optional
        Operation for combining weights. Default: 'mean'

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Symmetric edge indices and weights

    Notes
    -----
    Processing Steps:

    - Mirror edges
    - Duplicate weights
    - Coalesce edges
    - Combine weights

    Features:
    
    - Edge symmetrization
    - Weight handling
    - Efficient coalescing
    - Operation flexibility
    """
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)

    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight
