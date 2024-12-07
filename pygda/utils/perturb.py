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
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def inner_margin(t1, t2, margin):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return F.relu(1-(t1 * t2).sum(1)-margin).mean()

def diff(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return 0.5*((t1-t2)**2).sum(1).mean()


def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
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
