"""Official CF and sampler, preserved verbatim below; see ADAlign docs for provenance."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleNet(nn.Module):
    """
    TNet module for adversarial networks with fixed activation layers and predefined parameters.
    """

    def __init__(self, feature_dim=128, t_batchsize=64, t_var=1):
        super(SampleNet, self).__init__()
        self.feature_dim = feature_dim  # Feature dimension
        self.t_sigma_num = t_batchsize // 16  # Number of sigmas for t_net
        self._input_adv_t_net_dim = feature_dim  # Input noise dimension
        self._input_t_dim = feature_dim  # t_net input dimension
        self._input_t_batchsize = t_batchsize  # Batch size
        self._input_t_var = t_var  # Variance of input noise

        # Fixed activation layers
        self.activation_1 = nn.LeakyReLU(negative_slope=0.2)
        self.activation_2 = nn.Tanh()

        # Create a simple 3-layer fully connected network using fixed activation layers
        self.t_layers_list = nn.ModuleList()
        ch_in = self.feature_dim
        num_layer = 3
        for i in range(num_layer):
            self.t_layers_list.append(nn.Linear(ch_in, ch_in))
            self.t_layers_list.append(nn.BatchNorm1d(ch_in))
            # Use activation_1 for the first two layers, and activation_2 for the last layer
            self.t_layers_list.append(
                self.activation_1 if i < (num_layer - 1) else self.activation_2
            )

    def forward(self, device):
        # Generate white noise
        if self.t_sigma_num > 0:
            # Initialize the white noise input
            self._t_net_input = torch.randn(
                self.t_sigma_num, self._input_adv_t_net_dim
            ) * (self._input_t_var**0.5)
            self._t_net_input = self._t_net_input.to(device).detach()

            # Forward pass
            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)

            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)
            self._t = a
        else:
            # When t_sigma_num = 0, generate standard Gaussian noise as t
            self._t = torch.randn(self._input_t_batchsize, self._input_t_dim) * (
                (self._input_t_var / self._input_t_dim) ** 0.5
            )
            self._t = self._t.to(device).detach()
        return self._t
def CF(alpha, beta, source_features, target_features,t):
    #t = torch.randn((4096, source_features.size(1)), device=source_features.device)
    t_x_real = calculate_real(torch.matmul(t, source_features.t()))
    t_x_imag = calculate_imag(torch.matmul(t, source_features.t()))
    t_x_norm = calculate_norm(t_x_real, t_x_imag)

    t_target_real = calculate_real(torch.matmul(t, target_features.t()))
    t_target_imag = calculate_imag(torch.matmul(t, target_features.t()))
    t_target_norm = calculate_norm(t_target_real, t_target_imag)
    # Calculate amplitude difference and phase difference
    amp_diff = t_target_norm - t_x_norm
    loss_amp = torch.mul(amp_diff, amp_diff)

    loss_pha = 2 * (
        torch.mul(t_target_norm, t_x_norm)
        - torch.mul(t_x_real, t_target_real)
        - torch.mul(t_x_imag, t_target_imag)
    )

    loss_pha = loss_pha.clamp(min=1e-12)  # Ensure numerical stability

    # Combine losses
    loss = torch.mean(torch.sqrt(alpha * loss_amp + beta * loss_pha))
    return loss
def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)

from .a2gnn_base import A2GNNBase


class ADAlignBase(A2GNNBase):
    """Shared official A2GNN backbone, with no algorithmic changes."""


__all__ = ["ADAlignBase", "SampleNet", "CF"]

