import torch
import torch.nn as nn

from .a2gnn_base import A2GNNBase


class SampleNet(nn.Module):
    """Generate frequencies for characteristic-function alignment.

    Parameters
    ----------
    feature_dim : int, optional
        Dimension of each frequency vector. Default: 128.
    t_batchsize : int, optional
        Nominal number of output frequencies. Default: 64.
    t_var : float, optional
        Gaussian noise variance before sampling. Default: 1.
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
        """Draw frequencies on the requested device.

        Parameters
        ----------
        device : str or torch.device
            Device for frequency computation and output.

        Returns
        -------
        torch.Tensor
            Frequency vectors with `feature_dim` columns.

        Notes
        -----
        The learned path transforms Gaussian noise and repeats the outputs.
        If the number of learned vectors is zero, it draws Gaussian
        frequencies directly with variance `t_var / feature_dim`.
        Both ADAlign training phases use this forward in training mode.
        Only sampler ascent steps the sampler optimizer, but BatchNorm
        running statistics update during both phases.
        """
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


def CF(alpha, beta, source_features, target_features, t):
    """Compute the amplitude and phase discrepancy between two domains.

    Parameters
    ----------
    alpha : float
        Weight of the squared amplitude discrepancy.
    beta : float
        Weight of the phase discrepancy.
    source_features : torch.Tensor
        Source node embeddings.
    target_features : torch.Tensor
        Target node embeddings.
    t : torch.Tensor
        Frequencies with the same width as the embeddings.

    Returns
    -------
    torch.Tensor
        Scalar mean of the weighted distances over frequencies.

    Notes
    -----
    The author implementation clamps the phase term before taking the
    square root. The gradients can be singular at zero complex modulus
    or at zero discrepancy when beta is zero.
    """
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


class ADAlignBase(A2GNNBase):
    """A2GNN backbone used by ADAlign.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden feature dimension.
    num_classes : int
        Number of output classes.
    num_layers : int, optional
        Number of propagation layers before classification. Default: 1.
    adv : bool, optional
        Add the inherited domain classifier. Default: False.
    dropout : float, optional
        Dropout probability in the feature bottleneck. Default: 0.1.
    act : callable, optional
        Activation after each propagation layer. Default: torch.nn.functional.relu.
    mode : str, optional
        Backbone task type. Default: 'node'.
    **kwargs : dict
        Additional keywords accepted by A2GNNBase. Default: empty.

    Notes
    -----
    The inherited forward takes a PyG graph and a propagation count.
    It returns class logits after running the bottleneck with that count
    and the node classifier with one propagation step. CF alignment uses
    bottleneck embeddings from a separate forward call.
    """


__all__ = ["ADAlignBase", "SampleNet", "CF"]
