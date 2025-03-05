import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention mechanism for feature aggregation.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.

    Notes
    -----
    - Implements learnable attention weights
    - Uses softmax normalization
    - Includes dropout regularization
    - Single-head attention mechanism
    """

    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        """
        Apply attention mechanism to input features.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            List of input tensors to be attended.

        Returns
        -------
        torch.Tensor
            Attention-weighted feature aggregation.

        Notes
        -----
        Process:
        
        1. Stack input tensors
        2. Compute attention weights
        3. Apply softmax normalization
        4. Weighted sum of features
        """

        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs
