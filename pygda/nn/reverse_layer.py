import torch


class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.

    Implements a custom autograd function that:
    
    - Forward: Identity operation
    - Backward: Reverses and scales gradients

    """

    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass of gradient reversal.

        Parameters
        ----------
        ctx : torch.autograd.function.Context
            Context object for storing variables for backward.
        x : torch.Tensor
            Input tensor.
        alpha : float
            Gradient scaling factor.

        Returns
        -------
        torch.Tensor
            Input tensor without modification.

        Notes
        -----
        Identity operation in forward pass, stores alpha for backward.
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of gradient reversal.

        Parameters
        ----------
        ctx : torch.autograd.function.Context
            Context object containing saved alpha.
        grad_output : torch.Tensor
            Gradient from subsequent layer.

        Returns
        -------
        tuple
            Contains:
            - torch.Tensor: Reversed and scaled gradient
            - None: For alpha parameter (not needed)

        Notes
        -----
        Implements gradient reversal:
        grad = -alpha * grad_output
        """
        grad_output = grad_output.neg() * ctx.alpha
        return grad_output, None
