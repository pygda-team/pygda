import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb


class BernProp(MessagePassing):
    """
    K-order Bernstein polynomial approximation.

    Parameters
    ----------
    K : int
        Order of the polynomial filter. Determines the complexity of the spectral filter.
    is_source_domain : bool, optional
        Whether this layer is used for source domain. If True, temperature parameters
        are learnable. If False, they are fixed to a linear interpolation from 1 to 0.
        Default: ``True``.
    bias : bool, optional
        Whether to add bias. Currently not used but kept for compatibility.
        Default: ``True``.
    **kwargs : optional
        Additional keyword arguments passed to the MessagePassing parent class.

    Notes
    -----
    This class implements a graph neural network layer that performs spectral domain
    adaptation. It uses a polynomial filter approach with learnable temperature parameters 
    to adapt the spectral characteristics of the graph.

    The layer computes a polynomial filter of order K and applies it to the graph
    signal through message passing operations. The filter coefficients are determined
    by learnable temperature parameters that differ between source and target domains.

    Attributes
    ----------
    K : int
        Order of the polynomial filter.
    is_source_domain : bool
        Whether this layer is for source domain.
    cached_terms : torch.Tensor, optional
        Cached polynomial terms for filter computation.
    cached_coefs : torch.Tensor, optional
        Cached polynomial coefficients.
    temp : nn.Parameter
        Learnable temperature parameters for the filter.
    """

    def __init__(self, K, is_source_domain=True, bias=True, **kwargs):
        super(BernProp, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.is_source_domain = is_source_domain
        self.cached_terms = None
        self.cached_coefs = None
        self.temp = nn.Parameter(torch.Tensor(self.K + 1), requires_grad=is_source_domain)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the learnable parameters of the layer.

        Notes
        -----
        - For source domain layers, temperature parameters are initialized to 1.

        - For target domain layers, temperature parameters are set to a linear
          interpolation from 1 to 0 over K+1 values.
        """
        if self.is_source_domain:
            self.temp.data.fill_(1)
        else:
            self.temp.data = torch.linspace(1, 0, self.K + 1)

    def get_filter(self):
        """
        Compute the spectral filter using cached terms and coefficients.

        Returns
        -------
        torch.Tensor
            The computed spectral filter H.

        Notes
        -----
        This method requires cached_terms and cached_coefs to be set before calling.
        The filter is computed as a weighted sum of polynomial terms.
        """
        TEMP = F.relu(self.temp)
        H = 0

        for k in range(self.K + 1):
            H = H + TEMP[k] * self.cached_coefs[k] * self.cached_terms[k]

        return H

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the DGSD layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, num_features].
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weights of shape [num_edges]. If None, all edges are assumed
            to have weight 1. Default: ``None``.

        Returns
        -------
        torch.Tensor
            Updated node features after spectral domain adaptation.

        Notes
        -----
        This method implements the spectral domain adaptation by:

        - Computing the symmetric normalized Laplacian

        - Adding self-loops with appropriate weights

        - Propagating messages through the graph using polynomial filters

        - Combining the results using binomial coefficients and temperature parameters
        """
        TEMP = F.relu(self.temp)

        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
        return out

    def message(self, x_j, norm):
        """
        Message function for message passing.

        Parameters
        ----------
        x_j : torch.Tensor
            Source node features of shape [num_edges, num_features].
        norm : torch.Tensor
            Normalized edge weights of shape [num_edges].

        Returns
        -------
        torch.Tensor
            Messages of shape [num_edges, num_features].
        """
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        """
        String representation of the layer.

        Returns
        -------
        str
            String representation showing the class name, filter order K, and
            temperature parameters.
        """
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.temp)


class DGSDABase(nn.Module):
    """
    Base class for DGSDA.

    Parameters
    ----------
    features : int
        Input feature dimension.
    hidden : int
        Hidden layer dimension.
    classes : int
        Number of output classes.
    dprate : float, optional
        Dropout rate for propagation layers. Default: ``0.0``.
    K : int, optional
        Order of the polynomial filter for propagation layers. Default: ``15``.

    Notes
    -----
    This class implements a domain generalization model using spectral domain
    adaptation. It uses separate propagation layers for source and target domains
    to adapt the spectral characteristics of the graph data.

    The model consists of:

    - Linear transformation layers

    - Bernoulli propagation layers with polynomial filters

    - Domain-specific propagation paths
    """

    def __init__(self, features, hidden, classes, dprate=0.0, K=15):
        super(DGSDABase, self).__init__()
        self.lin1 = nn.Linear(features, hidden)
        self.lin2 = nn.Linear(hidden, classes)
        self.prop1 = BernProp(K)
        self.prop2 = BernProp(K)
        self.prop3 = BernProp(K)

        self.dprate = dprate

    def reset_parameters(self):
        """
        Reset the learnable parameters of the model.

        Notes
        -----
        Currently only resets the first propagation layer (prop1).
        """
        self.prop1.reset_parameters()

    def forward(self, data, is_source_domain=True):
        """
        Forward pass of the DGSDA model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data containing node features and edge indices.
        is_source_domain : bool, optional
            Whether the input is from source domain. Determines which
            propagation layer to use. Default: ``True``.

        Returns
        -------
        torch.Tensor
            Model predictions for node classification.

        Notes
        -----
        The forward pass consists of:

        - Domain-specific feature propagation

        - Dropout regularization

        - Final linear classification

        - Final propagation step
        """
        x, edge_index = data.x, data.edge_index

        x = self.get_props(x, edge_index, is_source_domain)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop3(x, edge_index)
        return x

    def get_props(self, x, edge_index, is_source_domain=True):
        """
        Apply domain-specific feature propagation.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, num_features].
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, num_edges].
        is_source_domain : bool, optional
            Whether to use source domain propagation layer. If True, uses prop1;
            if False, uses prop2. Default: ``True``.

        Returns
        -------
        torch.Tensor
            Propagated node features.

        Notes
        -----
        This method implements the domain-specific feature processing:

        - Initial dropout and ReLU activation

        - Domain-specific propagation using Bernoulli polynomial filters

        - Separate propagation paths for source and target domains
        """
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=self.dprate, training=self.training)
        if is_source_domain:
            x = self.prop1(x, edge_index)
        else:
            x = self.prop2(x, edge_index)
        return x
