import torch
from torch import nn
import torch.nn.functional as F
from .ppmi_conv import PPMIConv
from .cached_gcn_conv import CachedGCNConv
from .attention import Attention
from torch.nn import Parameter
import math

from torch_geometric.nn import FAConv
from torch_geometric.nn.inits import glorot, zeros


class SAGNN(torch.nn.Module):
    """
    Structure Aware Graph Neural Network layer.

    Parameters
    ----------
    in_features : int
        Input feature dimensionality
    out_features : int
        Output feature dimensionality
    alpha : float
        Attention coefficient for feature aggregation
    beta : float
        Balance coefficient for structure learning
    weight : torch.Tensor, optional
        Pre-defined weight matrix. Default: None
    bias : torch.Tensor, optional
        Pre-defined bias vector. Default: None

    Notes
    -----
    Combines feature attention and structure learning:

    - Uses FAConv for feature-attention convolution
    - Learnable transformation matrix
    - Optional bias term
    """

    def __init__(self, in_features, out_features, alpha, beta, weight=None, bias=None):
        super(SAGNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.alpha = alpha
        self.beta = beta

        self.conv = FAConv(in_features)

        if weight is None:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            glorot(self.weight)
        else:
            self.weight = weight

        if bias is None:
            self.bias = Parameter(torch.FloatTensor(out_features))
            zeros(self.bias)
        else:
            self.bias = bias
    
    def forward(self, x, edge_index):
        """
        Forward pass of SAGNN layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_features]
        edge_index : torch.Tensor
            Graph connectivity in COO format [2, num_edges]

        Returns
        -------
        torch.Tensor
            Updated node features [num_nodes, out_features]

        Notes
        -----
        - Feature-attention convolution
        - Linear transformation
        - Optional bias addition
        """
        x = self.conv(x, x, edge_index)
        output = torch.mm(x, self.weight)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SrcGNN(torch.nn.Module):
    """
    Source domain GNN with structure awareness.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality
    hid_dim : int
        Hidden feature dimensionality
    alpha : float, optional
        Attention coefficient. Default: 0.5
    beta : float, optional
        Structure learning coefficient. Default: 0.5
    num_layers : int, optional
        Number of SAGNN layers. Default: 3
    act : callable, optional
        Activation function. Default: F.relu
    base_model : torch.nn.Module, optional
        Base model for weight initialization. Default: None

    Notes
    -----
    - Multiple SAGNN layers
    - Dropout regularization
    - Weight sharing option with base model
    - Configurable depth and activation
    """

    def __init__(self, in_dim, hid_dim, alpha=0.5, beta=0.5, num_layers=3, act=F.relu, base_model=None):
        super(SrcGNN, self).__init__()

        self.dropout_layers = [nn.Dropout(0.1) for _ in range(num_layers)]
        self.act = act
        self.alpha = alpha
        self.beta = beta

        if base_model is None:
            weights = [None] * num_layers
            biases = [None] * num_layers
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        model_cls = SAGNN

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(model_cls(in_dim, hid_dim, alpha=self.alpha, beta=self.beta, weight=weights[0], bias=biases[0]))

        for idx in range(1, num_layers):
            self.conv_layers.append(model_cls(hid_dim, hid_dim, alpha=self.alpha, beta=self.beta, weight=weights[idx], bias=biases[idx]))
    
    def forward(self, x, edge_index):
        """
        Forward pass through source GNN.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_dim]
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges]

        Returns
        -------
        torch.Tensor
            Final node representations [num_nodes, hid_dim]

        Notes
        -----
        Sequential processing through SAGNN layers
        """
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
        
        return x


class TgtGNN(torch.nn.Module):
    """
    Target domain GNN with flexible architecture.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality
    hid_dim : int
        Hidden feature dimensionality
    gnn_type : str, optional
        Type of GNN layer ('gcn' or 'ppmi'). Default: 'gcn'
    num_layers : int, optional
        Number of GNN layers. Default: 3
    base_model : torch.nn.Module, optional
        Base model for weight initialization. Default: None
    act : callable, optional
        Activation function. Default: F.relu
    **kwargs : optional
        Additional arguments for GNN layers

    Notes
    -----
    - Choice of GNN type (GCN or PPMI)
    - Multiple layers with dropout
    - Weight sharing capability
    - Cached computation support
    """

    def __init__(self, in_dim, hid_dim, gnn_type='gcn', num_layers=3, base_model=None, act=F.relu, **kwargs):
        super(TgtGNN, self).__init__()

        if base_model is None:
            weights = [None] * num_layers
            biases = [None] * num_layers
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.gnn_type = gnn_type
        self.act = act

        model_cls = PPMIConv if gnn_type == 'ppmi' else CachedGCNConv

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(model_cls(in_dim, hid_dim, weight=weights[0], bias=biases[0], **kwargs))

        for idx in range(1, num_layers):
            self.conv_layers.append(model_cls(hid_dim, hid_dim, weight=weights[idx], bias=biases[idx], **kwargs))

    def forward(self, x, edge_index, cache_name):
        """
        Forward pass through target GNN.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_dim]
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges]
        cache_name : str
            Identifier for caching computations

        Returns
        -------
        torch.Tensor
            Final node representations [num_nodes, hid_dim]

        Notes
        -----
        - Layer-wise propagation
        - Intermediate activation
        - Dropout regularization
        - Cache-aware computation
        """
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = self.act(x)
                x = self.dropout_layers[i](x)
        return x


class SAGDABase(nn.Module):
    """
    Base class for SAGDA.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Number of classes.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    alpha   : float, optional
        Trade-off parameter for high pass filter. Default: ``0.5``.
    beta    : float, optional
        Trade-off parameter for low pass filter. Default: ``0.5``.
    ppmi: bool, optional
        Use PPMI matrix or not. Default: ``True``.
    adv_dim : int, optional
        Hidden dimension of adversarial module. Default: ``40``.
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 num_layers=3,
                 dropout=0.1,
                 act=F.relu,
                 beta=0.5,
                 alpha=0.5,
                 ppmi=True,
                 adv_dim=40,
                 **kwargs):
        super(SAGDABase, self).__init__()

        self.ppmi = ppmi

        self.encoder = TgtGNN(in_dim=in_dim, hid_dim=hid_dim, gnn_type='gcn', act=act, num_layers=num_layers)
        
        if self.ppmi:
            self.ppmi_encoder = TgtGNN(in_dim=in_dim, hid_dim=hid_dim, gnn_type='ppmi', base_model=self.encoder, path_len=10, num_layers=num_layers) 
        
        self.src_encoder = SrcGNN(in_dim=in_dim, hid_dim=hid_dim, num_layers=num_layers, alpha=alpha, beta=beta, base_model=self.encoder)
        
        self.cls_model = nn.Sequential(nn.Linear(hid_dim, num_classes))

        self.domain_model = nn.Sequential(
            nn.Linear(hid_dim, adv_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adv_dim, 2)
        )

        self.att_model = Attention(hid_dim)

        self.models = [self.encoder, self.cls_model, self.domain_model, self.ppmi_encoder, self.att_model, self.src_encoder]
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def src_encode(self, x, edge_index, mask=None):
        """
        Encode source domain data using structure-aware GNN.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix [num_nodes, in_dim]
        edge_index : torch.Tensor
            Graph connectivity in COO format [2, num_edges]
        mask : torch.Tensor, optional
            Boolean mask for node selection. Default: None

        Returns
        -------
        torch.Tensor
            Encoded node features [num_selected_nodes, hid_dim]

        Notes
        -----
        Uses SrcGNN with structure awareness and feature attention
        """
        encoded_output = self.src_encoder(x, edge_index)

        if mask is not None:
            encoded_output = encoded_output[mask]
        
        return encoded_output
    
    def gcn_encode(self, data, cache_name, mask=None):
        """
        Encode data using standard GCN encoder.

        Parameters
        ----------
        data : Data
            Graph data object containing node features and edge indices
        cache_name : str
            Identifier for caching computations
        mask : torch.Tensor, optional
            Boolean mask for node selection. Default: None

        Returns
        -------
        torch.Tensor
            GCN-encoded node features [num_selected_nodes, hid_dim]

        Notes
        -----
        Standard GCN encoding with caching support
        """
        encoded_output = self.encoder(data.x, data.edge_index, cache_name)
        
        if mask is not None:
            encoded_output = encoded_output[mask]
        
        return encoded_output
    
    def ppmi_encode(self, data, cache_name, mask=None):
        """
        Encode data using PPMI-based GNN encoder.

        Parameters
        ----------
        data : Data
            Graph data object containing node features and edge indices
        cache_name : str
            Identifier for caching computations
        mask : torch.Tensor, optional
            Boolean mask for node selection. Default: None

        Returns
        -------
        torch.Tensor
            PPMI-encoded node features [num_selected_nodes, hid_dim]

        Notes
        -----
        PPMI-based encoding capturing higher-order structure
        """
        encoded_output = self.ppmi_encoder(data.x, data.edge_index, cache_name)
        
        if mask is not None:
            encoded_output = encoded_output[mask]
            
        return encoded_output

    def encode(self, data, cache_name, mask=None):
        """
        Multi-view encoding combining GCN and optional PPMI features.

        Parameters
        ----------
        data : Data
            Graph data object containing node features and edge indices
        cache_name : str
            Identifier for caching computations
        mask : torch.Tensor, optional
            Boolean mask for node selection. Default: None

        Returns
        -------
        torch.Tensor
            Final encoded features [num_selected_nodes, hid_dim]

        Notes
        -----
        - GCN encoding
        - Optional PPMI encoding
        - Attention-based feature fusion if PPMI enabled
        """
        gcn_output = self.gcn_encode(data, cache_name, mask)
        
        if self.ppmi:
            ppmi_output = self.ppmi_encode(data, cache_name, mask)
            outputs = self.att_model([gcn_output, ppmi_output])
            return outputs
        else:
            return gcn_output
