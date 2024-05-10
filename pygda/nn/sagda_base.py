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
        x = self.conv(x, x, edge_index)
        output = torch.mm(x, self.weight)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class SrcGNN(torch.nn.Module):
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
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
        
        return x


class TgtGNN(torch.nn.Module):
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
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = self.act(x)
                x = self.dropout_layers[i](x)
        return x


class SAGDABase(nn.Module):
    """
    SA-GDA: Spectral Augmentation for Graph Domain Adaptation (MM-23)

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
        encoded_output = self.src_encoder(x, edge_index)

        if mask is not None:
            encoded_output = encoded_output[mask]
        
        return encoded_output
    
    def gcn_encode(self, data, cache_name, mask=None):
        encoded_output = self.encoder(data.x, data.edge_index, cache_name)
        
        if mask is not None:
            encoded_output = encoded_output[mask]
        
        return encoded_output
    
    def ppmi_encode(self, data, cache_name, mask=None):
        encoded_output = self.ppmi_encoder(data.x, data.edge_index, cache_name)
        
        if mask is not None:
            encoded_output = encoded_output[mask]
            
        return encoded_output

    def encode(self, data, cache_name, mask=None):
        gcn_output = self.gcn_encode(data, cache_name, mask)
        
        if self.ppmi:
            ppmi_output = self.ppmi_encode(data, cache_name, mask)
            outputs = self.att_model([gcn_output, ppmi_output])
            return outputs
        else:
            return gcn_output
