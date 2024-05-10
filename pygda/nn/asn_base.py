import torch
from torch import nn
import torch.nn.functional as F
from .ppmi_conv import PPMIConv
from .cached_gcn_conv import CachedGCNConv
from .reverse_layer import GradReverse
from .attention import Attention

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class GNNVAE(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes, gnn_type='gcn', num_layers=3, base_model=None, act=F.relu, **kwargs):
        super(GNNVAE, self).__init__()

        assert num_layers == 3, 'Invalid values'

        if base_model is None:
            weights = [None] * num_layers
            biases = [None] * num_layers
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.gnn_type = gnn_type
        self.act = act
        self.num_layers = num_layers

        model_cls = PPMIConv if gnn_type == 'ppmi' else GCNConv

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(model_cls(in_dim, hid_dim))

        for idx in range(1, num_layers - 2):
            self.conv_layers.append(model_cls(hid_dim, hid_dim))
        
        self.conv_layers.append(model_cls(hid_dim, num_classes))
        self.conv_layers.append(model_cls(hid_dim, num_classes))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 2):
            x = self.conv_layers[i](x, edge_index)
            x = self.act(x)
            x = self.dropout_layers[i](x)

        mu = self.conv_layers[-2](x, edge_index)
        logvar = self.conv_layers[-1](x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


class ASNBase(nn.Module):
    """
    Adversarial Separation Network for Cross-Network Node Classification (CIKM-21)

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim : int
        Hidden dimension of model.
    hid_dim_vae : int
        Hidden dimension of vae model.
    num_classes : int
        Number of classes.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    adv_dim : int, optional
        Hidden dimension of adversarial module. Default: ``40``.
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 hid_dim_vae,
                 num_classes,
                 num_layers=3,
                 act=F.relu,
                 dropout=0.1,
                 adv_dim=40,
                 **kwargs):
        super(ASNBase, self).__init__()

        self.private_encoder_s_l = GNNVAE(in_dim=in_dim, hid_dim=hid_dim, num_classes=hid_dim_vae, act=act, num_layers=num_layers)
        self.private_encoder_t_l = GNNVAE(in_dim=in_dim, hid_dim=hid_dim, num_classes=hid_dim_vae, act=act, num_layers=num_layers)
        self.private_encoder_s_g = GNNVAE(in_dim=in_dim, hid_dim=hid_dim, num_classes=hid_dim_vae, act=act, num_layers=num_layers)
        self.private_encoder_t_g = GNNVAE(in_dim=in_dim, hid_dim=hid_dim, num_classes=hid_dim_vae, act=act, num_layers=num_layers)

        self.decoder_s = InnerProductDecoder(dropout=dropout, act=lambda x: x)
        self.decoder_t = InnerProductDecoder(dropout=dropout, act=lambda x: x)

        self.shared_encoder_l = GNNVAE(in_dim=in_dim, hid_dim=hid_dim, num_classes=hid_dim_vae, act=act, num_layers=num_layers)
        self.shared_encoder_g = GNNVAE(in_dim=in_dim, hid_dim=hid_dim, num_classes=hid_dim_vae, act=act, num_layers=num_layers)

        self.cls_model = nn.Sequential(nn.Linear(hid_dim_vae, num_classes))

        self.domain_model = nn.Sequential(
            nn.Linear(hid_dim_vae, adv_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adv_dim, 2)
        )

        self.att_model = Attention(hid_dim_vae)
        self.att_model_self_s = Attention(hid_dim_vae)
        self.att_model_self_t = Attention(hid_dim_vae)

        self.models = [
            self.private_encoder_s_l,
            self.private_encoder_s_g,
            self.private_encoder_t_l,
            self.private_encoder_t_g,
            self.shared_encoder_g,
            self.shared_encoder_l,
            self.cls_model,
            self.domain_model,
            self.decoder_s,
            self.decoder_t,
            self.att_model,
            self.att_model_self_s,
            self.att_model_self_t
        ]
        
        self.cls_loss = nn.CrossEntropyLoss()
        self.loss_diff = DiffLoss()
    
    def recon_loss(self, preds, labels, mu, logvar, n_nodes, norm, pos_weight):
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        
        return cost + KLD
    
    def adj_label_for_reconstruction(self, data):
        A = to_dense_adj(data.edge_index)
        A = A.squeeze(dim=0)
        adj_label = A + torch.eye(A.shape[0]).to(data.x.device)
        pos_weight = (A.shape[0] * A.shape[0] - A.sum()) / A.sum()
        pos_weight = pos_weight.reshape(1, 1)
        norm = A.shape[0] * A.shape[0] / ((A.shape[0] * A.shape[0] - A.sum()) * 2)

        return adj_label, pos_weight, norm
