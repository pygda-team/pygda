import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader

from . import BaseGDA
from ..nn import DGDABase, DWPretrain
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class DGDA(BaseGDA):
    """
    Graph Domain Adaptation: A Generative View (TKDD-24).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    num_layers : int, optional
        Total number of layers in model. Default: ``3``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.0005``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    dec_dim : int, optional
        Dimension of the graph decoder hidden layer. Default: ``64``.
    d_dim : int, optional
        Dimension of the domain latent variables. Default: ``64``.
    y_dim : int, optional
        Dimension of the semantic latent variables. Default: ``256``.
    m_dim : int, optional
        Dimension of the random latent variables. Default: ``128``.
    recons_w : float, optional
        Trade-off weight for reconstruction loss. Default: ``1.0``.
    beta : float, optional
        Trade-off weight for kl loss. Default: ``0.5``.
    ent_w : float, optional
        Trade-off weight for entropy loss. Default: ``1.0``.
    d_w : float, optional
        Trade-off weight for domain loss. Default: ``1.0``.
    y_w : float, optional
        Trade-off weight for cross entropy loss. Defalut: ``1.0``.
    m_w : float, optional
        Trade-off weight for manipulating reconstruction loss. Defalut: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.001``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``400``.
    device : str, optional
        GPU or CPU. Default: ``cuda:0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``2``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=3,
        dropout=0.,
        act=F.relu,
        dec_dim=64,
        d_dim=64,
        y_dim=256,
        m_dim=128,
        recons_w=1.0,
        beta=0.5,
        ent_w=1.0,
        d_w=1.0,
        y_w=1.0,
        m_w=0.1,
        edge_drop_rate=0.1,
        edge_add_rate=0.1,
        weight_decay=5e-4,
        lr=0.001,
        epoch=400,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(DGDA, self).__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            weight_decay=weight_decay,
            lr=lr,
            epoch=epoch,
            device=device,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            **kwargs)
        
        self.dec_dim = dec_dim
        self.y_dim = y_dim
        self.d_dim = d_dim
        self.m_dim = m_dim

        self.recons_w = recons_w
        self.ent_w = ent_w
        self.beta = beta
        self.d_w = d_w
        self.y_w = y_w
        self.m_w = m_w
        self.edge_add_rate = edge_add_rate
        self.edge_drop_rate = edge_drop_rate

    def init_model(self, **kwargs):

        return DGDABase(
            in_dim=self.in_dim,
            num_class=self.num_classes,
            enc_hs=self.hid_dim,
            dec_hs=self.dec_dim,
            dim_d=self.d_dim,
            dim_y=self.y_dim,
            dim_m=self.m_dim,
            droprate=self.dropout,
            backbone='gcn',
            source_pretrained_emb=self.source_pretrained_emb,
            source_vertex_feats=self.source_data.x,
            target_pretrained_emb=self.target_pretrained_emb,
            target_vertex_feats=self.target_data.x,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        pass

    def fit(self, source_data, target_data):
        self.source_data = source_data
        self.target_data = target_data

        if self.batch_size == 0:
            self.source_batch_size = source_data.x.shape[0]
            source_loader = NeighborLoader(source_data,
                                self.num_neigh,
                                batch_size=self.source_batch_size)
            self.target_batch_size = target_data.x.shape[0]
            target_loader = NeighborLoader(target_data,
                                self.num_neigh,
                                batch_size=self.target_batch_size)
        else:
            source_loader = NeighborLoader(source_data,
                                self.num_neigh,
                                batch_size=self.batch_size)
            target_loader = NeighborLoader(target_data,
                                self.num_neigh,
                                batch_size=self.batch_size)
        print('Source data pretraining...')
        src_pretrain = DWPretrain(source_data)
        src_pretrain.fit()
        self.source_pretrained_emb = src_pretrain.get_embedding()

        print('Target data pretraining...')
        tgt_pretrain = DWPretrain(target_data)
        tgt_pretrain.fit()
        self.target_pretrained_emb = tgt_pretrain.get_embedding()

        self.dgda = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.dgda.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            p = float(epoch) / self.epoch
            alpha = 2. / (1. + np.exp(-10. * p)) - 1

            self.s_adj = to_dense_adj(source_data.edge_index).squeeze()
            self.t_adj = to_dense_adj(target_data.edge_index).squeeze()

            self.s_vts = torch.arange(source_data.x.shape[0]).to(self.device)
            self.t_vts = torch.arange(target_data.x.shape[0]).to(self.device)

            self.s_feats = source_data.x
            self.t_feats = target_data.x

            self.s_labels = source_data.y
            self.t_labels = target_data.y

            self.dgda.train()
            s_out = self.dgda(self.s_feats, self.s_vts, self.s_adj, 0, alpha=alpha)
            t_out = self.dgda(self.t_feats, self.t_vts, self.t_adj, 1, alpha=alpha)

            src_tr_loss = self.DGDA_loss(s_out, self.s_labels, self.s_adj, 0)
            tar_tr_loss = self.DGDA_loss(t_out, self.t_labels, self.t_adj, 1)
            loss = src_tr_loss + tar_tr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()

            epoch_source_preds = s_out['cls_output'].argmax(dim=1)

            # train with manipulated data
            s_nadj, s_dadj = self.drop_edges(self.s_adj, self.edge_drop_rate, self.edge_add_rate)
            t_nadj, t_dadj = self.drop_edges(self.t_adj, self.edge_drop_rate, self.edge_add_rate)
            
            optimizer.zero_grad()
            s_out = self.dgda(self.s_feats, self.s_vts, s_nadj, 0, alpha=alpha)
            t_out = self.dgda(self.t_feats, self.t_vts, t_nadj, 1, alpha=alpha)
            s_mn_loss = self.DGDA_loss(s_out, self.s_labels, s_nadj, 0)
            t_mn_loss = self.DGDA_loss(t_out, self.t_labels, t_nadj, 1)
            mn_loss = s_mn_loss + t_mn_loss
            mn_loss.backward()
            optimizer.step()

            micro_f1_score = eval_micro_f1(source_data.y, epoch_source_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)
    
    def process_graph(self, data):
        pass

    def predict(self, data):
        self.dgda.eval()

        with torch.no_grad():
            logits = self.dgda(self.t_feats, self.t_vts, self.t_adj, 1, recon=False)
            logits = logits['cls_output']

        return logits, data.y
    
    def DGDA_loss(self, res, labels, adj, domain, manipulate=False, dadj=None):
        # Reconstruction loss
        recon_loss = self.recons_w * self.recons_loss(res['a_recons'], adj)
        
        if manipulate:
            recon_loss += self.m_w * self.recons_loss(res['m_recons'], dadj)

        kld = self.kl_loss(res['dmu'], res['dlv'])
        kly = self.kl_loss(res['ymu'], res['ylv'])
        klm = self.kl_loss(res['mmu'], res['mlv'])
        kld = kld + kly + klm

        ent_loss = self.max_entropy(res['d']) + self.max_entropy(res['y']) + self.max_entropy(res['m'])

        if domain == 0:
            class_loss = F.cross_entropy(input=res['cls_output'], target=labels)
            domain_labels = torch.zeros_like(labels).float().to(self.device)
        else:
            class_loss = torch.zeros(())
            domain_labels = torch.ones_like(labels).float().to(self.device)

        domain_loss = F.binary_cross_entropy_with_logits(input=res['dom_output'].view(-1), target=domain_labels)

        loss = recon_loss + self.beta * kld + self.y_w * class_loss + self.d_w * domain_loss + self.ent_w * ent_loss

        loss = torch.maximum(loss, torch.zeros_like(loss).to(self.device))

        return loss
    
    def recons_loss(self, recons, adjs):
        batch_size, n_node = recons.shape
        total_node = batch_size * n_node
        n_edges = adjs.sum()
        device = adjs.device

        if n_edges == 0:  # no positive edges
            pos_weight = torch.zeros(()).to(device)
        else:
            pos_weight = float(total_node - n_edges) / n_edges

        norm = float(total_node) / (2 * (total_node - n_edges))

        rl = norm * F.binary_cross_entropy_with_logits(input=recons, target=adjs, pos_weight=pos_weight, reduction='mean')

        rl = torch.maximum(rl, torch.zeros_like(rl).to(device))

        return rl
    
    def kl_loss(self, mu, lv):
        n_node = mu.shape[1]
        kld = -0.5 / n_node * torch.mean(torch.sum(1 + 2 * lv - mu.pow(2) - lv.exp().pow(2), dim=-1))
        
        return kld
    
    def max_entropy(self, x):
        # Why 0.693148?
        ent = 0.693148 + torch.mean(torch.sigmoid(x) * F.logsigmoid(x))
        
        return ent
    
    def drop_edges(self, adj, drop_rate=0.0, add_rate=0.0):
        bs, N = adj.shape
        n_edges = adj.sum()
        sparsity = (n_edges + bs * N) / (bs * N)
        nadj = adj.clone()

        if drop_rate > 0.0:
            drop_mask = torch.bernoulli(adj, p=drop_rate)
            nadj -= drop_mask
            nadj[nadj < 0] = 0

        if add_rate > 0.0:
            add_mask = torch.bernoulli(adj, p=sparsity * add_rate)
            nadj += add_mask

        nadj = nadj.tril() + nadj.tril().permute(1, 0)

        I = torch.eye(N).to(adj.device)
        nadj += I
        nadj[nadj < 0] = 0
        nadj[nadj > 0] = 1

        dadj = adj - nadj
        dadj[dadj != 0] = 1
        dadj -= I
        dadj[dadj < 0] = 0

        return nadj, dadj
