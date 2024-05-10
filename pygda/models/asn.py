import torch
import warnings
import torch.nn.functional as F
import itertools
import time

from torch_geometric.loader import NeighborLoader

from torch_geometric.utils import to_dense_adj

from scipy import sparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse import vstack

from . import BaseGDA
from ..nn import ASNBase
from ..nn import GradReverse
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class ASN(BaseGDA):
    """
    Adversarial Separation Network for Cross-Network Node Classification (CIKM-21).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    hid_dim_vae : int
        Hidden dimension of vae model.
    num_classes : int
        Total number of classes.
    num_layers : int, optional
        Total number of layers in model. Default: ``3``.
    step : int, optional
        Propagation steps in PPMI matrix. Default: ``3``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    lambda_r : float, optional
        Hyperparameter for reconstruction loss. Default: ``1.``.
    lambda_d : float, optional
        Hyperparameter for domain loss. Default: ``0.5.``.
    lambda_f : float, optional
        Hyperparameter for different loss. Default: ``0.0001``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    adv_dim : int, optional
        Hidden dimension of adversarial module. Default: ``40``.
    lr : float, optional
        Learning rate. Default: ``0.001``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``200``.
    device : str, optional
        GPU or CPU. Default: ``cuda:0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        hid_dim_vae,
        num_classes,
        num_layers=3,
        step=3,
        dropout=0.,
        act=F.relu,
        lambda_r=1.0,
        lambda_d=0.1,
        lambda_f=0.001,
        adv_dim=10,
        weight_decay=5e-4,
        lr=3e-2,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(ASN, self).__init__(
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
        assert batch_size == 0, 'unsupport for batch training'
        
        self.hid_dim_vae=hid_dim_vae
        self.adv_dim=adv_dim
        self.lambda_r=lambda_r
        self.lambda_d=lambda_d
        self.lambda_f=lambda_f
        self.step=step

    def init_model(self, **kwargs):

        return ASNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            hid_dim_vae=self.hid_dim_vae,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            adv_dim=self.adv_dim,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        pass

    def fit(self, source_data, target_data):
        self.num_source_nodes, _ = source_data.x.shape
        self.num_target_nodes, _ = target_data.x.shape

        print('Processing source graph...')
        self.ppmi_s = self.process_graph(source_data).to(self.device)
        print('Processing target graph...')
        self.ppmi_t = self.process_graph(target_data).to(self.device)

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

        self.asn = self.init_model(**self.kwargs)

        params = itertools.chain(*[model.parameters() for model in self.asn.models])

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0

            alpha = min((epoch + 1) / self.epoch, 0.05)

            for model in self.asn.models:
                model.train()
            
            recovered_s, mu_s, logvar_s = self.asn.private_encoder_s_l(source_data.x, source_data.edge_index)
            recovered_t, mu_t, logvar_t = self.asn.private_encoder_t_l(target_data.x, target_data.edge_index)

            recovered_s_p, mu_s_p, logvar_s_p = self.asn.private_encoder_s_g(source_data.x, self.ppmi_s)
            recovered_t_p, mu_t_p, logvar_t_p = self.asn.private_encoder_t_g(target_data.x, self.ppmi_t)

            z_s, shared_encoded_source1, shared_encoded_source2 = self.asn.shared_encoder_l(source_data.x, source_data.edge_index)
            z_t, shared_encoded_target1, shared_encoded_target2 = self.asn.shared_encoder_l(target_data.x, target_data.edge_index)

            z_s_p, ppmi_encoded_source, ppmi_encoded_source2 = self.asn.shared_encoder_g(source_data.x, self.ppmi_s)
            z_t_p, ppmi_encoded_target, ppmi_encoded_target2 = self.asn.shared_encoder_g(target_data.x, self.ppmi_t)

            encoded_source = self.asn.att_model([shared_encoded_source1, ppmi_encoded_source])
            encoded_target = self.asn.att_model([shared_encoded_target1, ppmi_encoded_target])

            diff_loss_s = self.asn.loss_diff(mu_s, shared_encoded_source1)
            diff_loss_t = self.asn.loss_diff(mu_t, shared_encoded_target1)
            diff_loss_all = diff_loss_s + diff_loss_t

            ''' compute decoder reconstruction loss for S and T '''
            z_cat_s = torch.cat((self.asn.att_model_self_s([recovered_s, recovered_s_p]), self.asn.att_model_self_s([z_s, z_s_p])), 1)
            z_cat_t = torch.cat((self.asn.att_model_self_t([recovered_t, recovered_t_p]), self.asn.att_model_self_t([z_t, z_t_p])), 1)
            recovered_cat_s = self.asn.decoder_s(z_cat_s)
            recovered_cat_t = self.asn.decoder_t(z_cat_t)

            mu_cat_s = torch.cat((mu_s, mu_s_p, shared_encoded_source1, ppmi_encoded_source), 1)
            mu_cat_t = torch.cat((mu_t, mu_t_p, shared_encoded_target1, ppmi_encoded_target), 1)
            logvar_cat_s = torch.cat((logvar_s, logvar_s_p, shared_encoded_source2, ppmi_encoded_source2), 1)
            logvar_cat_t = torch.cat((logvar_t, logvar_t_p, shared_encoded_target2, ppmi_encoded_target2), 1)

            adj_label_s, pos_weight_s, norm_s = self.asn.adj_label_for_reconstruction(source_data)
            adj_label_t ,pos_weight_t, norm_t = self.asn.adj_label_for_reconstruction(target_data)

            recon_loss_s = self.asn.recon_loss(
                preds=recovered_cat_s,
                labels=adj_label_s,
                mu=mu_cat_s,
                logvar=logvar_cat_s,
                n_nodes=source_data.x.shape[0],
                norm=norm_s,
                pos_weight=pos_weight_s
                )
        
            recon_loss_t = self.asn.recon_loss(
                preds=recovered_cat_t,
                labels=adj_label_t,
                mu=mu_cat_t,
                logvar=logvar_cat_t,
                n_nodes=target_data.x.shape[0]*2,
                norm=norm_t,
                pos_weight=pos_weight_t
                )
            recon_loss_all =  recon_loss_s + recon_loss_t

            ''' compute node classification loss for S '''
            source_logits = self.asn.cls_model(encoded_source)
            cls_loss_source = self.asn.cls_loss(source_logits, source_data.y)

            ''' compute domain classifier loss for both S and T '''
            domain_output_s = self.asn.domain_model(GradReverse.apply(encoded_source, alpha))
            domain_output_t = self.asn.domain_model(GradReverse.apply(encoded_target, alpha))
            err_s_domain = self.asn.cls_loss(
                domain_output_s,
                torch.zeros(domain_output_s.size(0)).type(torch.LongTensor).to(self.device)
                )
            err_t_domain = self.asn.cls_loss(
                domain_output_t,
                torch.ones(domain_output_t.size(0)).type(torch.LongTensor).to(self.device)
                )
            loss_grl = err_s_domain + err_t_domain

            ''' compute entropy loss for T '''
            target_logits = self.asn.cls_model(encoded_target)
            target_probs = F.softmax(target_logits, dim=-1)
            target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
            loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

            ''' compute overall loss '''
            loss = cls_loss_source + self.lambda_d * loss_grl + self.lambda_r * recon_loss_all + self.lambda_f * diff_loss_all + loss_entropy * (epoch / self.epoch * 0.01)

            epoch_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_source_preds = source_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(source_data.y, epoch_source_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)
    
    def process_graph(self, data):
        adj = to_dense_adj(data.edge_index).squeeze()
        g = sparse.csc_matrix(adj.detach().cpu().numpy())
        A_k = self.agg_tran_prob_mat(g, self.step)
        A_ppmi = self.compute_ppmi(A_k)
        A_ppmi = lil_matrix(A_ppmi)
        n_A_ppmi = self.my_scale_sim_mat(A_ppmi)
        n_A_ppmi = self.sparse_mx_to_torch_sparse_tensor(n_A_ppmi)

        return n_A_ppmi
    
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape).coalesce()

    def agg_tran_prob_mat(self, g, step):
        """aggregated K-step transition probality"""
        g = self.my_scale_sim_mat(g)
        g = csc_matrix.toarray(g)
        a_k = g
        a = g
        for k in np.arange(2, step+1):
            a_k = np.matmul(a_k, g)
            a = a+a_k/k
        
        return a
    
    def my_scale_sim_mat(self, w):
        """L1 row norm of a matrix"""
        rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
        r_inv = np.power(rowsum + 1e-12, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        w = r_mat_inv.dot(w)
        
        return w
    
    def compute_ppmi(self, a):
        """compute PPMI, given aggregated K-step transition probality matrix as input"""
        np.fill_diagonal(a, 0)
        a = self.my_scale_sim_mat(a)
        (p, q) = np.shape(a)
        col = np.sum(a, axis=0)
        col[col == 0] = 1
        ppmi = np.log((float(p)*a) / (col[None, :]) + 1e-12)
        idx_nan = np.isnan(ppmi)
        ppmi[idx_nan] = 0
        ppmi[ppmi < 0] = 0
        
        return ppmi

    def predict(self, data):
        for model in self.asn.models:
            model.eval()
        
        ppmi = self.process_graph(data).to(self.device)

        with torch.no_grad():
            z_t, shared_encoded_data1, shared_encoded_data2 = self.asn.shared_encoder_l(data.x, data.edge_index)
            z_t_p, ppmi_encoded_data, ppmi_encoded_data2 = self.asn.shared_encoder_g(data.x, ppmi)
            encoded_data = self.asn.att_model([shared_encoded_data1, ppmi_encoded_data])
            logits = self.asn.cls_model(encoded_data)

        return logits, data.y
