import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import torch.nn as nn

from scipy import sparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse import vstack

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj

from . import BaseGDA
from ..nn import ACDNEBase
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class ACDNE(BaseGDA):
    """
    Adversarial Deep Network Embedding for Cross-network Node Classification (AAAI-20).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    n_emb : int, optional
        Adversarial learning module hidden dimension. Default: ``128``.
    pair_weight : float, optional
        Trade-off hyper-parameter for pairwise constraint. Default: ``0.1``.
    step : int, optional
        Propagation steps in PPMI matrix. Default: ``3``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
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
        num_layers=0,
        n_emb=128,
        pair_weight=0.1,
        step=3,
        dropout=0.,
        act=F.relu,
        weight_decay=0.,
        lr=4e-3,
        epoch=100,
        device='cuda:0',
        batch_size=100,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(ACDNE, self).__init__(
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
        
        self.n_emb=n_emb
        self.pair_weight=pair_weight
        self.step=step

    def init_model(self, **kwargs):

        return ACDNEBase(
            n_input=self.in_dim,
            n_hidden=[self.hid_dim]*2,
            n_emb=self.n_emb,
            num_class=self.num_classes,
            drop=self.dropout,
            batch_size=self.batch_size,
            **kwargs
        ).to(self.device)

    def forward_model(self, **kwargs):
        pass

    def fit(self, source_data, target_data):
        ppmi_s, x_n_s = self.process_graph(source_data)
        ppmi_t, x_n_t = self.process_graph(target_data)
        x_s = source_data.x.detach().cpu().numpy()
        x_t = target_data.x.detach().cpu().numpy()
        y_s = F.one_hot(source_data.y).detach().cpu().numpy()
        y_t = F.one_hot(target_data.y).detach().cpu().numpy()
        y_t_o = np.zeros(np.shape(y_t))

        num_nodes_s = x_s.shape[0]
        num_nodes_t = x_t.shape[0]

        x_s_new = np.concatenate((x_s, x_n_s), axis=1)
        x_t_new = np.concatenate((x_t, x_n_t), axis=1)

        self.whole_xs_xt_stt = torch.tensor(vstack([x_s, x_t]).toarray()).to(self.device)
        self.whole_xs_xt_stt_nei = torch.tensor(vstack([x_n_s, x_n_t]).toarray()).to(self.device)

        self.acdne = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.acdne.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.clf_loss_func = nn.CrossEntropyLoss()
        self.domain_loss_func = nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            s_batches = self.batch_generator([x_s_new, y_s], int(self.batch_size / 2), shuffle=True)
            t_batches = self.batch_generator([x_t_new, y_t_o], int(self.batch_size / 2), shuffle=True)
            num_batch = round(max(num_nodes_s / (self.batch_size / 2), num_nodes_t / (self.batch_size / 2)))

            p = float(epoch) / self.epoch
            alpha = 2. / (1. + np.exp(-10. * p)) - 1

            for idx in range(num_batch):
                xs_ys_batch, shuffle_index_s = next(s_batches)
                xs_batch = xs_ys_batch[0]
                ys_batch = xs_ys_batch[1]
                xt_yt_batch, shuffle_index_t = next(t_batches)
                xt_batch = xt_yt_batch[0]
                yt_batch = xt_yt_batch[1]

                x_batch = np.vstack([xs_batch, xt_batch])
                xb = torch.FloatTensor(x_batch[:, 0:self.in_dim])
                xb_nei = torch.FloatTensor(x_batch[:, -self.in_dim:])
                yb = np.vstack([ys_batch, yt_batch])

                mask_l = np.sum(yb, axis=1) > 0

                domain_label = np.vstack([np.tile([1., 0.], [self.batch_size // 2, 1]), np.tile([0., 1.], [self.batch_size // 2, 1])]) 
                # [1,0] for source, [0,1] for target
                # #topological proximity matrix between nodes in each mini-batch
                a_s, a_t = self.batch_ppmi(self.batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t)

                self.acdne.train()
                optimizer.zero_grad()
                emb, pred_logit, d_logit = self.acdne(xb.to(self.device), xb_nei.to(self.device), alpha)
                emb_s, emb_t = self.acdne.network_embedding.pairwise_constraint(emb)
                net_pro_loss_s = self.acdne.network_embedding.net_pro_loss(emb_s, torch.tensor(a_s).to(self.device))
                net_pro_loss_t = self.acdne.network_embedding.net_pro_loss(emb_t, torch.tensor(a_t).to(self.device))
                net_pro_loss = self.pair_weight * (net_pro_loss_s + net_pro_loss_t)

                clf_loss = self.clf_loss_func(pred_logit[mask_l], torch.tensor(yb[mask_l]).to(self.device))
                domain_loss = self.domain_loss_func(d_logit, torch.argmax(torch.tensor(domain_label).to(self.device), 1))
                total_loss = clf_loss + domain_loss + net_pro_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            epoch_source_logits, epoch_source_labels = self.predict(source_data, train=True)
            
            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(epoch_source_labels, epoch_source_preds)

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
        n_A_ppmi = self.my_scale_sim_mat(A_ppmi)
        X = data.x.detach().cpu().numpy()
        X_nei = np.matmul(n_A_ppmi, X)

        return A_ppmi, X_nei

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

    def batch_ppmi(self, batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t):
        """return the PPMI matrix between nodes in each batch"""
        # #proximity matrix between source network nodes in each mini-batch
        # noinspection DuplicatedCode
        a_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
        for ii in range(int(batch_size / 2)):
            for jj in range(int(batch_size / 2)):
                if ii != jj:
                    a_s[ii, jj] = ppmi_s[shuffle_index_s[ii], shuffle_index_s[jj]]
        # #proximity matrix between target network nodes in each mini-batch
        # noinspection DuplicatedCode
        a_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
        for ii in range(int(batch_size / 2)):
            for jj in range(int(batch_size / 2)):
                if ii != jj:
                    a_t[ii, jj] = ppmi_t[shuffle_index_t[ii], shuffle_index_t[jj]]
        return self.my_scale_sim_mat(a_s), self.my_scale_sim_mat(a_t)

    def predict(self, data, train=False):
        self.acdne.eval()

        with torch.no_grad():
            emb, pred_logit_xs_xt, _ = self.acdne(self.whole_xs_xt_stt, self.whole_xs_xt_stt_nei, 1)
            pred_prob_xs_xt = F.softmax(pred_logit_xs_xt, dim=1)
            if train:
                logits = pred_prob_xs_xt[0:data.x.shape[0], :]
            else:
                logits = pred_prob_xs_xt[-data.x.shape[0]:, :]

        return logits, data.y
    
    def shuffle_aligned_list(self, data):
        num = data[0].shape[0]
        shuffle_index = np.random.permutation(num)
        return shuffle_index, [d[shuffle_index] for d in data]

    def batch_generator(self, data, batch_size, shuffle=True):
        shuffle_index = None
        if shuffle:
            shuffle_index, data = self.shuffle_aligned_list(data)
        batch_count = 0
        while True:
            if batch_count * batch_size + batch_size >= data[0].shape[0]:
                batch_count = 0
                if shuffle:
                    shuffle_index, data = self.shuffle_aligned_list(data)
            start = batch_count * batch_size
            end = start + batch_size
            batch_count += 1
            yield [d[start:end] for d in data], shuffle_index[start:end]
