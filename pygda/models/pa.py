import torch
import warnings
import torch.nn.functional as F
import itertools
import time
import copy

import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import cvxpy as cp

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj

from . import BaseGDA
from ..nn import ReweightGNN
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class PairAlign(BaseGDA):
    """

    Pairwise Alignment Improves Graph Domain Adaptation (ICML-24).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    num_layers : int, optional
        Total number of gnn layers in model. Default: ``2``.
    cls_dim : int, optional
        Hidden dimension for classification layer. Default: ``128``.
    cls_layers : int, optional
        Total number of cls layers in model. Default: ``2``.
    ew_start : int, optional
        Starting epoch for edge reweighting. Default: ``0``.
    ew_freq : int, optional
        Frequency for edge reweighting. Default: ``10``.
    lw_start : int, optional
        Starting epoch for label reweighting. Default: ``0``.
    lw_freq : int, optional
        Frequency for label reweighting. Default: ``10``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    pooling : string, optional
        Aggregation in gnn. Default: ``mean``.
    ew_type : string, optional
        Use the true edge weight or not. Default: ``pseudobeta``.
    rw_lmda : float, optional
        Trade-off parameter for edge reweight. Default: ``1.0``.
    ls_lambda : float, optional
        Regularize the distance to 1 in w optimization. Default: ``1.0``.
    lw_lambda : float, optional
        Regularize the distance to 1 in beta optimization. Default: ``0.005``.
    label_rw : bool, optional
        Reweight the label or not. Default: ``False``.
    edge_rw : bool, optional
        Reweight the edge in source graph or not. Default: ``False``.
    gamma_reg : float, optional
        Mimic the variance of the edges to normalize the weight. Default: ``1e-4``.
    weight_CE_src : bool, optional
        Reweight the loss by src class or not. Default: ``False``.
    backbone : string, optional
        The backbone of PairAlign. Default: ``GS``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.0001``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Learning rate. Default: ``0.001``.
    bn : bool, optional
        Batch normalization or not. Default: ``False``.
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
        more log information. Default: ``2``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=2,
        cls_dim=128,
        cls_layers=2,
        dropout=0.,
        backbone='GS',
        pooling='mean',
        ew_type='pseudobeta',
        rw_lmda=1.0,
        ls_lambda=1.0,
        lw_lambda=0.005,
        label_rw=False,
        edge_rw=False,
        ew_start=0,
        ew_freq=10,
        lw_start=0,
        lw_freq=10,
        gamma_reg=1e-4,
        weight_CE_src=False,
        bn=False,
        act=F.relu,
        weight_decay=0.0001,
        lr=0.001,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(PairAlign, self).__init__(
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
        
        self.rw_lmda=rw_lmda
        self.cls_dim=cls_dim
        self.cls_layers=cls_layers
        self.backbone=backbone
        self.pooling=pooling
        self.bn=bn
        self.label_rw=label_rw
        self.weight_CE_src=weight_CE_src
        self.gamma_reg=gamma_reg
        self.edge_rw=edge_rw
        self.ew_start=ew_start
        self.ew_freq=ew_freq
        self.ew_type=ew_type
        self.ls_lambda=ls_lambda
        self.lw_lambda=lw_lambda
        self.lw_freq=lw_freq
        self.lw_start=lw_start

    def init_model(self, **kwargs):

        return ReweightGNN(
            input_dim=self.in_dim,
            gnn_dim=self.hid_dim,
            output_dim=self.num_classes,
            cls_dim=self.cls_dim,
            gnn_layers=self.num_layers,
            cls_layers=self.cls_layers,
            backbone=self.backbone,
            pooling=self.pooling,
            dropout=self.dropout,
            bn=self.bn,
            rw_lmda=self.rw_lmda,
            **kwargs
            ).to(self.device)

    def forward_model(self, source_data, target_data, epoch):
        GNN_embed_src, pred_src = self.gnn.forward(source_data, source_data.x)
        GNN_embed_tgt, pred_tgt = self.gnn.forward(target_data, target_data.x)

        mask_src = source_data.train_mask
        label_src = source_data.y[mask_src]
        pred_src = pred_src[mask_src]

        if self.label_rw:
            cls_loss_src = self.CE_loss_weight(pred_src, label_src, torch.from_numpy(source_data.lw).to(self.device), self.weight_CE_src)
        else:
            if self.weight_CE_src:
                cls_loss_src = self.CE_srcweight_loss(pred_src, label_src)
            else:
                cls_loss_src = self.CE_loss(pred_src, label_src)

        loss = cls_loss_src

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        GNN_embed_src, pred_src_gnn = self.gnn.forward(source_data, source_data.x)
        GNN_embed_tgt, pred_tgt_gnn = self.gnn.forward(target_data, target_data.x)

        pred_src = pred_src_gnn
        pred_tgt = pred_tgt_gnn

        pred_tgt_label = pred_tgt.argmax(dim=1)
        pred_tgt_label = pred_tgt_label.to(self.device)
        target_data.y_hat = pred_tgt_label

        if self.edge_rw and (epoch + 1) >= self.ew_start and (epoch + 1) % self.ew_freq == 0:
            self.update_ew(source_data, target_data, pred_src, pred_tgt)

        # update label weight
        label_weight = self.update_lw(source_data, target_data, pred_src, pred_tgt)
        if (epoch + 1) >= self.lw_start and (epoch + 1) % self.lw_freq == 0:
            source_data.lw = label_weight

        return loss, pred_src, pred_tgt

    def fit(self, source_data, target_data):
        if source_data.edge_weight is None:
            source_data.edge_weight = torch.ones(source_data.edge_index.shape[1]).to(self.device)
        if target_data.edge_weight is None:
            target_data.edge_weight = torch.ones(target_data.edge_index.shape[1]).to(self.device)

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

        self.gnn = self.init_model(**self.kwargs)

        self.opt = torch.optim.Adam(
            self.gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
            )

        start_time = time.time()

        self.calc_true_ew(source_data, target_data)
        self.calc_true_lw(source_data, target_data)
        self.calc_true_beta(source_data, target_data)

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.gnn.train()
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, epoch)
                epoch_loss += loss.item()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = self.predict(sampled_source_data)
                else:
                    source_logits, source_labels = self.predict(sampled_source_data)
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
            
            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(epoch_source_labels, epoch_source_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

    def process_graph(self, data):
        pass

    def predict(self, data):
        self.gnn.eval()

        with torch.no_grad():
           _, logits = self.gnn(data, data.x)

        return logits, data.y
    
    def CE_loss(self, pred, label):
        label = label.type(torch.int64)
        loss = nn.CrossEntropyLoss()
        
        return loss(pred, label)

    def CE_srcweight_loss(self, pred, label):
        label = label.type(torch.int64)
        y_onehot = F.one_hot(label).float().to(self.device)
        p_y = torch.sum(y_onehot, 0) / len(y_onehot)
        class_weights = torch.tensor(1.0 / p_y, dtype=torch.float, requires_grad=False).to(self.device)
        loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
        
        return torch.mean(loss(pred, label))

    def CE_loss_weight(self, pred, label, label_weight, weight_src):
        class_num = len(label.unique())
        label = label.type(torch.int64)
        y_onehot = F.one_hot(label).float().to(self.device)
        p_y = torch.sum(y_onehot, 0) / len(y_onehot)
        class_weights = (1.0 / p_y.clone().detach()).to(self.device)
        if weight_src:
            loss = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
        else:
            loss = nn.CrossEntropyLoss()
        weight = torch.mm(y_onehot, label_weight.view(-1, 1).float())
        
        return torch.mean(loss(pred, label).view(-1, 1) * weight)/ class_num
    
    def calc_true_lw(self, src_graph, tgt_graph):
        print('calc true lw...')
        label_onehot_src = F.one_hot(src_graph.y)
        label_onehot_tgt = F.one_hot(tgt_graph.y)
        num_nodes_src = torch.sum(label_onehot_src, 0)
        label_dis_src = num_nodes_src / src_graph.x.shape[0]

        num_nodes_tgt = torch.sum(label_onehot_tgt, 0)
        label_dis_tgt = num_nodes_tgt / tgt_graph.x.shape[0]

        label_weight = label_dis_tgt / label_dis_src
        src_graph.true_lw = label_weight.cpu().detach().numpy()

        src_graph.lw = np.ones_like(src_graph.true_lw)
            
    def calc_true_beta(self, src_graph, tgt_graph):
        print('calc true beta...')
        num_classes = self.num_classes
        src_num_edges = src_graph.edge_index.shape[1]
        src_edge_class = np.zeros((src_num_edges, num_classes, num_classes))
        for idx in range(src_num_edges):
            i = src_graph.edge_index[0][idx]
            j = src_graph.edge_index[1][idx]
            src_edge_class[idx, src_graph.y[i], src_graph.y[j]] = 1
        
        self.src_edge_class = src_edge_class
        
        edge_class_src = np.reshape(src_edge_class, (src_num_edges, num_classes**2))
        p_edge_src = edge_class_src.sum(axis=0) / src_num_edges

        tgt_num_edges = tgt_graph.edge_index.shape[1]
        tgt_edge_class = np.zeros((tgt_num_edges, num_classes, num_classes))
        for idx in range(tgt_num_edges):
            i = tgt_graph.edge_index[0][idx]
            j = tgt_graph.edge_index[1][idx]
            tgt_edge_class[idx, tgt_graph.y[i], tgt_graph.y[j]] = 1
        
        self.tgt_edge_class = tgt_edge_class

        edge_class_tgt = np.reshape(tgt_edge_class, (tgt_num_edges, num_classes**2))
        p_edge_tgt = edge_class_tgt.sum(axis=0) / tgt_num_edges

        beta = (p_edge_tgt) / (p_edge_src)
        beta[beta == float('inf')] = 1
        beta = np.nan_to_num(beta, nan = 1)
    
        src_graph.true_beta = beta
    
    def calc_true_ew(self, src_graph, tgt_graph):
        print('calc true ew...')
        src_edge_prob, tgt_edge_prob = self.cal_edge_prob_sep(src_graph, tgt_graph)

        edge_weight = (tgt_edge_prob + self.gamma_reg) / (src_edge_prob + self.gamma_reg)
    
        src_graph.true_ew = edge_weight.cpu().detach().numpy()
        src_graph.ew = np.ones_like(src_graph.true_ew)
    
    def cal_edge_prob_sep(self, src_graph, tgt_graph):
        src_adj = to_dense_adj(src_graph.edge_index).squeeze().T.detach().cpu().numpy()
        tgt_adj = to_dense_adj(tgt_graph.edge_index).squeeze().T.detach().cpu().numpy()
        num_nodes_src = src_graph.x.shape[0]
        num_nodes_tgt = tgt_graph.x.shape[0]
        src_label = src_graph.y
        tgt_label = tgt_graph.y
        num_class = self.num_classes

        src_label_one_hot = sp.csr_matrix((np.ones(num_nodes_src), (np.arange(num_nodes_src), src_label.cpu().numpy())),
                                      shape=(num_nodes_src, num_class))
        tgt_label_one_hot = sp.csr_matrix((np.ones(num_nodes_tgt), (np.arange(num_nodes_tgt), tgt_label.cpu().numpy())),
                                      shape=(num_nodes_tgt, num_class))

        src_node_num = src_label_one_hot.sum(axis=0).T * num_nodes_src
        tgt_node_sum = tgt_label_one_hot.sum(axis=0).T * num_nodes_tgt

        src_num_edge = (src_label_one_hot.T * src_adj * src_label_one_hot)
        tgt_num_edge = (tgt_label_one_hot.T * tgt_adj * tgt_label_one_hot)

        src_edge_prob = src_num_edge / src_node_num
        tgt_edge_prob = tgt_num_edge / tgt_node_sum

        src_edge_prob = torch.from_numpy(np.array(src_edge_prob))
        tgt_edge_prob = torch.from_numpy(np.array(tgt_edge_prob))

        return src_edge_prob, tgt_edge_prob
    
    def update_ew(self, src_data, tgt_data, pred_src, pred_tgt):
        if self.ew_type == "pseudobeta":
            ew_diff, beta_diff = self.calc_edge_rw_pseudo(src_data, tgt_data, pred_src, pred_tgt)
        else:
            beta_diff = 0
            if self.ew_type == "truth":
                edge_rw = src_data.true_ew
            else:
                edge_rw , _ = self.calc_ratio_weight(src_data, src_data.true_beta, self.gamma_reg)
            
            num_classes = self.num_classes
            src_num_edges = src_graph.edge_index.shape[1]
            
            edge_weight = np.matmul(self.src_edge_class.reshape(src_num_edges, num_classes**2), edge_rw.reshape(num_classes**2))
            src_data.edge_weight = torch.from_numpy(edge_weight).float().to(self.device)
            src_data.ew = edge_rw
            ew_diff = np.abs(edge_rw - src_data.true_ew.reshape(num_classes, num_classes)).sum()

        return 

    def calc_edge_rw_pseudo(self, src_graph, tgt_graph, yhat_src, yhat_tgt):
        true_ew = src_graph.true_ew
        true_beta = src_graph.true_beta
        num_classes = self.num_classes

        src_num_edges = src_graph.edge_index.shape[1]
        src_num_nodes = src_graph.x.shape[0]
        src_i_src = zip(np.arange(src_num_edges), src_graph.edge_index[0])
        src_i_tgt = zip(np.arange(src_num_edges), src_graph.edge_index[1])
        src_v = torch.ones(src_num_edges, dtype=torch.float32)
        src_edgehat_src = torch.sparse.mm(torch.sparse_coo_tensor(list(zip(*src_i_src)), src_v, (src_num_edges, src_num_nodes)).to(self.device), yhat_src)
        src_edgehat_tgt = torch.sparse.mm(torch.sparse_coo_tensor(list(zip(*src_i_tgt)), src_v, (src_num_edges, src_num_nodes)).to(self.device), yhat_src)
        src_edgehat = torch.einsum("bi,bj->bij",src_edgehat_src, src_edgehat_tgt).view(-1, num_classes**2)

        tgt_num_edges = tgt_graph.edge_index.shape[1]
        tgt_num_nodes = tgt_graph.x.shape[0]
        tgt_i_src = zip(np.arange(tgt_num_edges), tgt_graph.edge_index[0])
        tgt_i_tgt = zip(np.arange(tgt_num_edges), tgt_graph.edge_index[1])
        tgt_v = torch.ones(tgt_num_edges, dtype=torch.float32)
        tgt_edgehat_src = torch.sparse.mm(torch.sparse_coo_tensor(list(zip(*tgt_i_src)), tgt_v, (tgt_num_edges, tgt_num_nodes)).to(self.device), yhat_tgt)
        tgt_edgehat_tgt = torch.sparse.mm(torch.sparse_coo_tensor(list(zip(*tgt_i_tgt)), tgt_v, (tgt_num_edges, tgt_num_nodes)).to(self.device), yhat_tgt)
        tgt_edgehat = torch.einsum("bi,bj->bij",tgt_edgehat_src, tgt_edgehat_tgt).view(-1, num_classes**2)
    
        src_edgeclass = torch.tensor(self.src_edge_class.reshape(-1, num_classes**2)).float().to(self.device)
        C_hat = (src_edgehat.T @ src_edgeclass) / src_num_edges
        muhat_tgt = torch.mean(tgt_edgehat, dim = 0).view(num_classes**2, 1)
        mu_src = torch.mean(src_edgeclass, dim = 0).view(num_classes**2, 1)
    
        w = self.LS_optimization(C_hat, muhat_tgt, mu_src, self.ls_lambda)
        w = np.reshape(w, (num_classes, num_classes))

        beta_diff = np.abs(w - true_beta.reshape(num_classes, num_classes)).sum()

        gamma, p_ei_tgt = self.calc_ratio_weight(src_graph, w, self.gamma_reg)
        edge_rw = gamma
        edge_rw[edge_rw == float('inf')] = 1
        edge_rw = np.nan_to_num(edge_rw, nan = 1)

        edge_weight = np.matmul(self.src_edge_class.reshape(src_num_edges, num_classes**2), edge_rw.reshape(num_classes**2))
        src_graph.edge_weight = torch.from_numpy(edge_weight).float().to(self.device)
        src_graph.ew = edge_rw

        diff = np.abs(edge_rw - true_ew.reshape(num_classes, num_classes)).sum()
        
        return diff, beta_diff
    
    def LS_optimization(self, cov, muhat_tgt, mu_src, lambda_reg):
        mu_src = mu_src.cpu().detach().numpy().reshape(-1).astype(np.double)
        muhat_tgt = muhat_tgt.cpu().detach().numpy().reshape(-1).astype(np.double)
        cov = cov.cpu().detach().numpy().astype(np.double)

        print("rank of Cov in edge: " + str(np.linalg.matrix_rank(cov)))
        x0 = np.ones(cov.shape[1])  
        x = cp.Variable(cov.shape[1])

        objective = cp.Minimize(cp.norm(cov @ x - muhat_tgt, 2)**2 + lambda_reg * cp.norm(x - x0, 2)**2)
        constraints = [mu_src @ x == 1 , 0 <= x]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        x_value = x.value
           
        return x_value
    
    def calc_ratio_weight(self, src_graph, kmm_weight, gamma_reg):
        src_num_edges = src_graph.edge_index.shape[1]
        src_num_nodes = src_graph.x.shape[0]
        num_classes = self.num_classes
        p_eij_src = (np.sum(self.src_edge_class.reshape(src_num_edges, num_classes**2), axis=0) / src_num_edges)
        p_eij_tgt = p_eij_src * kmm_weight.reshape(-1)
        p_eij_src_reg = p_eij_src + gamma_reg
        p_eij_tgt_reg = p_eij_tgt + gamma_reg

        p_ei_src_reg = np.sum(p_eij_src_reg.reshape(num_classes, num_classes), axis=1)
        p_ei_tgt_reg = np.sum(p_eij_tgt_reg.reshape(num_classes, num_classes), axis=1)

        gamma = ((p_eij_tgt_reg.reshape(num_classes, num_classes).T / p_ei_tgt_reg).T) / ((p_eij_src_reg.reshape(num_classes, num_classes).T / p_ei_src_reg).T)

        return gamma, p_ei_tgt_reg
    
    def update_lw(self, src_data, tgt_data, pred_src, pred_tgt):
        yhat_tgt = torch.mean(F.softmax(pred_tgt,dim=1), dim=0)
        y_src_onehot = F.one_hot(src_data.y).float().to(self.device)
        y_src = torch.mean(y_src_onehot, dim=0)
        y_src_pred = F.softmax(pred_src,dim=1)
        cov_mat = torch.mm(y_src_pred.T, y_src_onehot) / src_data.x.shape[0]

        label_weight = self.calc_label_rw(y_src, yhat_tgt, cov_mat, self.lw_lambda)

        return label_weight
    
    def calc_label_rw(self, y_src, y_hat_tgt, cov, lambda_reg):
        y_src = y_src.cpu().detach().numpy().astype(np.double)
        y_hat_tgt = y_hat_tgt.cpu().detach().numpy().astype(np.double)
        cov = cov.cpu().detach().numpy().astype(np.double)
    
        x0 = np.ones(cov.shape[1])
        x = cp.Variable(cov.shape[1])

        objective = cp.Minimize(cp.norm(cov @ x - y_hat_tgt, 2)**2 + lambda_reg * cp.norm(x - x0, 2)**2)
        constraints = [y_src @ x == 1, 0 <= x]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        x_value = x.value
        
        return x_value
