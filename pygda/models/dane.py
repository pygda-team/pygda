import torch
import warnings
import torch.nn.functional as F
import itertools
import time
import copy

import torch.nn as nn
import scipy.sparse as sp
import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric.utils import is_undirected, to_undirected

from . import BaseGDA
from ..nn import GNNBase, GradReverse
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1

from torch_geometric.nn import global_mean_pool


class DANE(BaseGDA):
    """

    DANE: Domain Adaptive Network Embedding (IJCAI-19).

    Parameters
    ----------
    in_dim :  int
        Input feature dimension.
    hid_dim :  int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    num_layers : int
        Total number of layers in model.
    mode : str, optional
        Mode for node or graph level tasks. Default: ``node``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    gnn : string, optional
        GNN backbone. Default: ``gcn``.
    k : int, optional
        Number of negative samples. Default: ``5``.
    train_mode : string, optional
        Unsupervised or Semi-supervised. Default: ``unsup``.
    tgt_rate : float, optional
        Target graph rate of labeled nodes. Default: ``0.05``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``1e-5``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
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
        more log information. Default: ``2``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers,
        mode='node',
        dropout=0.,
        gnn='gcn',
        k=5,
        train_mode='unsup',
        tgt_rate=0.05,
        act=F.relu,
        weight_decay=1e-5,
        lr=0.001,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(DANE, self).__init__(
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
        
        assert train_mode in ['semi', 'unsup'], 'unsupport training mode'
        
        self.gnn_arc=gnn
        self.k=k
        self.train_mode=train_mode
        self.tgt_rate=tgt_rate
        self.mode=mode

    def init_model(self, **kwargs):

        return GNNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            gnn=self.gnn_arc,
            mode=self.mode,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        for dis_epoch in range(5):
            discriminator_loss = self.train_d(source_data, target_data)
        generator_loss = self.train_g(source_data, target_data)

        if self.mode == 'graph':
            source_logits = self.gnn(source_data.x, source_data.edge_index, batch=source_data.batch)
            target_logits = self.gnn(target_data.x, target_data.edge_index, batch=target_data.batch)
        else:
            source_logits = self.gnn(source_data.x, source_data.edge_index)
            target_logits = self.gnn(target_data.x, target_data.edge_index)

        loss = discriminator_loss + generator_loss

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        if self.mode == 'node':
            if not is_undirected(source_data.edge_index):
                source_data.edge_index = to_undirected(source_data.edge_index)
        
            if not is_undirected(target_data.edge_index):
                target_data.edge_index = to_undirected(target_data.edge_index)
        
            self.sample_size = min(source_data.x.shape[0], target_data.x.shape[0])

            if self.batch_size == 0:
                self.source_batch_size = source_data.x.shape[0]
                self.source_loader = NeighborLoader(source_data, self.num_neigh, batch_size=self.source_batch_size)
                self.target_batch_size = target_data.x.shape[0]
                self.target_loader = NeighborLoader(target_data, self.num_neigh, batch_size=self.target_batch_size)
            else:
                self.source_loader = NeighborLoader(source_data, self.num_neigh, batch_size=self.batch_size)
                self.target_loader = NeighborLoader(target_data, self.num_neigh, batch_size=self.batch_size)
        elif self.mode == 'graph':
            self.sample_size = min(len(source_data), len(target_data))

            if self.batch_size == 0:
                num_source_graphs = len(source_data)
                num_target_graphs = len(target_data)
                self.source_loader = DataLoader(source_data, batch_size=num_source_graphs, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=num_target_graphs, shuffle=True)
            else:
                self.source_loader = DataLoader(source_data, batch_size=self.batch_size, shuffle=True)
                self.target_loader = DataLoader(target_data, batch_size=self.batch_size, shuffle=True)
        else:
            assert self.mode in ('graph', 'node'), 'Invalid train mode'

        self.gnn = self.init_model(**self.kwargs)

        self.domain_discriminator = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 1)
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.d_optimizer = torch.optim.Adam(
            self.domain_discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.gnn.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)

                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data)
                epoch_loss += loss

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = source_logits, sampled_source_data.y
                else:
                    source_logits, source_labels = source_logits, sampled_source_data.y
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
    
    def train_d(self, source_data, target_data):
        self.gnn.eval()

        if self.mode == 'node':
            embedding_s = self.gnn.feat_bottleneck(source_data.x, source_data.edge_index)
            output_s = self.gnn.feat_classifier(embedding_s, source_data.edge_index)

            embedding_t = self.gnn.feat_bottleneck(target_data.x, target_data.edge_index)
            output_t = self.gnn.feat_classifier(embedding_t, target_data.edge_index)
        else:
            embedding_s = self.gnn.feat_bottleneck(source_data.x, source_data.edge_index)
            embedding_s = global_mean_pool(embedding_s, source_data.batch)
            output_s = self.gnn.feat_classifier(embedding_s, source_data.edge_index)

            embedding_t = self.gnn.feat_bottleneck(target_data.x, target_data.edge_index)
            embedding_t = global_mean_pool(embedding_t, target_data.batch)
            output_t = self.gnn.feat_classifier(embedding_t, target_data.edge_index)

        node_s = torch.tensor([1.0 for i in range(0, embedding_s.shape[0])])
        node_t = torch.tensor([1.0 for i in range(0, embedding_t.shape[0])])

        train_idx_s = torch.multinomial(node_s, 8 * self.sample_size, replacement=True)
        train_idx_t = torch.multinomial(node_t, 8 * self.sample_size, replacement=True)

        pre_s = self.domain_discriminator(embedding_s[train_idx_s])
        pre_t = self.domain_discriminator(embedding_t[train_idx_t])

        self.d_optimizer.zero_grad()

        loss = (pre_s ** 2).mean() + ((pre_t - 1) ** 2).mean()
        loss.backward()

        self.d_optimizer.step()

        return loss.item()
    
    def L_GCN(self, embedding, nodes_weight, idx_u, idx_v, k):
        embedding_u = embedding[idx_u]
        embedding_v = embedding[idx_v]

        embedding_neg = [embedding[torch.multinomial(nodes_weight, self.sample_size, replacement=False)] for i in range(0, k)]

        pos = torch.sum(torch.mul(embedding_u, embedding_v), dim=1)
        neg = [torch.sum(torch.mul(embedding_u, embedding_neg[i]) * (-1), dim=1) for i in range(0, k)]
        loss = - torch.sum(F.logsigmoid(pos))
        for i in range(0, k):
            loss = loss - torch.sum(F.logsigmoid(neg[i]))
        return loss

    def L_cluster(self, labelsA, embA, labelsB, embB):
        loss = 0.0
        labelsA = labelsA.detach().cpu().numpy()
        labelsB = labelsB.detach().cpu().numpy()
        for i in range(self.num_classes):
            idxA = np.where(labelsA == i)
            idxB = np.where(labelsB == i)
            if (idxA[0].size > 0 and idxB[0].size > 0):
                loss += torch.sum((torch.mean(embA[idxA], dim=0) - torch.mean(embB[idxB], dim=0)) ** 2)
        answer = loss / self.num_classes
        return answer
    
    def train_g(self, source_data, target_data):
        self.gnn.train()

        if self.mode == 'node':
            embedding_s = self.gnn.feat_bottleneck(source_data.x, source_data.edge_index)
            output_s = self.gnn.feat_classifier(embedding_s, source_data.edge_index)

            embedding_t = self.gnn.feat_bottleneck(target_data.x, target_data.edge_index)
            output_t = self.gnn.feat_classifier(embedding_t, target_data.edge_index)
        else:
            embedding_s = self.gnn.feat_bottleneck(source_data.x, source_data.edge_index)
            embedding_s = global_mean_pool(embedding_s, source_data.batch)
            output_s = self.gnn.feat_classifier(embedding_s, source_data.edge_index)

            embedding_t = self.gnn.feat_bottleneck(target_data.x, target_data.edge_index)
            embedding_t = global_mean_pool(embedding_t, target_data.batch)
            output_t = self.gnn.feat_classifier(embedding_t, target_data.edge_index)

        node_s = torch.tensor([1.0 for i in range(0, embedding_s.shape[0])])
        node_t = torch.tensor([1.0 for i in range(0, embedding_t.shape[0])])

        train_idx_s = torch.multinomial(node_s, 8 * self.sample_size, replacement=True)
        train_idx_t = torch.multinomial(node_t, 8 * self.sample_size, replacement=True)

        pre_s = self.domain_discriminator(embedding_s[train_idx_s])
        pre_t = self.domain_discriminator(embedding_t[train_idx_t])

        l_adv = (pre_t ** 2).mean() + ((pre_s - 1) ** 2).mean()

        if self.mode == 'node':
            sample_edge_s = torch.multinomial(torch.ones(source_data.edge_index.shape[1]), self.sample_size, replacement=False)
            idx_u_s = [source_data.edge_index[0][i] for i in sample_edge_s]
            idx_v_s = [source_data.edge_index[1][i] for i in sample_edge_s]

            sample_edge_t = torch.multinomial(torch.ones(target_data.edge_index.shape[1]), self.sample_size, replacement=False)
            idx_u_t = [target_data.edge_index[0][i] for i in sample_edge_t]
            idx_v_t = [target_data.edge_index[1][i] for i in sample_edge_t]

            _, nodes_weight_s = torch.unique(source_data.edge_index[0], return_counts=True)
            nodes_weight_s = torch.pow(nodes_weight_s, 0.75)

            _, nodes_weight_t = torch.unique(target_data.edge_index[0], return_counts=True)
            nodes_weight_t = torch.pow(nodes_weight_t, 0.75)

            l_gcn1 = self.L_GCN(embedding_s, nodes_weight_s, idx_u_s, idx_v_s, self.k)
            l_gcn2 = self.L_GCN(embedding_t, nodes_weight_t, idx_u_t, idx_v_t, self.k)

            l_gcn = l_gcn1 + l_gcn2
        else:
            l_gcn = 0

        l_ce = F.cross_entropy(output_s, source_data.y)

        if self.train_mode == 'semi':
            label_idx_t = torch.multinomial(node_t, int(self.tgt_rate * len(node_t)), replacement=False)
            l_ce_t = F.cross_entropy(output_t[label_idx_t], target_data.y[label_idx_t])
            l_ce += l_ce_t
            l_c = self.L_cluster(source_data.y, embedding_s, target_data.y[label_idx_t], embedding_t[label_idx_t])
            loss = l_gcn + l_adv * 0.1 + l_ce + l_c
        elif self.train_mode == 'unsup':
            loss = l_gcn + l_ce + l_adv * 0.1

        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        return loss.item()

    def predict(self, data, source=False):
        self.gnn.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    if self.mode == 'graph':
                        logits = self.gnn(sampled_data.x, sampled_data.edge_index, batch=sampled_data.batch)
                    else:
                        logits = self.gnn(sampled_data.x, sampled_data.edge_index)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))
        else:
            for idx, sampled_data in enumerate(self.target_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    if self.mode == 'graph':
                        logits = self.gnn(sampled_data.x, sampled_data.edge_index, batch=sampled_data.batch)
                    else:
                        logits = self.gnn(sampled_data.x, sampled_data.edge_index)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
