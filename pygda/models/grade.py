import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from . import BaseGDA
from ..nn import GRADEBase, GradReverse
from ..utils import logger, MMD
from ..metrics import eval_macro_f1, eval_micro_f1


class GRADE(BaseGDA):
    """
    Non-IID Transfer Learning on Graphs (AAAI-23).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    mode : str, optional
        Mode for node or graph level tasks. Default: ``node``.
    num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.01``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    disc : str, optional
        Discriminator. Default: ``JS``.
    weight : int, optional
        Loss trade-off parameter. Default: ``0.01``.
    lr : float, optional
        Learning rate. Default: ``0.001``.
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
        mode='node',
        num_layers=2,
        dropout=0.,
        act=F.relu,
        disc='JS',
        weight=0.01,
        weight_decay=0.01,
        lr=0.001,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(GRADE, self).__init__(
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
        
        self.disc=disc
        self.weight=weight
        self.mode=mode

    def init_model(self, **kwargs):

        return GRADEBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            disc=self.disc,
            mode=self.mode,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data, alpha):
        # source domain cross entropy loss
        source_logits, source_feats = self.grade(source_data)
        target_logits, target_feats = self.grade(target_data)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        # domain loss
        domain_loss = 0
        if self.disc == 'JS':
            domain_preds = self.grade.discriminator(GradReverse.apply(torch.cat([source_feats, target_feats], dim=0), alpha))
            if self.mode == 'node':
                domain_labels = np.array([0] * source_data.x.size(0) + [1] * target_data.x.size(0))
            else:
                domain_labels = np.array([0] * len(source_data) + [1] * len(target_data))
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=self.device)
            domain_loss = self.grade.criterion(domain_preds, domain_labels)
        elif self.disc == 'MMD':
            if self.mode == 'node':
                mind = min(source_data.x.size(0), target_data.x.size(0))
            else:
                mind = min(len(source_data), len(target_data))
            domain_loss = MMD(source_feats[:mind], target_feats[:mind])
        elif self.disc == 'C':
            ratio = 8
            s_l_f = torch.cat([source_feats, ratio * self.grade.one_hot_embedding(source_data.y)], dim=1)
            t_l_f = torch.cat([target_feats, ratio * F.softmax(target_logits, dim=1)], dim=1)
            domain_preds = self.grade.discriminator(GradReverse.apply(torch.cat([s_l_f, t_l_f], dim=0), alpha))
            if self.mode == 'node':
                domain_labels = np.array([0] * source_data.x.size(0) + [1] * target_data.x.size(0))
            else:
                domain_labels = np.array([0] * len(source_data) + [1] * len(target_data))
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=self.device)
            domain_loss = self.grade.criterion(domain_preds, domain_labels)

        loss = loss + domain_loss * self.weight

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        if self.mode == 'node':
            self.num_source_nodes, _ = source_data.x.shape
            self.num_target_nodes, _ = target_data.x.shape

            if self.batch_size == 0:
                self.source_batch_size = source_data.x.shape[0]
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.source_batch_size)
                self.target_batch_size = target_data.x.shape[0]
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.target_batch_size)
            else:
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
        elif self.mode == 'graph':
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

        self.grade = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.grade.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            alpha = 2 / (1 + np.exp(- 10 * epoch / self.epoch)) - 1

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.grade.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, alpha)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

    def predict(self, data, source=False):
        self.grade.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits, _ = self.grade(sampled_data)

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
                    logits, _ = self.grade(sampled_data)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
