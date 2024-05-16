import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader

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
        num_layers=2,
        dropout=0.,
        act=F.relu,
        disc='JS',
        weight=0.01,
        weight_decay=0.01,
        lr=0.001,
        epoch=100,
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

    def init_model(self, **kwargs):

        return GRADEBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            disc=self.disc,
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
            domain_labels = np.array([0] * source_data.x.size(0) + [1] * target_data.x.size(0))
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=source_data.x.device)
            domain_loss = self.grade.criterion(domain_preds, domain_labels)
        elif self.disc == 'MMD':
            mind = min(source_data.x.size(0), target_data.x.size(0))
            domain_loss = MMD(source_feats[:mind], target_feats[:mind])
        elif self.disc == 'C':
            ratio = 8
            s_l_f = torch.cat([source_feats, ratio * self.grade.one_hot_embedding(source_data.y)], dim=1)
            t_l_f = torch.cat([target_feats, ratio * F.softmax(target_logits, dim=1)], dim=1)
            domain_preds = self.grade.discriminator(GradReverse.apply(torch.cat([s_l_f, t_l_f], dim=0), alpha))
            domain_labels = np.array([0] * source_data.x.size(0) + [1] * target_data.x.size(0))
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=source_data.x.device)
            domain_loss = self.grade.criterion(domain_preds, domain_labels)

        loss = loss + domain_loss * self.weight

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):

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

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.grade.train()
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, alpha)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
        self.grade.eval()

        with torch.no_grad():
            logits, _ = self.grade(data)

        return logits, data.y
