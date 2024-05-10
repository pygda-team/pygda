import torch
import warnings
import torch.nn.functional as F
import itertools
import time
import copy

import numpy as np

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected

from . import BaseGDA
from ..nn import KBLBase
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class KBL(BaseGDA):
    """
    Bridged-GNN: Knowledge Bridge Learning for Effective Knowledge Transfer (CIKM-23).

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
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    k_cross : int, optional
        Number of edges for cross domains. Default: ``20``.
    k_within : int, optional
        Number of edges for within domains. Default: ``6``.
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
        more log information. Default: ``0``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        k_cross=50,
        k_within=10,
        num_layers=2,
        dropout=0.,
        act=F.relu,
        weight_decay=5e-3,
        lr=1e-3,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(KBL, self).__init__(
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
        
        self.k_cross = k_cross
        self.k_within = k_within

    def init_model(self, **kwargs):

        return KBLBase(
            data_src=self.source_data,
            data_tar=self.target_data,
            device=self.device,
            k_cross=self.k_cross,
            k_within=self.k_within,
            bridge_batch_size=1000,
            dim_hidden=64,
            num_layer=2,
            num_epoch=400,
            lr=0.001,
            weight_decay=5e-3,
            source_clf=True,
            norm_mode='None',
            norm_scale=1.,
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

        self.kbl = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.kbl.gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        data_merge = self.kbl.get_bridged_graph()
        data_merge.edge_index = to_undirected(data_merge.edge_index)

        self.stored_data = data_merge

        for epoch in range(self.epoch):
            epoch_loss = 0
            self.kbl.gnn.train()
            log_probs_xs = self.kbl.gnn(data_merge.x, data_merge.edge_index)
            loss = F.nll_loss(log_probs_xs[data_merge.central_mask], data_merge.y[data_merge.central_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()

            epoch_source_preds = log_probs_xs[data_merge.central_mask].argmax(dim=1)
            micro_f1_score = eval_micro_f1(data_merge.y[data_merge.central_mask], epoch_source_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)
    
    def process_graph(self, data):
        pass

    def predict(self, data):
        self.kbl.gnn.eval()

        with torch.no_grad():
            log_probs_xs = self.kbl.gnn(self.stored_data.x, self.stored_data.edge_index)
            logits = log_probs_xs[-data.x.shape[0]:,]

        return logits, data.y
