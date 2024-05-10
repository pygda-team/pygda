import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader

from . import BaseGDA
from ..nn import CWGCNBase
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class CWGCN(BaseGDA):
    """

    Correntropy-Induced Wasserstein GCN: Learning Graph Embedding via Domain Adaptation (TIP-2023).

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
    gnn : string, optional
        GNN backbone. Default: ``gcn``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
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
        more log information. Default: ``0``.
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
        gnn='gcn',
        act=F.relu,
        weight_decay=0.0001,
        lr=0.001,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(CWGCN, self).__init__(
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

        assert gnn in ('gcn', 'sage', 'gat', 'gin'), 'Invalid gnn backbone'
        assert num_layers==2, 'unsupport number of layers'

        self.gnn=gnn

    def init_model(self, **kwargs):

        return CWGCNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            gnn=self.gnn,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data):
        # source domain cross entropy loss
        source_logits, x_list = self.gnn(source_data.x, source_data.edge_index)

        loss, weight = self.gnn.c_loss(source_logits, source_data.y)

        return loss, source_logits, weight, x_list

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

        self.gnn = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        print('Step 1, train source data...')

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, sampled_source_data in enumerate(source_loader):
                self.gnn.train()
                
                loss, source_logits, weight, x_list = self.forward_model(sampled_source_data)
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
        
        print('Step 2, train target data...')

        for k,v in self.gnn.named_parameters():
            if k == 'cls.bias' or k == 'cls.lin.weight':
                v.requires_grad = False

        optimizer_t = torch.optim.Adam(
            self.gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_target_logits = None
            epoch_target_labels = None

            mean_s, std_s = self.get_src_mean_std(x_list[0], weight)
            mean_s_2, std_s_2 = self.get_src_mean_std(x_list[1], weight)

            for idx, sampled_target_data in enumerate(target_loader):
                self.gnn.train()
                target_logits, tgt_list = self.gnn(sampled_target_data.x, sampled_target_data.edge_index)
                std_t, mean_t = torch.std_mean(tgt_list[0], dim=0)
                loss = torch.sum(torch.square(mean_s.detach() - mean_t)) + torch.sum(torch.square(std_s.detach() - std_t))
                std_t_2, mean_t_2 = torch.std_mean(tgt_list[1], dim=0)
                loss = torch.sum(torch.square(mean_s_2.detach() - mean_t_2)) + torch.sum(torch.square(std_s_2.detach() - std_t_2))

                optimizer_t.zero_grad()
                loss.backward()
                optimizer_t.step()

                epoch_loss = loss.item()

                if idx == 0:
                    epoch_target_logits, epoch_target_labels = target_logits, sampled_target_data.y
                else:
                    target_logits, target_labels = target_logits, sampled_target_data.y
                    epoch_target_logits = torch.cat((epoch_target_logits, target_logits))
                    epoch_target_labels = torch.cat((epoch_target_labels, target_labels))
            
            epoch_target_preds = epoch_target_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(epoch_target_labels, epoch_target_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   target=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)
    
    def process_graph(self, data):
        pass

    def predict(self, data):
        self.gnn.eval()

        with torch.no_grad():
            logits, _ = self.gnn(data.x, data.edge_index)

        return logits, data.y
    
    def get_src_mean_std(self, embedding, weight):
        sum_w = torch.sum(weight)
        weight = weight.view(-1, 1) * (1 / sum_w)
        mean_e = torch.sum(embedding * weight, dim=0)
        var_e = torch.sum(torch.pow(embedding - mean_e, 2) * weight, dim=0)

        V = torch.sum(torch.pow(weight, 2))
        var_e = torch.pow(var_e / (1 - V), 1/2)

        return mean_e, var_e
