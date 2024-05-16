import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader

from . import BaseGDA
from ..nn import JHGDABase
from ..utils import logger, MMD
from ..metrics import eval_macro_f1, eval_micro_f1


class JHGDA(BaseGDA):
    """
    Improving Graph Domain Adaptation with Network Hierarchy (CIKM-23).

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
        Weight decay (L2 penalty). Default: ``0.``.
    share : bool, optional
        Share the diffpool module or not. Default: ``False``.
    sparse : bool, optional
        Diffpool module sparse or not. Default: ``False``.
    classwise : bool, optional
        Classwise conditional shift or not. Default: ``True``.
    pool_ratio : float, optional
        Graph pooling ratio. Default: ``0.2``.
    g_mmd : float, optional
        Global mmd weight. Default: ``0.5``.
    c_mmd : float, optional
        Conditional mmd weight. Default: ``0.5``.
    d_weight : float, optional
        Domain loss weight. Default: ``0.1``.
    ce_weight : float, optional
        Cluster entropy weight. Default: ``0.1``.
    prox_weight : float, optional
        Proximity loss weight. Default: ``0.1``.
    cce_weight : float, optional
        Conditional cluster entropy weight. Default: ``0.1``.
    lm_weight : float, optional
        Label matching weight. Default: ``0.1``.
    ls_weight : float, optional
        Label stable weight. Default: ``0.1``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
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
        num_layers=3,
        dropout=0.,
        act=F.relu,
        weight_decay=0.,
        share=False,
        sparse=False,
        classwise=True,
        pool_ratio=0.2,
        g_mmd=0.5,
        c_mmd=0.5,
        d_weight=0.1,
        ce_weight=0.1,
        prox_weight=0.1,
        cce_weight=0.1,
        lm_weight=0.1,
        ls_weight=0.1,
        lr=4e-3,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(JHGDA, self).__init__(
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
        
        self.sparse=sparse
        self.share=share
        self.classwise=classwise
        self.pool_ratio=pool_ratio
        self.g_mmd=g_mmd
        self.c_mmd=c_mmd
        self.d_weight=d_weight
        self.ce_weight=ce_weight
        self.prox_weight=prox_weight
        self.cce_weight=cce_weight
        self.lm_weight=lm_weight
        self.ls_weight=ls_weight

    def init_model(self, **kwargs):

        return JHGDABase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            device=self.device,
            pool_ratio=self.pool_ratio,
            num_layers=self.num_layers,
            num_s=self.num_s,
            num_t=self.num_t,
            dropout=self.dropout,
            act=self.act,
            share=self.share,
            sparse=self.sparse,
            classwise=self.classwise,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        # source domain cross entropy loss
        embeddings, pred, pooling_loss, label_matrix = self.jhgda(
            source_data.x,
            source_data.edge_index,
            self.jhgda.to_onehot(source_data.y, self.num_classes),
            target_data.x,
            target_data.edge_index,
            self.jhgda.to_onehot(target_data.y, self.num_classes)
        )
        train_loss = self.jhgda.loss_func(pred[0], source_data.y)
        loss = train_loss

        domain_loss = []
        domain_loss_classwise = []

        for i in range(self.num_layers):
            domain_loss_classwise.append(self.jhgda.classwise_simple_mmd(
                source=embeddings[i][0],
                target=embeddings[i][1],
                src_y=label_matrix[i][0],
                tgt_y=label_matrix[i][1]))
            domain_loss.append(self.jhgda.simple_mmd_kernel(embeddings[i][0], embeddings[i][1]))
        
        loss += self.d_weight * (sum(domain_loss) * self.g_mmd + sum(domain_loss_classwise) * self.c_mmd)
        loss += self.cce_weight * sum(x['cce'] for x in pooling_loss)
        loss += self.lm_weight * sum(x['lm'] for x in pooling_loss)
        loss += self.ls_weight * sum(x['ls'] for x in pooling_loss)
        loss += self.ce_weight * sum([x['ce'] for x in pooling_loss])
        loss += self.prox_weight * sum([x['prox'] for x in pooling_loss])
        
        return loss, pred[0], pred[1]

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
        
        self.num_s = source_data.x.shape[0]
        self.num_t = target_data.x.shape[0]

        self.jhgda = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.jhgda.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.jhgda.train()
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data)
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
        self.jhgda.eval()

        with torch.no_grad():
            logits = self.jhgda.inference(data)

        return logits, data.y

