import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader

from . import BaseGDA
from ..nn import A2GNNBase
from ..utils import logger, MMD
from ..metrics import eval_macro_f1, eval_micro_f1


class A2GNN(BaseGDA):
    """
    Rethinking Propagation for Unsupervised Graph Domain Adaptation (AAAI-24).

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
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    s_pnums : int, optional
        Number of propagations for source models. Default: ``0``.
    t_pnums : int, optional
        Number of propagations for target models. Default: ``30``.
    adv : bool, optional
        Adversarial training or not. Default: ``False``.
    weight : int, optional
        Loss trade-off parameter. Default: ``5``.
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
        s_pnums=0,
        t_pnums=30,
        adv=False,
        weight=5,
        weight_decay=0.,
        lr=4e-3,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(A2GNN, self).__init__(
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
        
        self.s_pnums=s_pnums
        self.t_pnums=t_pnums
        self.adv=adv
        self.weight=weight

    def init_model(self, **kwargs):

        return A2GNNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            adv=self.adv,
            dropout=self.dropout,
            act=self.act,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data, alpha):
        # source domain cross entropy loss
        source_logits = self.a2gnn(source_data, self.s_pnums)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        source_features = self.a2gnn.feat_bottleneck(source_data.x, source_data.edge_index, self.s_pnums)
        target_features = self.a2gnn.feat_bottleneck(target_data.x, target_data.edge_index, self.t_pnums)

        # Adv loss
        if self.adv:
            source_dlogits = self.a2gnn.domain_classifier(source_features, alpha)
            target_dlogits = self.a2gnn.domain_classifier(target_features, alpha)
            
            domain_label = torch.tensor(
                [0] * source_data.x.shape[0] + [1] * target_data.x.shape[0]
                ).to(self.device)
            
            domain_loss = F.cross_entropy(torch.cat([source_dlogits, target_dlogits], 0), domain_label)
            loss = loss + self.weight * domain_loss
        else:
            # MMD loss
            mmd_loss = MMD(source_features, target_features)
            loss = loss + mmd_loss * self.weight

        target_logits = self.a2gnn(target_data, self.t_pnums)

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

        self.a2gnn = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.a2gnn.parameters(),
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

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.a2gnn.train()
                
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
        self.a2gnn.eval()

        with torch.no_grad():
            logits = self.a2gnn(data, self.s_pnums)

        return logits, data.y
