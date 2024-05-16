import torch
import warnings
import torch.nn.functional as F
import itertools
import time

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import get_laplacian, to_dense_adj

from . import BaseGDA
from ..nn import SAGDABase
from ..nn import GradReverse
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1

import scipy
import numpy as np


class SAGDA(BaseGDA):
    """
    SA-GDA: Spectral Augmentation for Graph Domain Adaptation (MM-23).

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
        Weight decay (L2 penalty). Default: ``0.003``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    alpha : float, optional
        Trade-off parameter for high pass filter. Default: ``1.0``.
    beta : float, optional
        Trade-off parameter for low pass filter. Default: ``1.0``.
    ppmi : bool, optional
        Use PPMI matrix or not. Default: ``True``.
    adv_dim : int, optional
        Hidden dimension of adversarial module. Default: ``40``.
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
        beta=1.0,
        alpha=1.0,
        num_layers=2,
        dropout=0.0,
        act=F.relu,
        ppmi=True,
        adv_dim=40,
        weight_decay=3e-3,
        lr=4e-3,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(SAGDA, self).__init__(
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
        
        assert beta==1.0 and alpha==1.0, 'unsupport for alpha and beta values'
        
        self.ppmi=ppmi
        self.adv_dim=adv_dim
        self.alpha=alpha
        self.beta=beta

    def init_model(self, **kwargs):

        return SAGDABase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            beta=self.beta,
            alpha=self.alpha,
            ppmi=self.ppmi,
            adv_dim=self.adv_dim,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        pass

    def fit(self, source_data, target_data):
        self.num_source_nodes, _ = source_data.x.shape
        self.num_target_nodes, _ = target_data.x.shape

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

        self.sagda = self.init_model(**self.kwargs)

        params = itertools.chain(*[model.parameters() for model in self.sagda.models])

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0

            alpha = min((epoch + 1) / self.epoch, 0.05)

            for model in self.sagda.models:
                model.train()
            
            encoded_source = self.sagda.src_encode(source_data.x, source_data.edge_index)
            encoded_target = self.sagda.encode(target_data, 'target')
            source_logits = self.sagda.cls_model(encoded_source)

            # use source classifier loss:
            cls_loss = self.sagda.loss_func(source_logits, source_data.y)

            source_domain_preds = self.sagda.domain_model(GradReverse.apply(encoded_source, alpha))
            target_domain_preds = self.sagda.domain_model(GradReverse.apply(encoded_target, alpha))

            source_domain_cls_loss = self.sagda.loss_func(
                source_domain_preds,
                torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )
            target_domain_cls_loss = self.sagda.loss_func(
                target_domain_preds,
                torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(self.device)
            )

            loss_grl = source_domain_cls_loss + target_domain_cls_loss
            loss = cls_loss + loss_grl

            # use target classifier loss:
            target_logits = self.sagda.cls_model(encoded_target)
            target_probs = F.softmax(target_logits, dim=1)
            target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

            loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=1))

            loss = loss + loss_entropy * (epoch / self.epoch * 0.01)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss= loss.item()
            
            epoch_source_preds = source_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(source_data.y, epoch_source_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)
    
    def process_graph(self, data):
        pass

    def predict(self, data):
        for model in self.sagda.models:
            model.eval()

        with torch.no_grad():
            encoded_data = self.sagda.encode(data, 'target')
            logits = self.sagda.cls_model(encoded_data)

        return logits, data.y
