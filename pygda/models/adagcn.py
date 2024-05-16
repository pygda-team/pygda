import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import torch.nn as nn
from torch_geometric.loader import NeighborLoader

from . import BaseGDA
from ..nn import AdaGCNBase
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class AdaGCN(BaseGDA):
    """
    Graph Transfer Learning via Adversarial Domain Adaptation with Graph Convolution (TKDE-22).

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
    gnn_type : string, optional
        Use GCN or PPMIConv. Default: ``gcn``.
    adv_dim : int, optional
        Hidden dimension of adversarial module. Default: ``40``.
    gp_weight : float, optional
        Trade off parameter for gradient penalty. Default: ``5``.
    domain_weight : float, optional
        Trade off parameter for domain loss. Default: ``1``.
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
        num_layers=3,
        dropout=0.,
        act=F.relu,
        gnn_type='gcn',
        adv_dim=40,
        gp_weight=5,
        domain_weight=1,
        weight_decay=0.,
        lr=4e-3,
        epoch=100,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(AdaGCN, self).__init__(
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
        
        self.gnn_type=gnn_type
        self.adv_dim=adv_dim
        self.gp_weight=gp_weight
        self.domain_weight=domain_weight

    def init_model(self, **kwargs):

        return AdaGCNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            gnn_type=self.gnn_type,
            adv_dim=self.adv_dim,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        for _ in range(10):
            encoded_source = self.adagcn(source_data)
            encoded_target = self.adagcn(target_data)

            gp_loss = self.gradient_penalty(encoded_source, encoded_target)

            dis_s = torch.mean(self.discriminator(encoded_source).reshape(-1))
            dis_t = torch.mean(self.discriminator(encoded_target).reshape(-1))
            dis_loss = - torch.abs(dis_s - dis_t)

            loss = dis_loss + self.gp_weight * gp_loss
        
        # use source classifier loss:
        encoded_source = self.adagcn(source_data)
        encoded_target = self.adagcn(target_data)
        source_logits = self.adagcn.cls_model(encoded_source)
        cls_loss = self.adagcn.loss_func(source_logits, source_data.y)
        dis_s = torch.mean(self.discriminator(encoded_source).reshape(-1))
        dis_t = torch.mean(self.discriminator(encoded_target).reshape(-1))
        dis_loss = torch.abs(dis_s - dis_t)

        target_logits = self.adagcn.cls_model(encoded_target)

        loss = cls_loss + dis_loss * self.domain_weight

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

        self.adagcn = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.adagcn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.discriminator = nn.Sequential(
            nn.Linear(self.hid_dim, self.adv_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.adv_dim, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.c_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.adagcn.train()
                
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
        self.adagcn.eval()

        with torch.no_grad():
            encoded_data = self.adagcn(data)
            logits = self.adagcn.cls_model(encoded_data)

        return logits, data.y

    def gradient_penalty(self, encoded_source, encoded_target):
        num_s = encoded_source.shape[0]
        num_t = encoded_target.shape[0]

        if num_s < num_t:
            hidden = encoded_target[-num_s:,]
            hidden_s = torch.cat((encoded_source, encoded_source), dim=0)
            hidden_t = torch.cat((encoded_target[0:num_s,], hidden), dim=0)

            alpha = torch.rand((2 * num_s, 1)).to(self.device)

            difference = hidden_s - hidden_t
            interpolates = hidden_t + (alpha * difference)
        elif num_s > num_t:
            hidden = encoded_source[-num_t:,]
            hidden_s = torch.cat((encoded_source[0:num_t,], hidden), dim=0)
            hidden_t = torch.cat((encoded_target, encoded_target), dim=0)

            alpha = torch.rand((2 * num_t, 1)).to(self.device)

            difference = hidden_s - hidden_t
            interpolates = hidden_t + (alpha * difference)
        else:
            alpha = torch.rand((num_t, 1)).to(self.device)

            difference = encoded_source - encoded_target
            interpolates = encoded_target + (alpha * difference)

        inputs = torch.cat((encoded_source, encoded_target, interpolates), dim=0)
        scores = self.discriminator(inputs)

        gradient = torch.autograd.grad(
            inputs=inputs,
            outputs=scores,
            grad_outputs=torch.ones_like(scores).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return gradient_penalty

