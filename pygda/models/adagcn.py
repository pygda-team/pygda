import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import torch.nn as nn
from torch_geometric.loader import NeighborLoader, DataLoader

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
    mode : str, optional
        Mode for node or graph level tasks. Default: ``node``.
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
        mode='node',
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
        self.mode=mode

    def init_model(self, **kwargs):
        """
        Initialize the AdaGCN model.

        Parameters
        ----------
        **kwargs
            Other parameters for the AdaGCNBase model.

        Returns
        -------
        AdaGCNBase
            Initialized AdaGCN model on the specified device.
        """

        return AdaGCNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            gnn_type=self.gnn_type,
            mode=self.mode,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        """
        Forward pass of the model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Returns
        -------
        tuple
            Contains:
            - loss : torch.Tensor
                Combined loss from classification and domain adaptation.
            - source_logits : torch.Tensor
                Model predictions for source domain.
            - target_logits : torch.Tensor
                Model predictions for target domain.

        Notes
        -----
        Performs adversarial training with:

        - Discriminator optimization
        - Gradient penalty computation
        - Classification loss
        - Domain adaptation loss
        """
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
        """
        Train the AdaGCN model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process includes:

        - Setting up data loaders for both domains
        - Initializing GNN and discriminator
        - Training with adversarial learning
        - Supporting both node and graph level tasks
        """
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

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.adagcn.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data)
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
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data to be processed.

        Notes
        -----
        Placeholder method for graph preprocessing.
        """
        pass

    def predict(self, data, source=False):
        """
        Make predictions on given data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        source : bool, optional
            Whether the input is from source domain.
            Default: ``False``.

        Returns
        -------
        tuple
            Contains:
            - logits : torch.Tensor
                Model predictions.
            - labels : torch.Tensor
                True labels.

        Notes
        -----
        Handles predictions for both source and target domains
        using appropriate data loaders.
        """
        self.adagcn.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    encoded_data = self.adagcn(sampled_data)
                    logits = self.adagcn.cls_model(encoded_data)

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
                    encoded_data = self.adagcn(sampled_data)
                    logits = self.adagcn.cls_model(encoded_data)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels

    def gradient_penalty(self, encoded_source, encoded_target):
        """
        Compute gradient penalty for Wasserstein GAN training.

        Parameters
        ----------
        encoded_source : torch.Tensor
            Encoded features from source domain.
        encoded_target : torch.Tensor
            Encoded features from target domain.

        Returns
        -------
        torch.Tensor
            Computed gradient penalty value.

        Notes
        -----
        Implements Wasserstein GAN gradient penalty by:
        
        - Interpolating between source and target features
        - Computing gradients w.r.t. discriminator outputs
        - Penalizing gradients that deviate from norm 1
        - Handling different batch sizes between domains
        """
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
