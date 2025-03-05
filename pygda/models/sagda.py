import torch
import warnings
import torch.nn.functional as F
import itertools
import time

from torch_geometric.loader import NeighborLoader, DataLoader

from . import BaseGDA
from ..nn import SAGDABase
from ..nn import GradReverse
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1

import scipy
import numpy as np

from torch_geometric.nn import global_mean_pool


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
    mode : str, optional
        Mode for node or graph level tasks. Default: ``node``.
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
        mode='node',
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
        self.mode=mode

    def init_model(self, **kwargs):
        """
        Initialize the SAGDA base model.

        Parameters
        ----------
        **kwargs
            Additional parameters for model initialization.

        Returns
        -------
        SAGDABase
            Initialized model with specified architecture parameters.

        Notes
        -----
        Configures model with:

        - Spectral augmentation parameters (alpha, beta)
        - PPMI matrix option
        - Adversarial module settings
        - Base architecture parameters (layers, dropout)
        """

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
        """
        Forward pass placeholder.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Main forward logic is implemented in fit method
        to handle spectral augmentation and domain adaptation.
        """
        pass

    def fit(self, source_data, target_data):
        """
        Train the SAGDA model on source and target domain data.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process consists of multiple components:

        Data Handling

        - Supports both node and graph-level tasks
        - Configures appropriate data loaders
        - Handles batch processing

        Model Training

        - Initializes spectral augmentation components
        - Implements adversarial domain adaptation
        - Combines multiple loss terms:
        
            * Classification loss on source domain
            * Domain adversarial loss with gradient reversal
            * Target entropy minimization
            * Spectral augmentation losses

        Implementation Details

        - Dynamic adaptation parameter scaling
        - Graph pooling for graph-level tasks
        - Comprehensive progress monitoring
        - Flexible batch processing options
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

            for idx, (source_data, target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                source_data = source_data.to(self.device)
                target_data = target_data.to(self.device)

            for model in self.sagda.models:
                model.train()
            
            encoded_source = self.sagda.src_encode(source_data.x, source_data.edge_index)
            if self.mode == 'graph':
                encoded_source = global_mean_pool(encoded_source, source_data.batch)
            encoded_target = self.sagda.encode(target_data, 'target')
            if self.mode == 'graph':
                encoded_target = global_mean_pool(encoded_target, target_data.batch)
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
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data to be processed.

        Notes
        -----
        Placeholder method for potential preprocessing steps:

        - Spectral feature computation
        - Graph structure augmentation
        - Feature normalization
        """
        pass

    def predict(self, data, source=False):
        """
        Make predictions on input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        source : bool, optional
            Whether predicting on source domain. Default: ``False``.

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
        Prediction process:
        
        - Uses appropriate encoder based on domain
        - Applies graph pooling for graph-level tasks
        - Handles batch processing
        - Concatenates results for full predictions
        """
        for model in self.sagda.models:
            model.eval()
        
        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    encoded_data = self.sagda.src_encode(sampled_data.x, sampled_data.edge_index)

                    if self.mode == 'graph':
                        encoded_data = global_mean_pool(encoded_data, sampled_data.batch)
                    logits = self.sagda.cls_model(encoded_data)

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
                    encoded_data = self.sagda.encode(sampled_data, 'target')

                    if self.mode == 'graph':
                        encoded_data = global_mean_pool(encoded_data, sampled_data.batch)
                    logits = self.sagda.cls_model(encoded_data)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))
        
        return logits, labels
