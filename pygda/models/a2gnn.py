import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

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
        mode='node',
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
        self.mode=mode

    def init_model(self, **kwargs):
        """
        Initialize the A2GNN base model.

        Parameters
        ----------
        **kwargs
            Additional parameters for model initialization.

        Returns
        -------
        A2GNNBase
            Initialized model with specified architecture parameters.

        Notes
        -----
        Configures model with:

        - Asymmetric propagation settings
        - Domain adaptation components
        - Task-specific architecture (node/graph)
        - Optional adversarial training module
        """

        return A2GNNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            adv=self.adv,
            dropout=self.dropout,
            act=self.act,
            mode=self.mode,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data, alpha):
        """
        Forward pass of the A2GNN model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.
        alpha : float
            Gradient reversal scaling parameter.

        Returns
        -------
        tuple
            Contains:
            - loss : torch.Tensor
                Combined loss from multiple components.
            - source_logits : torch.Tensor
                Source domain predictions.
            - target_logits : torch.Tensor
                Target domain predictions.

        Notes
        -----
        Implements multiple components:

        - Asymmetric propagation (s_pnums vs t_pnums)
        - Classification loss on source domain
        - Domain adaptation (adversarial or MMD)
        - Feature bottleneck processing
        """

        # source domain cross entropy loss
        source_logits = self.a2gnn(source_data, self.s_pnums)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        if self.mode == 'node':
            source_batch = None
            target_batch = None
        else:
            source_batch = source_data.batch
            target_batch = target_data.batch

        source_features = self.a2gnn.feat_bottleneck(source_data.x, source_data.edge_index, source_batch, self.s_pnums)
        target_features = self.a2gnn.feat_bottleneck(target_data.x, target_data.edge_index, target_batch, self.t_pnums)

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
        """
        Train the A2GNN model on source and target domain data.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process includes:

        Data Preparation

        - Configures loaders for node/graph level tasks
        - Handles both full-batch and mini-batch scenarios
        - Sets up appropriate batch processing

        Training Loop

        - Dynamic adaptation parameter scaling
        - Asymmetric message propagation
        - Domain adaptation through either:
            
            * Adversarial training
            * MMD minimization

        - Comprehensive progress monitoring

        Implementation Features

        - Flexible task handling (node/graph)
        - Efficient batch processing
        - Adaptive learning mechanisms
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

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.a2gnn.train()

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
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data to be processed.

        Notes
        -----
        Placeholder method as preprocessing is handled through:
        
        - Asymmetric propagation mechanisms
        - Domain-specific feature processing
        - Batch-wise data handling
        """

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
        - Uses different propagation steps for source/target
        - Handles batch processing efficiently
        - Concatenates results for full predictions
        - Maintains evaluation mode consistency
        """

        self.a2gnn.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.a2gnn(sampled_data, self.s_pnums)

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
                    logits = self.a2gnn(sampled_data, self.t_pnums)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
