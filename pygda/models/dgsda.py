import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from . import BaseGDA
from ..nn import DGSDABase
from ..utils import logger, MMD
from ..metrics import eval_macro_f1, eval_micro_f1


class DGSDA(BaseGDA):
    """
    Disentangled Graph Spectral Domain Adaptation (ICML-25).

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
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    K : int, optional
        Order of the polynomial filter. Default: ``8``.
    alpha : float, optional
        Theta loss trade-off parameter. Default: ``0.05``.
    beta : float, optional
        MMD loss trade-off parameter. Default: ``0.5``.
    gamma : float, optional
        Entropy loss trade-off parameter. Default: ``0.05``.
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
        num_layers=2,
        dropout=0.,
        act=F.relu,
        K=8,
        alpha=0.05,
        beta=0.5,
        gamma=0.05,
        weight_decay=0.,
        lr=4e-3,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(DGSDA, self).__init__(
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
        
        assert num_layers==2, 'unsupport number of layers'
        assert mode=='node', 'unsupport mode'

        self.K=K
        self.mode=mode
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma

    def init_model(self, **kwargs):
        """
        Initialize the DGSDA base model.

        Parameters
        ----------
        **kwargs
            Additional parameters for model initialization.

        Returns
        -------
        DGSDABase
            Initialized model with specified architecture parameters.
        """

        return DGSDABase(
            features=self.in_dim,
            hidden=self.hid_dim,
            classes=self.num_classes,
            dprate=self.dropout,
            K=self.K,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data):
        """
        Forward pass of the DGSDA model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data containing node features, edge indices, and labels.
        target_data : torch_geometric.data.Data
            Target domain graph data containing node features, edge indices, and labels.

        Returns
        -------
        tuple
            Contains:
            
            - loss : torch.Tensor
                Total training loss combining multiple loss components.
            
            - source_logits : torch.Tensor
                Source domain predictions.

        Notes
        -----
        The forward pass computes multiple loss components:
    
        - Source domain cross-entropy loss for supervised learning
        - Temperature parameter L1 loss between source and target domains
        - Maximum Mean Discrepancy (MMD) loss for domain alignment
        - Entropy minimization loss for target domain regularization
        """

        # source domain cross entropy loss
        source_logits = self.dgsda(source_data)
        train_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        loss = train_loss

        theta_s = self.dgsda.prop1.temp
        theta_t = self.dgsda.prop2.temp

        theta_loss = F.l1_loss(theta_s, theta_t)
        loss = loss + theta_loss * self.alpha

        source_feature = F.relu(self.dgsda.lin1(source_data.x))
        target_feature = F.relu(self.dgsda.lin1(target_data.x))
        mmd_loss = MMD(source_feature, target_feature)
        loss = loss + mmd_loss * self.beta

        target_outputs = self.dgsda(target_data, False)
        entropy_loss = self.entropy_minimization_loss(target_outputs)
        loss = loss + entropy_loss * self.gamma 

        return loss, source_logits
    
    def entropy_minimization_loss(self, output):
        """
        Compute entropy minimization loss for target domain regularization.

        Parameters
        ----------
        output : torch.Tensor
            Model predictions of shape [num_nodes, num_classes].

        Returns
        -------
        torch.Tensor
            Entropy minimization loss value.
        """

        probs = F.softmax(output, dim=1)
        log_probs = F.log_softmax(output, dim=1)
        a = torch.sum(probs, dim=0)
        entropy_loss = -torch.sum(probs * log_probs / (a / torch.sum(a)), dim=1).mean()
        
        return entropy_loss

    def fit(self, source_data, target_data):
        """
        Train the DGSDA model on source and target domain data.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.
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

        self.dgsda = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.dgsda.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(self.source_loader, self.target_loader)):
                self.dgsda.train()

                sampled_source_data = sampled_source_data.to(self.device)
                sampled_target_data = sampled_target_data.to(self.device)
                
                loss, source_logits = self.forward_model(sampled_source_data, sampled_target_data)
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
        """

        self.dgsda.eval()

        if source:
            for idx, sampled_data in enumerate(self.source_loader):
                sampled_data = sampled_data.to(self.device)
                with torch.no_grad():
                    logits = self.dgsda(sampled_data)

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
                    logits = self.dgsda(sampled_data, False)

                    if idx == 0:
                        logits, labels = logits, sampled_data.y
                    else:
                        sampled_logits, sampled_labels = logits, sampled_data.y
                        logits = torch.cat((logits, sampled_logits))
                        labels = torch.cat((labels, sampled_labels))

        return logits, labels
