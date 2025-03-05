import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np
from tqdm import tqdm
import random

from torch_geometric.loader import NeighborLoader, DataLoader
from torch.nn.parameter import Parameter

from . import BaseGDA
from ..nn import GNNBase
from ..nn import GraphATABase
from ..utils import logger
from ..utils.perturb import *
from ..metrics import eval_macro_f1, eval_micro_f1


class GraphATA(BaseGDA):
    """
    Aggregate to Adapt: Node-Centric Aggregation for Multi-Source-Free Graph Domain Adaptation (WWW-25).

    This class implements a multi-source-free domain adaptation method for graphs using node-centric
    aggregation. It first trains independent source models and then adapts them to the target
    domain using a novel attention-based mechanism.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    num_src_domains : int
        Total number of source domains.
    K : int, optional
        Number of nearest neighbors for memory bank. Default: ``10``.
    momentum : float, optional
        Momentum coefficient for memory bank update. Default: ``0.9``.
    num_layers : int, optional
        Total number of layers in model. Default: ``3``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    gnn : string, optional
        GNN backbone type ('gcn', 'sage', 'gat', 'gin'). Default: ``gcn``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epochs. Default: ``200``.
    device : str, optional
        GPU or CPU device specification. Default: ``cuda:0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Default: ``2``.
    **kwargs
        Additional parameters for the model.

    Attributes
    ----------
    source_model_list : list
        List of trained source domain models.
    graphata : GraphATABase
        The main adaptation model that combines source models.
    source_loader : NeighborLoader
        DataLoader for source domain graphs.
    target_loader : NeighborLoader
        DataLoader for target domain graph.

    Notes
    -----
    The training process consists of two main phases:

    - Source Domain Pretraining:

        * Initializes independent GNN models for each source domain
        * Trains each model separately on its corresponding source data
        * Stores the trained models in source_model_list

    - Target Domain Adaptation:

        * Creates a GraphATABase model using trained source models
        * Uses memory bank mechanism with K-nearest neighbors
        * Applies information maximization and classification losses
        * Updates memory features and class predictions with momentum

    - The adaptation phase employs several key components:

        * Feature bottleneck for transformation
        * Node-centric attention for feature aggregation
        * Memory bank for pseudo-label generation
        * Entropy minimization for target domain alignment

    Examples
    --------
    >>> model = GraphATA(
    ...     in_dim=64,
    ...     hid_dim=32,
    ...     num_classes=7,
    ...     num_src_domains=3
    ... )
    >>> # Train on source domains
    >>> model.fit(source_data_list, target_data)
    >>> # Predict on target domain
    >>> logits, labels = model.predict(target_data)
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_src_domains,
        K=10,
        momentum=0.9,
        num_layers=3,
        dropout=0.,
        act=F.relu,
        weight_decay=0.,
        lr=1e-4,
        epoch=200,
        gnn='gcn',
        mode='node',
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(GraphATA, self).__init__(
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
    
        self.gnn=gnn
        self.num_src_domains = num_src_domains
        self.mode = mode
        self.K = K
        self.momentum = momentum

    def init_model(self, **kwargs):
        """
        Initialize source domain models.

        Creates a list of GNN models, one for each source domain. Each model
        is initialized with the same architecture but will be trained
        independently on different source domains.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to the GNNBase constructor.

        Returns
        -------
        list
            List of initialized GNNBase models, one per source domain.

        Notes
        -----
        Model initialization process:

        - Creates num_src_domains independent GNNBase models

        - Each model has identical architecture with:

            * Input dimension: in_dim
            * Hidden dimension: hid_dim
            * Output classes: num_classes
            * Number of layers: num_layers
            * Dropout rate: dropout
            * GNN type: gnn ('gcn', 'sage', 'gat', or 'gin')
            * Operation mode: mode ('node' or 'graph')

        - All models are moved to the specified device
        - Models are stored in source_model_list for later use

        The initialized models serve as the foundation for:
        
        - Independent source domain training
        - Weight extraction for the GraphATABase model
        - Knowledge transfer to the target domain
        """
        self.source_model_list = [GNNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            gnn=self.gnn,
            mode=self.mode,
            **kwargs
        ).to(self.device) for _ in range(self.num_src_domains)]

        return self.source_model_list

    def forward_model(self, data, **kwargs):
        pass
    
    def train_source(self, src_idx, source_data):
        """
        Train a source domain model.

        Parameters
        ----------
        src_idx : int
            Index of the source domain.
        source_data : torch_geometric.data.Data
            Source domain graph data.

        Notes
        -----
        Process:

        - Initializes model and optimizer for the source domain
        - Creates data loader with specified batch size
        - Trains model using cross-entropy loss
        - Stores trained model in source_model_list[src_idx]
        - Logs training progress using micro-F1 score
        """
        source_model = self.source_model_list[src_idx]
        optimizer = torch.optim.Adam(
            source_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        if self.mode == 'node':
            if self.batch_size == 0:
                self.source_batch_size = source_data.x.shape[0]
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.source_batch_size)
            else:
                self.source_loader = NeighborLoader(
                    source_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
        else:
            if self.batch_size == 0:
                num_source_graphs = len(source_data)
                self.source_loader = DataLoader(source_data, batch_size=num_source_graphs, shuffle=True)
            else:
                self.source_loader = DataLoader(source_data, batch_size=self.batch_size, shuffle=True)
            
        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None
        
            for idx, sampled_source_data in enumerate(self.source_loader):
                source_model.train()
                sampled_source_data = sampled_source_data.to(self.device)
                if self.mode == 'node':
                    source_batch = None
                else:
                    source_batch = sampled_source_data.batch
                source_logits = source_model(sampled_source_data.x, sampled_source_data.edge_index, batch=source_batch)
                loss = F.nll_loss(F.log_softmax(source_logits, dim=1), sampled_source_data.y)
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
                time=time.time() - self.start_time,
                verbose=self.verbose,
                train=True)

        self.source_model_list[src_idx] = source_model

    def fit(self, source_data_list, target_data):
        """
        Train the model on multiple source domains and adapt to target domain.

        Parameters
        ----------
        source_data_list : list of torch_geometric.data.Data
            List of source domain graph data objects.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process:

        - Source Pretraining:

            * Trains independent models for each source domain
            * Stores models in source_model_list

        - Target Adaptation:

            * Initializes GraphATABase with source models
            * Creates memory banks for features and classes
            * Updates model using:

                * Information maximization loss
                * K-NN based pseudo-label loss
                
            * Updates memory banks with momentum
        """
        if self.mode == 'node':
            if self.batch_size == 0:
                self.target_batch_size = target_data.x.shape[0]
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.target_batch_size)
            else:
                self.target_loader = NeighborLoader(
                    target_data,
                    self.num_neigh,
                    batch_size=self.batch_size)
        else:
            if self.batch_size == 0:
                num_target_graphs = len(target_data)
                self.target_loader = DataLoader(target_data, batch_size=num_target_graphs, shuffle=True)
            else:
                self.target_loader = DataLoader(target_data, batch_size=self.batch_size, shuffle=True)

        self.source_model_list = self.init_model(**self.kwargs)

        self.start_time = time.time()

        print('Source domain pretraining...')
        # Train each source model independently
        for src_idx in range(len(source_data_list)):
            self.train_source(src_idx, source_data_list[src_idx])

        print('Target domain adaptation...')
        param_group = []
        for i in range(len(source_data_list)):
            param_group += list(self.source_model_list[i].parameters())

        weight_list = []
        for model in self.source_model_list:
            w_list = []
            for name, param in model.named_parameters():
                if name[-10:] == 'lin.weight' and model.gnn == 'gcn':
                    w_list.append(param)
                elif name[-12:] == 'lin_l.weight' and model.gnn == 'sage':
                    w_list.append(param)
                elif name[-14:] == 'lin_src.weight' and model.gnn == 'gat':
                    w_list.append(param)
                elif name[-11:] == 'nn.0.weight' and model.gnn == 'gin':
                    w_list.append(param)
            weight_list.append(w_list)

        weight_listv2 = list(zip(*weight_list))

        self.graphata = GraphATABase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            model_weights=weight_listv2,
            model_list=self.source_model_list,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            gnn=self.gnn,
            mode=self.mode,
            **self.kwargs
        ).to(self.device)

        param_group += list(self.graphata.parameters())

        optimizer_t = torch.optim.Adam(param_group, lr=self.lr, weight_decay=self.weight_decay)
        if self.mode == 'node':
            mem_fea = torch.rand(target_data.x.size(0), self.hid_dim).to(self.device)
            mem_cls = torch.ones(target_data.x.size(0), self.num_classes).to(self.device) / self.num_classes
        else:
            mem_fea = torch.rand(len(target_data), self.hid_dim).to(self.device)
            mem_cls = torch.ones(len(target_data), self.num_classes).to(self.device) / self.num_classes

        for epoch in range(self.epoch):
            self.graphata.train()
            for i, data in enumerate(self.target_loader):    
                optimizer_t.zero_grad()
                data = data.to(self.device)
                if self.mode == 'node':
                    feat_output = self.graphata.feat_bottleneck(data.x, data.edge_index)
                    cls_output = self.graphata.feat_classifier(feat_output, data.edge_index)
                else:
                    feat_output = self.graphata.feat_bottleneck(data.x, data.edge_index, batch=data.batch)
                    cls_output = self.graphata.feat_classifier(feat_output, data.edge_index)
                softmax_out = F.softmax(cls_output, dim=1)
                entropy_loss = torch.mean(self.entropy(softmax_out))
                mean_softmax = softmax_out.mean(dim=0)
                div_loss = torch.sum(mean_softmax * torch.log(mean_softmax + 1e-5))
                im_loss = entropy_loss + div_loss

                feat_norm = F.normalize(feat_output, dim=1)
                mem_fea_norm = F.normalize(mem_fea, dim=1)
                distance = feat_norm @ mem_fea_norm.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k=self.K + 1)
                idx_near = idx_near[:, 1:]
                pred_near = torch.mean(mem_cls[idx_near], dim=1)
                _, preds = torch.max(pred_near, dim=1)
                cls_loss = F.cross_entropy(cls_output, preds) 
                loss = im_loss + cls_loss

                loss.backward()
                optimizer_t.step()

                self.graphata.eval()
                with torch.no_grad():
                    feat_output = self.graphata.feat_bottleneck(data.x, data.edge_index)
                    cls_output = self.graphata.feat_classifier(feat_output, data.edge_index)
                    softmax_out = F.softmax(cls_output, dim=1)
                    outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))
        
                mem_cls = (1.0 - self.momentum) * mem_cls + self.momentum * outputs_target.clone()
                mem_fea = (1.0 - self.momentum) * mem_fea + self.momentum * feat_output.clone()

                logits, _ = self.predict(data)
                preds = logits.argmax(dim=1)
                micro_f1_score = eval_micro_f1(data.y, preds)
        
                logger(epoch=epoch,
                    loss=loss,
                    target=micro_f1_score,
                    time=time.time() - self.start_time,
                    verbose=self.verbose,
                    train=True)
        
    def process_graph(self, data):
        pass
    
    def entropy(self, input_):
        """
        Calculate entropy of input probabilities.

        Parameters
        ----------
        input_ : torch.Tensor
            Input probability distribution.

        Returns
        -------
        torch.Tensor
            Entropy values for each input sample.

        Notes
        -----
        Used for information maximization loss during
        target domain adaptation.
        """
        entropy = -input_ * torch.log(input_ + 1e-8)
        entropy = torch.sum(entropy, dim=1)
        
        return entropy 

    def predict(self, data):
        """
        Make predictions on new data using adapted model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.

        Returns
        -------
        tuple
            - logits : torch.Tensor
                Model predictions (num_nodes, num_classes)
            - labels : torch.Tensor
                True labels if available

        Notes
        -----
        Uses the adapted GraphATABase model for prediction,
        which combines knowledge from all source domains
        through attention-based aggregation.
        """
        self.graphata.eval()

        for idx, sampled_data in enumerate(self.target_loader):
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                logits = self.graphata(sampled_data.x, sampled_data.edge_index, batch=sampled_data.batch)
                if idx == 0:
                    logits, labels = logits, sampled_data.y
                else:
                    sampled_logits, sampled_labels = logits, sampled_data.y
                    logits = torch.cat((logits, sampled_logits))
                    labels = torch.cat((labels, sampled_labels))

        return logits, labels
