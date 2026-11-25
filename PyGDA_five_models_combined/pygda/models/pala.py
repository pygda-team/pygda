import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .base import BaseGDA
from ..nn.pala_base import PALABase
from ..utils import logger


def _validate_pala_graph(data, in_dim=None):
    """
    Access graph features, edges, and cached PPMI weights.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Prepared graph containing node features, edge indices, and cached
        PPMI weights.
    in_dim : int or None, optional
        Expected input feature dimension. This argument is currently unused.
        Default: ``None``.

    Notes
    -----
    This helper accesses the graph tensors without reading node labels.
    """
    x, edges = data.x, data.edge_index
    weight = data.ppmi_edge_weight


class PALA(BaseGDA):
    """
    Class-imbalanced Graph Domain Adaptation via Prototype-anchored Learning and Alignment (IJCAI-25).

    Parameters
    ----------
    in_dim : int
        Number of input features. Default: none (required).
    hid_dim : int
        Encoder hidden dimension. Default: none (required).
    num_classes : int
        Number of source classes. Default: none (required).
    num_layers : int, optional
        Number of encoder layers; only two are supported. Default: ``2``.
    dropout : float, optional
        Encoder dropout probability. Default: ``0.1``.
    act : callable, optional
        Encoder activation; only ReLU is supported.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Adam learning rate. Default: ``0.01``.
    weight_decay : float, optional
        Adam weight decay. Default: ``0.001``.
    epoch : int, optional
        Number of full-graph training epochs. Default: ``200``.
    beta : float, optional
        Weight of the previous ELR memory in its moving average. Default: ``0.9``.
    epsilon : float, optional
        Upper bound for the random interpolation weight. Default: ``0.2``.
    gamma : int, optional
        Number of negative samples per anchor that are left uninterpolated
        and half the queue sampler minimum per class. Default: ``5``.
    anchor_num : int, optional
        Maximum number of anchors per source class. Default: ``200``.
    device : str or torch.device, optional
        Device for training and prediction. Default: ``'cuda:0'``.
    batch_size : int, optional
        Batch size; only full-graph training is supported. Default: ``0``.
    num_neigh : int, optional
        Neighbor sampling setting; only all neighbors are supported.
        Default: ``-1``.
    seed : int or None, optional
        Random seed set at the start of fitting. Default: ``None``.
    verbose : int, optional
        PyGDA logging verbosity, from 0 to 3. Default: ``2``.
    **kwargs : dict
        Extra keyword arguments are unsupported. Default: empty.

    Notes
    -----
    Prepare the source long-tail graph before fitting. Both graphs need
    precomputed PPMI edges and weights in their own node order. The authors'
    public ACMv9 to Citationv1 experiment uses a hidden dimension of 512 and
    source imbalance factor of 20; these are experiment settings.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=2,
        dropout=0.1,
        act=F.relu,
        lr=0.01,
        weight_decay=0.001,
        epoch=200,
        beta=0.9,
        epsilon=0.2,
        gamma=5,
        anchor_num=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        seed=None,
        verbose=2,
        **kwargs,
    ):
        
        super(PALA, self).__init__(
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
            verbose=verbose)
        self.beta, self.epsilon, self.gamma, self.anchor_num = beta, epsilon, gamma, anchor_num
        self.seed = seed
        self.history_ = []
        self.optimizer_ = None
        self.last_target_logits_ = None
        self.sample_queue_ = []

    def process_graph(self, data, include_labels=False):
        """
        Prepare features and cached edges for the model device.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with PPMI edges and weights in the same node order as ``x``.
        include_labels : bool, optional
            Include supplied labels without validating their classes.
            Default: ``False``.

        Returns
        -------
        torch_geometric.data.Data
            Prepared graph with float32 features. Edge and cache tensors are
            shared with the input when already on the model device.
        """
        _validate_pala_graph(data, self.in_dim)
        x = data.x.detach().to(device=self.device, dtype=torch.float32, copy=True)
        

        graph = Data(
            x=x,
            edge_index=data.edge_index.detach().to(self.device),
            ppmi_edge_index=data.ppmi_edge_index.detach().to(self.device),
            ppmi_edge_weight=data.ppmi_edge_weight.detach().to(self.device),
        )
        graph.global_adj = graph.ppmi_edge_index
        labels = getattr(data, 'y', None) if include_labels else None
        if labels is not None:
            graph.y = labels.detach().to(self.device, copy=True)
        return graph

    def init_model(self, source_data, num_target_nodes):
        """
        Initialize the two encoders and target memory.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Prepared source graph with labels and features on the model device.
        num_target_nodes : int
            Number of rows in the target memory bank.

        Returns
        -------
        PALABase
            Initialized PALA network.
        """
        model = PALABase(source_data, num_target_nodes, self.hid_dim, self.dropout)
        model.memory.beta = self.beta
        return model

    def forward_model(self, source_data, target_data, sample_queue, epoch):
        """
        Compute the training loss and update the target memory.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Prepared source graph with labels.
        target_data : torch_geometric.data.Data
            Prepared target graph.
        sample_queue : list of int
            Source node indices available for contrastive sampling.
        epoch : int
            Zero-based epoch index. ELR contributes to the loss after epoch 50.

        Returns
        -------
        torch.Tensor
            Scalar training loss. Its components are stored in ``losses_``.
        """
        
        loss, self.losses_ = self.model(source_data, target_data, sample_queue, epoch,
                                       gamma=self.gamma, epsilon=self.epsilon, anchor_num=self.anchor_num)
        return loss

    def fit(self, source_data, target_data, epoch_callback=None):
        """
        Fit the model using source labels and unlabeled target nodes.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Prepared long-tail source graph with labels and cached PPMI weights.
        target_data : torch_geometric.data.Data
            Target graph with cached PPMI weights. Its labels are ignored.
        epoch_callback : callable or None, optional
            Receives the zero-based epoch index, post-update target logits and
            loss record. Default: ``None``.

        Returns
        -------
        PALA
            Fitted model.

        Notes
        -----
        The target forward runs after every optimizer step to preserve the
        authors' dropout and random-number sequence. Callbacks should avoid
        extra stochastic forwards or other changes to random state.
        """
        
        source = self.process_graph(source_data, include_labels=True)
        labels = source.y
        

        target = self.process_graph(target_data, include_labels=False)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.device(self.device).type == 'cuda':
                torch.cuda.manual_seed(self.seed)
        self.model = self.init_model(source, target.num_nodes)
        self.model.set_cache('source', source.ppmi_edge_index, source.ppmi_edge_weight, source.num_nodes)
        self.model.set_cache('target', target.ppmi_edge_index, target.ppmi_edge_weight, target.num_nodes)
        
        
        # Anchor sampling reads this adjacency after callbacks can access the inputs.
        source.global_adj = source.ppmi_edge_index.clone()
        self.optimizer_ = torch.optim.Adam(list(self.model.encoder_f.parameters()) + list(self.model.encoder_g.parameters()),
                                            lr=self.lr, weight_decay=self.weight_decay)
        self.history_, self.sample_queue_ = [], []
        started = time.time()
        for epoch in range(self.epoch):
            self.sample_queue_ = self.model.update_queue(source, self.sample_queue_, self.gamma)
            loss = self.forward_model(source, target, self.sample_queue_, epoch)
            
            self.optimizer_.zero_grad()
            loss.backward()
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            
            self.optimizer_.step()
            # Keep the original evaluation forward and its RNG consumption.
            with torch.no_grad():
                self.last_target_logits_ = self.model.predict_logits(target)
            record = dict(epoch=epoch + 1, loss=float(loss.detach()), elr_active=epoch > 50,
                          **{key: float(value.detach()) for key, value in self.losses_.items()})
            self.history_.append(record)
            if epoch_callback is not None:
                epoch_callback(epoch, self.last_target_logits_.clone(), dict(record))
            logger(epoch=epoch, loss=record['loss'], time=time.time() - started, verbose=self.verbose)
        return self

    def predict(self, data, return_labels=True):
        """
        Predict node logits for a graph with a prepared PPMI cache.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph to predict, which may differ from the fitted target graph.
        return_labels : bool, optional
            Retrieve labels from the input if present. Default: ``True``.

        Returns
        -------
        logits : torch.Tensor
            Raw class scores, with one row per input node.
        labels : torch.Tensor or None
            Input reference labels, if requested and present.

        Notes
        -----
        Prediction disables source compensation. Dropout stays active as in
        the authors' implementation, so repeated calls can differ.
        """
        
        graph = self.process_graph(data, include_labels=False)
        self.model.set_cache('prediction', graph.ppmi_edge_index, graph.ppmi_edge_weight, graph.num_nodes)
        with torch.no_grad():
            logits = self.model.predict_logits(graph, cache_name='prediction')
        labels = getattr(data, 'y', None) if return_labels else None
        if labels is not None:
            labels = labels.to(self.device)
        return logits, labels
