"""Full-batch PyGDA interface for the public PALA algorithm."""
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .base import BaseGDA
from ..datasets.pala import validate_pala_graph
from ..nn.pala_base import PALABase
from ..utils import logger


class PALA(BaseGDA):
    """Class-imbalanced node adaptation with explicit author PPMI caches.

    Prepare the source long-tail graph with prepare_pala_source before fit.
    Both inputs carry ppmi_edge_index and ppmi_edge_weight in their node order.
    Only source labels are read during fit. Each epoch performs one target
    forward after the optimizer step to preserve the author's Dropout RNG
    schedule; an optional callback receives those existing logits.

    The contrastive loss uses a documented algebraically equivalent log-domain
    reduction. The author's eval-time Dropout remains active, so predictions
    are stochastic. Only full-batch two-layer ReLU training is supported.
    """

    def __init__(self, in_dim, hid_dim, num_classes, num_layers=2, dropout=0.1,
                 act=F.relu, lr=0.01, weight_decay=0.001, epoch=200,
                 beta=0.9, epsilon=0.2, gamma=5, anchor_num=200,
                 device='cuda:0', batch_size=0, num_neigh=-1, seed=None, verbose=2, **kwargs):
        if kwargs:
            raise TypeError('Unsupported PALA arguments: ' + ', '.join(sorted(kwargs)))
        if any(type(value) is not int or value < 1 for value in
               [in_dim, hid_dim, num_classes, epoch, gamma, anchor_num]) or num_classes < 2:
            raise ValueError('Dimensions, epochs, gamma and anchor_num must be positive integers; at least two classes are needed.')
        if num_layers != 2 or act is not F.relu or batch_size != 0 or num_neigh != -1:
            raise ValueError('PALA supports num_layers=2, ReLU, batch_size=0 and num_neigh=-1.')
        if not all(math.isfinite(value) for value in [dropout, lr, weight_decay, beta, epsilon]):
            raise ValueError('Numeric hyperparameters must be finite.')
        if not (0 <= dropout < 1 and lr > 0 and weight_decay >= 0 and 0 <= beta < 1 and 0 <= epsilon <= 1):
            raise ValueError('Invalid PALA optimizer, dropout, beta or epsilon parameter.')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError('seed must be None or an integer in [0, 2**32).')
        if type(verbose) is not int or not 0 <= verbose <= 3:
            raise ValueError('verbose must be an integer in [0,3].')
        super().__init__(in_dim, hid_dim, num_classes, num_layers=num_layers, dropout=dropout,
                         act=act, lr=lr, weight_decay=weight_decay, epoch=epoch, device=device,
                         batch_size=batch_size, num_neigh=num_neigh, verbose=verbose)
        self.beta, self.epsilon, self.gamma, self.anchor_num = beta, epsilon, gamma, anchor_num
        self.seed = seed
        self.history_ = []
        self.optimizer_ = None
        self.last_target_logits_ = None
        self.sample_queue_ = []

    def process_graph(self, data, include_labels=False):
        """Clone prepared graph inputs; include_labels is true only for source."""
        validate_pala_graph(data, self.in_dim)
        graph = Data(x=data.x.detach().to(dtype=torch.float32).clone(),
                     edge_index=data.edge_index.detach().clone(),
                     ppmi_edge_index=data.ppmi_edge_index.detach().clone(),
                     ppmi_edge_weight=data.ppmi_edge_weight.detach().clone())
        if not bool(torch.isfinite(graph.x).all()):
            raise ValueError('Features overflow the author float32 dtype.')
        graph.global_adj = graph.ppmi_edge_index
        if include_labels:
            labels = data.y
            if (labels is None or labels.dtype != torch.long or labels.ndim != 1 or
                    labels.numel() != graph.num_nodes or int(labels.min()) < 0 or
                    int(labels.max()) >= self.num_classes):
                raise ValueError('Source labels must be valid long class indices [N].')
            if int(torch.bincount(labels, minlength=self.num_classes).min()) < self.gamma * 2:
                raise ValueError('Each prepared source class needs at least 2*gamma nodes for the author queue sampler.')
            graph.y = labels.detach().clone()
        return graph.to(self.device)

    def init_model(self, source_data, num_target_nodes):
        """Initialize both encoders and memory using the prepared source."""
        model = PALABase(source_data, num_target_nodes, self.hid_dim, self.dropout)
        model.memory.beta = self.beta
        return model

    def forward_model(self, source_data, target_data, sample_queue, epoch):
        """Return scalar training loss; component tensors are available in losses_."""
        if self.model is None:
            raise RuntimeError('Initialize or fit PALA first.')
        loss, self.losses_ = self.model(source_data, target_data, sample_queue, epoch,
                                       gamma=self.gamma, epsilon=self.epsilon, anchor_num=self.anchor_num)
        return loss

    def fit(self, source_data, target_data, epoch_callback=None):
        """Train without target labels; callback(epoch, logits, record) is optional.

        Callback logits come from the existing post-update target forward.
        Callbacks should not perform extra stochastic forwards or alter RNG.
        Epoch indices passed to the callback are zero-based.
        """
        if epoch_callback is not None and not callable(epoch_callback):
            raise TypeError('epoch_callback must be callable or None.')
        source = self.process_graph(source_data, include_labels=True)
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
        self.optimizer_ = torch.optim.Adam(list(self.model.encoder_f.parameters()) + list(self.model.encoder_g.parameters()),
                                            lr=self.lr, weight_decay=self.weight_decay)
        self.history_, self.sample_queue_ = [], []
        started = time.time()
        for epoch in range(self.epoch):
            self.sample_queue_ = self.model.update_queue(source, self.sample_queue_, self.gamma)
            loss = self.forward_model(source, target, self.sample_queue_, epoch)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError('Non-finite PALA training loss.')
            self.optimizer_.zero_grad()
            loss.backward()
            gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError('Missing or non-finite PALA gradients.')
            self.optimizer_.step()
            # Keep the original evaluation forward and its RNG consumption.
            self.last_target_logits_ = self.model.predict_logits(target).detach()
            record = dict(epoch=epoch + 1, loss=float(loss.detach()), elr_active=epoch > 50,
                          **{key: float(value.detach()) for key, value in self.losses_.items()})
            self.history_.append(record)
            if epoch_callback is not None:
                epoch_callback(epoch, self.last_target_logits_.clone(), dict(record))
            logger(epoch=epoch, loss=record['loss'], time=time.time() - started, verbose=self.verbose)
        return self

    def predict(self, data, return_labels=True):
        """Return logits and optional labels for the supplied graph/cache.

        This call preserves the author's stochastic eval-time Dropout. Use
        return_labels=False to avoid retrieving any y attribute at inference.
        Source compensation is disabled, as in the author's predict function.
        """
        if self.model is None or self.last_target_logits_ is None:
            raise RuntimeError('Call fit before predict.')
        graph = self.process_graph(data, include_labels=False)
        self.model.set_cache('prediction', graph.ppmi_edge_index, graph.ppmi_edge_weight, graph.num_nodes)
        with torch.no_grad():
            logits = self.model.predict_logits(graph, cache_name='prediction')
        labels = getattr(data, 'y', None) if return_labels else None
        if labels is not None:
            labels = labels.to(self.device)
        return logits, labels
