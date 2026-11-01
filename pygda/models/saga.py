"""PyGDA training interface for the public SAGA implementation."""
import math
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch_geometric.data import Data

from .base import BaseGDA
from ..datasets.saga import prepare_saga_views, validate_saga_graph, validate_saga_pair
from ..nn.saga_base import SAGABase
from ..utils import logger


class SAGA(BaseGDA):
    """Multi-view node adaptation following the public SAGA implementation.

    Parameters
    ----------
    in_dim, hid_dim, num_classes : int
        Input features, hidden width and shared class count.
    embed_dim : int, optional
        Embedding width for each view. Default: 128.
    num_views : int, optional
        Number of views in each domain. Default: 2.
    k_hop : int, optional
        Fixed propagation depth from the author configuration. Default: 2.
    filter_alpha : float, optional
        Propagation teleport probability. Default: 0.
    alpha, beta, delta : float, optional
        Intra/cross alignment mixing, source CE and target entropy weights.
        Defaults: 0.95, 1.0 and 0.01, from the public ACM1-ACM2 configuration.
    tau : float, optional
        Cross-alignment temperature. Default: 1.
    num_layers : int, optional
        Fixed two-layer encoder/decoder. Must be 2.
    dropout : float, optional
        Encoder/decoder dropout probability. Default: 0.
    act : callable, optional
        Fixed ELU activation; must be torch.nn.functional.elu.
    lr, weight_decay : float, optional
        Adam parameters. Defaults: 0.001 and 0.0001.
    epoch : int, optional
        Number of training epochs. Default: 50.
    batch_size : int, optional
        Nodes per domain per step. Default: 1000. Zero means one full-graph
        step; unequal domains continue with source-only or target-only steps.
    num_neigh : int, optional
        Must be -1; propagation uses all edges before node batching.
    seed : int or None, optional
        Reset Python, NumPy and Torch before network initialization when set.
        Default: None (use the caller's random state).
    device : str, optional
        PyTorch device. Default: 'cuda:0'.
    verbose : int, optional
        PyGDA logging level from 0 to 3. Default: 2.

    Notes
    -----
    Target labels are not read in fit. Predictions use the supplied graph.
    View order is significant. The author core loss and fixed-hop behavior
    are preserved; the paper's additional dynamic hop search is not added.
    Paired batches with a singleton domain are rejected before training,
    because the author's projection BatchNorm cannot train on one node.
    """

    def __init__(self, in_dim, hid_dim, num_classes, embed_dim=128, num_views=2,
                 k_hop=2, filter_alpha=0.0, alpha=0.95, beta=1.0, delta=0.01,
                 tau=1.0, num_layers=2, dropout=0.0, act=F.elu,
                 lr=0.001, weight_decay=0.0001, epoch=50, batch_size=1000,
                 num_neigh=-1, seed=None, device='cuda:0', verbose=2, **kwargs):
        if kwargs:
            raise TypeError('Unsupported SAGA arguments: ' + ', '.join(sorted(kwargs)))
        positive = (in_dim, hid_dim, num_classes, embed_dim, num_views, k_hop, epoch)
        if any(type(n) is not int or n < 1 for n in positive):
            raise ValueError('Dimensions, views, propagation depth and epochs must be positive integers.')
        if num_layers != 2 or act is not F.elu or num_neigh != -1:
            raise ValueError('The public SAGA backbone requires num_layers=2, act=F.elu and num_neigh=-1.')
        if type(batch_size) is not int or batch_size < 0 or batch_size == 1:
            raise ValueError('batch_size must be zero or an integer of at least two.')
        values = (filter_alpha, alpha, beta, delta, tau, dropout, lr, weight_decay)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('All numeric parameters must be finite.')
        if (not 0 <= filter_alpha <= 1 or not 0 <= alpha <= 1 or
                min(beta, delta, weight_decay) < 0 or min(tau, lr) <= 0 or not 0 <= dropout < 1):
            raise ValueError('Invalid propagation, loss, dropout or optimizer parameters.')
        if type(verbose) is not int or not 0 <= verbose <= 3:
            raise ValueError('verbose must be an integer in [0, 3].')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError('seed must be None or an integer in [0, 2**32).')
        super().__init__(in_dim=in_dim, hid_dim=hid_dim, num_classes=num_classes,
                         num_layers=num_layers, dropout=dropout, act=act, lr=lr,
                         weight_decay=weight_decay, epoch=epoch, device=device,
                         batch_size=batch_size, num_neigh=num_neigh, verbose=verbose)
        self.embed_dim, self.num_views = embed_dim, num_views
        self.k_hop, self.filter_alpha = k_hop, filter_alpha
        self.alpha, self.beta, self.delta, self.tau = alpha, beta, delta, tau
        self.seed = seed
        self.history_, self.batch_history_ = [], []
        self.view_names_ = None

    def init_model(self, **kwargs):
        if kwargs:
            raise TypeError('SAGA init_model takes no additional arguments.')
        config = SimpleNamespace(loss_weights=dict(alpha=self.alpha, beta=self.beta, delta=self.delta))
        return SAGABase(self.in_dim, self.num_views, self.hid_dim, self.embed_dim,
                        self.tau, self.dropout, 0, 0, self.device,
                        self.num_classes, config).to(self.device)

    def process_graph(self, data, include_labels=False):
        """Clone the multi-view attributes; only source processing reads y."""
        views = validate_saga_graph(data)
        if len(views) != self.num_views or data.x.size(1) != self.in_dim:
            raise ValueError('Graph dimensions or view count differ from the model.')
        graph = Data(x=data.x.float().clone(), view_names=views,
                     edge_index_dict={v: data.edge_index_dict[v].clone() for v in views},
                     edge_weight_dict={v: data.edge_weight_dict[v].clone() for v in views})
        if getattr(data, 'saga_num_classes', None) is not None:
            graph.saga_num_classes = data.saga_num_classes
            if graph.saga_num_classes != self.num_classes:
                raise ValueError('Dataset and model class counts differ.')
        if include_labels:
            y = data.y
            if (y is None or y.dtype != torch.long or y.ndim != 1 or
                    y.numel() != data.x.size(0) or y.min() < 0 or y.max() >= self.num_classes):
                raise ValueError('Source y must be a long vector of valid class indices.')
            graph.y = y.clone()
        if not bool(torch.isfinite(graph.x).all()):
            raise ValueError('Features overflow float32, the author feature dtype.')
        return graph.to(self.device)

    def forward_model(self, source_data, target_data):
        """Return scalar loss for prepared node batches (either may be None).

        Each present batch contains x and saga_features; source also contains
        integer y. fit prepares these batches after whole-graph propagation.
        """
        if self.model is None:
            raise RuntimeError('Initialize or fit SAGA before forwarding.')
        if source_data is None and target_data is None:
            raise ValueError('At least one domain batch is required.')
        source_x = source_data.x if source_data is not None else None
        target_x = target_data.x if target_data is not None else None
        source_features = source_data.saga_features if source_data is not None else None
        target_features = target_data.saga_features if target_data is not None else None
        labels = F.one_hot(source_data.y, self.num_classes) if source_data is not None else None
        return self.model(target_x, target_features, source_x, source_features, labels, False)

    def fit(self, source_data, target_data):
        """Train for fixed epochs without target evaluation, files or checkpoints."""
        source = self.process_graph(source_data, include_labels=True)
        target = self.process_graph(target_data, include_labels=False)
        views = validate_saga_pair(source, target)
        ns, nt = source.x.size(0), target.x.size(0)
        size = self.batch_size or max(ns, nt)
        steps = max(math.ceil(ns / size), math.ceil(nt / size))
        for start in range(0, max(ns, nt), size):
            source_n, target_n = min(size, max(ns - start, 0)), min(size, max(nt - start, 0))
            if source_n and target_n and min(source_n, target_n) == 1:
                raise ValueError('A paired singleton batch would fail author BatchNorm; choose a compatible batch_size.')
        _, fs = prepare_saga_views(source, self.k_hop, self.filter_alpha, self.device)
        _, ft = prepare_saga_views(target, self.k_hop, self.filter_alpha, self.device)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.device(self.device).type == 'cuda':
                torch.cuda.manual_seed(self.seed)
        self.model = self.init_model()
        self.view_names_ = views
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)
        self.history_, self.batch_history_ = [], []
        started = time.time()
        for epoch in range(self.epoch):
            self.model.train()
            # Match the author's RandomSampler construction and RNG consumption.
            order_s = torch.tensor(list(RandomSampler(range(ns), replacement=False)))
            order_t = torch.tensor(list(RandomSampler(range(nt), replacement=False)))
            total = 0.0
            for batch_index in range(steps):
                start, end = batch_index * size, (batch_index + 1) * size
                index_s, index_t = order_s[start:end], order_t[start:end]
                src = Data(x=source.x[index_s], saga_features=[f[index_s] for f in fs],
                           y=source.y[index_s]) if index_s.numel() else None
                tgt = Data(x=target.x[index_t], saga_features=[f[index_t] for f in ft]) if index_t.numel() else None
                optimizer.zero_grad()
                loss = self.forward_model(src, tgt)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite SAGA training loss.')
                loss.backward()
                optimizer.step()
                value = float(loss.detach())
                total += value
                self.batch_history_.append(dict(epoch=epoch + 1, step=batch_index + 1,
                                               source_nodes=index_s.numel(), target_nodes=index_t.numel(), loss=value))
            self.history_.append(dict(epoch=epoch + 1, loss=total, steps=steps))
            logger(epoch=epoch, loss=total, time=time.time() - started, verbose=self.verbose, train=True)
        return self

    def predict(self, data, source=False):
        """Return logits and optional labels for the graph supplied by the caller."""
        if self.model is None or self.view_names_ is None:
            raise RuntimeError('Call fit before predict.')
        graph = self.process_graph(data, include_labels=False)
        if tuple(graph.view_names) != self.view_names_:
            raise ValueError('Prediction view order differs from training.')
        _, features = prepare_saga_views(graph, self.k_hop, self.filter_alpha, self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.classifier(torch.cat(self.model.encoder(features), dim=1))
        labels = data.y.to(self.device) if data.y is not None else None
        return logits, labels
