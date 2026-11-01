"""PyGDA interface for the released ImGDA full-graph implementation."""
import math
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .base import BaseGDA
from ..datasets.imgda import validate_imgda_graph
from ..nn.imgda_base import ImGDAEncoder, imgda_objective
from ..utils import logger


class ImGDA(BaseGDA):
    """Class-imbalanced node adaptation following the public ImGDA code.

    Inputs require explicit global_adj and normalized global_norm tensors.
    The prepared source also needs original imgda_class_counts. Use
    load_imgda_pair for the released processed graphs and cache layout.

    The fixed two-layer ReLU encoder and three losses preserve the public
    implementation. Ordinary-list Dropout remains active during evaluation;
    predictions are stochastic. fit performs one unlabeled target evaluation
    per epoch to preserve the author's RNG consumption, and retains final
    weights. An optional callback receives (epoch, losses, target_logits) for
    external evaluation. fit never reads target labels or selects a checkpoint.

    Parameters follow BaseGDA; defaults mirror the author: hid_dim=512,
    dropout=0.1, lr=0.01, weight_decay=0.001, epoch=200, k=200 and
    dynamic_temperature=0.5. Only full-graph node adaptation is supported.
    seed=None retains the caller's RNG; a nonnegative seed resets initialization.
    """

    def __init__(self, in_dim, hid_dim, num_classes, num_layers=2, dropout=0.1,
                 act=F.relu, lr=0.01, weight_decay=0.001, epoch=200,
                 device='cuda:0', batch_size=0, num_neigh=-1, verbose=2,
                 k=200, dynamic_temperature=0.5, seed=None, **kwargs):
        if kwargs:
            raise TypeError('Unsupported ImGDA arguments: '+', '.join(sorted(kwargs)))
        if any(type(n) is not int or n < 1 for n in (in_dim, hid_dim, epoch, k)) or type(num_classes) is not int or num_classes < 2:
            raise ValueError('Dimensions, epoch and k must be positive integers; num_classes must be at least two.')
        if num_layers != 2 or act is not F.relu or batch_size != 0 or num_neigh != -1:
            raise ValueError('ImGDA requires two ReLU layers, batch_size=0 and num_neigh=-1.')
        if not all(math.isfinite(v) for v in (dropout, lr, weight_decay, dynamic_temperature)):
            raise ValueError('Numeric parameters must be finite.')
        if not 0 <= dropout < 1 or lr <= 0 or weight_decay < 0 or not 0 < dynamic_temperature <= 1:
            raise ValueError('Invalid dropout, optimizer or temperature parameters.')
        if type(verbose) is not int or not 0 <= verbose <= 3:
            raise ValueError('verbose must be an integer in [0, 3].')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError('seed must be None or an integer in [0, 2**32).')
        super().__init__(in_dim=in_dim, hid_dim=hid_dim, num_classes=num_classes,
                         num_layers=num_layers, dropout=dropout, act=act, lr=lr,
                         weight_decay=weight_decay, epoch=epoch, device=device,
                         batch_size=batch_size, num_neigh=num_neigh, verbose=verbose)
        self.k, self.dynamic_temperature, self.seed = k, dynamic_temperature, seed
        self.history_, self.losses_, self.last_logits_ = [], None, None

    def process_graph(self, data, source=False):
        validate_imgda_graph(data, source=source, num_classes=self.num_classes if source else None)
        if data.x.size(1) != self.in_dim:
            raise ValueError('Graph feature width differs from in_dim.')
        graph = Data(x=data.x.to(self.device).clone(), edge_index=data.edge_index.to(self.device).clone(),
                     global_adj=data.global_adj.to(self.device).clone(), global_norm=data.global_norm.to(self.device).clone())
        if source:
            graph.y = data.y.to(self.device).clone()
            graph.imgda_class_counts = data.imgda_class_counts.to(self.device).clone()
        return graph

    def init_model(self, source_data=None, **kwargs):
        if kwargs or source_data is None:
            raise ValueError('init_model requires the prepared source graph and no extra arguments.')
        encoder = ImGDAEncoder(source_data,self.hid_dim,self.dropout,self.in_dim,self.device).to(self.device)
        classifier = nn.Sequential(nn.Linear(self.hid_dim,self.num_classes)).to(self.device)
        return nn.ModuleDict(dict(encoder=encoder,classifier=classifier))

    def _cache(self, name, graph):
        for conv in self.model['encoder'].conv_layers:
            conv.set_global_cache(name,graph.global_adj,graph.global_norm)

    def forward_model(self, source_data, target_data, **kwargs):
        """Return the full-graph objective for a prepared pair; no optimizer step."""
        if kwargs or self.model is None:
            raise RuntimeError('Initialize the model before forward_model; no extra arguments are supported.')
        self._cache('source',source_data)
        self._cache('target',target_data)
        self.losses_ = imgda_objective(self.model['encoder'],self.model['classifier'],
            source_data,target_data,source_data.imgda_class_counts,'source','target',self.k,self.dynamic_temperature)
        return self.losses_['loss']

    def _target_logits(self, graph, cache_name):
        self._cache(cache_name,graph)
        encoded = self.model['encoder'](graph.x,graph.edge_index,cache_name,False,self.device,use_ppmi=True)
        return self.model['classifier'](encoded)

    def fit(self, source_data, target_data, callback=None):
        """Fit fixed epochs, with an optional external evaluation callback."""
        if callback is not None and not callable(callback):
            raise TypeError('callback must be callable or None.')
        source = self.process_graph(source_data,source=True)
        target = self.process_graph(target_data)
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.device(self.device).type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(self.seed)
        self.model = self.init_model(source)
        self.optimizer_ = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        self.history_, self.last_logits_ = [], None
        started = time.time()
        for epoch in range(self.epoch):
            self.model.train()
            loss = self.forward_model(source,target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite ImGDA training loss.')
            self.optimizer_.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
                raise FloatingPointError('Nonfinite ImGDA gradient.')
            self.optimizer_.step()
            values = {k:float(self.losses_[k].detach()) for k in ('loss','cb_cL_loss','cd_cl_loss','lc_loss')}
            self.history_.append(dict(epoch=epoch,**values))
            self.model.eval()
            with torch.no_grad():
                self.last_logits_ = self._target_logits(target,'target').detach()
            if not torch.isfinite(self.last_logits_).all():
                raise FloatingPointError('Nonfinite ImGDA target logits.')
            if callback is not None:
                callback(epoch,dict(values),self.last_logits_.clone())
            logger(epoch=epoch,loss=values['loss'],time=time.time()-started,verbose=self.verbose,train=True)
            # Keep scalar history; avoid retaining the final training graph.
            self.losses_ = values
        return self

    def predict(self, data):
        """Return global-branch logits and optional labels for the supplied graph.

        Preserves released stochastic Dropout evaluation. Explicitly reseed
        the caller's RNG when replaying a prediction for numerical comparison.
        """
        if self.model is None or self.last_logits_ is None:
            raise RuntimeError('Call fit before predict.')
        graph = self.process_graph(data)
        self.model.eval()
        with torch.no_grad():
            logits = self._target_logits(graph,'prediction')
        labels = data.y.to(self.device) if data.y is not None else None
        return logits,labels
