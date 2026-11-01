"""ADAlign integration based on gxingyu/ADAlign, commit d6d7a700."""

import math
import time

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .base import BaseGDA
from ..nn.adalign_base import ADAlignBase, CF, SampleNet
from ..utils import logger


class ADAlign(BaseGDA):
    """Learning Adaptive Distribution Alignment with Neural Characteristic
    Function for Graph Domain Adaptation (ICLR 2026).

    Parameters
    ----------
    in_dim : int
        Number of input features, shared by both domains.
    hid_dim : int
        Hidden feature and sampled-frequency dimension.
    num_classes : int
        Number of source classes; labels are integers in this range.
    num_layers : int, optional
        Number of backbone layers. Default: 2.
    dropout : float, optional
        Dropout probability in [0, 1). Default: 0.25.
    act : callable, optional
        Backbone activation. Default: torch.nn.functional.relu.
    s_pnums, t_pnums : int, optional
        Source and target propagation counts. Defaults: 0 and 10.
    weight : float, optional
        Nonnegative alignment-loss coefficient (lambda). Default: 0.4.
    weight_decay : float, optional
        Nonnegative Adam weight decay. Default: 0.001.
    lr : float, optional
        Positive learning rate for both optimizers. Default: 0.001.
    epoch : int, optional
        Number of training epochs. Default: 150.
    device : str, optional
        PyTorch device. Default: 'cuda:0'.
    batch_size : int, optional
        Only full-batch training (0) is supported. Default: 0.
    num_neigh : int, optional
        Only all-neighbor sampling (-1) is supported. Default: -1.
    verbose : int, optional
        PyGDA logging level from 0 (silent) to 3. Default: 2.
    t_batchsize : int, optional
        Nominal frequency count, a multiple of 16 and at least 32.
        The sampler repeats each generated frequency 16 times. Default: 2048.
    alpha : float, optional
        Amplitude coefficient (kappa) in [0, 1]. Default: 0.5.
    mode : str, optional
        Only 'node' is supported. Default: 'node'.
    adv : bool, optional
        Must be False; domain-classifier training is unsupported.
    **kwargs
        Additional backbone arguments.

    Notes
    -----
    Training uses all source labels and does not use target labels.
    Defaults follow the author CLI configuration. The author loss has
    singular gradients at zero complex modulus and, for alpha=1, at zero
    discrepancy; these boundaries are preserved for implementation parity.
    """

    def __init__(self, in_dim, hid_dim, num_classes, num_layers=2,
                 dropout=0.25, act=F.relu, s_pnums=0, t_pnums=10,
                 weight=0.4, weight_decay=0.001, lr=0.001, epoch=150,
                 device='cuda:0', batch_size=0, num_neigh=-1, verbose=2,
                 t_batchsize=2048, alpha=0.5, mode='node', adv=False,
                 **kwargs):
        if mode != 'node' or adv:
            raise ValueError('ADAlign supports node mode with adv=False only.')
        if batch_size != 0:
            raise ValueError('ADAlign currently requires batch_size=0.')
        if num_neigh != -1:
            raise ValueError('ADAlign currently requires num_neigh=-1.')
        if not isinstance(t_batchsize, int) or t_batchsize < 32 or t_batchsize % 16:
            raise ValueError('t_batchsize must be a multiple of 16, at least 32.')
        numeric = (alpha, weight, dropout, lr, weight_decay)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('alpha, weight, dropout, lr and weight_decay must be finite.')
        if not 0 <= alpha <= 1 or weight < 0:
            raise ValueError('alpha must be in [0, 1] and weight nonnegative.')
        if not 0 <= dropout < 1 or lr <= 0 or weight_decay < 0:
            raise ValueError('Require dropout in [0, 1), lr > 0 and weight_decay >= 0.')
        if type(verbose) is not int or not 0 <= verbose <= 3:
            raise ValueError('verbose must be an integer in [0, 3].')
        if any(not isinstance(n, int) or n < 0 for n in (s_pnums, t_pnums)):
            raise ValueError('Propagation counts must be nonnegative integers.')
        if any(not isinstance(n, int) or n < 1
               for n in (in_dim, hid_dim, num_classes, num_layers, epoch)):
            raise ValueError('Dimensions, layer count and epoch must be positive integers.')
        super().__init__(in_dim=in_dim, hid_dim=hid_dim,
                         num_classes=num_classes, num_layers=num_layers,
                         dropout=dropout, act=act, weight_decay=weight_decay,
                         lr=lr, epoch=epoch, device=device, batch_size=batch_size,
                         num_neigh=num_neigh, verbose=verbose, **kwargs)
        self.s_pnums, self.t_pnums = s_pnums, t_pnums
        self.weight, self.alpha = weight, alpha
        self.t_batchsize = t_batchsize
        self.mode, self.adv = mode, adv
        self.a2gnn = None
        self.sample_net = None
        self.history_ = []

    def init_model(self, **kwargs):
        return ADAlignBase(in_dim=self.in_dim, hid_dim=self.hid_dim,
                          num_classes=self.num_classes,
                          num_layers=self.num_layers, adv=False,
                          dropout=self.dropout, act=self.act,
                          mode='node', **kwargs).to(self.device)

    def process_graph(self, data, **kwargs):
        """Validate and copy a graph; do not alter topology or features."""
        if not isinstance(data, Data):
            raise TypeError('Expected torch_geometric.data.Data.')
        if data.x is None or data.x.ndim != 2 or data.x.size(0) == 0:
            raise ValueError('A nonempty node feature matrix is required.')
        if data.x.size(1) != self.in_dim or not data.x.is_floating_point():
            raise ValueError('Features must be floating point with in_dim columns.')
        if not torch.isfinite(data.x).all():
            raise ValueError('Features must be finite.')
        if data.num_nodes != data.x.size(0):
            raise ValueError('num_nodes must match the node feature count.')
        edge = data.edge_index
        if edge is None or edge.ndim != 2 or edge.size(0) != 2 or edge.dtype != torch.long:
            raise ValueError('edge_index must be a long tensor of shape [2, E].')
        if edge.numel() and (edge.min() < 0 or edge.max() >= data.x.size(0)):
            raise ValueError('edge_index contains an invalid node index.')
        return data.clone().to(self.device)

    def forward_model(self, source_data, target_data, alpha=None):
        """Preserve the official separate classification/alignment forwards.

        The optional alpha argument is retained for official call compatibility;
        the spectral amplitude coefficient is self.alpha.
        """
        if self.a2gnn is None or self.sample_net is None:
            raise RuntimeError('Initialize or fit the model before forwarding.')
        source_logits = self.a2gnn(source_data, self.s_pnums)
        source_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        source_features = self.a2gnn.feat_bottleneck(
            source_data.x, source_data.edge_index, None, self.s_pnums)
        target_features = self.a2gnn.feat_bottleneck(
            target_data.x, target_data.edge_index, None, self.t_pnums)
        frequencies = self.sample_net(self.device)
        align_loss = CF(self.alpha, 1 - self.alpha,
                        source_features, target_features, frequencies)
        target_logits = self.a2gnn(target_data, self.t_pnums)
        return (source_loss + self.weight * align_loss,
                source_logits, target_logits, align_loss)

    def fit(self, source_data, target_data):
        """Fit for a fixed number of epochs; return self, without target evaluation."""
        source = self.process_graph(source_data)
        if not isinstance(target_data, Data):
            raise TypeError('Expected torch_geometric.data.Data for target_data.')
        # Clone without target labels before validation or loading. No target
        # labels, masks or target metrics are used to choose model parameters.
        target_unlabeled = target_data.clone()
        target_unlabeled.y = None
        target = self.process_graph(target_unlabeled)
        if (source.y is None or source.y.ndim != 1 or
                source.y.numel() != source.x.size(0) or source.y.dtype != torch.long):
            raise ValueError('Source labels must be a long vector with one label per node.')
        if source.y.min() < 0 or source.y.max() >= self.num_classes:
            raise ValueError('Source labels must be in [0, num_classes).')
        self.source_loader = NeighborLoader(source, self.num_neigh,
                                            batch_size=source.x.size(0))
        self.target_loader = NeighborLoader(target, self.num_neigh,
                                            batch_size=target.x.size(0))
        self.a2gnn = self.init_model(**self.kwargs)
        self.model = self.a2gnn
        # Match the sampler dimension to the backbone embeddings.
        self.sample_net = SampleNet(feature_dim=self.hid_dim,
                                    t_batchsize=self.t_batchsize, t_var=1).to(self.device)
        optimizer_main = torch.optim.Adam(self.a2gnn.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
        optimizer_sample = torch.optim.Adam(self.sample_net.parameters(), lr=self.lr,
                                            weight_decay=self.weight_decay)
        self.history_ = []
        start = time.time()
        for epoch in range(self.epoch):
            total = alignment = 0.0
            for src, tgt in zip(self.source_loader, self.target_loader):
                src, tgt = src.to(self.device), tgt.to(self.device)
                self.a2gnn.train()
                self.sample_net.train()
                main_parameters = list(self.a2gnn.parameters())
                flags = [p.requires_grad for p in main_parameters]
                try:
                    for p in main_parameters:
                        p.requires_grad_(False)
                    optimizer_sample.zero_grad()
                    _, _, _, cf = self.forward_model(src, tgt)
                    (-cf).backward()
                    optimizer_sample.step()
                finally:
                    for p, flag in zip(main_parameters, flags):
                        p.requires_grad_(flag)
                # Match the official second phase: sampler remains in train
                # mode, including BatchNorm buffer updates, but is not stepped.
                optimizer_main.zero_grad()
                loss, _, _, cf = self.forward_model(src, tgt)
                loss.backward()
                optimizer_main.step()
                total += loss.item()
                alignment += cf.item()
            self.history_.append({'epoch': epoch + 1, 'loss': total,
                                  'alignment_loss': alignment})
            logger(epoch=epoch, loss=total, time=time.time() - start,
                   verbose=self.verbose, train=True)
        return self

    def predict(self, data, source=False):
        """Return logits and optional labels for the graph actually supplied."""
        if self.a2gnn is None:
            raise RuntimeError('Call fit before predict.')
        graph = self.process_graph(data)
        self.a2gnn.eval()
        with torch.no_grad():
            logits = self.a2gnn(graph, self.s_pnums if source else self.t_pnums)
        return logits, graph.y
