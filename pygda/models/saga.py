import math
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from .base import BaseGDA
from ..nn.saga_base import SAGABase
from ..utils import logger


class SAGA(BaseGDA):
    """
    SAGA: Structural Aggregation Guided Alignment with Dynamic View and
    Neighborhood Order Selection for Multiview Graph Domain Adaptation (ICLR 2026).

    Parameters
    ----------
    in_dim : int
        Number of node features.
    hid_dim : int
        Hidden width of the encoder and decoder.
    num_classes : int
        Number of classes shared by the training domains.
    embed_dim : int, optional
        Embedding width per view. Default: ``128``.
    num_views : int, optional
        Number of ordered views per graph. Default: ``2``.
    k_hop : int, optional
        Fixed number of propagation steps. Default: ``2``.
    filter_alpha : float, optional
        Input feature weight in the propagation recurrence. Default: ``0.0``.
    alpha : float, optional
        Intra-alignment weight. Cross-alignment uses 1 - alpha. Default: ``0.95``.
    beta : float, optional
        Source classification loss weight. Default: ``1.0``.
    delta : float, optional
        Target prediction entropy weight. Default: ``0.01``.
    tau : float, optional
        Cross-alignment temperature. Default: ``1.0``.
    num_layers : int, optional
        Fixed encoder and decoder depth. Default: ``2``.
    dropout : float, optional
        Dropout probability in the encoder and decoder. Default: ``0.0``.
    act : callable, optional
        Fixed activation of the author backbone. Default: ``torch.nn.functional.elu``.
    lr : float, optional
        Adam learning rate. Default: ``0.001``.
    weight_decay : float, optional
        Adam weight decay. Default: ``0.0001``.
    epoch : int, optional
        Number of training epochs. Default: ``50``.
    batch_size : int, optional
        Nodes per domain per batch. Zero uses a full batch. Default: ``1000``.
    num_neigh : int, optional
        Neighbor count. Only -1 is supported because propagation uses all
        edges before batching. Default: ``-1``.
    seed : int or None, optional
        Seed applied before network initialization. None uses the caller's
        random state. Default: ``None``.
    device : str, optional
        PyTorch device. Default: ``cuda:0``.
    verbose : int, optional
        PyGDA logging level from 0 to 3. Default: ``2``.
    **kwargs : dict
        Additional arguments are accepted by the signature and ignored.

    Notes
    -----
    Each PyG Data graph supplies x, ordered view_names, edge_index_dict and
    edge_weight_dict. Each view uses PyG's source-to-destination edge order.
    Training requires integer source labels and uses all supplied nodes.
    The implementation follows the public code's fixed propagation depth
    at commit 55ff00a4e426a2d0fbeb913f11330ab69b68a506 in f1shungry/SAGA.
    Each view has its self-loops replaced by unit loops before symmetric
    normalization, propagation and row L2 normalization.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        embed_dim=128,
        num_views=2,
        k_hop=2,
        filter_alpha=0.0,
        alpha=0.95,
        beta=1.0,
        delta=0.01,
        tau=1.0,
        num_layers=2,
        dropout=0.0,
        act=F.elu,
        lr=0.001,
        weight_decay=0.0001,
        epoch=50,
        batch_size=1000,
        num_neigh=-1,
        seed=None,
        device='cuda:0',
        verbose=2,
        **kwargs,
    ):
        super().__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            lr=lr,
            weight_decay=weight_decay,
            epoch=epoch,
            device=device,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
        )
        self.embed_dim, self.num_views = embed_dim, num_views
        self.k_hop, self.filter_alpha = k_hop, filter_alpha
        self.alpha, self.beta, self.delta, self.tau = alpha, beta, delta, tau
        self.seed = seed
        self.history_, self.batch_history_ = [], []
        self.view_names_ = None

    def init_model(self, **kwargs):
        """
        Initialize the author backbone on the configured device.

        Parameters
        ----------
        **kwargs : dict
            Additional initialization arguments. Currently unused.

        Returns
        -------
        SAGABase
            A fresh encoder, decoder, projection heads and classifier.
        """

        config = SimpleNamespace(loss_weights=dict(alpha=self.alpha, beta=self.beta, delta=self.delta))
        return SAGABase(self.in_dim, self.num_views, self.hid_dim, self.embed_dim,
                        self.tau, self.dropout, 0, 0, self.device,
                        self.num_classes, config).to(self.device)

    def process_graph(self, data, include_labels=False):
        """
        Prepare the ordered views and node features for propagation.

        Parameters
        ----------
        data : torch_geometric.data.Data
            A graph with one edge index and weight vector per named view.
        include_labels : bool, optional
            Carry input labels into the prepared graph. Default: ``False``.

        Returns
        -------
        torch_geometric.data.Data
            Prepared graph on the model device. The input graph is read-only.
        """

        nodes, width = data.x.shape

        x = data.x.to(device=self.device, dtype=torch.float32)

        views = tuple(getattr(data, 'view_names', ()))

        indices = getattr(data, 'edge_index_dict', None)
        weights = getattr(data, 'edge_weight_dict', None)

        graph = Data(x=x, view_names=views,
                     edge_index_dict={v: indices[v].to(self.device) for v in views},
                     edge_weight_dict={v: weights[v].to(self.device) for v in views})
        graph.saga_num_classes = getattr(data, 'saga_num_classes', None)
        if include_labels and data.y is not None:
            graph.y = data.y.to(self.device)
        return graph

    @torch.no_grad()
    def _propagate_views(self, graph):
        """
        Propagate node features in each view.

        Parameters
        ----------
        graph : torch_geometric.data.Data
            Graph prepared by ``process_graph``.

        Returns
        -------
        list of torch.Tensor
            One row-normalized feature matrix per view, in the declared order.
        """
        x = graph.x
        nodes = x.size(0)
        size = (nodes, nodes)
        loop = torch.arange(nodes, device=x.device)
        loops = torch.stack((loop, loop))
        unit = torch.ones(nodes, device=x.device)
        features = []
        for view in graph.view_names:
            # Restore the author matrix convention before coalescing duplicates.
            matrix = torch.sparse_coo_tensor(
                graph.edge_index_dict[view].flip(0), graph.edge_weight_dict[view], size
            ).coalesce()
            index, values = remove_self_loops(matrix.indices(), matrix.values())
            matrix = torch.sparse_coo_tensor(
                torch.cat((loops, index), dim=1), torch.cat((unit, values)), size
            ).coalesce()
            index = matrix.indices()
            inv = 1.0 / torch.sqrt(torch.sparse.sum(matrix, dim=1).values())
            weights = matrix.values() * inv[index[0]] * inv[index[1]]
            matrix = torch.sparse_coo_tensor(index, weights, size).coalesce().float()
            z = x
            for _ in range(self.k_hop):
                z = torch.sparse.mm(matrix, z)
                z = (1 - self.filter_alpha) * z + self.filter_alpha * x
            features.append(F.normalize(z, dim=1, p=2))
        return features

    def forward_model(self, source_data, target_data):
        """
        Compute the scalar training loss for a prepared batch.

        Parameters
        ----------
        source_data : torch_geometric.data.Data or None
            Source nodes with raw features, propagated views and integer labels.
        target_data : torch_geometric.data.Data or None
            Target nodes with raw features and propagated views.

        Returns
        -------
        torch.Tensor
            Scalar loss for the available domains. Alignment requires both.
        """

        source_x = source_data.x if source_data is not None else None
        target_x = target_data.x if target_data is not None else None
        source_features = source_data.saga_features if source_data is not None else None
        target_features = target_data.saga_features if target_data is not None else None
        labels = F.one_hot(source_data.y, self.num_classes) if source_data is not None else None
        return self.model(target_x, target_features, source_x, source_features, labels, False)

    def fit(self, source_data, target_data):
        """
        Train on all supplied nodes for the configured number of epochs.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source graph with one integer class label per node.
        target_data : torch_geometric.data.Data
            Target graph with matching feature width and ordered views.

        Returns
        -------
        SAGA
            The fitted estimator with epoch and batch loss histories.

        Notes
        -----
        Target labels are excluded from training and evaluation is external.
        Batches from the longer domain continue after the other domain is exhausted.
        Batches that pair a singleton domain with another domain are invalid
        because the projection heads use BatchNorm.
        """
        source = self.process_graph(source_data, include_labels=True)
        target = self.process_graph(target_data, include_labels=False)
        views = source.view_names

        for graph in (source, target):
            classes = getattr(graph, 'saga_num_classes', None)

        y = source.y

        ns, nt = source.x.size(0), target.x.size(0)
        size = self.batch_size or max(ns, nt)
        steps = max(math.ceil(ns / size), math.ceil(nt / size))
        for start in range(0, max(ns, nt), size):
            source_n, target_n = min(size, max(ns - start, 0)), min(size, max(nt - start, 0))

        fs = self._propagate_views(source)
        ft = self._propagate_views(target)
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
        """
        Predict node logits for the graph supplied by the caller.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with the fitted model's ordered views and optional labels.
        source : bool, optional
            Compatibility argument. Both domains use the same classifier. Default: ``False``.

        Returns
        -------
        torch.Tensor
            Per-node class logits.
        torch.Tensor or None
            Input labels on the model device, when supplied.
        """

        graph = self.process_graph(data, include_labels=False)

        features = self._propagate_views(graph)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.classifier(torch.cat(self.model.encoder(features), dim=1))
        labels = data.y
        if labels is not None:
            labels = labels.to(self.device)
        return logits, labels
