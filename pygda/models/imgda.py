import math
import pickle
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

from .base import BaseGDA
from ..nn.imgda_base import ImGDABase, imgda_objective, make_longtailed_data_remove
from ..utils import logger


def validate_imgda_graph(data, source=False, num_classes=None):
    """
    Access graph fields and count selected source labels when requested.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input graph with node features, local edges, and optional cache fields.
    source : bool, optional
        Count nodes per source class. Default: ``False``.
    num_classes : int or None, optional
        Minimum number of bins for source label counts. Default: ``None``.
        Supply an integer when ``source=True``.

    Notes
    -----
    This helper does not explicitly validate the features, edges, cache
    weights, or class counts. The target path does not read labels.
    """
    x, edges = data.x, data.edge_index
    
    norm = getattr(data, 'global_norm', None)
    
    if source:
        y, counts = data.y, getattr(data, 'imgda_class_counts', None)
        
        selected_counts = torch.bincount(y, minlength=num_classes)
        


def prepare_imgda_source(data, global_adj, global_norm, imbalance_factor=20, device='cuda:0'):
    """
    Select a long-tailed source graph and attach its precomputed global cache.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Original source graph with node labels and a training mask.
    global_adj : torch.Tensor
        Global edge indices for the selected source nodes.
    global_norm : torch.Tensor
        Precomputed normalized weight for each global edge.
    imbalance_factor : int, optional
        Requested head-to-tail class size ratio. Default: ``20``.
    device : str or torch.device, optional
        Device used for node selection and the returned graph. Default: ``'cuda:0'``.

    Returns
    -------
    torch_geometric.data.Data
        Selected source graph with reindexed local edges, the global cache,
        original class counts, and the IDs of the retained nodes.

    Notes
    -----
    The supplied cache must match the selected and reindexed nodes.
    CPU and CUDA can resolve degree ties differently.
    """
   
    x, y, edges = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    mask = data.train_mask.to(device)
    classes = int(y.max()) + 1
    counts = torch.bincount(y, minlength=classes)
    train_counts = torch.bincount(y[mask], minlength=classes)
    
    _, selected, _, _, edge_mask = make_longtailed_data_remove(
        edges, y, train_counts.tolist(), classes, imbalance_factor, mask.clone())
    node_ids = torch.nonzero(selected).squeeze(1)
    
    # Renumber retained nodes to match the rows of the filtered features.
    mapping = torch.full((y.numel(),), -1, dtype=torch.long, device=device)
    mapping[node_ids] = torch.arange(node_ids.numel(), device=device)
    mapped = mapping[edges[:, edge_mask]]
    mapped = mapped[:, (mapped >= 0).all(dim=0)]
    
    
    graph = Data(x=x[selected].clone(), y=y[selected].clone(),
                 edge_index=mapped.clone(),
                 global_adj=global_adj.to(device).clone(), 
                 global_norm=global_norm.to(device).clone(),
                 imgda_class_counts=counts.clone(), 
                 imgda_original_node_ids=node_ids.clone())
    
    validate_imgda_graph(graph, source=True, num_classes=classes)
    return graph


def load_imgda_pair(root, source, target, imbalance_factor=20, device='cuda:0'):
    """
    Load the released source and target graphs with their global caches.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing ``data/<domain>/processed/data.pt`` and
        ``tmp/<cache>.pkl``.
    source : str
        Source domain name.
    target : str
        Target domain name. 
    imbalance_factor : int, optional
        Source imbalance ratio used to select nodes and locate the matching
        source cache. Default: ``20``.
    device : str or torch.device, optional
        Device on which to prepare and return both graphs. Default: ``'cuda:0'``.

    Returns
    -------
    tuple of torch_geometric.data.Data
        Prepared source graph and target graph without labels.

    Notes
    -----
    Only load trusted files because the released files use pickle.
    Load target labels separately for external evaluation.
    """
    root = Path(root)
    

    def read_graph(name):
        """Read one domain graph from its processed PyG cache."""
        graph, slices = torch.load(root/'data'/name/'processed/data.pt', map_location='cpu', weights_only=False)
        return graph
        
    def read_cache(name):
        """Read global edge indices and normalized weights."""
        with (root/'tmp'/(name+'.pkl')).open('rb') as stream:
            value = pickle.load(stream)
       
        return value

    original_source, original_target = read_graph(source), read_graph(target)
    source_cache = read_cache(source+str(imbalance_factor))
    selected = prepare_imgda_source(original_source, *source_cache, imbalance_factor, device)
    target_edges, target_norm = read_cache(target)
    unlabeled = Data(x=original_target.x.to(device).clone(), edge_index=original_target.edge_index.to(device).clone(),
                     global_adj=target_edges.to(device).clone(), global_norm=target_norm.to(device).clone())
    validate_imgda_graph(unlabeled)
    return selected, unlabeled


class ImGDA(BaseGDA):
    """
    Dual Prototype-Enhanced Contrastive Framework for Class-Imbalanced Graph Domain Adaptation (NeurIPS-25).

    Parameters
    ----------
    in_dim : int
        Input feature dimension shared by both domains. Default: required.
    hid_dim : int
        Encoder width. Default: required. The released experiment uses ``512``.
    num_classes : int
        Number of node classes. Default: required.
    num_layers : int, optional
        Number of encoder layers. Only ``2`` is supported. Default: ``2``.
    dropout : float, optional
        Dropout probability between encoder layers. Default: ``0.1``.
    act : callable, optional
        Encoder activation. Only ReLU is supported.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Adam learning rate. Default: ``0.01``.
    weight_decay : float, optional
        Adam weight decay. Default: ``0.001``.
    epoch : int, optional
        Number of training epochs. Default: ``200``.
    device : str or torch.device, optional
        Device for training and prediction. Default: ``cuda:0``.
    batch_size : int, optional
        Batch size. Only ``0`` (full graphs) is supported. Default: ``0``.
    num_neigh : int, optional
        Neighbor sampling setting. Only ``-1`` (full neighborhoods) is
        supported. Default: ``-1``.
    verbose : int, optional
        PyGDA logging level from 0 to 3. Default: ``2``.
    k : int, optional
        Maximum number of source anchors sampled per class in each branch.
        Default: ``200``.
    dynamic_temperature : float, optional
        Lower bound of the class-dependent temperature in the cross-branch
        loss. Default: ``0.5``.
    seed : int or None, optional
        Seed set at the start of fitting. ``None`` uses the current random
        states. Default: ``None``.
    **kwargs : dict
        Extra keyword arguments are unsupported. Default: empty.

    Notes
    -----
    ImGDA supports full-graph node classification with explicit global edge
    indices and normalized weights. Use ``load_imgda_pair`` from ``pygda.models.imgda``
    to prepare the released graphs and caches.
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
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        k=200,
        dynamic_temperature=0.5,
        seed=None,
        **kwargs
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
        self.k = k
        self.dynamic_temperature = dynamic_temperature
        self.seed = seed
        self.history_, self.losses_, self.last_logits_ = [], None, None

    def process_graph(self, data, source=False):
        """
        Copy the graph tensors needed by the model to its device.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with node features, local edges, and the normalized global cache.
        source : bool, optional
            Also copy source labels and original class counts. Default: ``False``.

        Returns
        -------
        torch_geometric.data.Data
            Prepared graph with tensors independent of the input graph.
        """
       
        graph = Data(
            x=data.x.to(self.device, copy=True),
            edge_index=data.edge_index.to(self.device, copy=True),
            global_adj=data.global_adj.to(self.device, copy=True),
            global_norm=data.global_norm.to(self.device, copy=True),
        )
        if source:
            graph.y = data.y.to(self.device, copy=True)
            graph.imgda_class_counts = data.imgda_class_counts.to(self.device, copy=True)
        return graph

    def init_model(self, source_data=None, **kwargs):
        """
        Initialize the encoder and node classifier for the selected source graph.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Prepared source graph. Its node count sets the size of the
            compensation parameters; a graph must be supplied.
        **kwargs : dict
            Extra keyword arguments are unsupported.

        Returns
        -------
        torch.nn.ModuleDict
            Encoder and classifier on the configured device.
        """
        
        encoder = ImGDABase(source_data, self.hid_dim, self.dropout, self.in_dim, self.device)
        classifier = nn.Sequential(nn.Linear(self.hid_dim, self.num_classes)).to(self.device)
        return nn.ModuleDict({'encoder': encoder, 'classifier': classifier})

    def _cache(self, name, graph):
        """Register the graph's normalized global edges under the given name."""
        for conv in self.model['encoder'].conv_layers:
            conv.set_global_cache(name, graph.global_adj, graph.global_norm)

    def forward_model(self, source_data, target_data, **kwargs):
        """
        Compute the full-graph objective for a prepared domain pair.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Prepared source graph with labels and original class counts.
        target_data : torch_geometric.data.Data
            Prepared target graph. Target labels are unused.
        **kwargs : dict
            Extra keyword arguments are unsupported.

        Returns
        -------
        torch.Tensor
            Sum of the cross-branch, cross-domain, and classification losses.
        """
        
        self._cache('source', source_data)
        self._cache('target', target_data)
        self.losses_ = imgda_objective(
            self.model['encoder'],
            self.model['classifier'],
            source_data,
            target_data,
            source_data.imgda_class_counts,
            'source',
            'target',
            self.k,
            self.dynamic_temperature,
        )
        return self.losses_['loss']

    def _target_logits(self, graph, cache_name):
        """Return global-branch logits for a prepared graph without compensation."""
        self._cache(cache_name, graph)
        encoded = self.model['encoder'](
            graph.x, graph.edge_index, cache_name, False, self.device, use_ppmi=True
        )
        return self.model['classifier'](encoded)

    def fit(self, source_data, target_data, callback=None):
        """
        Train ImGDA on a class-imbalanced source graph and an unlabeled target.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Selected source graph and its matching cache. Each class needs at
            least two labeled nodes; ``imgda_class_counts`` stores counts before
            source nodes were removed.
        target_data : torch_geometric.data.Data
            Target graph and its cache. Labels, if present, are ignored.
        callback : callable or None, optional
            Receives ``(epoch, losses, target_logits)`` after each update, with
            a zero-based epoch, scalar losses, and a copy of the target logits.
            Default: ``None``.

        Returns
        -------
        ImGDA
            Fitted model with final-epoch weights.

        Notes
        -----
        One target forward pass per epoch preserves the released code's dropout
        random draws, even without a callback. Fitting does not select a checkpoint.
        """
        
        validate_imgda_graph(source_data, source=True, num_classes=self.num_classes)
        source = self.process_graph(source_data, source=True)
        validate_imgda_graph(target_data)
        target = self.process_graph(target_data)

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.device(self.device).type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(self.seed)

        self.model = self.init_model(source)
        self.optimizer_ = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.history_, self.last_logits_ = [], None
        start_time = time.time()

        for epoch in range(self.epoch):
            self.model.train()
            loss = self.forward_model(source, target)
            
            self.optimizer_.zero_grad()
            loss.backward()
            self.optimizer_.step()

            values = {
                name: self.losses_[name].detach().item()
                for name in ('loss', 'cb_cL_loss', 'cd_cl_loss', 'lc_loss')
            }
            self.history_.append(dict(epoch=epoch, **values))
            self.model.eval()
            with torch.no_grad():
                self.last_logits_ = self._target_logits(target, 'target')
            
            if callback is not None:
                callback(epoch, dict(values), self.last_logits_.clone())
            logger(
                epoch=epoch,
                loss=values['loss'],
                time=time.time() - start_time,
                verbose=self.verbose,
                train=True,
            )
            # Keep scalar history; avoid retaining the final training graph.
            self.losses_ = values
        return self

    def predict(self, data):
        """
        Predict node logits from the supplied graph's global branch.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with a normalized global cache and optional node labels.

        Returns
        -------
        logits : torch.Tensor
            Raw class scores of shape ``[num_nodes, num_classes]``.
        labels : torch.Tensor or None
            Input labels on the configured device, if present.

        Notes
        -----
        The released encoder keeps dropout active in evaluation mode, so
        repeated predictions can differ.
        """
        
        validate_imgda_graph(data)
        graph = self.process_graph(data)
        self.model.eval()
        with torch.no_grad():
            logits = self._target_logits(graph, 'prediction')
        labels = data.y.to(self.device) if data.y is not None else None
        return logits, labels
