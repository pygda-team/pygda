"""Load the released ImGDA processed graphs and precomputed global caches."""
from pathlib import Path
import pickle
import re

import torch
from torch_geometric.data import Data

from ..nn.imgda_base import make_longtailed_data_remove


def validate_imgda_graph(data, source=False, num_classes=None):
    """Validate explicit normalized-cache inputs; target validation never reads y."""
    x, edges = data.x, data.edge_index
    if (x is None or x.ndim != 2 or x.size(0) < 1 or x.size(1) < 1 or
            x.dtype != torch.float32 or not torch.isfinite(x).all()):
        raise ValueError('x must be a nonempty finite float32 matrix.')
    for name, index in (('edge_index', edges), ('global_adj', getattr(data, 'global_adj', None))):
        if (index is None or index.dtype != torch.long or index.ndim != 2 or index.size(0) != 2 or
                index.numel() == 0 or index.min() < 0 or index.max() >= x.size(0)):
            raise ValueError(name + ' must contain valid nonempty [2, E] long indices.')
    norm = getattr(data, 'global_norm', None)
    if (norm is None or norm.dtype != torch.float32 or norm.ndim != 1 or
            norm.numel() != data.global_adj.size(1) or not torch.isfinite(norm).all()):
        raise ValueError('global_norm must supply one finite float32 normalized weight per global edge.')
    if source:
        y, counts = data.y, getattr(data, 'imgda_class_counts', None)
        if (num_classes is None or type(num_classes) is not int or num_classes < 2 or
                y is None or y.dtype != torch.long or y.ndim != 1 or y.numel() != x.size(0) or
                y.min() < 0 or y.max() >= num_classes):
            raise ValueError('Source labels must be contiguous class indices with a specified class count.')
        selected_counts = torch.bincount(y, minlength=num_classes)
        if (selected_counts < 2).any():
            raise ValueError('Released ImGDA requires at least two selected source nodes in every class.')
        if (counts is None or counts.dtype != torch.long or counts.ndim != 1 or
                counts.numel() != num_classes or (counts.to(selected_counts.device) < selected_counts).any()):
            raise ValueError('imgda_class_counts must contain original full-source counts for every class.')


def prepare_imgda_source(data, global_adj, global_norm, imbalance_factor=20, device='cuda:0'):
    """Select the author's long-tail training population before attaching its cache.

    The supplied cache must already correspond to the selected and reindexed
    nodes. Degree ties can resolve differently on CPU and CUDA. Cache index
    bounds alone cannot prove node-semantic alignment.
    """
    if type(imbalance_factor) is not int or imbalance_factor < 1:
        raise ValueError('imbalance_factor must be a positive integer.')
    if data.x is None or data.y is None or data.train_mask is None:
        raise ValueError('Full source x, y, edge_index and the original train_mask are required.')
    x, y, edges = data.x.to(device), data.y.to(device), data.edge_index.to(device)
    mask = data.train_mask.to(device)
    if (x.ndim != 2 or x.dtype != torch.float32 or not torch.isfinite(x).all() or
            y.dtype != torch.long or y.ndim != 1 or y.numel() != x.size(0) or
            y.min() < 0 or mask.dtype != torch.bool or mask.shape != y.shape):
        raise ValueError('Invalid full source features, labels or train_mask.')
    classes = int(y.max()) + 1
    counts = torch.bincount(y, minlength=classes)
    train_counts = torch.bincount(y[mask], minlength=classes)
    if classes < 2 or (train_counts < 2).any():
        raise ValueError('Every source class must have at least two training nodes.')
    if (edges.dtype != torch.long or edges.ndim != 2 or edges.size(0) != 2 or
            edges.numel() == 0 or edges.min() < 0 or edges.max() >= y.numel()):
        raise ValueError('Invalid full source edge_index.')
    _, selected, _, _, edge_mask = make_longtailed_data_remove(
        edges, y, train_counts.tolist(), classes, imbalance_factor, mask.clone())
    node_ids = torch.nonzero(selected).squeeze(1)
    # Preserve the original sorted-node indexing and induced edge order.
    mapping = torch.full((y.numel(),), -1, dtype=torch.long, device=device)
    mapping[node_ids] = torch.arange(node_ids.numel(), device=device)
    mapped = mapping[edges[:, edge_mask]]
    mapped = mapped[:, (mapped >= 0).all(dim=0)]
    graph = Data(x=x[selected].clone(), y=y[selected].clone(), edge_index=mapped.clone(),
                 global_adj=global_adj.to(device).clone(), global_norm=global_norm.to(device).clone(),
                 imgda_class_counts=counts.clone(), imgda_original_node_ids=node_ids.clone())
    validate_imgda_graph(graph, source=True, num_classes=classes)
    return graph


def load_imgda_pair(root, source='ACMv9', target='Citationv1', imbalance_factor=20, device='cuda:0'):
    """Read trusted author .pt/.pkl files without regenerating masks or caches.

    root contains data/<domain>/processed/data.pt and tmp/<cache>.pkl.
    Only load trusted files: author files use Python pickle serialization.
    Target labels are omitted. Load them separately for external evaluation.
    """
    root = Path(root)
    if any(not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in (source, target)):
        raise ValueError('Domain names must be plain identifiers.')
    if type(imbalance_factor) is not int or imbalance_factor < 1:
        raise ValueError('imbalance_factor must be a positive integer.')
    def read_graph(name):
        graph, slices = torch.load(root/'data'/name/'processed/data.pt', map_location='cpu', weights_only=False)
        if slices is not None and any(len(v) != 2 for v in slices.values()):
            raise ValueError('Expected one graph in the author processed cache.')
        return graph
    def read_cache(name):
        with (root/'tmp'/(name+'.pkl')).open('rb') as stream:
            value = pickle.load(stream)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError('Expected (edge_index, normalized_weight) global cache.')
        return value
    original_source, original_target = read_graph(source), read_graph(target)
    source_cache = read_cache(source+str(imbalance_factor))
    selected = prepare_imgda_source(original_source, *source_cache, imbalance_factor, device)
    target_edges, target_norm = read_cache(target)
    unlabeled = Data(x=original_target.x.to(device).clone(), edge_index=original_target.edge_index.to(device).clone(),
                     global_adj=target_edges.to(device).clone(), global_norm=target_norm.to(device).clone())
    validate_imgda_graph(unlabeled)
    return selected, unlabeled
