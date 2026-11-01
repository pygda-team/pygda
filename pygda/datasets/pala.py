"""Load the author's trusted PALA material and preserve its existing splits."""
import io
import pickle
from pathlib import Path

import torch
from torch_geometric.data import Data

__all__ = ['load_pala_graph', 'load_pala_cache', 'prepare_pala_source', 'prepare_pala_target']


def load_pala_graph(root, name, include_labels=True):
    """Read a trusted author processed/data.pt; never regenerate masks or cache.

    This reader uses pickle-backed torch.load and is intended for verified
    author files. include_labels=False does not retrieve y from the graph.
    """
    path = Path(root) / name / 'processed' / 'data.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(payload, tuple) or len(payload) != 2 or not isinstance(payload[0], Data):
        raise ValueError('Expected the author (Data, slices) processed-file format.')
    original = payload[0]
    graph = Data(x=original.x.clone(), edge_index=original.edge_index.clone())
    if include_labels:
        graph.y = original.y.clone()
        for key in ['train_mask', 'val_mask', 'test_mask']:
            value = getattr(original, key, None)
            if value is not None:
                graph[key] = value.clone()
    return graph


def load_pala_cache(path):
    """Read a trusted author tensor pickle onto CPU, preserving signed weights."""
    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda value: torch.load(io.BytesIO(value), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    with Path(path).open('rb') as stream:
        payload = CPUUnpickler(stream).load()
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError('Expected a cached (edge_index, normalization) pair.')
    return tuple(value.detach().clone() for value in payload)


def validate_pala_graph(data, in_dim=None):
    """Validate feature, graph and cache tensors without reading labels."""
    x, edges = data.x, data.edge_index
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError('Expected nonempty floating point node features [N,F].')
    if in_dim is not None and x.shape[1] != in_dim:
        raise ValueError('Feature width differs from in_dim.')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Features contain non-finite values.')
    for name, index in [('edge_index', edges), ('ppmi_edge_index', data.ppmi_edge_index)]:
        if not isinstance(index, torch.Tensor) or index.dtype != torch.long or index.ndim != 2 or index.shape[0] != 2:
            raise ValueError(name + ' must be a long tensor [2,E].')
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= x.shape[0]):
            raise ValueError(name + ' has invalid node indices.')
    weight = data.ppmi_edge_weight
    if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.float32 or weight.ndim != 1
            or weight.numel() != data.ppmi_edge_index.shape[1] or not bool(torch.isfinite(weight).all())):
        raise ValueError('Expected finite float32 cached weights [E].')


def _attach_cache(graph, cache, device):
    if not isinstance(cache, (tuple, list)) or len(cache) != 2:
        raise ValueError('Supply the author cached (edge_index, normalization) pair.')
    graph.ppmi_edge_index = cache[0].detach().clone()
    graph.ppmi_edge_weight = cache[1].detach().clone()
    validate_pala_graph(graph)
    graph.global_adj = graph.ppmi_edge_index
    return graph.to(device)


def prepare_pala_source(data, cache, imbalance_factor=20, device='cuda:0'):
    """Apply author low-degree removal and induce/reindex selected train nodes.

    Run before setting the per-training seed, matching the author entrypoint.
    The cache must already use this exact selected-node order. Shape validation
    cannot establish cache/node semantic alignment without external provenance.
    """
    from ..nn.pala_base import make_longtailed_data_remove

    if type(imbalance_factor) is not int or imbalance_factor < 1:
        raise ValueError('imbalance_factor must be a positive integer.')
    x, edges, labels, mask = data.x, data.edge_index, data.y, data.train_mask
    if (labels is None or labels.dtype != torch.long or labels.ndim != 1 or labels.numel() != x.shape[0]
            or int(labels.min()) < 0 or mask is None or mask.dtype != torch.bool or mask.shape != labels.shape):
        raise ValueError('Source needs nonnegative long labels and an existing boolean train_mask.')
    classes = int(labels.max()) + 1
    counts = torch.bincount(labels[mask], minlength=classes).tolist()
    if classes < 2 or min(counts) < 1:
        raise ValueError('Source train_mask must represent at least two contiguous classes.')
    labels, mask, edges = labels.to(device), mask.to(device), edges.to(device)
    _, selected, _, _, edge_mask = make_longtailed_data_remove(
        edges, labels, counts, classes, imbalance_factor, mask.clone())
    indices = selected.nonzero().view(-1)
    mapping = torch.full((labels.numel(),), -1, device=device, dtype=torch.long)
    mapping[indices] = torch.arange(indices.numel(), device=device)
    mapped = mapping[edges[:, edge_mask]]
    mapped = mapped[:, (mapped >= 0).all(dim=0)]
    graph = Data(x=x.to(device)[selected].clone(), y=labels[selected].clone(), edge_index=mapped,
                 pala_original_node_ids=indices.clone())
    return _attach_cache(graph, cache, device)


def prepare_pala_target(data, cache, device='cuda:0'):
    """Copy target features/edges and explicit cache, without retrieving y."""
    graph = Data(x=data.x.clone(), edge_index=data.edge_index.clone())
    return _attach_cache(graph, cache, device)
