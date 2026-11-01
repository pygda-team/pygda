"""Read-only multi-view data adapter for the author SAGA datasets.

The preprocessing follows f1shungry/SAGA at commit
55ff00a4e426a2d0fbeb913f11330ab69b68a506. Derived preprocessing is
distributed under its MIT license; see SAGA_LICENSE.txt.
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset


DOMAIN_SPECS = {
    'ACM1': (3025, 512, 3, ('PAP', 'PSP')),
    'ACM2': (4019, 512, 3, ('PAP', 'PSP')),
    'CN': (25467, 128, 10, ('PAP', 'PP')),
    'US': (55392, 128, 10, ('PAP', 'PP')),
    'JP': (12651, 128, 10, ('PAP', 'PP')),
    'FR': (12052, 128, 10, ('PAP', 'PP')),
    'DE': (17919, 128, 10, ('PAP', 'PP')),
    'RU': (3690, 128, 10, ('PAP', 'PP')),
}


def _labels(value, nodes, classes):
    if not isinstance(value, torch.Tensor):
        raise TypeError('Labels must be a tensor.')
    if value.ndim == 2 and tuple(value.shape) == (nodes, classes):
        if not bool(((value == 0) | (value == 1)).all()) or not bool((value.sum(1) == 1).all()):
            raise ValueError('Expected one-hot labels with exactly one class per node.')
        value = value.argmax(1)
    elif value.ndim != 1 or value.numel() != nodes or value.is_floating_point():
        raise ValueError('Expected integer class indices or a one-hot label matrix.')
    value = value.long()
    if value.numel() and (value.min() < 0 or value.max() >= classes):
        raise ValueError('Label index outside the domain class range.')
    return value.clone()


def load_saga_graph(root, name, include_labels=True):
    """Load one domain without downloads, transforms, cache writes or splits.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory containing ACM1/, ACM2/, or the MAG country directories.
    name : str
        Domain name in DOMAIN_SPECS.
    include_labels : bool, optional
        If False, label.pt is never opened and the returned graph has no y.

    Returns
    -------
    torch_geometric.data.Data
        x, optional integer y, ordered view_names, edge_index_dict and
        edge_weight_dict. Edges use PyG's [source, destination] convention.
        Reconstruction flips the rows back to the author's matrix convention.
    """
    if name not in DOMAIN_SPECS:
        raise ValueError(f'Unknown SAGA domain: {name}')
    _, features, classes, views = DOMAIN_SPECS[name]
    folder = Path(root) / name
    # These files must come from the trusted author dataset archive.
    x = torch.load(folder / 'feat.pt', map_location='cpu', weights_only=False)
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError('feat.pt must contain a node feature tensor.')
    x = x.float()
    if x.size(1) != features or not x.size(0) or not bool(torch.isfinite(x).all()):
        raise ValueError('Unexpected feature shape or non-finite features.')
    indices, weights = {}, {}
    for view in views:
        adj = torch.load(folder / (view.lower() + '.pt'), map_location='cpu', weights_only=False)
        if not isinstance(adj, torch.Tensor) or adj.layout != torch.sparse_coo:
            raise ValueError(f'{view} must be a sparse COO tensor, as in the author loader.')
        if tuple(adj.shape) != (x.size(0), x.size(0)):
            raise ValueError(f'{view} node count does not match features.')
        adj = adj.coalesce()
        indices[view] = adj.indices().flip(0).clone()
        weights[view] = adj.values().clone()
    graph = Data(x=x.clone(), edge_index_dict=indices, edge_weight_dict=weights,
                 view_names=tuple(views), saga_domain=name, saga_num_classes=classes)
    if include_labels:
        graph.y = _labels(torch.load(folder / 'label.pt', map_location='cpu', weights_only=False),
                          x.size(0), classes)
    validate_saga_graph(graph)
    return graph


class SAGADataset(Dataset):
    """One-domain dataset using existing author files, with no processed cache.

    Parameters
    ----------
    root : str
        Parent directory of the domain folders.
    name : str
        Domain name, e.g. ACM1 or CN.
    transform : callable, optional
        Optional PyG transform applied to a clone returned by indexing.
    include_labels : bool, optional
        Whether to open label.pt. Default: True.
    """

    def __init__(self, root, name, transform=None, include_labels=True):
        self.name = name
        self._graph = load_saga_graph(root, name, include_labels=include_labels)
        super().__init__(str(root), transform=transform)

    def len(self):
        return 1

    def get(self, index):
        if index != 0:
            raise IndexError(index)
        return self._graph.clone()


def validate_saga_graph(data):
    """Validate the multi-view input contract without changing the input."""
    if not isinstance(data, Data) or data.x is None or data.x.ndim != 2:
        raise ValueError('Expected PyG Data with a node feature matrix.')
    if not data.x.is_floating_point() or not data.x.size(0) or not bool(torch.isfinite(data.x).all()):
        raise ValueError('Features must be nonempty, floating point and finite.')
    if data.num_nodes != data.x.size(0):
        raise ValueError('num_nodes must equal the number of feature rows.')
    views = tuple(getattr(data, 'view_names', ()))
    if not views or len(set(views)) != len(views):
        raise ValueError('view_names must contain unique ordered relation names.')
    for attribute in ('edge_index_dict', 'edge_weight_dict'):
        mapping = getattr(data, attribute, None)
        if not isinstance(mapping, dict) or set(mapping) != set(views):
            raise ValueError(f'{attribute} must have exactly the named views.')
    for view in views:
        edge, weight = data.edge_index_dict[view], data.edge_weight_dict[view]
        if (not isinstance(edge, torch.Tensor) or edge.dtype != torch.long or
                edge.ndim != 2 or edge.size(0) != 2):
            raise ValueError(f'{view}: expected long edge indices of shape [2, E].')
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 1 or
                weight.numel() != edge.size(1)):
            raise ValueError(f'{view}: expected one weight per edge in a 1-D tensor.')
        # Author archives may store adjacency values as integers or bools.
        # Preserve that dtype here; adding float unit loops in preprocessing
        # performs the same promotion as the author implementation.
        if weight.is_complex():
            raise ValueError(f'{view}: complex edge weights are unsupported.')
        if not bool(torch.isfinite(weight).all()):
            raise ValueError(f'{view}: edge weights contain non-finite values ({weight.dtype}).')
        if bool((weight < 0).any()):
            raise ValueError(f'{view}: edge weights contain negative values ({weight.dtype}).')
        if edge.numel() and (edge.min() < 0 or edge.max() >= data.x.size(0)):
            raise ValueError(f'{view}: edge endpoint outside the node range.')
    return views


def saga_adjacencies(data, device='cpu'):
    """Rebuild author-layout adjacency matrices in the declared view order."""
    views = validate_saga_graph(data)
    nodes = data.x.size(0)
    return [torch.sparse_coo_tensor(data.edge_index_dict[v].flip(0).to(device),
                                    data.edge_weight_dict[v].to(device),
                                    size=(nodes, nodes), device=device).coalesce()
            for v in views]


def prepare_saga_views(data, k_hop=2, filter_alpha=0.0, device='cpu'):
    """Return author-equivalent normalized matrices and propagated features.

    k_hop is fixed, following the public implementation. No dynamic hop
    search is introduced. Existing self-loops are removed, unit loops added,
    and symmetric normalization precedes APPNP and row L2 normalization.
    Labels are not accessed, and the graph is not modified.
    """
    if type(k_hop) is not int or k_hop < 1:
        raise ValueError('k_hop must be a positive integer.')
    if not math.isfinite(filter_alpha) or not 0 <= filter_alpha <= 1:
        raise ValueError('filter_alpha must be finite and in [0, 1].')
    x = data.x.to(device=device, dtype=torch.float32)
    normalized, propagated = [], []
    with torch.no_grad():
        for adj in saga_adjacencies(data, device=device):
            index, values = adj.indices(), adj.values()
            mask = index[0] != index[1]
            loop = torch.arange(x.size(0), device=device)
            index = torch.cat((torch.stack((loop, loop)), index[:, mask]), dim=1)
            values = torch.cat((torch.ones(x.size(0), device=device), values[mask]))
            matrix = torch.sparse_coo_tensor(index, values, adj.shape, device=device).coalesce()
            degree = torch.sparse.sum(matrix, dim=1).values()
            inv = 1.0 / torch.sqrt(degree)
            weights = matrix.values() * inv[matrix.indices()[0]] * inv[matrix.indices()[1]]
            matrix = torch.sparse_coo_tensor(matrix.indices(), weights, matrix.shape,
                                             device=device).coalesce()
            normalized.append(matrix)
            z = x
            for _ in range(k_hop):
                z = torch.sparse.mm(matrix.float(), z.float())
                z = (1 - filter_alpha) * z + filter_alpha * x
            propagated.append(F.normalize(z, dim=1, p=2))
    return normalized, propagated


def validate_saga_pair(source, target):
    """Check the initial equal-view contract without inspecting target labels."""
    source_views = validate_saga_graph(source)
    target_views = validate_saga_graph(target)
    if source_views != target_views:
        raise ValueError('Source and target must use the same ordered view names.')
    if source.x.size(1) != target.x.size(1):
        raise ValueError('Source and target feature dimensions must match.')
    source_classes = getattr(source, 'saga_num_classes', None)
    target_classes = getattr(target, 'saga_num_classes', None)
    if source_classes is not None and target_classes is not None and source_classes != target_classes:
        raise ValueError('Source and target class spaces must match.')
    return source_views
