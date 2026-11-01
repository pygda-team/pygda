"""Read the author's nine-class UAGA MAT files without altering labels or edges."""
from pathlib import Path


def load_uaga_graph(root, name):
    """Load citation-v1, dblp-v4 or acm-v8 as PyG Data.

    ``root`` contains the original MAT files. Labels remain in 0..8; UAGA.fit
    induces the known-class source graph, and predict maps unknown target labels.
    No files are downloaded or modified. Dense float32 features follow the author
    training code. Do not substitute the five-class ACMv9/Citationv1 datasets.
    """
    if name not in ('citation-v1', 'dblp-v4', 'acm-v8'):
        raise ValueError('Unknown UAGA domain: ' + str(name))
    import torch
    from scipy.io import loadmat
    from torch_geometric.data import Data
    raw = loadmat(Path(root) / (name + '.mat'))
    adjacency = raw['adjacency_matrix'].tocoo()
    edges = torch.stack((torch.from_numpy(adjacency.row.astype('int64')),
                         torch.from_numpy(adjacency.col.astype('int64'))))
    return Data(x=torch.tensor(raw['features'].toarray(), dtype=torch.float32),
                edge_index=edges, y=torch.tensor(raw['labels'][0], dtype=torch.long))
