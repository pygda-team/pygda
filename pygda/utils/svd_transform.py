import os

import torch
import numpy as np
from torch_geometric.utils import get_laplacian
from sklearn.decomposition import TruncatedSVD

def svd_transform(
    data, 
    processed_paths
    ):
    """
    Perform SVD transformation on graph adjacency matrix and store eigenvalues/vectors.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input graph data object containing edge indices and node features
    processed_paths : str
        Path to save processed eigenvalues and eigenvectors

    Returns
    -------
    torch_geometric.data.Data
        Data object augmented with eigenvalues and eigenvectors

    Notes
    -----
    Processing Steps:

    - Graph Processing:
       
       * Extract number of nodes
       * Compute Laplacian matrix
       * Convert to dense adjacency

    - SVD Computation:
       
       * Choose components based on graph size
       * Small graphs (<1000 nodes): 100 components
       * Large graphs: 1000 components
       * Perform truncated SVD

    - Data Storage:
       
       * Save square root of explained variance
       * Save component vectors
       * Load into data object

    Features:

    - Adaptive dimensionality
    - Memory efficient SVD
    - Sparse to dense conversion
    - Eigendecomposition storage

    Mathematical Details:

    - Uses truncated SVD for dimensionality reduction
    - Computes graph Laplacian eigendecomposition
    - Stores sqrt{explained_variance} as eigenvalues
    - Preserves principal components as eigenvectors
    """
    num_node = data.y.shape[0]
    edge_index, edge_weight = get_laplacian(data.edge_index, num_nodes=num_node)
    edge_index = edge_index.numpy()
    edge_weight = edge_weight.numpy()
    adj = np.zeros((num_node, num_node), dtype=np.float32)
    adj[edge_index[0,:], edge_index[1,:]] = edge_weight

    if num_node < 1000:
        pca = TruncatedSVD(n_components=100, n_iter=20, random_state=42)
    else:
        pca = TruncatedSVD(n_components=1000, n_iter=20, random_state=42)
    pca.fit(adj)

    torch.save(torch.tensor(pca.explained_variance_ ** 0.5, dtype=torch.float32 ), processed_paths + 'eival.pt')
    torch.save(torch.tensor(pca.components_, dtype=torch.float32 ), processed_paths + 'eivec.pt')

    data.eival = torch.load(processed_paths + 'eival.pt')
    data.eivec = torch.load(processed_paths + 'eivec.pt')

    return data
