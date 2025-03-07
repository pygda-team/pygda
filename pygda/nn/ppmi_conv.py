from collections import Counter
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import numpy as np
from tqdm import tqdm
from .cached_gcn_conv import CachedGCNConv


class PPMIConv(CachedGCNConv):
    """
    Graph convolution layer using Pointwise Mutual Information (PPMI) for edge weighting.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    weight : torch.Tensor, optional
        Pre-defined weight matrix. Default: None
    bias : torch.Tensor, optional
        Pre-defined bias vector. Default: None
    improved : bool, optional
        If True, uses A + 2I for self-loops. Default: False
    use_bias : bool, optional
        Whether to include bias. Default: True
    path_len : int, optional
        Maximum length of random walks. Default: 5
    **kwargs : optional
        Additional arguments for CachedGCNConv

    Notes
    -----
    Implements PPMI-based graph convolution by:

    - Generating random walks to estimate node co-occurrences
    - Computing PPMI scores for edge weights
    - Applying normalized convolution with PPMI weights
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        weight=None,
        bias=None,
        improved=False,
        use_bias=True,
        path_len=5,
        **kwargs
        ):
        super().__init__(in_channels, out_channels, weight, bias, improved, use_bias, **kwargs)
        self.path_len = path_len

    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        """
        Compute PPMI-based normalized adjacency matrix.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices (2 x E)
        num_nodes : int
            Number of nodes in the graph
        edge_weight : torch.Tensor, optional
            Initial edge weights. Default: None
        improved : bool, optional
            If True, uses A + 2I for self-loops. Default: False
        dtype : torch.dtype, optional
            Data type for computations. Default: None

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Contains:

            - Modified edge indices
            - PPMI-based normalized edge weights

        Notes
        -----
        - Build adjacency dictionary
        - Generate random walks
        - Compute normalized co-occurrence probabilities
        - Calculate PPMI scores
        - Apply symmetric normalization
        
        The PPMI computation follows:
        PPMI(a,b) = max(log(P(a,b)/(P(a)P(b)) * N/path_len), 0)
        where:
        
        - P(a,b) is co-occurrence probability
        - P(a), P(b) are marginal probabilities
        - N is number of nodes
        """

        adj_dict = {}

        def add_edge(a, b):
            if a in adj_dict:
                neighbors = adj_dict[a]
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        cpu_device = torch.device("cpu")
        gpu_device = torch.device(edge_index.device)
        for a, b in edge_index.t().detach().to(cpu_device).numpy():
            a = int(a)
            b = int(b)
            add_edge(a, b)
            add_edge(b, a)

        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

        def sample_neighbor(a):
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]

        # word_counter = Counter()
        walk_counters = {}

        def norm(counter):
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter

        for _ in tqdm(range(40)):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, self.path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter

                    walk_counter[b] += 1

                    current_a = b

        normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

        prob_sums = Counter()

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                prob_sums[b] += prob

        ppmis = {}

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                ppmi = max(np.log(prob / prob_sums[b] * len(prob_sums) / self.path_len), 0)
                ppmis[(a, b)] = ppmi

        new_edge_index = []
        edge_weight = []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)

        edge_index = torch.tensor(new_edge_index).t().to(gpu_device)
        edge_weight = torch.tensor(edge_weight).to(gpu_device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32)
