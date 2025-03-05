import torch
from torch_geometric.nn import Node2Vec

class DWPretrain(torch.nn.Module):
    """
    DeepWalk pretraining implementation for graph embeddings.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Input graph data object.
    epoch : int, optional
        Number of training epochs. Default: 200.
    embedding_dim : int, optional
        Dimension of node embeddings. Default: 128.
    walk_length : int, optional
        Length of each random walk. Default: 20.
    context_size : int, optional
        Size of context window. Default: 10.
    walks_per_node : int, optional
        Number of walks per node. Default: 10.
    num_negative_samples : int, optional
        Number of negative samples per positive pair. Default: 1.

    Notes
    -----
    Implements DeepWalk algorithm using Node2Vec with p=q=1.0 (equivalent to DeepWalk).
    Uses sparse implementation for memory efficiency.
    """

    def __init__(
        self,
        data,
        epoch=200,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        ):
        super(DWPretrain, self).__init__()

        self.data = data
        self.device = data.edge_index.device
        self.epoch = epoch
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples

        self.model = Node2Vec(
            data.edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            num_negative_samples=self.num_negative_samples,
            p=1.0,
            q=1.0,
            sparse=True,
        ).to(self.device)

        num_workers = 4
        self.loader = self.model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
    
    def train(self):
        """
        Execute one epoch of training.

        Returns
        -------
        float
            Average loss value for the epoch.

        Notes
        -----
        Training process:
        
        1. Generate random walks
        2. Sample positive and negative context pairs
        3. Update embeddings using SparseAdam optimizer
        """
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.loader)
    
    def fit(self):
        """
        Complete training procedure for all epochs.

        Notes
        -----
        Executes training loop for specified number of epochs.
        Prints progress including epoch number and loss value.
        """
        for epoch in range(self.epoch):
            loss = self.train()
            print(f'Epoch: {epoch:03d}, pretrain loss: {loss:.4f}')
    
    def get_embedding(self):
        """
        Retrieve learned node embeddings.

        Returns
        -------
        torch.Tensor
            Node embedding matrix of shape (num_nodes, embedding_dim).

        Notes
        -----
        Returns final node embeddings after training or during evaluation.
        """
        self.model.eval()
        z = self.model()

        return z
