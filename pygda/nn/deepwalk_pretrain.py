import torch
from torch_geometric.nn import Node2Vec

class DWPretrain(torch.nn.Module):
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
        for epoch in range(self.epoch):
            loss = self.train()
            print(f'Epoch: {epoch:03d}, pretrain loss: {loss:.4f}')
    
    def get_embedding(self):
        self.model.eval()
        z = self.model()

        return z
