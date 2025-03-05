import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor, spmm

from .gmm_clustering import GMMClustering


class GCNPooling(torch.nn.Module):
    """
    GCN-based hierarchical pooling layer.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Number of nodes in pooled graph.
    device : torch.device
        Device to use.
    sparse : bool
        Whether to use sparse assignment matrix.
    """

    def __init__(self, in_dim, hid_dim, device, sparse):
        super(GCNPooling, self).__init__()
        self.gcn = GCNConv(in_dim, hid_dim)
        self.softmax = nn.Softmax(dim=1)
        self.device = device
        self.sparse = sparse
    
    def to_onehot(self, label_matrix, num_classes):
        """
        Convert labels to one-hot encoding.

        Parameters
        ----------
        label_matrix : torch.Tensor
            Label indices.
        num_classes : int
            Number of classes.

        Returns
        -------
        torch.Tensor
            One-hot encoded labels.
        """
        identity = torch.eye(num_classes).to(self.device)
        onehot = torch.index_select(identity, 0, label_matrix)
        
        return onehot

    def forward(self, X_old, edge_index, edge_weight, A_old, Y_old, Z, use_sparse=False):
        """
        Forward pass of pooling layer.

        Parameters
        ----------
        X_old : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.
        edge_weight : torch.Tensor
            Edge weights.
        A_old : torch.Tensor
            Adjacency matrix.
        Y_old : torch.Tensor
            Node labels.
        Z : torch.Tensor
            Node embeddings.
        use_sparse : bool, optional
            Whether to use sparse operations.

        Returns
        -------
        tuple
            Contains:
            
            - S: Assignment matrix
            - X_new: Pooled features
            - A_new: Pooled adjacency
            - Y_new: Pooled labels
            - Y_new_prob: Label probabilities
        """
        S = self.softmax(F.relu(self.gcn(X_old, edge_index, edge_weight)))

        if self.sparse:
            S = self.to_onehot(torch.argmax(S, dim=1), num_classes=S.shape[1])

        X_new = torch.matmul(S.T, Z)
        n_class = Y_old.shape[1]
        Y_new_prob = torch.softmax(torch.matmul(S.T, Y_old), dim=1)
        Y_new = self.to_onehot(torch.argmax(Y_new_prob, dim=1), n_class)

        if use_sparse:
            num_nodes_old = X_old.shape[0]
            row, col = edge_index[0,:], edge_index[1,:]
            tmp = spmm(index=torch.vstack([row, col]), value=edge_weight, m=num_nodes_old, n=num_nodes_old, matrix=S)
            A_new = torch.matmul(tmp.t(), S)
        else:
            A_new = torch.matmul(torch.matmul(S.T, A_old), S)

        return S, X_new, A_new, Y_new, Y_new_prob


class JHGDABase(nn.Module):
    """
    Base class for JHGDA.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Number of classes.
    device : str
        GPU or CPU.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    share   : bool, optional
        Share the diffpool module or not. Default: ``False``.
    sparse  : bool, optional
        Diffpool module sparse or not. Default: ``False``.
    classwise   : bool, optional
        Classwise conditional shift or not. Default: ``True``.
    **kwargs : optional
        Other parameters for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 num_classes,
                 device,
                 pool_ratio,
                 num_s,
                 num_t,
                 num_layers=3,
                 dropout=0.1,
                 act=F.relu,
                 share=False,
                 classwise=False,
                 sparse=False,
                 **kwargs):
        super(JHGDABase, self).__init__()

        self.num_layers = num_layers
        self.share = share
        self.pool_ratio=pool_ratio
        self.classwise = classwise
        self.sparse = sparse
        self.device = device
        self.act = act
        self.num_s = num_s
        self.num_t = num_t

        self.gnn_emb = nn.ModuleList()
        self.gnn_emb.append(GCNConv(in_dim, hid_dim))

        for i in range(1, self.num_layers):
            self.gnn_emb.append(GCNConv(hid_dim, hid_dim))
        
        if share:
            self.gnn_pool = nn.ModuleList()
            node_num_s = int(self.num_s * self.pool_ratio)
            self.gnn_pool.append(GCNPooling(in_dim, node_num_s, device=self.device, sparse=self.sparse))
            for i in range(1, self.num_layers):
                self.gnn_pool.append(GCNPooling(hid_dim, int(node_num_s * self.pool_ratio), device=self.device, sparse=self.sparse))
                node_num_s = int(node_num_s * self.pool_ratio)
        else:
            self.gnn_pool_src = nn.ModuleList()
            node_num_s = int(self.num_s * self.pool_ratio)
            self.gnn_pool_src.append(GCNPooling(in_dim, node_num_s, device=self.device, sparse=self.sparse))
            for i in range(1, self.num_layers):
                self.gnn_pool_src.append(GCNPooling(hid_dim, int(node_num_s * self.pool_ratio), device=self.device, sparse=self.sparse))
                node_num_s = int(node_num_s * self.pool_ratio)

            self.gnn_pool_tgt = nn.ModuleList()
            node_num_t = int(self.num_t * self.pool_ratio)
            self.gnn_pool_tgt.append(GCNPooling(in_dim, node_num_t, device=self.device, sparse=self.sparse))
            for i in range(1, self.num_layers):
                self.gnn_pool_tgt.append(GCNPooling(hid_dim, int(node_num_t * self.pool_ratio), device=self.device, sparse=self.sparse))
                node_num_t = int(node_num_t * self.pool_ratio)
        
        self.cls_model = nn.Sequential(nn.Linear(hid_dim, num_classes))
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x_s, edge_index_s, y_s, x_t, edge_index_t, y_t):
        """
        Forward pass of JHGDA model.

        Parameters
        ----------
        x_s : torch.Tensor
            Source node features.
        edge_index_s : torch.Tensor
            Source edge indices.
        y_s : torch.Tensor
            Source labels.
        x_t : torch.Tensor
            Target node features.
        edge_index_t : torch.Tensor
            Target edge indices.
        y_t : torch.Tensor
            Target labels.

        Returns
        -------
        tuple
            Contains:

            - embeddings: List of source/target embeddings
            - pred: Source/target predictions
            - pooling_loss: Dictionary of pooling losses
            - y: List of source/target labels
        """
        edge_weight_s = torch.ones(edge_index_s.shape[1]).to(self.device)
        edge_weight_t = torch.ones(edge_index_t.shape[1]).to(self.device)
        A_s, A_t = edge_index_s, edge_index_t
        y_prob_s = y_s
        y_prob_t = y_t

        self.embedddings = []
        self.pooling_loss = []
        self.y = []

        for i in range(self.num_layers):
            z_s = self.act(self.gnn_emb[i](x_s, edge_index_s, edge_weight_s))
            z_t = self.act(self.gnn_emb[i](x_t, edge_index_t, edge_weight_t))
            self.embedddings.append([z_s, z_t])

            if self.classwise and i == 0:
                y_t_pseudo = self.pseudo_label(z_s, y_s, z_t, y_t, edge_index_t, edge_weight_t)
                y_prob_t = y_t_pseudo
                self.y.append([y_s, y_t_pseudo])
            
            use_sparse = True if i == 0 else False

            if len(self.pooling_loss) < i + 1:
                self.pooling_loss.append({})

            if self.share:
                S_s, x_s, A_s_new, y_s, y_prob_s_new = self.gnn_pool[i](X_old=x_s, edge_index=edge_index_s, edge_weight=edge_weight_s, A_old=A_s, Y_old=y_prob_s, Z=z_s, use_sparse=use_sparse)
                S_t, x_t, A_t_new, y_t, y_prob_t_new = self.gnn_pool[i](X_old=x_t, edge_index=edge_index_t, edge_weight=edge_weight_t, A_old=A_t, Y_old=y_prob_t, Z=z_t, use_sparse=use_sparse)
            else:
                S_s, x_s, A_s_new, y_s, y_prob_s_new = self.gnn_pool_src[i](X_old=x_s, edge_index=edge_index_s, edge_weight=edge_weight_s, A_old=A_s, Y_old=y_prob_s, Z=z_s, use_sparse=use_sparse)
                S_t, x_t, A_t_new, y_t, y_prob_t_new = self.gnn_pool_tgt[i](X_old=x_t, edge_index=edge_index_t, edge_weight=edge_weight_t, A_old=A_t, Y_old=y_prob_t, Z=z_t, use_sparse=use_sparse)
            
            self.y.append([y_s, y_t])

            self.pooling_loss[i]['ce'] = (self.entropy(S_s) + self.entropy(S_t)) / 2
            self.pooling_loss[i]['prox'] = (self.proximity_loss(A_s, S_s) + self.proximity_loss(A_t, S_t)) / 2
            self.pooling_loss[i]['cce'] = (self.entropy(y_prob_s_new) + self.entropy(y_prob_t_new)) / 2
            self.pooling_loss[i]['lm'] = (self.label_matching(S_s, y_prob_s, y_prob_s_new) + self.label_matching(S_t, y_prob_t, y_prob_t_new)) / 2
            self.pooling_loss[i]['ls'] = (self.label_stable(S_s, y_prob_s, y_prob_s_new) + self.label_stable(S_t, y_prob_t, y_prob_t_new)) / 2

            A_s, A_t = A_s_new, A_t_new
            y_prob_s, y_prob_t = y_prob_s_new, y_prob_t_new
            edge_index_s, edge_weight_s = self.adj2coo(A_s)
            edge_index_t, edge_weight_t = self.adj2coo(A_t)
        
        pred_s = self.cls_model(self.embedddings[0][0])
        pred_t = self.cls_model(self.embedddings[0][1])
        pred = [pred_s, pred_t]

        return self.embedddings, pred, self.pooling_loss, self.y

    def pseudo_label(self, z_s, y_s, z_t, y_t, edge_index_t, edge_weight_t):
        """
        Generate pseudo-labels for target domain.

        Parameters
        ----------
        z_s : torch.Tensor
            Source embeddings.
        y_s : torch.Tensor
            Source labels.
        z_t : torch.Tensor
            Target embeddings.
        y_t : torch.Tensor
            Target labels.
        edge_index_t : torch.Tensor
            Target edge indices.
        edge_weight_t : torch.Tensor
            Target edge weights.

        Returns
        -------
        torch.Tensor
            Pseudo-labels for target domain.
        """
        n_class = y_s.shape[1]
        entropy_lower_bound = 0.04
        gmmcluster = GMMClustering(num_class=n_class, device=self.device)
        
        with torch.no_grad():
            pred_t = self.cls_model(z_t)
            _, tgt_indices = torch.max(torch.log_softmax(pred_t, dim=-1), dim=1)

        tgt_y_pseudo = tgt_indices
        tgt_y_pseudo = self.to_onehot(tgt_y_pseudo, n_class)

        return tgt_y_pseudo.to(self.device)
    
    def entropy(self, x, reduction='mean'):
        """
        Compute entropy of probability distribution.

        Parameters
        ----------
        x : torch.Tensor
            Probability distribution.
        reduction : str, optional
            Reduction method. Default: 'mean'.

        Returns
        -------
        torch.Tensor
            Entropy value.
        """
        eps = 1e-7
        log_x = torch.log(x + eps)
        entropy_x = torch.sum(- x * log_x, dim=1)
        
        return torch.mean(entropy_x)
    
    def proximity_loss(self, A, S, adj_hop=1):
        """
        Compute graph structure preservation loss.

        Parameters
        ----------
        A : torch.Tensor
            Original adjacency matrix.
        S : torch.Tensor
            Assignment matrix.
        adj_hop : int, optional
            Number of hops. Default: 1.

        Returns
        -------
        torch.Tensor
            Proximity loss value.
        """
        eps = 1e-7
        num_nodes = S.size()[0]
        pred_adj0 = torch.matmul(S, S.T)
        tmp = pred_adj0
        pred_adj = pred_adj0
        for adj_pow in range(adj_hop - 1):
            tmp = tmp @ pred_adj0
            pred_adj = pred_adj + tmp

        pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(self.device))
        pos_adj = torch.log(pred_adj + eps)
        neg_adj = torch.log(1 - pred_adj + eps)
        num_entries = num_nodes * num_nodes

        if A.shape[0]<A.shape[1]:
            pos = torch.sum(pos_adj[A[0],A[1]])
            neg = torch.sum(neg_adj) - torch.sum(neg_adj[A[0],A[1]])
            link_loss = (-pos-neg) / float(num_entries)
        else:
            link_loss = -A * torch.log(pred_adj + eps) - (1 - A) * torch.log(1 - pred_adj + eps)
            link_loss = torch.sum(link_loss) / float(num_entries)

        return link_loss
    
    def label_matching(self, S, Y_old, Y_new):
        """
        Compute label consistency loss.

        Parameters
        ----------
        S : torch.Tensor
            Assignment matrix.
        Y_old : torch.Tensor
            Original labels.
        Y_new : torch.Tensor
            New labels.

        Returns
        -------
        torch.Tensor
            Label matching loss value.
        """
        n_node_old = S.shape[0]
        n_node_new = S.shape[1]
        label_matching_mat = torch.matmul(Y_old, Y_new.T)
        label_matching_mat = self.to_onehot(torch.argmax(label_matching_mat, dim=1), num_classes=n_node_new)
        S = self.to_onehot(torch.argmax(S, dim=1), num_classes=S.shape[1])
        c = torch.sum(label_matching_mat * S)

        return 1 - c / n_node_old

    def label_stable(self, S, Y_old, Y_new):
        """
        Compute label stability loss.

        Parameters
        ----------
        S : torch.Tensor
            Assignment matrix.
        Y_old : torch.Tensor
            Original labels.
        Y_new : torch.Tensor
            New labels.

        Returns
        -------
        torch.Tensor
            Label stability loss value.
        """
        n_class = Y_old.shape[1]
        label_stable_mat = torch.softmax(torch.matmul(torch.matmul(Y_old.T, S), Y_new), dim=1)
        pos = torch.mean(torch.diag(label_stable_mat))

        return 1 - pos
    
    def adj2coo(self, A):
        """
        Convert dense adjacency matrix to COO format.

        Parameters
        ----------
        A : torch.Tensor
            Dense adjacency matrix, shape (n_nodes, n_nodes).

        Returns
        -------
        tuple
            Contains:

            - torch.Tensor: Edge indices in COO format, shape (2, n_edges)
            - torch.Tensor: Edge weights, shape (n_edges,)

        """
        edge_weight = torch.squeeze(A.reshape(1, -1))
        row_elements = []
        col_elements = []
        n_node = A.shape[0]
        for i in range(n_node):
            row_elements.append(torch.Tensor([i] * n_node))
            col_elements.append(torch.arange(0, n_node))
        row = torch.hstack(row_elements)
        col = torch.hstack(col_elements)
        edge_index = torch.vstack([row, col]).long()

        return edge_index.to(self.device), edge_weight.to(self.device)
    
    def classwise_simple_mmd(self, source, target, src_y, tgt_y):
        """
        Compute class-wise Maximum Mean Discrepancy.

        Parameters
        ----------
        source : torch.Tensor
            Source domain features.
        target : torch.Tensor
            Target domain features.
        src_y : torch.Tensor
            Source domain labels (one-hot).
        tgt_y : torch.Tensor
            Target domain labels (one-hot).

        Returns
        -------
        float
            Sum of class-wise MMD values.

        """
        mmd = 0.
        for c in range(src_y.shape[1]):
            src_idx = src_y[:, c].to(torch.bool)
            src = source[src_idx]
            tgt_idx = tgt_y[:, c].to(torch.bool)
            tgt = target[tgt_idx]
            if not torch.isnan(self.simple_mmd(src, tgt)):
                mmd += self.simple_mmd(src, tgt)
        
        return mmd
    
    def simple_mmd(self, source, target):
        """
        Compute simple Maximum Mean Discrepancy.

        Parameters
        ----------
        source : torch.Tensor
            Source domain features.
        target : torch.Tensor
            Target domain features.

        Returns
        -------
        torch.Tensor
            L2 distance between mean feature vectors.

        """
        source = torch.mean(source, dim=0)
        target = torch.mean(target, dim=0)

        return torch.norm(source - target)
    
    def simple_mmd_kernel(self, source, target):
        """
        Compute kernel-based MMD with RBF kernel.

        Parameters
        ----------
        source : torch.Tensor
            Source domain features.
        target : torch.Tensor
            Target domain features.

        Returns
        -------
        torch.Tensor
            RBF kernel value between mean feature vectors.

        """
        source = torch.mean(source, dim=0)
        target = torch.mean(target, dim=0)

        return torch.exp(- 0.1 * torch.norm(source - target))
    
    def inference(self, data):
        """
        Perform inference on input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data containing:
            - x: Node features
            - edge_index: Edge indices

        Returns
        -------
        torch.Tensor
            Model predictions.

        Notes
        -----
        Simplified forward pass for inference:

        1. Single GNN layer
        2. Classification
        """
        edge_weight = torch.ones(data.edge_index.shape[1]).to(self.device)
        z = self.gnn_emb[0](data.x, data.edge_index, edge_weight)
        pred = self.cls_model(z)

        return pred
    
    def to_onehot(self, label_matrix, num_classes):
        """
        Convert label indices to one-hot encoding.

        Parameters
        ----------
        label_matrix : torch.Tensor
            Label indices.
        num_classes : int
            Number of classes.

        Returns
        -------
        torch.Tensor
            One-hot encoded labels.

        """
        identity = torch.eye(num_classes).to(self.device)
        onehot = torch.index_select(identity, 0, label_matrix)
        
        return onehot



