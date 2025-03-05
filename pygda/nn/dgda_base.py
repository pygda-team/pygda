import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .reverse_layer import GradReverse


class BatchGraphConvolution(torch.nn.Module):
    """
    Batch-wise graph convolution layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, optional
        Whether to include bias. Default: True.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BatchGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)
        init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        Forward pass of the graph convolution layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape (num_nodes, in_features).
        adj : torch.Tensor
            Adjacency matrix, shape (num_nodes, num_nodes).

        Returns
        -------
        torch.Tensor
            Convolved node features, shape (num_nodes, out_features).

        """
        # expand_weight = self.weight.expand(x.shape[0], -1, -1)
        output = torch.mm(adj, torch.mm(x, self.weight))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """
        String representation of the module.

        Returns
        -------
        str
            String describing the layer's dimensions.

        Notes
        -----
        Format: "BatchGraphConvolution(in_features -> out_features)"
        Used for model printing and debugging.
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BatchMultiHeadGraphAttention(torch.nn.Module):
    """
    Multi-head graph attention layer with batch processing.

    Parameters
    ----------
    n_head : int
        Number of attention heads.
    in_features : int
        Number of input features.
    out_features : int
        Number of output features per head.
    attn_dropout : float
        Dropout rate for attention weights.

    """

    def __init__(self, n_head, in_features, out_features, attn_dropout):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.in_features = in_features
        self.out_features = out_features
        self.w = Parameter(torch.Tensor(n_head, in_features, out_features))
        self.a_src = Parameter(torch.Tensor(n_head, out_features, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, out_features, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        self.bias = Parameter(torch.Tensor(out_features))
        init.constant_(self.bias, 0)
        init.xavier_uniform_(self.w)
        init.xavier_uniform_(self.a_src)
        init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        """
        Forward pass of multi-head attention layer.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, num_nodes, in_features).
        adj : torch.Tensor
            Adjacency matrices, shape (batch_size, num_nodes, num_nodes).

        Returns
        -------
        torch.Tensor
            Attended node features, shape (batch_size, num_nodes, n_head * out_features).

        Notes
        -----
        Process:

        1. Linear projection to n_head spaces
        2. Compute attention scores
        3. Apply masked softmax
        4. Compute output
        """
        bs, n = x.size()[:2]  # x = (bs, n, in_dim)
        h_prime = torch.matmul(x.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(F.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(F.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~adj.unsqueeze(1)  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        output += self.bias
        output = output.view(bs, n, -1)
        return output
    
    def __repr__(self):
        """
        String representation of the module.

        Returns
        -------
        str
            String describing the layer's dimensions.

        Notes
        -----
        Format: "BatchMultiHeadGraphAttention(in_features -> out_features)"
        Used for model printing and debugging.
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BatchGIN(torch.nn.Module):
    """
    Batch Graph Isomorphism Network layer implementation.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_size : int
        Size of hidden layer.
    out_features : int
        Number of output features.

    """

    def __init__(self, in_features, hidden_size, out_features):
        super(BatchGIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin_0 = nn.Linear(in_features, hidden_size)
        self.lin_1 = nn.Linear(hidden_size, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        """
        Forward pass of GIN layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape (batch_size, num_nodes, in_features).
        adj : torch.Tensor
            Batch of adjacency matrices, shape (batch_size, num_nodes, num_nodes).

        Returns
        -------
        torch.Tensor
            Updated node features, shape (batch_size, num_nodes, out_features).

        Notes
        -----
        Process:

        1. Neighborhood aggregation with self-loop
        2. Two-layer MLP transformation
        3. Dropout regularization
        """
        h = x + torch.bmm(adj, x)
        h = self.dropout(self.act(self.lin_0(h)))
        h = self.lin_1(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GNN_VGAE_Encoder(nn.Module):
    """
    Variational Graph Auto-Encoder with multiple latent spaces.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hs : int
        Hidden state dimension.
    dim_d : int
        Domain space dimension.
    dim_y : int
        Label space dimension.
    dim_m : int
        Manipulation space dimension.
    droprate : float
        Dropout rate.
    backbone : str
        GNN backbone type ('gcn', 'gat', 'gin').

    Notes
    -----
    Encodes graph into three separate latent spaces:

    - Domain space (d)
    - Label space (y)
    - Manipulation space (m)
    """

    def __init__(self, in_dim, hs, dim_d, dim_y, dim_m, droprate, backbone='gcn'):
        super(GNN_VGAE_Encoder, self).__init__()
        self.backbone = backbone
        if backbone == 'gcn':
            self.gnn0 = BatchGraphConvolution(in_dim, hs)
            self.gnn1 = BatchGraphConvolution(hs, hs)
            self.d_gnn2 = BatchGraphConvolution(hs, 2 * dim_d)
            self.y_gnn2 = BatchGraphConvolution(hs, 2 * dim_y)
            self.m_gnn2 = BatchGraphConvolution(hs, 2 * dim_m)
        elif backbone == 'gat':
            self.gnn0 = BatchMultiHeadGraphAttention(1, in_dim, hs, 0.2)
            self.gnn1 = BatchMultiHeadGraphAttention(1, hs, hs, 0.2)
            self.d_gnn2 = BatchMultiHeadGraphAttention(1, hs, 2 * dim_d, 0.2)
            self.y_gnn2 = BatchMultiHeadGraphAttention(1, hs, 2 * dim_y, 0.2)
            self.m_gnn2 = BatchMultiHeadGraphAttention(1, hs, 2 * dim_m, 0.2)
        elif backbone == 'gin':
            self.gnn0 = BatchGIN(in_dim, hs, hs)
            self.gnn1 = BatchGIN(hs, hs, hs)
            self.d_gnn2 = BatchGIN(hs, hs, 2 * dim_d)
            self.y_gnn2 = BatchGIN(hs, hs, 2 * dim_y)
            self.m_gnn2 = BatchGIN(hs, hs, 2 * dim_m)
        else:
            raise NotImplementedError

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)
    
    def repara(self, mu, lv):
        """
        Reparameterization trick for VAE.

        Parameters
        ----------
        mu : torch.Tensor
            Mean vectors.
        lv : torch.Tensor
            Log variance vectors.

        Returns
        -------
        torch.Tensor
            Sampled vectors from the distribution.

        """
        if self.training:
            eps = torch.randn_like(lv)
            std = torch.exp(lv)
            return mu + eps * std
        else:
            return mu
    
    def vectorized_sym_norm(self, adjs):
        """
        Compute symmetric normalization of adjacency matrices.

        Parameters
        ----------
        adjs : torch.Tensor
            Batch of adjacency matrices, shape (batch_size, num_nodes, num_nodes).

        Returns
        -------
        torch.Tensor
            Normalized adjacency matrices.

        """
        adjs += torch.eye(adjs.shape[1], device=adjs.device)
        inv_sqrt_D = 1.0 / adjs.sum(dim=-1, keepdim=True).sqrt()  # B x N x 1
        inv_sqrt_D[torch.isinf(inv_sqrt_D)] = 0.0
        normalized_adjs = (inv_sqrt_D * adjs) * inv_sqrt_D.transpose(0, 1)
        
        return normalized_adjs

    def forward(self, x, adj):
        """
        Forward pass of the VGAE encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch_size, num_nodes, in_dim).
        adj : torch.Tensor
            Adjacency matrices, shape (batch_size, num_nodes, num_nodes).

        Returns
        -------
        tuple
            - dict: Contains latent representations and their parameters:
                - 'd', 'y', 'm': Sampled latent vectors
                - 'dmu', 'dlv': Domain space parameters
                - 'ymu', 'ylv': Label space parameters
                - 'mmu', 'mlv': Manipulation space parameters
            - torch.Tensor: Final hidden representations

        Notes
        -----
        Process:

        1. Graph normalization (for GCN backbone)
        2. Two-layer message passing
        3. Parallel encoding into three spaces
        4. Reparameterization for each space
        """
        if self.backbone == 'gcn':
            adj = self.vectorized_sym_norm(adj)
        res = dict()
        h = self.dropout(self.act(self.gnn0(x, adj)))
        h = self.dropout(self.act(self.gnn1(h, adj)))
        d = self.d_gnn2(h, adj)
        y = self.y_gnn2(h, adj)
        m = self.m_gnn2(h, adj)
        res['dmu'], res['dlv'] = d.chunk(chunks=2, dim=-1)
        res['ymu'], res['ylv'] = y.chunk(chunks=2, dim=-1)
        res['mmu'], res['mlv'] = m.chunk(chunks=2, dim=-1)
        res['d'] = self.repara(res['dmu'], res['dlv'])
        res['y'] = self.repara(res['ymu'], res['ylv'])
        res['m'] = self.repara(res['mmu'], res['mlv'])

        return res, h


class GraphDiscriminator(nn.Module):
    """
    Graph-level discriminator for adversarial training.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hs : int
        Hidden state dimension.
    droprate : float
        Dropout rate for regularization.

    Notes
    -----
    Implements a graph-level discriminator with:

    1. Feature transformation
    2. Mean pooling
    3. Binary classification
    """

    def __init__(self, in_dim, hs, droprate):
        super(GraphDiscriminator, self).__init__()
        self.lin_0 = nn.Linear(in_dim, hs)
        self.lin_1 = nn.Linear(hs, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)

    def forward(self, x):
        """
        Forward pass of graph discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input node features, shape (n_nodes, in_dim).

        Returns
        -------
        torch.Tensor
            Graph-level discrimination logits, shape (1,).

        Notes
        -----
        Process:
        
        1. Node feature transformation with dropout
        2. Mean pooling across nodes
        3. Final classification layer
        """
        h = self.act(self.dropout(self.lin_0(x)))
        h = torch.mean(h, dim=1)
        logits = self.lin_1(h)
        return logits


class GraphDecoder(nn.Module):
    """
    Multi-space graph decoder for structure reconstruction.

    Parameters
    ----------
    dec_hs : int
        Decoder hidden state dimension.
    dim_d : int
        Domain space dimension.
    dim_y : int
        Label space dimension.
    dim_m : int
        Manipulation space dimension.
    droprate : float
        Dropout rate for regularization.

    Notes
    -----
    Reconstructs graph structure from three disentangled spaces:
    
    - Domain space (d)
    - Label space (y)
    - Manipulation space (m)
    """

    def __init__(self, dec_hs, dim_d, dim_y, dim_m, droprate):
        super(GraphDecoder, self).__init__()
        self.d_lin0 = nn.Linear(dim_d, dim_d)
        self.y_lin0 = nn.Linear(dim_y, dim_y)
        self.m_lin0 = nn.Linear(dim_m, dim_m)
        self.dym_lin1 = nn.Linear(dim_d + dim_y + dim_m, dec_hs)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, d, y, m):
        """
        Forward pass of graph decoder.

        Parameters
        ----------
        d : torch.Tensor
            Domain space features, shape (n_nodes, dim_d).
        y : torch.Tensor
            Label space features, shape (n_nodes, dim_y).
        m : torch.Tensor
            Manipulation space features, shape (n_nodes, dim_m).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency matrix, shape (n_nodes, n_nodes).

        Notes
        -----
        Process:

        1. Independent transformation of each space
        2. Concatenation of transformed features
        3. Final projection to common space
        4. Symmetric matrix construction via outer product
        """
        d = self.dropout(self.act(self.d_lin0(d)))
        y = self.dropout(self.act(self.y_lin0(y)))
        m = self.dropout(self.act(self.m_lin0(m)))
        dym = torch.cat([d, y, m], dim=-1)
        dym = self.dym_lin1(dym)
        adj_recons = torch.mm(dym, dym.permute(1,0))
        return adj_recons


class NoiseDecoder(nn.Module):
    """
    Decoder network for reconstructing noise patterns in graph structure.

    Parameters
    ----------
    dim_m : int
        Dimension of manipulation space features.
    droprate : float
        Dropout rate for regularization.

    """

    def __init__(self, dim_m, droprate):
        super(NoiseDecoder, self).__init__()
        self.m_lin0 = nn.Linear(dim_m, dim_m)
        self.m_lin1 = nn.Linear(dim_m, dim_m)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of noise decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features from manipulation space, shape (n_nodes, dim_m).

        Returns
        -------
        torch.Tensor
            Reconstructed noise matrix, shape (n_nodes, n_nodes).

        Notes
        -----
        Process:

        1. Two-layer MLP transformation
        2. Symmetric matrix construction via outer product
        """
        h = self.dropout(self.act(self.m_lin0(x)))
        h = self.m_lin1(h)
        noise_recons = torch.mm(h, h.permute(1, 0))
        return noise_recons


class ClassClassifier(nn.Module):
    """
    Node classification network.

    Parameters
    ----------
    hs : int
        Hidden state dimension.
    n_class : int
        Number of output classes.
    droprate : float
        Dropout rate for regularization.

    Notes
    -----
    Two-layer MLP with dropout and ReLU activation.
    """

    def __init__(self, hs, n_class, droprate):
        super(ClassClassifier, self).__init__()
        self.lin0 = nn.Linear(hs, hs)
        self.lin1 = nn.Linear(hs, n_class)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (n_nodes, hs).

        Returns
        -------
        torch.Tensor
            Classification logits, shape (n_nodes, n_class).

        Notes
        -----
        Process:

        1. Hidden layer with dropout and ReLU
        2. Output layer for class logits
        """
        h = self.dropout(self.act(self.lin0(x)))
        logits = self.lin1(h)
        return logits


class DomainClassifier(nn.Module):
    """
    Binary domain classifier for adversarial training.

    Parameters
    ----------
    dim_d : int
        Dimension of domain space features.

    Notes
    -----
    Single linear layer for binary domain classification.
    Typically used with gradient reversal for domain adaptation.
    """

    def __init__(self, dim_d):
        super(DomainClassifier, self).__init__()
        self.lin = nn.Linear(dim_d, 1)

    def forward(self, x):
        """
        Forward pass of domain classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input features from domain space, shape (n_nodes, dim_d).

        Returns
        -------
        torch.Tensor
            Domain classification logits, shape (n_nodes, 1).

        Notes
        -----
        Simple linear projection for binary domain classification.
        Used in conjunction with gradient reversal layer during training.
        """
        logits = self.lin(x)
        return logits


class DGDABase(nn.Module):
    """
    Base class for DGDA.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    num_class : int
        Number of classes.
    enc_hs : int
        Encoder hidden size.
    dec_hs : int
        Decoder hidden size.
    dim_d : int
        Domain space dimension.
    dim_y : int
        Label space dimension.
    dim_m : int
        Manipulation space dimension.
    droprate : float
        Dropout rate.
    backbone : str
        GNN backbone type.
    source_pretrained_emb : torch.Tensor
        Pretrained embeddings for source domain.
    source_vertex_feats : torch.Tensor
        Vertex features for source domain.
    target_pretrained_emb : torch.Tensor
        Pretrained embeddings for target domain.
    target_vertex_feats : torch.Tensor
        Vertex features for target domain.

    Notes
    -----
    Implements graph disentanglement with:

    1. Feature augmentation with pretrained embeddings
    2. Multi-space encoding (domain, label, manipulation)
    3. Graph reconstruction and noise modeling
    4. Domain adversarial training
    """

    def __init__(
        self,
        in_dim,
        num_class,
        enc_hs,
        dec_hs,
        dim_d,
        dim_y,
        dim_m,
        droprate,
        backbone,
        source_pretrained_emb,
        source_vertex_feats,
        target_pretrained_emb,
        target_vertex_feats
        ):
        super(DGDABase, self).__init__()
        self.semb = nn.Embedding(source_pretrained_emb.size(0), source_pretrained_emb.size(1))
        self.semb.weight = Parameter(source_pretrained_emb, requires_grad=False)
        self.temb = nn.Embedding(target_pretrained_emb.size(0), target_pretrained_emb.size(1))
        self.temb.weight = Parameter(target_pretrained_emb, requires_grad=False)
        in_dim += int(source_pretrained_emb.size(1))

        self.svf = nn.Embedding(source_vertex_feats.size(0), source_vertex_feats.size(1))
        self.svf.weight = Parameter(source_vertex_feats, requires_grad = False)
        self.tvf = nn.Embedding(target_vertex_feats.size(0), target_vertex_feats.size(1))
        self.tvf.weight = Parameter(target_vertex_feats, requires_grad = False)
        in_dim += int(source_vertex_feats.size(1))

        self.encoder = GNN_VGAE_Encoder(in_dim, enc_hs, dim_d, dim_y, dim_m, droprate, backbone)
        self.graphDiscriminator = GraphDiscriminator(enc_hs, enc_hs // 2, droprate)
        self.graph_decoder = GraphDecoder(dec_hs, dim_d, dim_y, dim_m, droprate)
        self.noise_decoder = NoiseDecoder(dim_m, droprate)
        self.classClassifier = ClassClassifier(dim_y, num_class, droprate)
        self.domainClassifier = DomainClassifier(dim_d)

    def forward(self, x, vts, adj, domain, recon=True, alpha=1.0):
        """
        Forward pass of DGDA model.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        vts : torch.Tensor
            Vertex indices.
        adj : torch.Tensor
            Adjacency matrix.
        domain : int
            Domain indicator (0: source, 1: target).
        recon : bool, optional
            Whether to compute reconstruction. Default: True.
        alpha : float, optional
            Gradient reversal scaling. Default: 1.0.

        Returns
        -------
        dict
            Contains:
            - 'd', 'y', 'm': Latent representations
            - 'a_recons': Graph reconstruction
            - 'm_recons': Noise reconstruction
            - 'dom_output': Domain predictions
            - 'cls_output': Class predictions

        Notes
        -----
        Process flow:
        
        1. Feature augmentation
        2. Graph encoding
        3. Reconstruction (optional)
        4. Domain and class prediction
        """
        if domain == 0:
            x = torch.cat((x, self.semb(vts)), dim=-1)
            x = torch.cat((x, self.svf(vts)), dim=-1)
        else:
            x = torch.cat((x, self.temb(vts)), dim=-1)
            x = torch.cat((x, self.tvf(vts)), dim=-1)

        res, h = self.encoder(x, adj)

        if recon:
            res['a_recons'] = self.graph_decoder(res['d'], res['y'], res['m'])
            res['m_recons'] = self.noise_decoder(res['m'])
        
        res['dom_output'] = self.domainClassifier(GradReverse.apply(res['d'], alpha))
        res['cls_output'] = self.classClassifier(res['y'])

        return res
