import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .reverse_layer import GradReverse


class BatchGraphConvolution(torch.nn.Module):
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
        # expand_weight = self.weight.expand(x.shape[0], -1, -1)
        output = torch.mm(adj, torch.mm(x, self.weight))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BatchMultiHeadGraphAttention(torch.nn.Module):
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
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BatchGIN(torch.nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(BatchGIN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lin_0 = nn.Linear(in_features, hidden_size)
        self.lin_1 = nn.Linear(hidden_size, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        h = x + torch.bmm(adj, x)
        h = self.dropout(self.act(self.lin_0(h)))
        h = self.lin_1(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GNN_VGAE_Encoder(nn.Module):
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
        if self.training:
            eps = torch.randn_like(lv)
            std = torch.exp(lv)
            return mu + eps * std
        else:
            return mu
    
    def vectorized_sym_norm(self, adjs):
        adjs += torch.eye(adjs.shape[1], device=adjs.device)
        inv_sqrt_D = 1.0 / adjs.sum(dim=-1, keepdim=True).sqrt()  # B x N x 1
        inv_sqrt_D[torch.isinf(inv_sqrt_D)] = 0.0
        normalized_adjs = (inv_sqrt_D * adjs) * inv_sqrt_D.transpose(0, 1)
        
        return normalized_adjs

    def forward(self, x, adj):
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
    def __init__(self, in_dim, hs, droprate):
        super(GraphDiscriminator, self).__init__()
        self.lin_0 = nn.Linear(in_dim, hs)
        self.lin_1 = nn.Linear(hs, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)

    def forward(self, x):
        h = self.act(self.dropout(self.lin_0(x)))
        h = torch.mean(h, dim=1)
        logits = self.lin_1(h)
        return logits


class GraphDecoder(nn.Module):
    def __init__(self, dec_hs, dim_d, dim_y, dim_m, droprate):
        super(GraphDecoder, self).__init__()
        self.d_lin0 = nn.Linear(dim_d, dim_d)
        self.y_lin0 = nn.Linear(dim_y, dim_y)
        self.m_lin0 = nn.Linear(dim_m, dim_m)
        self.dym_lin1 = nn.Linear(dim_d + dim_y + dim_m, dec_hs)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, d, y, m):
        d = self.dropout(self.act(self.d_lin0(d)))
        y = self.dropout(self.act(self.y_lin0(y)))
        m = self.dropout(self.act(self.m_lin0(m)))
        dym = torch.cat([d, y, m], dim=-1)
        dym = self.dym_lin1(dym)
        adj_recons = torch.mm(dym, dym.permute(1,0))
        return adj_recons


class NoiseDecoder(nn.Module):
    def __init__(self, dim_m, droprate):
        super(NoiseDecoder, self).__init__()
        self.m_lin0 = nn.Linear(dim_m, dim_m)
        self.m_lin1 = nn.Linear(dim_m, dim_m)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.dropout(self.act(self.m_lin0(x)))
        h = self.m_lin1(h)
        noise_recons = torch.mm(h, h.permute(1, 0))
        return noise_recons


class ClassClassifier(nn.Module):
    def __init__(self, hs, n_class, droprate):
        super(ClassClassifier, self).__init__()
        self.lin0 = nn.Linear(hs, hs)
        self.lin1 = nn.Linear(hs, n_class)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.dropout(self.act(self.lin0(x)))
        logits = self.lin1(h)
        return logits


class DomainClassifier(nn.Module):
    def __init__(self, dim_d):
        super(DomainClassifier, self).__init__()
        self.lin = nn.Linear(dim_d, 1)

    def forward(self, x):
        logits = self.lin(x)
        return logits


class DGDABase(nn.Module):
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
