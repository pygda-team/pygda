import torch
import time
import copy
import torch.nn as nn

import numpy as np
import torch_geometric
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from torch_geometric.nn.dense.linear import Linear

from .gnn_base import GNNBase


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=10):
        """
        mode:
            'None' : No normalization
            'PN'   : Original version
            'PN-SI'  : Scale-Individually version
            'PN-SCS' : Scale-and-Center-Simultaneously version
            ('SCS'-mode is not in the paper but we found it works well in practice, especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        
        return x


class GraphEncoder(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden=64,
        layer_num=2,
        root_weight=True,
        norm_mode='PN-SCS',
        norm_scale=1,
        log_softmax=False
        ):
        super(GraphEncoder, self).__init__()
        self.convs = nn.ModuleList()
        if layer_num == 1:
            self.convs.append(SAGEConv(dim_in, dim_out, root_weight=root_weight))
        else:
            for num in range(layer_num):
                if num == 0:
                    self.convs.append(SAGEConv(dim_in, dim_hidden, root_weight=root_weight))
                elif num == layer_num - 1:
                    self.convs.append(SAGEConv(dim_hidden, dim_out, root_weight=root_weight))
                else:
                    self.convs.append(SAGEConv(dim_hidden, dim_hidden, root_weight=root_weight))
        
        self.norm = PairNorm(mode=norm_mode, scale=norm_scale)
        self.log_softmax = log_softmax

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        adj_sp = edge_index
        for ind, conv in enumerate(self.convs):
            if ind == len(self.convs) - 1:
                x = conv(x, adj_sp)
            else:
                x = conv(x, adj_sp)
                x = self.norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        if self.log_softmax:
            x = F.log_softmax(x, dim=1)
        
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layer=2,
        use_norm=False,
        dropout=0.5,
        act_fn='relu',
        norm_mode='PN',
        norm_scale=1.
        ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        self.use_norm = use_norm
        self.dropout = dropout
        self.num_layer = num_layer

        # linear layers
        if num_layer == 1:
            self.layers.append(Linear(dim_in, dim_out, bias=True)) # input layer
        else:
            self.layers.append(Linear(dim_in, dim_hidden, bias=True)) # input layer
            for _ in range(num_layer-2):
                self.layers.append(Linear(dim_hidden, dim_hidden, bias=True))
            self.layers.append(Linear(dim_hidden, dim_out, bias=True)) # input layer

        # bn
        if use_norm:
            self.pair_norm = PairNorm(norm_mode, norm_scale)
                
        # activation function
        self.act_fn = act_fn
        if act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'leakyrelu':
            self.act_fn = nn.LeakyReLU(0.2, inplace=False)
        elif act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        elif act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        else:
            raise NotImplementedError('Not Implemented Activation Function:{}'.format(act_fn))
        
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
    
    def forward(self, z):
        x = z
        for idx in range(self.num_layer - 1):
            x = self.layers[idx](x)
            if self.use_norm:
                x = self.pair_norm(x)
            x = self.act_fn(x)
        recons = self.layers[-1](x)
        
        return recons


class Discriminator(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        num_layer=2,
        use_bn=False,
        use_pair_norm=False,
        dropout=0.5,
        act_fn='leakyrelu',
        sigmoid_output=True,
        norm_mode='PN',
        norm_scale=1.
        ):
        super(Discriminator, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.num_layer = num_layer
        self.sigmoid_output = sigmoid_output

        # linear layers
        if num_layer == 1:
            self.layers.append(Linear(dim_in, 1, bias=True)) # input layer
        else:
            self.layers.append(Linear(dim_in, dim_hidden, bias=True)) # input layer
            for _ in range(num_layer-2):
                self.layers.append(Linear(dim_hidden, dim_hidden, bias=True))
            self.layers.append(Linear(dim_hidden, 1, bias=True)) # input layer
        self.use_pair_norm = use_pair_norm
        if use_pair_norm:
            self.pair_norm = PairNorm(mode=norm_mode, scale=norm_scale)

        # bn
        if use_bn:
            self.bns = torch.nn.ModuleList()
            for _ in range(num_layer-1):
                self.bns.append(nn.BatchNorm1d(dim_hidden))
                
        # activation function
        self.act_fn = act_fn
        if act_fn == 'relu':
            self.act_fn = nn.ReLU()
        elif act_fn == 'leakyrelu':
            self.act_fn = nn.LeakyReLU(0.2, inplace=False)
        elif act_fn == 'tanh':
            self.act_fn = nn.Tanh()
        elif act_fn == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        else:
            raise NotImplementedError('Not Implemented Activation Function:{}'.format(act_fn))
        self.reset_parameters()
    
    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, z):
        x = z
        for idx in range(self.num_layer-1):
            x = self.layers[idx](x)
            if self.use_bn:
                x = self.bns[idx](x)
            elif self.use_pair_norm:
                x = self.pair_norm(x)
            x = self.act_fn(x)
        logits = self.layers[-1](x)
        probs = torch.sigmoid(logits) if self.sigmoid_output else logits

        return probs


class Similar(torch.nn.Module):
    def __init__(self, in_channels, num_clf_classes, dropout=0.6, use_clf=True):
        super(Similar, self).__init__()
        self.biasatt = nn.Sequential(
            Linear(128, 64, bias=True, weight_initializer='glorot'),
            nn.Tanh(),   
            Linear(64, 128, bias=True, weight_initializer='glorot'),
        )

        for m in self.biasatt:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.use_clf = use_clf
        if use_clf:
            self.lin_clf = Linear(in_channels, num_clf_classes, bias=True, weight_initializer='glorot')

        self.lin_self = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            Linear(in_channels, 64, bias=False, weight_initializer='glorot'),
            nn.BatchNorm1d(64),
            nn.Tanh(),  
            Linear(64, 128, bias=False, weight_initializer='glorot'),
        )

        self.dropout = dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.use_clf:
            self.lin_clf.reset_parameters()

        for m in self.lin_self:
            if isinstance(m, nn.Linear):
                m.reset_parameters()
    
    def similarity_cross_domain(self, x_src, x_tar, idx1, idx2):
        z_src = self.lin_self(x_src)
        z_tar = self.lin_self(x_tar)
        alpha = torch.nn.CosineSimilarity(dim=-1)((z_src[idx1] + self.biasatt(z_src[idx1])).unsqueeze(1), z_tar[idx2] + self.biasatt(z_tar[idx2]))
        alpha = torch.sigmoid(alpha)

        return alpha

    def forward_cross_domain(self, x_src, x_tar, idx1, idx2):
        z_src, z_tar = x_src, x_tar
        log_probs_clf_src = log_probs_clf_tar = None

        if self.use_clf:
            logits_src = self.lin_clf(F.dropout(F.relu(z_src), p=self.dropout, training=self.training))
            logits_tar = self.lin_clf(F.dropout(F.relu(z_tar), p=self.dropout, training=self.training))
            log_probs_clf_src = F.log_softmax(logits_src, dim=-1)
            log_probs_clf_tar = F.log_softmax(logits_tar, dim=-1)

        alpha = self.similarity_cross_domain(z_src, z_tar, idx1, idx2)

        return alpha.unsqueeze(-1), log_probs_clf_src, log_probs_clf_tar

    def similarity(self, x, idx1, idx2):
        z = self.lin_self(x)
        alpha = torch.nn.CosineSimilarity(dim=1)(z[idx1] + self.biasatt(z[idx1]), z[idx2] + self.biasatt(z[idx2]))
        alpha = torch.sigmoid(alpha)

        return alpha

    def forward(self, x, idx1, idx2):
        z = x
        log_probs_clf = None

        if self.use_clf:
            logits = self.lin_clf(F.dropout(F.relu(z), p=self.dropout, training=self.training))
            log_probs_clf = F.log_softmax(logits, dim=-1)

        alpha = self.similarity(z, idx1, idx2)

        return alpha.unsqueeze(-1), log_probs_clf


class SourceLearner(torch.nn.Module):
    def __init__(
        self,
        data,
        dim_hidden=64,
        norm_mode='None',
        norm_scale=1,
        use_clf=True
        ):
        super(SourceLearner, self).__init__()
        self.dim_in = data.num_features
        self.num_classes = data.y.max().item() + 1
        self.dim_hidden = dim_hidden

        self.backbone = GraphEncoder(self.dim_in, self.dim_hidden, dim_hidden=self.dim_hidden, \
            layer_num=2, root_weight=True, norm_mode=norm_mode, norm_scale=norm_scale, log_softmax=False)

        self.sim_net = Similar(self.dim_hidden, num_clf_classes=data.y.max().item()+1, dropout=0.6, use_clf=use_clf)

        self.reset_parameters()

    def reset_parameters(self):
        self.backbone.reset_parameters()
        self.sim_net.reset_parameters()
    
    def forward(self, data, idx1, idx2, return_representation=False):
        h = self.backbone(data.x, data.edge_index)
        probs_pair, logits_clf = self.sim_net(h, idx1, idx2)

        if return_representation:
            return probs_pair, logits_clf, h
        else:
            return probs_pair, logits_clf


class TargetLearnerAE(torch.nn.Module):
    
    def __init__(self, data, dim_eq_trans=128, dim_hidden=64, norm_mode='None', norm_scale=1):
        super(TargetLearnerAE, self).__init__()
        self.dim_in = data.num_features
        self.dim_eq_trans = dim_eq_trans
        self.num_classes = data.y.max().item() + 1
        self.dim_hidden = dim_hidden

        self.equavilent_trans_layer = nn.Sequential(
            Linear(self.dim_in, dim_eq_trans, bias=True),
            PairNorm(mode=norm_mode, scale=norm_scale),
            nn.Tanh()
        )

        self.encoder = GraphEncoder(dim_eq_trans, dim_hidden, dim_hidden=dim_hidden, \
            layer_num=2, root_weight=True, norm_mode=norm_mode, norm_scale=norm_scale, log_softmax=False)

        self.decoder = Decoder(dim_hidden, dim_hidden, dim_eq_trans, num_layer=2, \
            use_norm=True, dropout = 0.5, act_fn = 'relu', norm_mode=norm_mode, norm_scale=norm_scale)

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
    
    def encode(self, data):
        h0 = self.equavilent_trans_layer(data.x)
        z = self.encoder(h0, data.edge_index)

        return z, h0
    
    def decode(self, z):
        recons = self.decoder(z)
        recons = torch.tanh(recons)

        return recons
    
    def forward(self, data):
        z, h0 = self.encode(data)
        recons = self.decode(z)

        return h0, z, recons


class AdversarialLearner(nn.Module):
    def __init__(
        self,
        data_src,
        data_tar,
        dim_hidden=64,
        num_layer=2,
        source_clf=True,
        norm_mode='PN',
        norm_scale=1.
        ):
        super(AdversarialLearner, self).__init__()

        self.num_layer = num_layer
        self.source_clf = source_clf

        self.source_learner = SourceLearner(
            data_src,
            dim_hidden=dim_hidden,
            norm_mode=norm_mode,
            norm_scale=norm_scale,
            use_clf=source_clf
        )

        self.target_learner = TargetLearnerAE(
            data_tar,
            dim_eq_trans=128,
            dim_hidden=dim_hidden,
            norm_mode=norm_mode,
            norm_scale=norm_scale
        )

        self.discriminator = Discriminator(
            dim_hidden,
            dim_hidden,
            num_layer=2,
            use_pair_norm=False,
            dropout=0.5,
            act_fn='relu',
            sigmoid_output=True,
            norm_mode=norm_mode,
            norm_scale=norm_scale
        )

    def get_probs_within_domain(self, data, idx1, idx2, domain='target'):
        if domain == 'source':
            probs_pair, log_probs_clf = self.source_learner(data, idx1, idx2, return_representation=False)
        elif domain == 'target':
            z, _ = self.target_learner.encode(data)
            probs_pair, log_probs_clf = self.source_learner.sim_net(z, idx1, idx2)
        if not self.source_clf:
            log_probs_clf = torch.zeros((data.x.shape[0], data.y.max().item() + 1))

        return probs_pair, log_probs_clf.exp()

    def get_probs_cross_domain(self, data_src, data_tar, idx1, idx2, return_representation=False):
        z_src = self.source_learner.backbone(data_src.x, data_src.edge_index)
        z_tar, _ = self.target_learner.encode(data_tar)
        probs_pair, log_probs_clf_src, log_probs_clf_tar = self.source_learner.sim_net.forward_cross_domain(z_src, z_tar, idx1, idx2)
        
        if not self.source_clf:
            log_probs_clf_src = torch.zeros((z_src.shape[0], data_src.y.max().item()+1))
            log_probs_clf_tar = torch.zeros((z_tar.shape[0], data_tar.y.max().item()+1))
        
        if return_representation:
            return probs_pair, log_probs_clf_src.exp(), log_probs_clf_tar.exp(), z_src.detach(), z_tar.detach()
        else:
            return probs_pair, log_probs_clf_src.exp(), log_probs_clf_tar.exp()


class PairEnumerator:
    def __init__(self, data, mode='train'):
        super(PairEnumerator, self).__init__()
        self.num_classes = data.y.max().item() + 1 # y start from 0, -1 denotes missing
        self.class_bucket = {} # class->indexes
        idx2lbl = torch.Tensor(list(enumerate(data.y.tolist()))).long().transpose(0, 1) # tmp variable
        self.mode = mode
        for lbl in range(self.num_classes):
            if mode == 'train':
                self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.train_mask.cpu()]
            elif mode == 'val':
                self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.val_mask.cpu()]
            elif mode == 'test':
                self.class_bucket[lbl] = idx2lbl[0][(idx2lbl[1] == lbl) * data.test_mask.cpu()]
            elif mode == 'all':
                mask_all = (data.train_mask + data.val_mask + data.test_mask).cpu()
                self.class_bucket[lbl] = idx2lbl[0][idx2lbl[1] == lbl * mask_all]
            else:
                raise NotImplementedError('Not Implemented Mode:{}'.format(mode))
    
    def balanced_sampling(self, max_class_num=2, sample_size=40000, shuffle=True):        
        if self.num_classes > max_class_num:
            selected_classes = np.random.choice(torch.arange(self.num_classes), replace=False, size=max_class_num) # sampling classes without putback
        else:
            selected_classes = np.arange(self.num_classes).astype(np.int8)
        # same-class pair generation
        sample_idxs_1 = []
        sample_idxs_2 = []
        sample_per_class_same = int(0.5 * sample_size / max_class_num)
        sample_per_class_diff = int(0.5 * sample_size / (max_class_num * (max_class_num - 1)))
        for lbl_1 in selected_classes:
            for lbl_2 in selected_classes:
                if lbl_1 == lbl_2:
                    idx1 = torch.from_numpy(np.random.choice(self.class_bucket[lbl_1], size=sample_per_class_same)).long() # sampling with putback
                    idx2 = torch.from_numpy(np.random.choice(self.class_bucket[lbl_2], size=sample_per_class_same)).long() # sampling with putback
                else:
                    idx1 = torch.from_numpy(np.random.choice(self.class_bucket[lbl_1], size=sample_per_class_diff)).long() # sampling with putback
                    idx2 = torch.from_numpy(np.random.choice(self.class_bucket[lbl_2], size=sample_per_class_diff)).long() # sampling with putback
                # pair_idxs = torch.stack((idx1, idx2), dim=0) # 2 * pair_num
                sample_idxs_1.append(idx1) 
                sample_idxs_2.append(idx2)
        sample_idxs_1 = torch.cat(sample_idxs_1, dim=0)
        sample_idxs_2 = torch.cat(sample_idxs_2, dim=0)

        # shuffle
        if shuffle:
            shuffle_idx = torch.arange(sample_idxs_1.shape[0]).tolist()
            random.shuffle(shuffle_idx)
            sample_idxs_1 = sample_idxs_1[shuffle_idx]
            sample_idxs_2 = sample_idxs_1[shuffle_idx]

        return sample_idxs_1, sample_idxs_2
    
    def pair_enumeration(self, x1, x2):
        '''
            input:  [B,D]
            return: [B*B,D]
            input  [[a],
                    [b]]
            return [[a,a],
                    [b,a],
                    [a,b],
                    [b,b]]
        '''
        assert x1.ndimension() == 2 and x2.ndimension() == 2, 'Input dimension must be 2'
        # [a,b,c,a,b,c,a,b,c]
        # [a,a,a,b,b,b,c,c,c]
        x1_ = x1.repeat(x2.size(0), 1)
        x2_ = x2.repeat(1, x1.size(0)).view(-1, x1.size(1))
        
        return torch.cat((x1_, x2_), dim=1)

    def sampling(self, max_class_num=2, sample_size=40000, shuffle=True):
        if self.num_classes > max_class_num:
            selected_classes = np.random.choice(torch.arange(self.num_classes), replace=False, size=max_class_num) # sampling classes without putback
        else:
            selected_classes = np.arange(self.num_classes).astype(np.int8)
        sample_idxs_1 = []
        sample_idxs_2 = []
        sample_per_class = int(np.sqrt(sample_size) / max_class_num)
        for lbl in selected_classes:
            sample_idxs_1.append(torch.from_numpy(np.random.choice(self.class_bucket[lbl], size=sample_per_class)).long()) # sampling with putback
            sample_idxs_2.append(torch.from_numpy(np.random.choice(self.class_bucket[lbl], size=sample_per_class)).long()) # sampling with putback
        sample_idxs_1 = torch.cat(sample_idxs_1) # (sample_per_class,)
        sample_idxs_2 = torch.cat(sample_idxs_2) # (sample_per_class,)
        # enumeration
        pair_idxs = self.pair_enumeration(sample_idxs_1.unsqueeze(1), sample_idxs_2.unsqueeze(1)).transpose(0, 1) # (2, sample_size)
        if shuffle:
            shuffle_idx = torch.arange(pair_idxs.shape[1]).tolist()
            random.shuffle(shuffle_idx)
            sample_idxs_1 = pair_idxs[0][shuffle_idx]
            sample_idxs_2 = pair_idxs[1][shuffle_idx]
        else:
            sample_idxs_1 = pair_idxs[0]
            sample_idxs_2 = pair_idxs[1]
        
        return sample_idxs_1, sample_idxs_2


class BridgedGraph(torch.nn.Module):
    def __init__(
        self,
        data_src,
        data_tar,
        k_cross=20,
        k_within=6,
        epsilon=0.5,
        dim_hidden=64,
        batch_size=1000,
        num_layer=2,
        num_epoch=200,
        lr=0.001,
        weight_decay=5e-3,
        source_clf=True,
        norm_mode='PN',
        norm_scale=1.,
        device=None
        ):
        super(BridgedGraph, self).__init__()

        self.data_src = data_src
        self.data_tar = data_tar

        self.k_cross = k_cross
        self.k_within = k_within
        self.epsilon = epsilon

        self.dim_hidden = dim_hidden
        self.num_layer = num_layer
        self.num_epoch = num_epoch
        self.source_clf = source_clf
        self.norm_mode = norm_mode
        self.norm_scale = norm_scale
        self.device = device

        self.model = AdversarialLearner(
            self.data_src,
            self.data_tar,
            dim_hidden=self.dim_hidden,
            num_layer=self.num_layer,
            source_clf=self.source_clf,
            norm_mode=self.norm_mode,
            norm_scale=self.norm_scale
            )

        self.optimizer_src_tar = torch.optim.Adam(
            [
                {'params': self.model.source_learner.parameters(), 'lr': 1e-3, 'weight_decay': 5e-3},
                {'params': self.model.target_learner.parameters(), 'lr': lr, 'betas': (0.5, 0.999), 'weight_decay': 5e-3}
            ])
        
        self.optimizer_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=5e-3)
    
    def train_adv_few_shot(
        self,
        epoch,
        pair_enumerator_src_train=None,
        pair_enumerator_tar_train=None,
        pair_enumerator_cross_train=None,
        max_class_num=2,
        sample_size=40000,
        use_clf=False
        ):
        self.model.train()
        # 1. Train Similarity-Learner & Auto-Encoder
        # within-source domain similarity
        self.optimizer_src_tar.zero_grad()
        idx1_src, idx2_src = pair_enumerator_src_train.sampling(max_class_num=max_class_num, sample_size=sample_size, shuffle=False)
        probs_pair_src, log_probs_clf_src, h_src = self.model.source_learner(self.data_src, idx1_src, idx2_src, return_representation=True)
        y_pair_src = (self.data_src.y[idx1_src] == self.data_src.y[idx2_src]).float().unsqueeze(-1)
        loss_sim_within_src = F.binary_cross_entropy(probs_pair_src, y_pair_src)
        h0_tar, h_tar, recons = self.model.target_learner(self.data_tar)
        loss_recons = F.mse_loss(recons, h0_tar)
        g_labels = torch.ones((h_tar.shape[0], 1)).to(h_tar.device)
        loss_g = F.binary_cross_entropy(self.model.discriminator(h_tar), g_labels)
        loss_ae = loss_g + loss_recons * 0.1

        loss_sim = loss_sim_within_src + loss_ae
        
        if use_clf:
            loss_clf_src = F.nll_loss(log_probs_clf_src[self.data_src.train_mask], self.data_src.y[self.data_src.train_mask])
            loss_sim += loss_clf_src
            print('Loss_sim:{:.4f} | Loss_clf_src:{:.4f}'.format(loss_sim.detach().cpu().item(), loss_clf_src.detach().cpu().item()))
        loss_sim.backward()
        self.optimizer_src_tar.step()
        
        # 2. Train Discriminator
        self.optimizer_d.zero_grad()
        real_labels = torch.ones((h_src.shape[0], 1)).to(h_src.device)
        fake_labels = torch.zeros((h_tar.shape[0], 1)).to(h_tar.device)
        real_loss = F.binary_cross_entropy(self.model.discriminator(h_src.detach()), real_labels)
        fake_loss = F.binary_cross_entropy(self.model.discriminator(h_tar.detach()), fake_labels)
        loss_d = (real_loss + fake_loss) / 2
        loss_d.backward()
        self.optimizer_d.step()

        return loss_sim.detach().item(), loss_d.detach().item(), loss_ae.detach().item(), loss_g.detach().item(), loss_recons.detach().item()

    def fit(self):
        self.data_src = self.data_src.to(self.device)
        self.data_tar = self.data_tar.to(self.device)
        self.model = self.model.to(self.device)

        pair_enumerator_src_train = PairEnumerator(self.data_src, mode='train')
        
        for epoch in range(1, 1 + self.num_epoch):
            t0 = time.time()
            loss_sim, loss_d, loss_ae, loss_g, loss_recons = self.train_adv_few_shot(
                epoch,
                pair_enumerator_src_train=pair_enumerator_src_train,
                max_class_num=2,
                sample_size=40000,
                use_clf=self.source_clf)
            print(''.format(loss_g, loss_recons))

            log_ae = '[AE]Epoch: {:03d}, Loss_ae:{:.4f} | Loss_recons:{:.4f} | Loss_g:{:.4f} | Loss_d:{:.4f}  Time(s/epoch):{:.4f}'.format(
                epoch, loss_ae, loss_recons, loss_g, loss_d, time.time() - t0)
            
            print(log_ae)
        
    def pair_enumeration(self, x1, x2):
        '''
            input:  [B,D]
            return: [B*B,D]
            input  [[a],
                    [b]]
            return [[a,a],
                    [b,a],
                    [a,b],
                    [b,b]]
        '''
        assert x1.ndimension() == 2 and x2.ndimension() == 2, 'Input dimension must be 2'
        # [a,b,c,a,b,c,a,b,c]
        # [a,a,a,b,b,b,c,c,c]
        x1_ = x1.repeat(x2.size(0), 1)
        x2_ = x2.repeat(1, x1.size(0)).view(-1, x1.size(1))
        
        return torch.cat((x1_, x2_), dim=1)
    
    def add_topk_sim_cross_domain_edges_full(self, k=3):
        num_src_nodes = self.data_src.x.shape[0]
        num_tar_nodes = self.data_tar.x.shape[0]
        all_idx_src = torch.arange(num_src_nodes).unsqueeze(-1).to(self.device)
        all_idx_tar = torch.arange(num_tar_nodes).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            self.model.eval()
            probs_pair, probs_clf_src, probs_clf_tar, _, _ = self.model.get_probs_cross_domain(self.data_src, self.data_tar, all_idx_src, all_idx_tar, return_representation=True)
            sim_mat = probs_pair.squeeze(-1).view(-1, num_tar_nodes) # num_src_nodes * num_tar_nodes
            H, W = sim_mat.shape
            mat = sim_mat.view(-1)
            e_sim_mat, indices = mat.topk(k=k)
            edge_index_added = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1).t()

            return torch_geometric.utils.coalesce(edge_index_added), e_sim_mat, indices, probs_clf_src, probs_clf_tar
    
    def add_topk_sim_cross_domain_edges(self, epsilon=0.5, k=3, batch_size=1000):
        num_src_nodes = self.data_src.x.shape[0]
        num_tar_nodes = self.data_tar.x.shape[0]
        all_idx_src = torch.arange(num_src_nodes).unsqueeze(-1).to(self.device)
        start_idx = 0
        edge_index_bucket = []
        e_sim_mat = []
        idx_src_mat = []

        while start_idx < num_tar_nodes:
            # left-close & right-open interval
            end_idx = min(start_idx + batch_size, num_tar_nodes)
            batch_idx_tar = torch.arange(start_idx, end_idx, step=1).unsqueeze(-1).to(self.device)
            pair_idxs = self.pair_enumeration(all_idx_src, batch_idx_tar).transpose(0, 1)
        
            idx1, idx2 = pair_idxs[0], pair_idxs[1]
            # print(idx1, idx2)
            # Do not need to remove self-loops for adding cross-domain edges, but need that step for adding within-domain edges
            with torch.no_grad():
                self.model.eval()
                probs_pair, probs_clf_src, probs_clf_tar, _, _ = self.model.get_probs_cross_domain(self.data_src, self.data_tar, idx1, idx2, return_representation=True)
                # leave double-check by probs_clf_src and probls_clf_tar as further work
                # leave using edge weighted graph as further work
                sim_mat = probs_pair.squeeze(-1).view(-1, num_src_nodes) # bs * num_src_nodes
                topk_sim = sim_mat.topk(k=k, dim=1, largest=True, sorted=False)
                topk_idx_tar = torch.cat([batch_idx_tar for _ in range(k)], dim=1).view(-1)
                topk_idx_src = topk_sim.indices.view(-1) # .cpu()
                edge_index_topk_batch = torch.stack((topk_idx_src, topk_idx_tar), dim=0)
                edge_index_bucket.append(edge_index_topk_batch)
                e_sim_mat.append(topk_sim.values) # .cpu
                idx_src_mat.append(topk_sim.indices) # .cpu()
            start_idx = end_idx
        edge_index_added = torch.cat(edge_index_bucket, dim=1)
        e_sim_mat = torch.cat(e_sim_mat, dim=0)
        idx_src_mat = torch.cat(idx_src_mat, dim=0)
        
        return torch_geometric.utils.coalesce(edge_index_added), e_sim_mat, idx_src_mat, probs_clf_src, probs_clf_tar

    def add_topk_sim_within_domain_edges_full(self, data_src, k=3, domain='source'):
        num_src_nodes = data_src.x.shape[0]
        all_idx_src = torch.arange(num_src_nodes).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            self.model.eval()
            probs_pair, _  = self.model.get_probs_within_domain(data_src, all_idx_src, all_idx_src, domain=domain)
            sim_mat = probs_pair.squeeze(-1).view(-1, num_src_nodes)
            H, W = sim_mat.shape
            mat = sim_mat.view(-1)
            e_sim_mat, indices = mat.topk(k=k)
            edge_index_added = torch.cat(((indices // W).unsqueeze(1), (indices % W).unsqueeze(1)), dim=1).t()

            return torch_geometric.utils.coalesce(edge_index_added), e_sim_mat, indices
    
    def add_topk_sim_within_domain_edges(self, data_src, k=3, batch_size=1000, domain='source'):
        # Possible edge types of src-tar cross-domain edges: 
        #   * train-train; train-val; train-test;
        #   * val-val, val-test, test-test
        num_src_nodes = data_src.x.shape[0]
        all_idx_src = torch.arange(num_src_nodes).unsqueeze(-1).to(self.device)

        # 1. construct source-graph
        start_idx = 0
        edge_index_bucket_src = []
        e_sim_mat_src = []
        idx_src_mat_src = []

        while start_idx < num_src_nodes:
            # left-close & right-open interval
            end_idx = min(start_idx + batch_size, num_src_nodes)
            batch_idx_src = torch.arange(start_idx, end_idx, step=1).unsqueeze(-1).to(self.device)
            pair_idxs = self.pair_enumeration(all_idx_src, batch_idx_src).transpose(0, 1)
        
            idx1, idx2 = pair_idxs[0], pair_idxs[1]
            # Do not need to remove self-loops for adding cross-domain edges, but need that step for adding within-domain edges
            with torch.no_grad():
                self.model.eval()
                probs_pair, _ = self.model.get_probs_within_domain(data_src, idx1, idx2, domain=domain)
                # leave double-check by probs_clf_src and probls_clf_tar as further work
                # leave using edge weighted graph as further work
                sim_mat = probs_pair.squeeze(-1).view(-1, num_src_nodes) # bs * num_src_nodes
                topk_sim = sim_mat.topk(k=k, dim=1, largest=True, sorted=False)
                topk_idx_to = torch.cat([batch_idx_src for _ in range(k)], dim=1).view(-1)
                topk_idx_from = topk_sim.indices.view(-1) # .cpu()
                edge_index_topk_batch = torch.stack((topk_idx_from, topk_idx_to), dim=0)
                edge_index_bucket_src.append(edge_index_topk_batch)
                e_sim_mat_src.append(topk_sim.values) # .cpu()
                idx_src_mat_src.append(topk_sim.indices) # .cpu()
            start_idx = end_idx
        edge_index_added_src = torch.cat(edge_index_bucket_src, dim=1)
        edge_index_added_src = torch_geometric.utils.coalesce(edge_index_added_src) # remove duplicated edges
        e_sim_mat_src = torch.cat(e_sim_mat_src, dim=0)
        idx_src_mat_src = torch.cat(idx_src_mat_src, dim=0)

        return edge_index_added_src, e_sim_mat_src, idx_src_mat_src
    
    def merge_graphs(
        self,
        edge_index_cross_added,
        edge_index_added_src=None,
        edge_index_added_tar=None
        ):
        N_src = self.data_src.x.shape[0]
        N_tar = self.data_tar.x.shape[0]
        N = N_src + N_tar
        x = torch.cat((self.data_src.x, self.data_tar.x), dim=0) # .cpu()
        edge_index_src_ori = self.data_src.edge_index # .cpu()
        edge_index_tar_ori = (self.data_tar.edge_index + N_src)# .cpu()
        edge_index_cross_added[1, :] += N_src
        edge_index = torch.cat((edge_index_src_ori, edge_index_tar_ori, edge_index_cross_added), dim=1)
        if edge_index_added_src is not None:
            edge_index = torch.cat((edge_index, edge_index_added_src), dim=1)
        if edge_index_added_tar is not None:
            edge_index_added_tar = (edge_index_added_tar + N_src) # .cpu()
            edge_index = torch.cat((edge_index, edge_index_added_tar), dim=1)

        central_mask = torch.zeros(N).bool().to(self.device)
        central_mask[:N_src] = True
        train_mask = torch.zeros(N).bool().to(self.device)
        val_mask = torch.zeros(N).bool().to(self.device)
        test_mask = torch.zeros(N).bool().to(self.device)
        train_mask[torch.where(self.data_src.train_mask)[0]] = True
        val_mask[torch.where(self.data_src.val_mask)[0]] = True
        test_mask[torch.where(self.data_src.test_mask)[0]] = True

        # train_mask[torch.where(self.data_tar.train_mask)[0] + N_src] = True
        # val_mask[torch.where(self.data_tar.val_mask)[0] + N_src] = True
        # test_mask[torch.where(self.data_tar.test_mask)[0] + N_src] = True

        y = torch.cat((self.data_src.y, self.data_tar.y), dim=0)
        
        return torch_geometric.data.Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            central_mask=central_mask
            ).coalesce()
    
    def reorder(self, data_merge, data_src, mapper_idx_src, mapper_idx_tar):
        # reorder for data_merge (keep consistent with order/node_id used in the original data)
        N_src = data_src.x.shape[0]
        mapper_merge = {}

        for key in mapper_idx_src:
            mapper_merge[key] = mapper_idx_src[key]

        for key in mapper_idx_tar:
            assert key not in mapper_merge
            mapper_merge[key] = mapper_idx_tar[key] + N_src
    
        mapper_merge_inverse = {}
        for key in mapper_merge:
            mapper_merge_inverse[mapper_merge[key]] = key
        mapper_merge_items_sort = sorted(list(mapper_merge.items()), reverse=False, key=lambda x:x[0])
        mapper_merge_items_sort = torch.LongTensor(mapper_merge_items_sort)
        reorder_idxs = mapper_merge_items_sort[:, 1]

        data_merge.train_mask = data_merge.train_mask[reorder_idxs]
        data_merge.val_mask = data_merge.val_mask[reorder_idxs]
        data_merge.test_mask = data_merge.test_mask[reorder_idxs]
        data_merge.central_mask = data_merge.central_mask[reorder_idxs]
        data_merge.x = data_merge.x[reorder_idxs]
        data_merge.y = data_merge.y[reorder_idxs]
        edge_index_ori = data_merge.edge_index

        data_merge.edge_index = torch.LongTensor([[mapper_merge_inverse[idx.item()] for idx in edge_index_ori[0]], [mapper_merge_inverse[idx.item()] for idx in edge_index_ori[1]]])
        
        return data_merge

    def gen_bridged_graph(self, batch_size=1000):

        # edge_index_cross_domain_added, e_sim_mat, idx_src_mat, probs_clf_src, probs_clf_tar = self.add_topk_sim_cross_domain_edges(
        #     epsilon=epsilon,
        #     k=self.k_cross,
        #     batch_size=batch_size)
        
        edge_index_cross_domain_added, e_sim_mat, idx_src_mat, probs_clf_src, probs_clf_tar = self.add_topk_sim_cross_domain_edges_full(k=self.k_cross)

        # print(edge_index_cross_domain_added.shape, e_sim_mat.shape, idx_src_mat.shape, probs_clf_src.shape, probs_clf_tar.shape)

        if self.k_within > 0:
            # add within-domain edges
            edge_index_added_src, e_sim_mat_src, idx_src_mat_src = self.add_topk_sim_within_domain_edges_full(self.data_src, k=self.k_within, domain='source')
            edge_index_added_tar, e_sim_mat_tar, idx_src_mat_tar = self.add_topk_sim_within_domain_edges_full(self.data_tar, k=self.k_within, domain='target')
            # print(edge_index_added_src.shape, edge_index_added_tar.shape)
        else:
            edge_index_added_src = edge_index_added_tar = None
        
        # merge graphs, adding valid edges
        data_merge = self.merge_graphs(
            copy.deepcopy(edge_index_cross_domain_added),
            copy.deepcopy(edge_index_added_src),
            copy.deepcopy(edge_index_added_tar)
            )
        # data_merge = self.reorder(data_merge, self.data_src, mapper_idx_src, mapper_idx_tar)

        return data_merge


class KBLBase(torch.nn.Module):
    def __init__(
        self,
        data_src,
        data_tar,
        device,
        k_cross=20,
        k_within=6,
        epsilon=0.5,
        bridge_batch_size=1000,
        dim_hidden=64,
        num_layer=2,
        num_epoch=200,
        lr=0.001,
        weight_decay=5e-3,
        source_clf=True,
        norm_mode='PN',
        norm_scale=1.
        ):
        super(KBLBase, self).__init__()

        self.bridged_graph = BridgedGraph(
            data_src,
            data_tar,
            k_cross=k_cross,
            k_within=k_within,
            batch_size=bridge_batch_size,
            dim_hidden=dim_hidden,
            num_layer=num_layer,
            num_epoch=num_epoch,
            lr=lr,
            weight_decay=weight_decay,
            source_clf=source_clf,
            norm_mode=norm_mode,
            norm_scale=norm_scale,
            device=device
        )

        self.bridge_batch_size=bridge_batch_size

        self.gnn = GNNBase(
            in_dim=data_src.x.shape[1],
            hid_dim=dim_hidden,
            num_classes=data_src.y.max().item() + 1,
            num_layers=num_layer,
            act=F.relu,
            gnn='gcn'
        ).to(device)
        
    def get_bridged_graph(self):
        self.bridged_graph.fit()
        data_merge = self.bridged_graph.gen_bridged_graph(batch_size=self.bridge_batch_size)

        return data_merge
    
    def forward(self, data):
        log_probs_xs = self.gnn(data.x, data.edge_index)
        
        return log_probs_xs
