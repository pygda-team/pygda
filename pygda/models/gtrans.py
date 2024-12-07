import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np
from tqdm import tqdm
import random

from torch_geometric.loader import NeighborLoader, DataLoader
from torch.nn.parameter import Parameter
from torch_geometric.utils import dropout_adj

from . import BaseGDA
from ..nn import GNNBase
from ..utils import logger
from ..utils.perturb import *
from ..metrics import eval_macro_f1, eval_micro_f1


class GTrans(BaseGDA):
    """
    Empowering Graph Representation Learning with Test-Time Graph Transformation (ICLR-23).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    num_layers : int, optional
        Total number of layers in model. Default: ``3``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    gnn : string, optional
        GNN backbone. Default: ``gcn``.
    loop_adj : int, optional
        Loops for optimizing structure.
        Default: ``1``.
    loop_feat : int, optional
        Loops for optimizing features.
        Default: ``4``.
    ratio : float, optional
        Budget B for changing graph structure. Default: ``0.1``.
    margin : float, optional
        Test time loss hyperparameter. Default: ``-1``.
    strategy : str, optional
        Graph augmentation strategy. Default: ``dropedge``.
    make_undirected : bool, optional
        Transform into undirected graph. Default: ``True``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``200``.
    device : str, optional
        GPU or CPU. Default: ``cuda:0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``2``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=3,
        dropout=0.,
        act=F.relu,
        loop_adj=1,
        loop_feat=4,
        ratio=0.1,
        margin=-1,
        make_undirected=True,
        strategy='dropedge',
        weight_decay=0.,
        lr=1e-4,
        epoch=500,
        gnn='gcn',
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(GTrans, self).__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            weight_decay=weight_decay,
            lr=lr,
            epoch=epoch,
            device=device,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            **kwargs)
        
        assert batch_size == 0, 'unsupport for batch training'

        self.gnn=gnn
        self.loop_adj=loop_adj
        self.loop_feat=loop_feat
        self.ratio=ratio
        self.margin=margin
        self.strategy=strategy
        self.make_undirected=make_undirected

    def init_model(self, **kwargs):

        return GNNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            gnn=self.gnn,
            **kwargs
        ).to(self.device)
    
    def forward_model(self, data,  **kwargs):
        pass
    
    def train_source(self, optimizer):
        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None
        
            for idx, sampled_source_data in enumerate(self.source_loader):
                self.gtrans.train()

                sampled_source_data = sampled_source_data.to(self.device)
                source_logits = self.gtrans(sampled_source_data.x, sampled_source_data.edge_index)
                loss = F.nll_loss(F.log_softmax(source_logits, dim=1), sampled_source_data.y)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = source_logits, sampled_source_data.y
                else:
                    source_logits, source_labels = source_logits, sampled_source_data.y
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
        
            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(epoch_source_labels, epoch_source_preds)
        
            logger(epoch=epoch,
                loss=epoch_loss,
                source_train_acc=micro_f1_score,
                time=time.time() - self.start_time,
                verbose=self.verbose,
                train=True)

    def fit(self, source_data, target_data):
        if self.batch_size == 0:
            self.source_batch_size = source_data.x.shape[0]
            self.source_loader = NeighborLoader(
                source_data,
                self.num_neigh,
                batch_size=self.source_batch_size)
            
            self.target_batch_size = target_data.x.shape[0]
            self.target_loader = NeighborLoader(
                target_data,
                self.num_neigh,
                batch_size=self.target_batch_size)
        else:
            self.source_loader = NeighborLoader(
                source_data,
                self.num_neigh,
                batch_size=self.batch_size)
            
            self.target_loader = NeighborLoader(
                target_data,
                self.num_neigh,
                batch_size=self.batch_size)

        self.gtrans = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.gtrans.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        delta_feat = Parameter(torch.FloatTensor(target_data.x.size(0), target_data.x.size(1)).to(self.device))
        delta_feat.data.fill_(1e-7)
        optimizer_feat = torch.optim.Adam([delta_feat], lr=self.lr)

        modified_edge_index = target_data.edge_index.clone()
        modified_edge_index = modified_edge_index[:, modified_edge_index[0] < modified_edge_index[1]]
        row, col = modified_edge_index[0], modified_edge_index[1]
        edge_index_id = (2 * target_data.x.size(0) - row - 1) * row // 2 + col - row - 1
        edge_index_id = edge_index_id.long()
        modified_edge_index = linear_to_triu_idx(target_data.x.size(0), edge_index_id)
        perturbed_edge_weight = torch.full_like(edge_index_id, 1e-7, dtype=torch.float32, requires_grad=True).to(self.device)

        optimizer_adj = torch.optim.Adam([perturbed_edge_weight], lr=self.lr)

        n_perturbations = int(self.ratio * target_data.edge_index.shape[1] //2)

        n = target_data.x.size(0)

        self.start_time = time.time()

        print('Source domain pretraining...')
        self.train_source(optimizer)

        print('Target domain adaptation...')
        for k,v in self.gtrans.named_parameters():
            v.requires_grad = False
        
        if target_data.edge_weight is None:
            edge_index = target_data.edge_index
            edge_weight = torch.ones(edge_index.shape[1]).to(self.device)
        else:
            edge_index = target_data.edge_index
            edge_weight = target_data.edge_weight
         
        for it in tqdm(range(self.epoch//(self.loop_feat + self.loop_adj))):
            perturbed_edge_weight = perturbed_edge_weight.detach()
            for loop_feat in range(self.loop_feat):
                optimizer_feat.zero_grad()
                loss = self.test_time_loss(target_data.x + delta_feat, edge_index, edge_weight)
                loss.backward()
                optimizer_feat.step()
                print('Feat: ' + str(loss.item()))
            
            new_feat = (target_data.x + delta_feat).detach()
            for loop_adj in range(self.loop_adj):
                perturbed_edge_weight.requires_grad = True
                edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, self.device, edge_index, edge_weight, self.make_undirected)
                loss = self.test_time_loss(new_feat, edge_index, edge_weight)
                print('Adj: ' + str(loss.item()))

                gradient = grad_with_checkpoint(loss, perturbed_edge_weight)[0]

                with torch.no_grad():
                    self.update_edge_weights(gradient, optimizer_adj, perturbed_edge_weight)
                    perturbed_edge_weight = project(n_perturbations, perturbed_edge_weight, 1e-7)
            
            if self.loop_adj != 0:
                edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, self.device, edge_index, edge_weight, self.make_undirected)
                edge_weight = edge_weight.detach()
            
            if self.loop_feat != 0:
                feat = (target_data.x + delta_feat).detach()
                self.new_feat = feat
        
        self.edge_index, self.edge_weight = self.sample_final_edges(n_perturbations, perturbed_edge_weight, target_data, modified_edge_index, n)

    
    def update_edge_weights(self, gradient, optimizer_adj, perturbed_edge_weight):
        optimizer_adj.zero_grad()
        perturbed_edge_weight.grad = gradient
        optimizer_adj.step()
        perturbed_edge_weight.data[perturbed_edge_weight < 1e-7] = 1e-7
    
    @torch.no_grad()
    def sample_final_edges(self, n_perturbations, perturbed_edge_weight, data, modified_edge_index, n):
        best_loss = float('Inf')
        perturbed_edge_weight = perturbed_edge_weight.detach()
        perturbed_edge_weight[perturbed_edge_weight <= 1e-7] = 0

        feat = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_weight = torch.ones(edge_index.shape[1]).to(self.device)

        for i in range(20):
            if best_loss == float('Inf'):
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(perturbed_edge_weight).to(self.device)
                sampled_edges[torch.topk(perturbed_edge_weight, n_perturbations).indices] = 1
            else:
                sampled_edges = torch.bernoulli(perturbed_edge_weight).float()

            if sampled_edges.sum() > n_perturbations:
                n_samples = sampled_edges.sum()
                print(f'{i}-th sampling: too many samples {n_samples}')
        
            perturbed_edge_weight = sampled_edges

            edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, self.device, edge_index, edge_weight)
            
            with torch.no_grad():
                loss = self.test_time_loss(feat, edge_index, edge_weight)

            # Save best sample
            if best_loss > loss:
                best_loss = loss
                print('best_loss:', best_loss.item())
                best_edges = perturbed_edge_weight.clone().cpu()

        # Recover best sample
        perturbed_edge_weight.data.copy_(best_edges.to(self.device))

        edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, self.device, edge_index, edge_weight)
        edge_mask = edge_weight == 1
        make_undirected = self.make_undirected

        allowed_perturbations = 2 * n_perturbations if make_undirected else n_perturbations
        edges_after_attack = edge_mask.sum()
        clean_edges = edge_index.shape[1]
        assert (edges_after_attack >= clean_edges - allowed_perturbations
                and edges_after_attack <= clean_edges + allowed_perturbations), \
            f'{edges_after_attack} out of range with {clean_edges} clean edges and {n_perturbations} pertutbations'
    
        return edge_index[:, edge_mask], edge_weight[edge_mask]
    
    def process_graph(self, data):
        pass
    
    def test_time_loss(self, feat, edge_index, edge_weight=None):
        loss = 0
        
        if self.strategy == 'dropedge':
            output1 = self.augment(feat, edge_index=edge_index, edge_weight=edge_weight, p=0.05, strategy=self.strategy)
        
        output2 = self.augment(feat, edge_index=edge_index, edge_weight=edge_weight, p=0.0, strategy='dropedge')
        output3 = self.augment(feat, edge_index=edge_index, edge_weight=edge_weight, p=0.0, strategy='shuffle')

        if self.margin != -1:
            loss = inner(output1, output2) - inner_margin(output2, output3, margin=self.margin)
        else:
            loss = inner(output1, output2) - inner(output2, output3)

        return loss

    def augment(self, feat, edge_index=None, edge_weight=None, p=0.5, strategy='dropedge'):
        if strategy == 'dropedge':
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = self.gtrans.feat_bottleneck(feat, edge_index, edge_weight)

        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = self.gtrans.feat_bottleneck(shuf_fts, edge_index, edge_weight)
    
        return output

    def predict(self, data):
        self.gtrans.eval()

        logits = self.gtrans(self.new_feat, self.edge_index, self.edge_weight)
        labels = data.y

        return logits, labels
