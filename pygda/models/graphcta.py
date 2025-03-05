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
from torch_geometric.nn import SimpleConv

from . import BaseGDA
from ..nn import GNNBase
from ..utils import logger
from ..utils.perturb import *
from ..metrics import eval_macro_f1, eval_micro_f1


class GraphCTA(BaseGDA):
    """
    Collaborate to Adapt: Source-Free Graph Domain Adaptation via Bi-directional Adaptation (WWW-24).

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
    loop_model : int, optional
        Loops for optimizing model.
        Default: ``3``.
    loop_adj : int, optional
        Loops for optimizing structure.
        Default: ``1``.
    loop_feat : int, optional
        Loops for optimizing features.
        Default: ``4``.
    K : int, optional
        Number of k-nearest neighbors.
        Default: ``5``.
    ratio : float, optional
        Budget B for changing graph structure. Default: ``0.1``.
    tau : float, optional
        Contrastive loss hyperparameter. Default: ``0.2``.
    lamb : float, optional
        Loss trade off hyperparameter. Default: ``0.2``.
    momentum : float, optional
        Momentum update hyperparameter. Default: ``0.9``.
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
        loop_model=3,
        loop_adj=1,
        loop_feat=4,
        ratio=0.1,
        K=5,
        tau=0.2,
        lamb=0.2,
        momentum=0.9,
        make_undirected=True,
        weight_decay=0.,
        lr=1e-4,
        epoch=500,
        gnn='gcn',
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(GraphCTA, self).__init__(
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
        self.loop_model=loop_model
        self.loop_adj=loop_adj
        self.loop_feat=loop_feat
        self.ratio=ratio
        self.tau=tau
        self.lamb=lamb
        self.momentum=momentum
        self.K=K
        self.make_undirected=make_undirected
        self.neighprop = SimpleConv(aggr='mean')

    def init_model(self, **kwargs):
        """
        Initialize the GraphCTA base model.

        Parameters
        ----------
        **kwargs
            Additional parameters for model initialization.

        Returns
        -------
        GNNBase
            Initialized model with specified architecture.

        Notes
        -----
        Configures base GNN model with:

        - Input and hidden dimensions
        - Number of classes and layers
        - Dropout rate
        - Specified GNN backbone type
        - Device placement
        """
        return GNNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            gnn=self.gnn,
            **kwargs
        ).to(self.device)
    
    def forward_model(self, data, **kwargs):
        """
        Forward pass placeholder for GraphCTA model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        **kwargs
            Additional arguments.

        Notes
        -----
        Placeholder method as GraphCTA implements custom forward logic through:

        - Source domain training in train_source()
        - Collaborative adaptation in fit()
        - Feature and structure optimization
        - Memory-based prototype learning
        """
        pass
    
    def train_source(self, optimizer):
        """
        Train the model on source domain data.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer for model parameters.

        Notes
        -----
        Training process includes:

        Per-epoch Operations:
        
        - Tracks cumulative loss
        - Maintains logits and labels
        - Computes performance metrics

        Batch Processing:
        
        - Moves data to device
        - Computes model predictions
        - Applies negative log-likelihood loss
        - Updates model parameters

        Monitoring:
        
        - Computes micro-F1 score
        - Logs training progress
        - Tracks timing information

        Implementation Features:
        
        - Supports batch processing
        - Uses softmax with log probabilities
        - Accumulates predictions for full evaluation
        - Comprehensive logging
        """
        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None
        
            for idx, sampled_source_data in enumerate(self.source_loader):
                self.graphcta.train()

                sampled_source_data = sampled_source_data.to(self.device)
                source_logits = self.graphcta(sampled_source_data.x, sampled_source_data.edge_index)
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
        """
        Train the GraphCTA model with collaborative adaptation.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Implementation consists of three main phases:

        Initialization
        
        - Sets up data loaders
        - Initializes model and optimizers
        - Creates memory banks for features and classes
        - Prepares feature and structure perturbation variables

        Source Pretraining
        
        - Trains model on source domain
        - Uses standard cross-entropy loss
        - Prepares model for adaptation

        Target Adaptation (Iterative)
        
        - Model Update Loop:
            
            * Updates model parameters
            * Computes prototype-based alignment
            * Updates memory banks with momentum
            * Combines local and contrastive losses

        - Feature Optimization Loop:
            
            * Optimizes feature perturbations
            * Uses test-time adaptation loss
            * Maintains feature consistency

        - Structure Optimization Loop:
            
            * Modifies edge weights
            * Ensures budget constraints
            * Preserves graph properties

        Implementation Features:
        
        - Memory-based prototype learning
        - Gradient checkpointing for efficiency
        - Momentum updates for stability
        - Budget-constrained modifications
        - Multiple optimization objectives
        - Collaborative feature-structure adaptation
        """
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

        self.graphcta = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.graphcta.parameters(),
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

        mem_fea = torch.rand(target_data.x.size(0), self.hid_dim).to(self.device)    
        mem_cls = torch.ones(target_data.x.size(0), self.num_classes).to(self.device) / self.num_classes

        n = target_data.x.size(0)

        self.start_time = time.time()

        print('Source domain pretraining...')
        self.train_source(optimizer)

        print('Target domain adaptation...')
        
        if target_data.edge_weight is None:
            edge_index = target_data.edge_index
            edge_weight = torch.ones(edge_index.shape[1]).to(self.device)
        else:
            edge_index = target_data.edge_index
            edge_weight = target_data.edge_weight
        
        feat = target_data.x
        
        for it in tqdm(range(self.epoch//(self.loop_feat + self.loop_adj))):
            for loop_model in range(self.loop_model):
                for k,v in self.graphcta.named_parameters():
                    v.requires_grad = True
                self.graphcta.train()

                feat = feat.detach()
                edge_weight = edge_weight.detach()

                optimizer.zero_grad()
                feat_output = self.graphcta.feat_bottleneck(feat, edge_index, edge_weight)
                cls_output = self.graphcta.feat_classifier(feat_output, edge_index, edge_weight)

                onehot = torch.nn.functional.one_hot(cls_output.argmax(1), num_classes=self.num_classes).float()
                proto = (torch.mm(mem_fea.t(), onehot) / (onehot.sum(dim=0) + 1e-8)).t()

                prob = self.neighprop(mem_cls, edge_index)
                weight, pred = torch.max(prob, dim=1)
                cl, weight_ = self.instance_proto_alignment(feat_output, proto, pred)
                ce = F.cross_entropy(cls_output, pred, reduction='none')
                loss_local = torch.sum(weight_ * ce) / (torch.sum(weight_).item())
                loss = loss_local * (1 - self.lamb) + cl * self.lamb

                loss.backward()
                optimizer.step()
                print('Model: ' + str(loss.item()))

                self.graphcta.eval()
                with torch.no_grad():
                    feat_output = self.graphcta.feat_bottleneck(feat, edge_index, edge_weight)
                    cls_output = self.graphcta.feat_classifier(feat_output, edge_index, edge_weight)
                    softmax_out = F.softmax(cls_output, dim=1)
                    outputs_target = softmax_out**2 / ((softmax_out**2).sum(dim=0))
        
                mem_cls = (1.0 - self.momentum) * mem_cls + self.momentum * outputs_target.clone()
                mem_fea = (1.0 - self.momentum) * mem_fea + self.momentum * feat_output.clone()
            
            for k,v in self.graphcta.named_parameters():
                v.requires_grad = False

            perturbed_edge_weight = perturbed_edge_weight.detach()
            for loop_feat in range(self.loop_feat):
                optimizer_feat.zero_grad()
                loss = self.test_time_loss(target_data.x + delta_feat, edge_index, edge_weight, mem_fea, mem_cls)
                loss.backward()
                optimizer_feat.step()
                print('Feat: ' + str(loss.item()))
            
            new_feat = (target_data.x + delta_feat).detach()
            self.new_feat = new_feat
            for loop_adj in range(self.loop_adj):
                perturbed_edge_weight.requires_grad = True
                edge_index, edge_weight = get_modified_adj(modified_edge_index, perturbed_edge_weight, n, self.device, edge_index, edge_weight, self.make_undirected)
                loss = self.test_time_loss(new_feat, edge_index, edge_weight, mem_fea, mem_cls)
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
        
        self.edge_index, self.edge_weight = self.sample_final_edges(n_perturbations, perturbed_edge_weight, target_data, modified_edge_index, n, mem_fea, mem_cls)

    def instance_proto_alignment(self, feat, center, pred):
        """
        Compute instance-prototype alignment loss.

        Parameters
        ----------
        feat : torch.Tensor
            Node features.
        center : torch.Tensor
            Prototype centers.
        pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        tuple
            Contains:
            - loss : torch.Tensor
                Contrastive alignment loss.
            - weight : torch.Tensor
                Instance-prototype similarity weights.

        Notes
        -----
        - Implements temperature-scaled contrastive loss
        - Handles both instance-prototype and instance-instance relations
        - Uses cosine similarity for feature comparison
        """
        feat_norm = F.normalize(feat, dim=1)
        center_norm = F.normalize(center, dim=1)
        sim = torch.matmul(feat_norm, center_norm.t())

        num_nodes = feat.size(0)
        weight = sim[range(num_nodes), pred]
        sim = torch.exp(sim / self.tau)
        pos_sim = sim[range(num_nodes), pred]

        sim_feat = torch.matmul(feat_norm, feat_norm.t())
        sim_feat = torch.exp(sim_feat / self.tau)
        ident = sim_feat[range(num_nodes), range(num_nodes)]

        logit = pos_sim / (sim.sum(dim=1) - pos_sim + sim_feat.sum(dim=1) - ident + 1e-8)
        loss = - torch.log(logit + 1e-8).mean()

        return loss, weight

    def update_edge_weights(self, gradient, optimizer_adj, perturbed_edge_weight):
        """
        Update edge weights during structure optimization.

        Parameters
        ----------
        gradient : torch.Tensor
            Computed gradients for edge weights.
        optimizer_adj : torch.optim.Optimizer
            Optimizer for edge weights.
        perturbed_edge_weight : torch.Tensor
            Current edge weights to be updated.

        Notes
        -----
        - Applies gradient updates to edge weights
        - Maintains minimum weight threshold
        - Uses Adam optimizer for updates
        """
        optimizer_adj.zero_grad()
        perturbed_edge_weight.grad = gradient
        optimizer_adj.step()
        perturbed_edge_weight.data[perturbed_edge_weight < 1e-7] = 1e-7
    
    @torch.no_grad()
    def sample_final_edges(self, n_perturbations, perturbed_edge_weight, data, modified_edge_index, n, mem_fea, mem_cls):
        """
        Sample final edge structure based on learned weights.

        Parameters
        ----------
        n_perturbations : int
            Maximum number of allowed edge modifications.
        perturbed_edge_weight : torch.Tensor
            Learned edge weights.
        data : torch_geometric.data.Data
            Target graph data.
        modified_edge_index : torch.Tensor
            Modified edge indices.
        n : int
            Number of nodes.
        mem_fea : torch.Tensor
            Memory bank features.
        mem_cls : torch.Tensor
            Memory bank class predictions.

        Returns
        -------
        tuple
            Contains:
            - edge_index : torch.Tensor
                Final edge structure.
            - edge_weight : torch.Tensor
                Final edge weights.

        Notes
        -----
        - Uses iterative sampling strategy
        - Maintains best performing structure
        - Ensures perturbation budget constraints
        - Handles undirected graph requirements
        """
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
                loss = self.test_time_loss(feat, edge_index, edge_weight, mem_fea, mem_cls)

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
        """
        Process input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph to be processed.

        Notes
        -----
        Placeholder method as graph processing is handled through:

        - Feature perturbation optimization
        - Structure modification
        - Memory bank updates
        - Prototype-based alignment
        """
        pass
    
    def entropy(self, input_):
        """
        Calculate entropy of probability distribution.

        Parameters
        ----------
        input_ : torch.Tensor
            Input probability distribution.

        Returns
        -------
        torch.Tensor
            Computed entropy values per sample.

        Notes
        -----
        - Handles numerical stability with epsilon
        - Used for confidence-based pseudo-labeling
        """
        entropy = -input_ * torch.log(input_ + 1e-8)
        entropy = torch.sum(entropy, dim=1)
        
        return entropy 
    
    def test_time_loss(self, feat, edge_index, edge_weight, mem_fea, mem_cls):
        """
        Compute test-time adaptation loss.

        Parameters
        ----------
        feat : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.
        edge_weight : torch.Tensor
            Edge weights.
        mem_fea : torch.Tensor
            Memory bank features.
        mem_cls : torch.Tensor
            Memory bank class predictions.

        Returns
        -------
        torch.Tensor
            Combined loss value.

        Notes
        -----
        Loss components include:
        
        1. Pseudo-label based classification
        2. Feature similarity with memory bank
        3. Class-wise prototype alignment
        4. Confidence-based sample selection
        """
        self.graphcta.eval()
        feat_output = self.graphcta.feat_bottleneck(feat, edge_index, edge_weight)
        cls_output = self.graphcta.feat_classifier(feat_output, edge_index, edge_weight)
        softmax_out = F.softmax(cls_output, dim=1)
        _, predict = torch.max(softmax_out, 1)
        mean_ent = self.entropy(softmax_out)
        est_p = (mean_ent<mean_ent.mean()).sum().item() / mean_ent.size(0)
        value = mean_ent

        predict = predict.cpu().numpy()
        train_idx = np.zeros(predict.shape)

        cls_k = self.num_classes
        for c in range(cls_k):
            c_idx = np.where(predict==c)
            c_idx = c_idx[0]
            c_value = value[c_idx]

            _, idx_ = torch.sort(c_value)
            c_num = len(idx_)
            c_num_s = int(c_num * est_p / 5)

            for ei in range(0, c_num_s):
                ee = c_idx[idx_[ei]]
                train_idx[ee] = 1
                
        train_idx = np.array(train_idx, dtype=bool)
        pred_label = predict[train_idx]
        pseudo_label = torch.from_numpy(pred_label).to(self.device)

        pred_output = cls_output[train_idx]
        loss = F.cross_entropy(pred_output, pseudo_label)

        distance = feat_output @ mem_fea.T
        _, idx_near = torch.topk(distance, dim=-1, largest=True, k=self.K + 1)
        idx_near = idx_near[:, 1:]  # batch x K
        
        mem_near = mem_fea[idx_near]  # batch x K x d
        feat_output_un = feat_output.unsqueeze(1).expand(-1, self.K, -1) # batch x K x d
        loss -= torch.mean((feat_output_un * mem_near).sum(-1).sum(1)/self.K) * 0.1

        _, pred_mem = torch.max(mem_cls, dim=1)
        _, pred = torch.max(softmax_out, dim=1)
        idx = pred.unsqueeze(-1) == pred_mem
        neg_num = torch.sum(~idx, dim=1)
        dis = (distance * ~idx).sum(1)/neg_num
        loss += dis.mean() * 0.1

        return loss


    def predict(self, data):
        """
        Make predictions using the adapted model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.

        Returns
        -------
        tuple
            Contains:
            - logits : torch.Tensor
                Model predictions using optimized features and structure.
            - labels : torch.Tensor
                True labels.

        Notes
        -----
        - Uses transformed features (self.new_feat)
        - Uses optimized edge structure (self.edge_index, self.edge_weight)
        - Evaluates model in inference mode
        """
        self.graphcta.eval()

        logits = self.graphcta(self.new_feat, self.edge_index, self.edge_weight)
        labels = data.y

        return logits, labels
