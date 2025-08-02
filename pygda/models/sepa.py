import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from . import BaseGDA
from ..nn import GNNBase
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class PageRank(MessagePassing):
    """
    Implementation of PageRank algorithm as a message passing layer.

    This class implements a personalized PageRank algorithm that can be used as a
    message passing layer in graph neural networks. It performs k iterations of
    random walk with restart to compute node importance scores.

    Parameters
    ----------
    k : int, optional
        Number of iterations for random walk. Default: ``5``.
    alpha : float, optional
        Restart probability (teleportation factor). Default: ``0.9``.
    **kwargs
        Additional arguments for MessagePassing.

    Attributes
    ----------
    k : int
        Number of iterations for random walk.
    alpha : float
        Restart probability.

    Notes
    -----
    The PageRank computation follows these steps:

    - Normalize edge weights using GCN normalization

    - For k iterations:

        * Propagate messages along edges

        * Combine with restart probability

    - Return final node importance scores

    The message passing follows the formula:
    x = (1 - alpha) * propagate(x) + alpha * hidden,
    where hidden is the initial node features.
    """

    def __init__(self, k=5, alpha=0.9, **kwargs):
        super(PageRank, self).__init__(aggr='add', **kwargs)
        self.k = k
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the PageRank layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix, shape (num_nodes, feature_dim).
        edge_index : torch.Tensor
            Edge indices, shape (2, num_edges).
        edge_weight : torch.Tensor, optional
            Edge weights. Default: ``None``.

        Returns
        -------
        torch.Tensor
            Updated node features after k iterations of PageRank.
        """
        edge_index, norm = gcn_norm(edge_index, edge_weight)

        hidden = x
        for k in range(self.k):
            x = self.propagate(edge_index, x=x, norm=norm)
            x = (1 - self.alpha) * x + self.alpha * hidden

        return x

    def message(self, x_j, norm):
        """
        Compute messages for each edge.

        Parameters
        ----------
        x_j : torch.Tensor
            Node features of the target nodes.
        norm : torch.Tensor
            Normalized edge weights.

        Returns
        -------
        torch.Tensor
            Messages to be aggregated.
        """
        return norm.view(-1, 1) * x_j


class SEPA(BaseGDA):
    """
    Structure Enhanced Prototypical Alignment for Unsupervised Cross-Domain Node Classification (NN-24).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    gnn : string, optional
        GNN backbone. Default: ``gcn``.
    lamda_im : float, optional
        Parameter of loss_im. Default: ``0.2``.
    lamda_pp : float, optional
        Parameter of loss_pp. Default: ``0.1``.
    lamda_sep : float, optional
        Parameter of loss_sep. Default: ``1.0``.
    tau : float, optional
        Temperature in info-nce loss. Default: ``1.5``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    lr : float, optional
        Learning rate. Default: ``0.001``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    device : str, optional
        GPU or CPU. Default: ``cuda:0``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    **kwargs
        Other parameters for the model.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=2,
        dropout=0.,
        gnn='gcn',
        lamda_im=0.2,
        lamda_pp=0.1,
        lamda_sep=1.0,
        tau=1.5,
        act=F.relu,
        weight_decay=0.01,
        lr=0.001,
        epoch=200,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(SEPA, self).__init__(
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
        
        self.gnn=gnn
        self.lamda_im = lamda_im
        self.lamda_pp = lamda_pp
        self.lamda_sep = lamda_sep
        self.tau = tau
        self.pagerank = PageRank().to(self.device)

        assert batch_size == 0, "do not support batch training"

    def init_model(self, **kwargs):
        """
        Initialize the SEPA model.

        Parameters
        ----------
        **kwargs
            Other parameters for the SEPA model.

        Returns
        -------
        SEPA
            Initialized SEPA model on the specified device.
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

    def forward_model(self, source_data, target_data):
        """
        Forward pass of the SEPA model for domain adaptation.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data containing:
        target_data : torch_geometric.data.Data
            Target domain graph data with the same structure as source_data

        Returns
        -------
        tuple
            Contains (loss, source_logits, target_logits):
            - loss : torch.Tensor
                Combined loss value
            - source_logits : torch.Tensor
                Model predictions for source domain nodes
            - target_logits : torch.Tensor
                Model predictions for target domain nodes

        Notes
        -----
        This function requires batch training (batch_size > 0) as specified in the model initialization.
        The implementation should handle batched data from both source and target domains.
        """
        pass

    def fit(self, source_data, target_data):
        """
        Train the SEPA model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        The training process includes:

        - Creating data loaders for both domains
        - Initializing the GNN model and optimizer
        - Training for specified number of epochs
        - Logging training progress (loss and micro-F1 score)
        """

        if self.batch_size == 0:
            self.source_batch_size = source_data.x.shape[0]
            source_loader = NeighborLoader(source_data,
                                self.num_neigh,
                                batch_size=self.source_batch_size)
            self.target_batch_size = target_data.x.shape[0]
            target_loader = NeighborLoader(target_data,
                                self.num_neigh,
                                batch_size=self.target_batch_size)
        else:
            source_loader = NeighborLoader(source_data,
                                self.num_neigh,
                                batch_size=self.batch_size)
            target_loader = NeighborLoader(target_data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.gnn = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.gnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                self.gnn.train()
                
                source_feat = self.gnn.feat_bottleneck(sampled_source_data.x, sampled_source_data.edge_index)
                target_feat = self.gnn.feat_bottleneck(sampled_target_data.x, sampled_target_data.edge_index)
        
                target_cls = self.gnn.feat_classifier(target_feat, sampled_target_data.edge_index)
                target_prob = F.softmax(target_cls, dim=1)
                loss_im, _, _, ent_loss_t = self.cls_im(target_prob)

                source_proto = self.obtain_source_prototype(source_feat, sampled_source_data.y)
                target_proto = self.obtain_target_prototype(target_prob, target_feat, sampled_target_data.edge_index, ent_loss_t)
                loss_pp = self.proto_alignment(target_proto, source_proto)
                
                loss_sep = self.seperate_center(source_proto, source_feat, sampled_source_data.y)

                output = self.gnn(sampled_source_data.x, sampled_source_data.edge_index)
                train_loss = F.nll_loss(output, sampled_source_data.y)
                
                loss = self.lamda_im * loss_im + self.lamda_pp * loss_pp + self.lamda_sep * loss_sep + train_loss
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = self.predict(sampled_source_data)
                else:
                    source_logits, source_labels = self.predict(sampled_source_data)
                    epoch_source_logits = torch.cat((epoch_source_logits, source_logits))
                    epoch_source_labels = torch.cat((epoch_source_labels, source_labels))
            
            epoch_source_preds = epoch_source_logits.argmax(dim=1)
            micro_f1_score = eval_micro_f1(epoch_source_labels, epoch_source_preds)

            logger(epoch=epoch,
                   loss=epoch_loss,
                   source_train_acc=micro_f1_score,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

    def cls_im(self, prob):
        """
        Compute information maximization loss for domain adaptation.

        This function calculates the information maximization loss which consists of two components:

        - Diversity loss: Encourages the model to make diverse predictions across classes
        - Entropy loss: Minimizes the uncertainty of predictions

        Parameters
        ----------
        prob : torch.Tensor
            Probability distribution over classes, shape (num_samples, num_classes)

        Returns
        -------
        tuple
            Contains (loss_im, div_loss, ent_loss, ent_loss_temp):

            - loss_im : torch.Tensor
                Total information maximization loss (div_loss + ent_loss)
            - div_loss : torch.Tensor
                Diversity loss term
            - ent_loss : torch.Tensor
                Mean entropy loss across samples
            - ent_loss_temp : torch.Tensor
                Per-sample entropy values

        Notes
        -----
        The loss is computed as:

        - Diversity loss: -sum(mean_prob * log(mean_prob))
        - Entropy loss: mean(-sum(prob * log(prob)))

        where mean_prob is the average probability distribution across samples
        """
        mean_prob = prob.mean(dim=0)
        div_loss = torch.sum(mean_prob * torch.log(mean_prob + 1e-12))
        ent_loss_temp = - torch.sum(prob * torch.log(prob + 1e-12), dim=1)
        ent_loss = torch.mean(ent_loss_temp)
        loss_im = div_loss + ent_loss
    
        return loss_im, div_loss, ent_loss, ent_loss_temp
    
    def obtain_source_prototype(self, feat, label):
        """
        Compute class prototypes for source domain features.

        This function calculates the mean feature vector (prototype) for each class
        in the source domain by averaging the features of samples belonging to the same class.

        Parameters
        ----------
        feat : torch.Tensor
            Node features from source domain, shape (num_nodes, feature_dim)
        label : torch.Tensor
            Class labels for source domain nodes, shape (num_nodes,)

        Returns
        -------
        torch.Tensor
            Class prototypes, shape (num_classes, feature_dim)
            Each row represents the mean feature vector for one class

        Notes
        -----
        The prototype for each class is computed as:
        center = sum(features * onehot) / sum(onehot),
        where onehot is the one-hot encoding of class labels
        """
        onehot = torch.eye(self.num_classes).to(self.device)[label]
        center = torch.mm(feat.t(), onehot) / (onehot.sum(dim=0))
        
        return center.t()
    
    def obtain_target_prototype(self, prob, feat, edge_index, ent):
        ent = ent.unsqueeze(-1)
        _, pred = torch.max(prob, dim=1)
        onehot = torch.eye(self.num_classes).to(self.device)[pred]
        biased_center = (torch.mm(feat.t(), onehot) / (onehot.sum(dim=0) + 1e-12)).t()
    
        # Create mask for each class
        class_masks = (pred.unsqueeze(1) == torch.arange(self.num_classes, device=self.device).unsqueeze(0))
        
        # Compute mean probabilities for each class using masked operations
        # For classes with no samples, the mean will be 0
        matrix = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        for c in range(self.num_classes):
            mask = class_masks[:, c]
            if mask.any():
                matrix[c] = prob[mask].mean(dim=0)
        
        eye = torch.eye(self.num_classes, self.num_classes).to(self.device)
        matrix = F.normalize(matrix * (1 - eye), dim=1)
        matrix_every = prob * (matrix[pred])
        matrix_every = F.normalize(matrix_every, dim=1)
        bias_center_every = biased_center[pred]
        unbias_center_every = torch.mm(matrix_every, biased_center) / torch.sum(matrix_every + 1e-12, dim=1, keepdim=True)
        unbias_center_every = F.normalize(self.pagerank(unbias_center_every - bias_center_every, edge_index), dim=1)
        center_every_tt = bias_center_every + ent * unbias_center_every
        center_every = (torch.mm(center_every_tt.t(), onehot) / (onehot.sum(dim=0) + 1e-12)).t()

        return center_every
    
    def seperate_center(self, center, feat, label):
        """
        Compute separation loss between class prototypes to enhance class discriminability.

        This function calculates a loss that encourages class prototypes to be well-separated
        in the feature space by maximizing the distance between different class prototypes
        while minimizing the distance between samples of the same class.

        Parameters
        ----------
        center : torch.Tensor
            Class prototypes, shape (num_classes, feature_dim)
        feat : torch.Tensor
            Node features, shape (num_nodes, feature_dim)
        label : torch.Tensor
            Class labels for nodes, shape (num_nodes,)

        Returns
        -------
        torch.Tensor
            Separation loss value that measures the discriminability between class prototypes

        Notes
        -----
        The separation loss is computed using InfoNCE loss:

        - Normalize prototypes to unit length
        - Compute cosine similarity matrix between prototypes
        - Apply temperature scaling (tau) to the similarity scores
        - Calculate InfoNCE loss by comparing positive pairs (same class)
          against negative pairs (different classes)
        
        The loss encourages:

        - High similarity between samples of the same class
        - Low similarity between samples of different classes
        """
        num_nodes = feat.size(0)
        proto_norm = F.normalize(center, dim=1)
        sim = torch.matmul(proto_norm, proto_norm.t())
        sim = torch.exp(sim / self.tau)
        pos_sim = sim[range(self.num_classes), range(self.num_classes)]
        loss = (sim.sum(dim=1) - pos_sim) / (self.num_classes - 1)
        loss = torch.log(loss + 1e-8) 
        loss = torch.mean(loss)
    
        return loss
    
    def proto_alignment(self, target_proto, source_proto):
        """
        Compute prototype alignment loss between source and target domains.

        This function calculates a loss that aligns class prototypes between source and target domains
        using InfoNCE loss. It encourages similar classes to have similar prototypes across domains
        while maintaining discriminability between different classes.

        Parameters
        ----------
        target_proto : torch.Tensor
            Class prototypes from target domain, shape (num_classes, feature_dim)
        source_proto : torch.Tensor
            Class prototypes from source domain, shape (num_classes, feature_dim)

        Returns
        -------
        torch.Tensor
            Prototype alignment loss value that measures the alignment between domains

        Notes
        -----
        The alignment loss is computed using InfoNCE loss with three components:

        - Target-Source similarity: Measures alignment between corresponding classes
        - Target-Target similarity: Ensures target prototypes are discriminable
        - Source-Source similarity: Ensures source prototypes are discriminable

        The loss is computed as:

        - Normalize prototypes to unit length

        - Compute cosine similarity matrices between:
        
            * Target and source prototypes (ts)
            * Target and target prototypes (tt)
            * Source and source prototypes (ss)

        - Apply temperature scaling (tau)

        - Calculate InfoNCE loss using positive pairs (same class) and
          negative pairs (different classes)

        The loss encourages:

        - High similarity between corresponding class prototypes across domains
        - Low similarity between different class prototypes within each domain
        """
        num_proto = target_proto.size(0)

        target_norm = F.normalize(target_proto + 1e-6, dim=1)
        source_norm = F.normalize(source_proto + 1e-6, dim=1)

        sim_matrix_ts = torch.matmul(target_norm, source_norm.t())
        sim_matrix_ts = torch.exp(sim_matrix_ts / self.tau)
        pos_sim_ts = sim_matrix_ts[range(num_proto), range(num_proto)]

        sim_matrix_tt = torch.matmul(target_norm, target_norm.t())
        sim_matrix_tt = torch.exp(sim_matrix_tt / self.tau)
        pos_sim_tt = sim_matrix_tt[range(num_proto), range(num_proto)]

        sim_matrix_ss = torch.matmul(source_norm, source_norm.t())
        sim_matrix_ss = torch.exp(sim_matrix_ss / self.tau)
        pos_sim_ss = sim_matrix_ss[range(num_proto), range(num_proto)]

        denominator = sim_matrix_ts.sum(dim=1) - pos_sim_ts \
                    + sim_matrix_tt.sum(dim=1) - pos_sim_tt \
                    + sim_matrix_ss.sum(dim=1) - pos_sim_ss
        
        logit = pos_sim_ts / (denominator.clamp(min=1e-6))
        loss = - torch.log(logit.clamp(min=1e-6)).mean()

        return loss
    
    def process_graph(self, data):
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data to be processed.

        Notes
        -----
        This is a placeholder method that should be implemented by subclasses
        if graph preprocessing is needed.
        """
        pass

    def predict(self, data):
        """
        Make predictions using the trained model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.

        Returns
        -------
        tuple
            Contains (logits, labels):
            
            - logits : torch.Tensor
                Model predictions
            - labels : torch.Tensor
                True labels from the data
        """
        self.gnn.eval()

        with torch.no_grad():
            logits = self.gnn(data.x, data.edge_index)

        return logits, data.y
