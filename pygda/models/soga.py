import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import numpy as np

from torch_geometric.loader import NeighborLoader, DataLoader

from . import BaseGDA
from ..nn import SOGABase
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class SOGA(BaseGDA):
    """
    Source Free Graph Unsupervised Domain Adaptation (WSDM-24).

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
    num_negative_samples : int, optional
        The number of negative samples in NCE loss.
        Default: ``5``.
    num_positive_samples : int, optional
        The number of positive samples in NCE loss.
        Default: ``2``.
    struct_lambda : float, optional
        Structure NCE loss weight. Default: ``1.0``.
    neigh_lambda : float, optional
        Neighborhood NCE loss weight. Default: ``1.0``.
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
        num_negative_samples=5,
        num_positive_samples=2,
        struct_lambda=1.0,
        neigh_lambda=1.0,
        weight_decay=0.,
        lr=4e-3,
        epoch=200,
        gnn='gcn',
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(SOGA, self).__init__(
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
        self.num_negative_samples=num_negative_samples
        self.num_positive_samples=num_positive_samples
        self.struct_lambda=struct_lambda
        self.neigh_lambda=neigh_lambda

    def init_model(self, **kwargs):
        """
        Initialize the SOGA base model.

        Parameters
        ----------
        **kwargs
            Additional parameters for model initialization.

        Returns
        -------
        SOGABase
            Initialized model with specified parameters.

        Notes
        -----
        Configures model with:

        - GNN backbone architecture
        - NCE loss parameters
        - Sampling configurations
        - Model dimensions and dropout
        """

        return SOGABase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            gnn=self.gnn,
            num_negative_samples=self.num_negative_samples,
            num_positive_samples=self.num_positive_samples,
            device=self.device,
            **kwargs
        ).to(self.device)
    
    def forward_model(self, data,  **kwargs):
        """
        Forward pass placeholder for SOGA model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        **kwargs
            Additional arguments.

        Notes
        -----
        Placeholder method as SOGA implements custom forward logic through:
        
        - Source domain training in train_source()
        - Target domain adaptation in train_target()
        - Prediction using adapted model in predict()
        - NCE-based contrastive learning
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
        - Performs supervised training on source domain
        - Uses cross-entropy loss for classification
        - Tracks training metrics and loss
        - Handles batch processing if specified
        """
        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None
        
            for idx, sampled_source_data in enumerate(self.source_loader):
                self.soga.train()

                sampled_source_data = sampled_source_data.to(self.device)
                source_logits = self.soga(sampled_source_data.x, sampled_source_data.edge_index)
                loss = self.soga.cce(source_logits, sampled_source_data.y)
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
    
    def train_target(self, target_data, optimizer):
        """
        Adapt the model to target domain.

        Parameters
        ----------
        target_data : torch_geometric.data.Data
            Target domain graph data.
        optimizer : torch.optim.Optimizer
            Optimizer for model parameters.

        Notes
        -----
        Implementation includes:

        - Initialization of target domain samples
        - NCE loss computation for structure and neighborhood
        - Information maximization loss
        - Combined optimization objective
        """
        self.soga.init_target(target_data.clone(), target_data)

        for epoch in range(self.epoch):
            epoch_probs = None
        
            for idx, sampled_target_data in enumerate(self.target_loader):
                self.soga.train()

                sampled_target_data = sampled_target_data.to(self.device)
                target_logits = self.soga(sampled_target_data.x, sampled_target_data.edge_index)
                probs = F.softmax(target_logits, dim=-1)

                if idx == 0:
                    epoch_probs = probs
                else:
                    epoch_probs = torch.cat((epoch_probs, probs))
            
            NCE_loss_struct = self.NCE_loss(epoch_probs, self.soga.center_nodes_struct, self.soga.positive_samples_struct, self.soga.negative_samples_struct)
            NCE_loss_neigh = self.NCE_loss(epoch_probs, self.soga.center_nodes_neigh, self.soga.positive_samples_neigh, self.soga.negative_samples_neigh)

            IM_loss = self.ent(epoch_probs) - self.div(epoch_probs)
        
            loss =  IM_loss + self.struct_lambda * NCE_loss_struct  + self.neigh_lambda * NCE_loss_neigh 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            logger(epoch=epoch,
                loss=loss,
                time=time.time() - self.start_time,
                verbose=self.verbose,
                train=True)
    
    def entropy(self, x):
        """
        Calculate entropy of probability distribution.

        Parameters
        ----------
        x : torch.Tensor
            Input probability distribution.

        Returns
        -------
        torch.Tensor
            Computed entropy values.

        Notes
        -----
        - Handles numerical stability with epsilon
        - Computes per-sample entropy
        """
        batch_size, num_feature = x.size()
        epsilon = 1e-5
        ent = -x * torch.log(x + epsilon)
        ent = torch.sum(ent, dim=1)

        return ent
    
    def ent(self, softmax_output):
        """
        Calculate mean entropy across samples.

        Parameters
        ----------
        softmax_output : torch.Tensor
            Softmax probabilities.

        Returns
        -------
        torch.Tensor
            Mean entropy loss.
        """
        entropy_loss = torch.mean(self.entropy(softmax_output))

        return entropy_loss

    def div(self, softmax_output):
        """
        Calculate diversity loss.

        Parameters
        ----------
        softmax_output : torch.Tensor
            Softmax probabilities.

        Returns
        -------
        torch.Tensor
            Diversity loss value.

        Notes
        -----
        - Computes negative entropy of mean predictions
        - Encourages uniform class distribution
        """
        mean_softmax_output = softmax_output.mean(dim = 0)
        diversity_loss = torch.sum(-mean_softmax_output * torch.log(mean_softmax_output + 1e-8))
        
        return diversity_loss
    
    def NCE_loss(self, outputs, center_nodes, positive_samples, negative_samples):
        """
        Compute Noise Contrastive Estimation loss.

        Parameters
        ----------
        outputs : torch.Tensor
            Model output probabilities.
        center_nodes : torch.Tensor
            Indices of center nodes.
        positive_samples : torch.Tensor
            Indices of positive samples.
        negative_samples : torch.Tensor
            Indices of negative samples.

        Returns
        -------
        torch.Tensor
            NCE loss value.

        Notes
        -----
        - Implements InfoNCE-style contrastive loss
        - Handles both structural and neighborhood contrasts
        - Uses temperature-scaled dot product similarity
        """
        negative_embedding = F.embedding(negative_samples, outputs).to(self.device)
        positive_embedding = F.embedding(positive_samples, outputs).to(self.device)
        center_embedding = F.embedding(center_nodes, outputs).to(self.device)

        positive_embedding = positive_embedding.permute([0, 2, 1])
        positive_score =  torch.bmm(center_embedding, positive_embedding).squeeze()
        exp_positive_score = torch.exp(positive_score).squeeze()
        
        negative_embedding = negative_embedding.permute([0, 2, 1])
        negative_score = torch.bmm(center_embedding, negative_embedding).squeeze()
        exp_negative_score = torch.exp(negative_score).squeeze()
        
        exp_negative_score = torch.sum(exp_negative_score, dim=1)
        
        loss = -torch.log(exp_positive_score / exp_negative_score) 
        loss = loss.mean()

        return loss

    def fit(self, source_data, target_data):
        """
        Train the SOGA model on source and target domains.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process:

        Source Pretraining
        
        - Supervised training on source domain
        - Classification loss optimization

        Target Adaptation
        
        - Unsupervised adaptation
        - NCE loss for structure and neighborhood
        - Information maximization
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

        self.soga = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.soga.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.start_time = time.time()

        print('Source domain pretraining...')
        self.train_source(optimizer)
        print('Target domain adaptation...')
        self.train_target(target_data, optimizer)

    
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

        - Positive/negative sample generation
        - Neighborhood structure analysis
        - Information maximization computations
        - Batch-wise data handling in training methods
        """
        pass

    def predict(self, data):
        """
        Make predictions on target domain data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.

        Returns
        -------
        tuple
            Contains:
            - logits : torch.Tensor
                Model predictions.
            - labels : torch.Tensor
                True labels.

        Notes
        -----
        - Evaluates model in inference mode
        - Handles batch processing
        - Concatenates predictions for full graph
        """
        self.soga.eval()
        
        for idx, sampled_data in enumerate(self.target_loader):
            sampled_data = sampled_data.to(self.device)
            with torch.no_grad():
                logits = self.soga(sampled_data.x, sampled_data.edge_index)

                if idx == 0:
                    logits, labels = logits, sampled_data.y
                else:
                    sampled_logits, sampled_labels = logits, sampled_data.y
                    logits = torch.cat((logits, sampled_logits))
                    labels = torch.cat((labels, sampled_labels))

        return logits, labels
