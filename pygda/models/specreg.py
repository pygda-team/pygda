import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import torch.nn as nn

from torch_geometric.loader import NeighborLoader

from . import BaseGDA
from ..nn import UDAGCNBase
from ..nn import GradReverse
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class SpecReg(BaseGDA):
    """
    Graph Domain Adaptation via Theory-Grounded Spectral Regularization (ICLR-23).

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
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.003``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    ppmi : bool, optional
        Use PPMI matrix or not. Default: ``True``.
    adv_dim : int, optional
        Hidden dimension of adversarial module. Default: ``40``.
    reg_mode : bool, optional
        Use reg mode or adv mode. Default: ``True``.
    gamma_adv : float, optional
        Trade off parameter for adv. Default: ``0.1``.
    thr_smooth : float, optional
        Spectral smoothness threshold. Default: ``-1``.
    gamma_smooth : float, optional
        Trade off parameter for spectral smoothness. Default: ``0.01``.
    thr_mfr : float, optional
        Maximum Frequency Response threshold. Default: ``-1``.
    gamma_mfr : float, optional
        Trade off parameter for Maximum Frequency Response. Default: ``0.01``.
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
        ppmi=True,
        adv_dim=40,
        reg_mode=True,
        gamma_adv=0.1,
        thr_smooth=-1,
        gamma_smooth=0.01,
        thr_mfr=-1,
        gamma_mfr=0.01,
        weight_decay=3e-3,
        lr=4e-3,
        epoch=100,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(SpecReg, self).__init__(
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
        
        self.ppmi=ppmi
        self.adv_dim=adv_dim
        self.reg_mode=reg_mode
        self.gamma_adv=gamma_adv
        self.thr_smooth=thr_smooth
        self.gamma_smooth=gamma_smooth
        self.thr_mfr=thr_mfr
        self.gamma_mfr=gamma_mfr

    def init_model(self, **kwargs):
        """
        Initialize the SpecReg model.

        Parameters
        ----------
        **kwargs
            Other parameters for the UDAGCNBase model.

        Returns
        -------
        UDAGCNBase
            Initialized UDAGCN model on the specified device.
        """

        return UDAGCNBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
            act=self.act,
            ppmi=self.ppmi,
            adv_dim=self.adv_dim,
            **kwargs
        ).to(self.device)

    def forward_model(self, source_data, target_data, alpha, epoch):
        """
        Forward pass of the model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.
        alpha : float
            Gradient reversal scaling parameter.
        epoch : int
            Current training epoch.

        Returns
        -------
        tuple
            Contains:
            - loss : torch.Tensor
                Combined loss from multiple components.
            - source_logits : torch.Tensor
                Model predictions for source domain.
            - target_logits : torch.Tensor
                Model predictions for target domain.

        Notes
        -----
        Computes multiple loss terms:
        
        - Classification loss on source domain
        - Wasserstein distance with gradient penalty
        - Spectral smoothness regularization (if reg_mode)
        - Maximum Frequency Response regularization (if reg_mode)
        - Entropy minimization on target domain
        """
        encoded_source = self.udagcn.encode(source_data, "source")
        encoded_target = self.udagcn.encode(target_data, "target")
        source_logits = self.udagcn.cls_model(encoded_source)

        # use source classifier loss:
        cls_loss = self.udagcn.loss_func(source_logits, source_data.y)

        _x_src, _x_tgt = encoded_source.detach(), encoded_target.detach()
        for _ in range(5):
            self.optimizer_critic.zero_grad()
            loss_1 = self.critic(_x_src).mean() - self.critic(_x_tgt).mean()
            loss_2 = self.calculate_gradient_penalty(_x_src, _x_tgt)
            loss_adv = - loss_1 + 10 * loss_2
            loss_adv.backward()
            self.optimizer_critic.step()
            
        loss_grl = self.critic(encoded_source).mean() - self.critic(encoded_target).mean()
        loss = cls_loss + loss_grl * self.gamma_adv

        if self.reg_mode:
            x_src = torch.einsum('nm,md->nd', source_data.eivec, encoded_source)
            x_tgt = torch.einsum('nm,md->nd', target_data.eivec, encoded_target)

            if self.thr_smooth > 0:
                delta_src = (x_src[:-1] - x_src[1:]).abs()
                delta_tgt = (x_tgt[:-1] - x_tgt[1:]).abs()
                loss = loss + (F.relu(delta_src - self.thr_smooth).mean() + F.relu(delta_tgt - self.thr_smooth).mean()) * self.gamma_smooth
            
            if self.thr_mfr > 0:
                loss = loss + (F.relu(x_src.abs() - self.thr_mfr).mean() + F.relu(x_tgt.abs() - self.thr_mfr).mean()) * self.gamma_mfr

        # use target classifier loss:
        target_logits = self.udagcn.cls_model(encoded_target)
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)

        loss_entropy = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))

        loss = loss + loss_entropy * (epoch / self.epoch * 0.01)

        return loss, source_logits, target_logits

    def fit(self, source_data, target_data):
        """
        Train the SpecReg model.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process includes:

        - Setting up data loaders
        - Initializing model, critic, and optimizers
        - Alternating training between:
            - Critic optimization (Wasserstein distance)
            - Model optimization with spectral regularization
        - Computing and logging training metrics
        """
        self.num_source_nodes, _ = source_data.x.shape
        self.num_target_nodes, _ = target_data.x.shape

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

        self.udagcn = self.init_model(**self.kwargs)

        params = itertools.chain(*[model.parameters() for model in self.udagcn.models])

        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # module for SpecReg
        self.critic = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, 1)
        ).to(self.device)

        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), self.lr)

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            alpha = min((epoch + 1) / self.epoch, 0.05)

            for idx, (sampled_source_data, sampled_target_data) in enumerate(zip(source_loader, target_loader)):
                for model in self.udagcn.models:
                    model.train()
                
                loss, source_logits, target_logits = self.forward_model(sampled_source_data, sampled_target_data, alpha, epoch)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx == 0:
                    epoch_source_logits, epoch_source_labels = self.predict(sampled_source_data, source=True)
                else:
                    source_logits, source_labels = self.predict(sampled_source_data, source=True)
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
    
    def process_graph(self, data):
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data to be processed.

        Notes
        -----
        Placeholder method for graph preprocessing.
        """
        pass

    def predict(self, data, source=False):
        """
        Make predictions on given data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        source : bool, optional
            Whether the input is from source domain.
            Default: ``False``.

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
        Uses appropriate encoder based on domain (source/target).
        """
        for model in self.udagcn.models:
            model.eval()

        with torch.no_grad():
            if source:
                encoded_data = self.udagcn.encode(data, 'source')
            else:
                encoded_data = self.udagcn.encode(data, 'target')
            logits = self.udagcn.cls_model(encoded_data)

        return logits, data.y
    
    def calculate_gradient_penalty(self, x_src, x_tgt):
        """
        Calculate gradient penalty for Wasserstein GAN training.

        Parameters
        ----------
        x_src : torch.Tensor
            Source domain features.
        x_tgt : torch.Tensor
            Target domain features.

        Returns
        -------
        torch.Tensor
            Computed gradient penalty value.

        Notes
        -----
        Implements Wasserstein GAN gradient penalty by:

        - Interpolating between source and target features
        - Computing gradients of critic output
        - Penalizing gradients that deviate from norm 1
        """
        x = torch.cat([x_src, x_tgt], dim=0).requires_grad_(True)
        x_out = self.critic(x)
        grad_out = torch.ones(x_out.shape, requires_grad=False).to(x_out.device)

        # Get gradient w.r.t. x
        grad = torch.autograd.grad(
            outputs=x_out,
            inputs=x,
            grad_outputs=grad_out,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,)[0]
        grad = grad.view(grad.shape[0], -1)
        grad_penalty = torch.mean((grad.norm(2, dim=1) - 1) ** 2)

        return grad_penalty
