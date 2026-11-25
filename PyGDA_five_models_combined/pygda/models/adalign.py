import math
import time

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .base import BaseGDA
from ..nn.adalign_base import ADAlignBase, CF, SampleNet
from ..utils import logger


class ADAlign(BaseGDA):
    """Learning Adaptive Distribution Alignment with Neural Characteristic
    Function for Graph Domain Adaptation (ICLR 2026).

    Parameters
    ----------
    in_dim : int
        Number of input features, shared by both domains.
    hid_dim : int
        Hidden feature and sampled-frequency dimension.
    num_classes : int
        Number of source classes; labels are integers in this range.
    num_layers : int, optional
        Number of backbone layers. Default: 2.
    dropout : float, optional
        Dropout probability in [0, 1). Default: 0.25.
    act : callable, optional
        Backbone activation. Default: torch.nn.functional.relu.
    s_pnums : int, optional
        Number of source propagation steps. Default: 0.
    t_pnums : int, optional
        Number of target propagation steps. Default: 10.
    weight : float, optional
        Nonnegative alignment-loss coefficient (lambda). Default: 0.4.
    weight_decay : float, optional
        Nonnegative Adam weight decay. Default: 0.001.
    lr : float, optional
        Positive learning rate for both optimizers. Default: 0.001.
    epoch : int, optional
        Number of training epochs. Default: 150.
    device : str, optional
        PyTorch device. Default: 'cuda:0'.
    batch_size : int, optional
        Only full-batch training (0) is supported. Default: 0.
    num_neigh : int, optional
        Only all-neighbor sampling (-1) is supported. Default: -1.
    verbose : int, optional
        PyGDA logging level from 0 (silent) to 3. Default: 2.
    t_batchsize : int, optional
        Nominal frequency count, a multiple of 16 and at least 32.
        The sampler repeats each generated frequency 16 times. Default: 2048.
    alpha : float, optional
        Amplitude coefficient (kappa) in [0, 1]. Default: 0.5.
    mode : str, optional
        Only 'node' is supported. Default: 'node'.
    adv : bool, optional
        Enable a domain classifier. Only False is supported. Default: False.
    **kwargs : dict
        Additional keyword arguments for the backbone. Default: empty.

    Notes
    -----
    Training uses all source labels and does not use target labels.
    Constructor defaults follow the author CLI configuration. Published
    experiment settings may vary by transfer task. The author loss has
    singular gradients at zero complex modulus and, for alpha=1, at zero
    discrepancy; these boundaries are preserved for implementation parity.
    """

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=2,
        dropout=0.25,
        act=F.relu,
        s_pnums=0,
        t_pnums=10,
        weight=0.4,
        weight_decay=0.001,
        lr=0.001,
        epoch=150,
        device='cuda:0',
        batch_size=0,
        num_neigh=-1,
        verbose=2,
        t_batchsize=2048,
        alpha=0.5,
        mode='node',
        adv=False,
        **kwargs
    ):

        super().__init__(
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
            **kwargs
        )

        self.s_pnums, self.t_pnums = s_pnums, t_pnums
        self.weight, self.alpha = weight, alpha
        self.t_batchsize = t_batchsize
        self.mode, self.adv = mode, adv
        self.a2gnn = None
        self.sample_net = None
        self.history_ = []

    def init_model(self, **kwargs):
        """Initialize the node classification backbone.

        Parameters
        ----------
        **kwargs : dict
            Additional backbone arguments.

        Returns
        -------
        ADAlignBase
            Backbone on the configured device.
        """
        return ADAlignBase(
            in_dim=self.in_dim,
            hid_dim=self.hid_dim,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            adv=False,
            dropout=self.dropout,
            act=self.act,
            mode='node',
            **kwargs
        ).to(self.device)

    def process_graph(self, data, **kwargs):
        """Prepare a node graph for the backbone.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with node features and edge indices.
        **kwargs : dict
            Reserved for the BaseGDA preprocessing interface.

        Returns
        -------
        torch_geometric.data.Data
           Graph copy on the configured device.
        """
        
        return data.clone().to(self.device)

    def forward_model(self, source_data, target_data, alpha=None):
        """Compute source classification and domain alignment losses.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Labeled source graph.
        target_data : torch_geometric.data.Data
            Target graph used to compute alignment features.
        alpha : float or None, optional
            Compatibility argument for the author interface. Default: None.
            The amplitude coefficient is set by the constructor.

        Returns
        -------
        tuple of torch.Tensor
            Total loss, source logits, target logits, and alignment loss.
        """
       
        source_logits = self.a2gnn(source_data, self.s_pnums)
        source_loss = F.nll_loss(F.log_softmax(source_logits, dim=1), source_data.y)
        source_features = self.a2gnn.feat_bottleneck(
            source_data.x, source_data.edge_index, None, self.s_pnums)
        target_features = self.a2gnn.feat_bottleneck(
            target_data.x, target_data.edge_index, None, self.t_pnums)
        frequencies = self.sample_net(self.device)
        align_loss = CF(
            self.alpha, 1 - self.alpha, source_features, target_features, frequencies
        )
        target_logits = self.a2gnn(target_data, self.t_pnums)
        return (source_loss + self.weight * align_loss,
                source_logits, target_logits, align_loss)

    def fit(self, source_data, target_data):
        """Train on one labeled source graph and one unlabeled target graph.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source graph with one class index per node.
        target_data : torch_geometric.data.Data
            Target graph whose labels, if present, are excluded from training.

        Returns
        -------
        ADAlign
            Fitted model after the configured number of epochs.

        Notes
        -----
        Each epoch maximizes alignment loss through the sampler, then minimizes
        the joint loss through the backbone. Source masks do not restrict training.
        """
        source = self.process_graph(source_data)
        target = self.process_graph(target_data)
        target.y = None
        
        self.source_loader = NeighborLoader(source, self.num_neigh,
                                            batch_size=source.x.size(0))
        self.target_loader = NeighborLoader(target, self.num_neigh,
                                            batch_size=target.x.size(0))
        self.a2gnn = self.init_model(**self.kwargs)
        self.model = self.a2gnn
       
       
        # Match the sampler dimension to the backbone embeddings.
        self.sample_net = SampleNet(feature_dim=self.hid_dim,
                                    t_batchsize=self.t_batchsize, t_var=1).to(self.device)
        optimizer_main = torch.optim.Adam(self.a2gnn.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
        optimizer_sample = torch.optim.Adam(self.sample_net.parameters(), lr=self.lr,
                                            weight_decay=self.weight_decay)
        self.history_ = []
        start = time.time()
        for epoch in range(self.epoch):
            total = alignment = 0.0
            for src, tgt in zip(self.source_loader, self.target_loader):
                src, tgt = src.to(self.device), tgt.to(self.device)
                self.a2gnn.train()
                self.sample_net.train()
                main_parameters = list(self.a2gnn.parameters())
                flags = [p.requires_grad for p in main_parameters]
                try:
                    for p in main_parameters:
                        p.requires_grad_(False)
                    optimizer_sample.zero_grad()
                    _, _, _, cf = self.forward_model(src, tgt)
                    (-cf).backward()
                    optimizer_sample.step()
                finally:
                    for p, flag in zip(main_parameters, flags):
                        p.requires_grad_(flag)
                
                
                
                # Match the official second phase: sampler remains in train
                # mode, including BatchNorm buffer updates, but is not stepped.
                optimizer_main.zero_grad()
                loss, _, _, cf = self.forward_model(src, tgt)
                loss.backward()
                optimizer_main.step()
                total += loss.item()
                alignment += cf.item()
            self.history_.append({'epoch': epoch + 1, 'loss': total,
                                  'alignment_loss': alignment})
            logger(epoch=epoch, loss=total, time=time.time() - start,
                   verbose=self.verbose, train=True)
        return self

    def predict(self, data, source=False):
        """Predict node logits for the supplied graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph to classify. Labels are optional.
        source : bool, optional
            Use source propagation steps. Default: False.

        Returns
        -------
        logits : torch.Tensor
            Class logits in input node order.
        labels : torch.Tensor or None
            Labels supplied with the graph, when available.
        """
        graph = self.process_graph(data)
        self.a2gnn.eval()
        with torch.no_grad():
            logits = self.a2gnn(graph, self.s_pnums if source else self.t_pnums)
        return logits, graph.y
