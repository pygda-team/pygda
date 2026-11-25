import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .base import BaseGDA
from ..nn.uaga_base import UAGABase
from ..utils import logger


class UAGA(BaseGDA):
    """Unknown-excluded adversarial graph domain alignment.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension per encoder attention head.
    num_classes : int
        Number of known classes. The model adds one output for the unknown class.
    num_layers : int, optional
        Number of encoder layers. Default: ``1``.
    dropout : float, optional
        Feature dropout rate. Default: ``0.4``.
    attn_dropout : float or None, optional
        Attention dropout rate. If ``None``, use ``dropout``. Default: ``None``.
    num_heads : int, optional
        Number of attention heads in each encoder layer. Default: ``8``.
    num_out_heads : int, optional
        Number of attention heads in the classifier. Default: ``2``.
    lr : float, optional
        Learning rate. Default: ``0.001``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.0001``.
    epoch : int, optional
        Number of adaptation epochs. Default: ``200``.
    separation_epochs : int, optional
        Number of separation epochs. Default: ``30``.
    batch_size : int, optional
        Number of seed nodes per domain in each minibatch, capped by the
        smaller domain size. Use ``0`` for that cap. Default: ``2048``.
    num_neigh : int, optional
        Number of neighbors in sampling. UAGA requires all neighbors.
        Default: ``-1``.
    threshold : float, optional
        Reference unknown-class probability in the separation loss.
        Default: ``0.5``.
    target_weight : float, optional
        Weight of the target pseudo-label loss. Default: ``0.1``.
    top_k : int, optional
        Number of target nodes with the highest unknown-class scores used
        to initialize the unknown cluster center. Default: ``2000``.
    seed : int or None, optional
        Random seed. If ``None``, do not reset the random state before
        training. Default: ``None``.
    device : str, optional
        Device for training and prediction. Default: ``cuda:0``.
    verbose : int, optional
        Verbosity level in [0, 3]. Default: ``2``.
    **kwargs
        Additional model parameters are not supported.

"""

    def __init__(
        self,
        in_dim,
        hid_dim,
        num_classes,
        num_layers=1,
        dropout=0.4,
        attn_dropout=None,
        num_heads=8,
        num_out_heads=2,
        lr=0.001,
        weight_decay=0.0001,
        epoch=200,
        separation_epochs=30,
        batch_size=2048,
        num_neigh=-1,
        threshold=0.5,
        target_weight=0.1,
        top_k=2000,
        seed=None,
        device='cuda:0',
        verbose=2,
        **kwargs,
    ):
        
        super().__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            epoch=epoch,
            batch_size=batch_size,
            num_neigh=num_neigh,
            device=device,
            verbose=verbose,
        )
        self.attn_dropout = dropout if attn_dropout is None else attn_dropout
        self.num_heads = num_heads
        self.num_out_heads = num_out_heads
        self.separation_epochs = separation_epochs
        self.threshold = threshold
        self.target_weight = target_weight
        self.top_k = top_k
        self.seed = seed
        self.history_ = []
        self.is_fitted_ = False

    def init_model(self, **kwargs):
        """
        Initialize the UAGA base model.

        Parameters
        ----------
        **kwargs
            Additional initialization arguments. Currently unused.

        Returns
        -------
        UAGABase
            Initialized model on the configured device.

        Notes
        -----
        Configures model with:

        - A GAT encoder with the configured layers and attention heads
        - A graph attention classifier with one extra output for the unknown class
        - A domain discriminator for the adaptation stage
        """
        heads = [self.num_heads] * self.num_layers + [self.num_out_heads]
        return UAGABase(
            self.num_layers,
            self.in_dim,
            self.hid_dim,
            heads,
            self.dropout,
            self.attn_dropout,
            nn.ReLU(True),
            self.num_classes + 1,
        ).to(self.device)

    def process_graph(self, data, include_labels=False):
        """
        Process the input graph data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph containing node features and edge indices.
        include_labels : bool, optional
            Whether to read source labels and keep only known-class nodes.
            Default: ``False``.

        Notes
        -----
        Prepares the graph through:

        - Feature conversion to float32
        - Source graph filtering and node relabeling
        - Device placement for the prepared tensors

        Target graphs are processed without reading their labels.
        """
    
        x, edges = data.x, data.edge_index
        x, edges = x.detach().cpu().float().clone(), edges.detach().cpu().clone()
       
        labels = None
        if include_labels:
            labels = data.y
            labels = labels.detach().cpu().clone()
            keep = labels < self.num_classes
            
            mapping = torch.full((len(x),), -1, dtype=torch.long)
            mapping[keep] = torch.arange(int(keep.sum()))
            edge_keep = keep[edges[0]] & keep[edges[1]]
            edges = mapping[edges[:, edge_keep]]
            x, labels = x[keep], labels[keep]
        return SimpleNamespace(x=x.to(self.device), edge_index=edges.to(self.device),
                               y=labels.to(self.device) if labels is not None else None)

    @staticmethod
    def _graph(prepared):
        """
        Build a DGL graph from the prepared input.

        Parameters
        ----------
        prepared : SimpleNamespace
            Node features and edge indices returned by ``process_graph``.

        Returns
        -------
        dgl.DGLGraph
            Graph on the same device as the node features.

        Notes
        -----
        Preserves isolated nodes and replaces existing self-loops with one per node.
        """
        import dgl
        graph = dgl.graph((prepared.edge_index[0].cpu(), prepared.edge_index[1].cpu()),
                          num_nodes=len(prepared.x)).to(prepared.x.device)
        return graph.remove_self_loop().add_self_loop()

    def _batch(self, graph, maximum, size):
        """
        Sample node batches from the input graph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Source or target graph to sample.
        maximum : int
            Total number of seed nodes to draw, including repeats.
        size : int
            Number of seed nodes in each minibatch.

        Yields
        ------
        tuple
            Input node IDs, seed node IDs, and DGL blocks.

        Notes
        -----
        Shuffles nodes once, repeats the order to match the larger domain, and
        samples all neighbors for the encoder and classifier.
        """
        import dgl
        order = list(range(graph.num_nodes()))
        np.random.shuffle(order)
        repeated = ((maximum // len(order) + bool(maximum % len(order))) * order)[:maximum]
        
        indices = torch.tensor(repeated, dtype=torch.long, device=graph.device)
        
        collator = dgl.dataloading.NodeCollator(
            graph, indices, dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers + 1))
        
        yield from DataLoader(collator.dataset, size, collate_fn=collator.collate)

    def forward_model(self, source_data, target_data, step=0):
        """
        Compute the training loss for paired source and target batches.

        Parameters
        ----------
        source_data : SimpleNamespace
            Sampled blocks, features, and source labels.
        target_data : SimpleNamespace
            Sampled blocks and features; adaptation also uses pseudo-labels and
            gradient reversal coefficients.
        step : int, optional
            Training stage: ``0`` for separation or ``1`` for adaptation. Default: ``0``.

        Returns
        -------
        torch.Tensor
            Combined loss for the current batch.

        Notes
        -----
        Both stages use source classification loss. Separation adds a loss on
        target unknown-class probabilities. Adaptation adds pseudo-label and
        domain classification losses. Domain labels cover the source side of
        the last DGL block, including neighbors of seed nodes.
        """
        
        _, source_logits, *source_domain = self.model(
            source_data.blocks, source_data.x, 1, False, step
        )
        coefficient = 1 if step == 0 else target_data.grl
        _, target_logits, *target_domain = self.model(
            target_data.blocks, target_data.x, coefficient, True, step
        )
        source_loss = F.cross_entropy(source_logits, source_data.y)

        # Separation: move target unknown probabilities toward the threshold.
        if step == 0:
            unknown_scores = target_logits.softmax(1)[:, -1]
            threshold = torch.full_like(target_logits[:, -1], self.threshold)
            separation_loss = F.binary_cross_entropy(unknown_scores, threshold)
            return source_loss + separation_loss

        # Adaptation: use accepted pseudo-labels and align the two domains.
        mask = target_data.pseudo_label != -1
        target_loss = 0
        if mask.any():
            target_loss = F.cross_entropy(target_logits[mask], target_data.pseudo_label[mask])
        domain_scores = torch.cat((source_domain[0], target_domain[0]))
        domain_labels = torch.cat((source_data.domain, target_data.domain))
        domain_loss = F.binary_cross_entropy(domain_scores, domain_labels)
        return source_loss + self.target_weight * target_loss + domain_loss

    def _evaluate(self, graph, x, step, target=False, coefficient=1):
        """
        Run inference on the full graph.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Graph to evaluate.
        x : torch.Tensor
            Node features.
        step : int
            Training stage used for the forward pass.
        target : bool, optional
            Whether the graph is from the target domain. Default: ``False``.
        coefficient : float or torch.Tensor, optional
            Gradient reversal coefficient. Default: ``1``.

        Returns
        -------
        tuple
            Node embeddings, predicted classes, and unknown-class probabilities.

        Notes
        -----
        Runs without gradients. Outputs are used to generate pseudo-labels.
        """
        self.model.eval()
        with torch.no_grad():
            result = self.model(graph, x, coefficient, target, step)
            embedding, logits = result[:2]
            return embedding, logits.argmax(1), logits.softmax(1)[:, -1]

    def _pseudo_labels(self, source_features, source_y, target_features, predictions, scores):
        """
        Generate target pseudo-labels and gradient reversal coefficients.

        Parameters
        ----------
        source_features : torch.Tensor
            Source node embeddings.
        source_y : torch.Tensor
            Known-class source labels.
        target_features : torch.Tensor
            Target node embeddings.
        predictions : torch.Tensor
            Predicted target classes.
        scores : torch.Tensor
            Target unknown-class probabilities.

        Returns
        -------
        tuple
            Target pseudo-labels and a reversal coefficient for each target node.

        Notes
        -----
        Accepts a KMeans label when it agrees with the model prediction. Among
        the highest-scoring unknown-class candidates, marks unassigned nodes
        as unknown. Other unassigned nodes retain label ``-1``. Unknown nodes
        receive reversal coefficient ``-1``; all other nodes receive ``1``.
        """
        from sklearn.cluster import KMeans
        indices = scores.argsort()[-self.top_k:]
        centers = torch.cat([source_features[source_y == k].mean(0, True)
                             for k in range(self.num_classes)] + [target_features[indices].mean(0, True)])
        clusters = KMeans(self.num_classes + 1, init=np.array(centers.tolist()), n_init=1).fit_predict(
            np.array(target_features.tolist()))
        clusters = torch.tensor(clusters, dtype=torch.long, device=target_features.device)
        pseudo = torch.full_like(predictions, -1)
        accepted = clusters == predictions
        pseudo[accepted] = clusters[accepted]
        unassigned = indices[pseudo[indices] == -1]
        pseudo[unassigned] = self.num_classes
        coefficient = torch.ones(len(pseudo), device=pseudo.device)
        coefficient[pseudo == self.num_classes] = -1
        return pseudo, coefficient

    def fit(self, source_data, target_data, epoch_callback=None):
        """
        Train UAGA on the source and target graphs.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source graph with features, edges, and class labels.
        target_data : torch_geometric.data.Data
            Target graph with features and edges.
        epoch_callback : callable or None, optional
            Called with ``(self, step, epoch)`` after each epoch. Default: ``None``.

        Returns
        -------
        UAGA
            The fitted model.

        Notes
        -----
        Separation trains the encoder and classifier. Adaptation starts with
        fresh optimizer state, also trains the domain discriminator, and updates
        pseudo-labels each epoch. Target ground-truth labels are not used.
        """
        import dgl
        self.is_fitted_ = False
        source = self.process_graph(source_data, include_labels=True)
        target = self.process_graph(target_data)
        if self.seed is not None:
            random.seed(self.seed)
            dgl.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.device(self.device).type == 'cuda':
                torch.cuda.manual_seed_all(self.seed)
        source_graph = self._graph(source)
        target_graph = self._graph(target)
        self.model = self.init_model()
        maximum = max(len(source.x), len(target.x))
        size = min(self.batch_size or maximum, len(source.x), len(target.x))
        self.history_ = []
        source_features = target_features = predictions = scores = None
        start_time = time.perf_counter()
        for step, length in ((0, self.separation_epochs), (1, self.epoch)):
            if self.verbose:
                phase = 'separation' if step == 0 else 'adaptation'
                print(f'UAGA {phase}: {length} epochs', flush=True)
            if step == 0:
                parameters = [
                    {'params': self.model.generator.parameters()},
                    {'params': self.model.classifier.parameters()},
                ]
            else:
                parameters = self.model.parameters()

            # The official schedule starts adaptation with fresh Adam state.
            optimizer = torch.optim.Adam(
                parameters, lr=self.lr, weight_decay=self.weight_decay
            )
            for epoch in range(1, length + 1):
                if step == 1:
                    pseudo, coefficients = self._pseudo_labels(source_features, source.y,
                                                               target_features, predictions, scores)
                self.model.train()
                total, count = 0.0, 0
                source_loader = self._batch(source_graph, maximum, size)
                target_loader = self._batch(target_graph, maximum, size)
                for source_sample, target_sample in zip(source_loader, target_loader):
                    source_inputs, source_seeds, source_blocks = source_sample
                    target_inputs, target_seeds, target_blocks = target_sample
                    source_batch = SimpleNamespace(
                        blocks=source_blocks, x=source.x[source_inputs], y=source.y[source_seeds]
                    )
                    target_batch = SimpleNamespace(blocks=target_blocks, x=target.x[target_inputs])
                    if step == 1:
                        source_nodes = source_blocks[-1].srcdata['_ID']
                        target_nodes = target_blocks[-1].srcdata['_ID']
                        source_batch.domain = torch.zeros(len(source_nodes), device=self.device)
                        target_batch.domain = torch.ones(len(target_nodes), device=self.device)
                        target_batch.grl = coefficients[target_nodes].reshape(-1, 1)
                        target_batch.pseudo_label = pseudo[target_seeds]
                    loss = self.forward_model(source_batch, target_batch, step)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Non-finite UAGA loss.')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())
                    count += 1
                source_features = self._evaluate(source_graph, source.x, step)[0]
                target_features, predictions, scores = self._evaluate(
                    target_graph, target.x, step, target=True,
                    coefficient=coefficients if step else 1,
                )
                epoch_loss = total / count
                self.history_.append(dict(phase=step, epoch=epoch, loss=epoch_loss, steps=count))
                logger(epoch=epoch, loss=epoch_loss, time=time.perf_counter() - start_time,
                       verbose=self.verbose, train=True)
                if epoch_callback is not None:
                    epoch_callback(self, step, epoch)
        self.is_fitted_ = True
        return self

    def predict(self, data, source=False):
        """
        Make predictions on the supplied graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph to predict. Target labels are optional.
        source : bool, optional
            Whether to predict on the source domain. Default: ``False``.

        Returns
        -------
        tuple
            Classification logits and labels. Labels are ``None`` for unlabeled targets.

        Notes
        -----
        Source prediction filters private-class nodes and returns known-class
        logits. Target prediction returns one extra logit for the unknown class
        and maps private labels to that class when labels are present.
        """
        
        prepared = self.process_graph(data, include_labels=source)
        graph = self._graph(prepared)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(graph, prepared.x, 1, not source, 1)[1]
        if source:
            return logits[:, :self.num_classes], prepared.y
        labels = getattr(data, 'y', None)
        if labels is not None:
            labels = labels.to(self.device).clamp(max=self.num_classes)
        return logits, labels
