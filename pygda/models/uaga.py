"""Open-set node adaptation through the official UAGA DGL backend."""
import math
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import BaseGDA
from ..nn.uaga_base import UAGABase


class UAGA(BaseGDA):
    """Unknown-excluded adversarial graph domain alignment.

    Parameters
    ----------
    in_dim, hid_dim : int
        Input width and GAT width per attention head. Author hidden width: 32.
    num_classes : int
        Number K of known classes. Outputs have K+1 columns; column K is unknown.
        Source labels must be nonnegative integers, with known labels 0..K-1.
        Source nodes labelled K or above and their incident edges are removed.
    epoch, separation_epochs : int, optional
        Adaptation and separation lengths. Defaults: 200 and 30.
    num_layers, num_heads, num_out_heads : int, optional
        Encoder layer count, encoder and classifier heads. Defaults: 1, 8, 2.
    dropout, attn_dropout : float, optional
        Feature and attention dropout. Defaults: 0.4 (citation-v1 to dblp-v4).
        Other tasks require the author task settings documented in UAGA.md.
    lr, weight_decay : float, optional
        Adam settings, default 0.001 and 0.0001. Adam is reset between phases.
    batch_size : int, optional
        Paired node batch size, default 2048. Smaller domains repeat their same
        shuffled permutation to the larger domain's length. Zero uses that length.
    threshold, target_weight, top_k : float, float, int, optional
        Separation unknown probability, target pseudo-label loss weight and
        number of high-unknown-score target nodes. Defaults: 0.5, 0.1, 2000.
    seed : int or None, optional
        Reset Python, NumPy, Torch and DGL before initialization. Default: None.
    device, verbose : str, int, optional
        Default 'cuda:0', 2. All-neighbor sampling is required (num_neigh=-1).

    Notes
    -----
    Accepts PyG Data objects with dense x, edge_index and source y. fit never
    reads target y or masks. All target nodes participate without labels.
    Directed edges and their order are preserved; self-loops are replaced by
    one per node. Edge weights are ignored, matching official dgl.from_scipy.
    DGL is optional at import time and required for fit and predict.
    The author-version environment currently has an unresolved CPU exit crash;
    the tested reference device is RTX 4090. Modern DGL versions need validation.
    """
    def __init__(self, in_dim, hid_dim, num_classes, num_layers=1, dropout=0.4,
                 attn_dropout=None, num_heads=8, num_out_heads=2, lr=0.001,
                 weight_decay=0.0001, epoch=200, separation_epochs=30,
                 batch_size=2048, num_neigh=-1, threshold=0.5, target_weight=0.1,
                 top_k=2000, seed=None, device='cuda:0', verbose=2, **kwargs):
        if kwargs:
            raise TypeError('Unsupported UAGA arguments: ' + ', '.join(sorted(kwargs)))
        positive = (in_dim, hid_dim, num_classes, num_layers, num_heads, num_out_heads,
                    epoch, separation_epochs, top_k)
        if any(type(v) is not int or v < 1 for v in positive):
            raise ValueError('Dimensions, heads, epoch counts and top_k must be positive integers.')
        if type(batch_size) is not int or batch_size < 0 or num_neigh != -1:
            raise ValueError('batch_size must be nonnegative and num_neigh must be -1.')
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        values = (dropout, attn_dropout, lr, weight_decay, threshold, target_weight)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Numeric parameters must be finite.')
        if not (0 <= dropout < 1 and 0 <= attn_dropout < 1 and lr > 0 and
                weight_decay >= 0 and 0 <= threshold <= 1 and target_weight >= 0):
            raise ValueError('Invalid dropout, optimizer or loss parameters.')
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError('seed must be None or an integer in [0, 2**32).')
        if type(verbose) is not int or not 0 <= verbose <= 3:
            raise ValueError('verbose must be in [0, 3].')
        super().__init__(in_dim, hid_dim, num_classes, num_layers=num_layers, dropout=dropout,
                         lr=lr, weight_decay=weight_decay, epoch=epoch, batch_size=batch_size,
                         num_neigh=num_neigh, device=device, verbose=verbose)
        self.attn_dropout, self.num_heads, self.num_out_heads = attn_dropout, num_heads, num_out_heads
        self.separation_epochs, self.threshold = separation_epochs, threshold
        self.target_weight, self.top_k, self.seed = target_weight, top_k, seed
        self.history_ = []
        self.is_fitted_ = False

    def init_model(self, **kwargs):
        if kwargs:
            raise TypeError('init_model takes no additional arguments.')
        heads = [self.num_heads] * self.num_layers + [self.num_out_heads]
        return UAGABase(self.num_layers, self.in_dim, self.hid_dim, heads, self.dropout,
                        self.attn_dropout, nn.ReLU(True), self.num_classes + 1).to(self.device)

    def process_graph(self, data, include_labels=False):
        """Prepare copied tensors; source filtering is induced and order preserving."""
        x, edges = data.x, data.edge_index
        if (not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.ndim != 2
                or x.shape[0] == 0 or x.shape[1] != self.in_dim):
            raise ValueError('x must be a nonempty dense feature matrix with in_dim columns.')
        if (not isinstance(edges, torch.Tensor) or edges.dtype != torch.long or
                edges.ndim != 2 or edges.shape[0] != 2):
            raise ValueError('edge_index must have shape [2, E] and dtype long.')
        x, edges = x.detach().cpu().float().clone(), edges.detach().cpu().clone()
        if not bool(torch.isfinite(x).all()):
            raise ValueError('Features must be finite after conversion to float32.')
        if edges.numel() and (edges.min() < 0 or edges.max() >= len(x)):
            raise ValueError('Edge endpoints are out of range.')
        labels = None
        if include_labels:
            labels = data.y
            if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or
                    labels.ndim != 1 or len(labels) != len(x) or labels.min() < 0):
                raise ValueError('Source y must be a nonnegative long vector with one label per node.')
            labels = labels.detach().cpu().clone()
            keep = labels < self.num_classes
            if set(labels[keep].tolist()) != set(range(self.num_classes)):
                raise ValueError('Every known source class must have at least one node.')
            mapping = torch.full((len(x),), -1, dtype=torch.long)
            mapping[keep] = torch.arange(int(keep.sum()))
            edge_keep = keep[edges[0]] & keep[edges[1]]
            edges = mapping[edges[:, edge_keep]]
            x, labels = x[keep], labels[keep]
        return SimpleNamespace(x=x.to(self.device), edge_index=edges.to(self.device),
                               y=labels.to(self.device) if labels is not None else None)

    @staticmethod
    def _graph(prepared):
        import dgl
        graph = dgl.graph((prepared.edge_index[0].cpu(), prepared.edge_index[1].cpu()),
                          num_nodes=len(prepared.x)).to(prepared.x.device)
        return graph.remove_self_loop().add_self_loop()

    def _batch(self, graph, maximum, size):
        import dgl
        order = list(range(graph.num_nodes()))
        np.random.shuffle(order)
        repeated = ((maximum // len(order) + bool(maximum % len(order))) * order)[:maximum]
        indices = torch.tensor(repeated, dtype=torch.long, device=graph.device)
        collator = dgl.dataloading.NodeCollator(
            graph, indices, dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers + 1))
        yield from DataLoader(collator.dataset, size, collate_fn=collator.collate)

    def forward_model(self, source_data, target_data, step=0):
        """Loss for prepared batches: blocks, x, y; phase 1 also needs domain/grl.

        Target pseudo_label contains generated labels, never ground truth labels.
        Phase-1 domain labels and reversal coefficients cover the classifier
        block's source nodes, not just its seed nodes.
        """
        if self.model is None:
            raise RuntimeError('Initialize the model before forwarding.')
        if step not in (0, 1):
            raise ValueError('step must be 0 or 1.')
        _, source_logits, *source_domain = self.model(source_data.blocks, source_data.x, 1, False, step)
        coefficient = 1 if step == 0 else target_data.grl
        _, target_logits, *target_domain = self.model(target_data.blocks, target_data.x, coefficient, True, step)
        loss = nn.functional.cross_entropy(source_logits, source_data.y)
        if step == 0:
            return loss + nn.functional.binary_cross_entropy(
                target_logits.softmax(1)[:, -1], torch.full_like(target_logits[:, -1], self.threshold))
        mask = target_data.pseudo_label != -1
        target_loss = nn.functional.cross_entropy(target_logits[mask], target_data.pseudo_label[mask]) if mask.any() else 0
        domain_loss = nn.functional.binary_cross_entropy(
            torch.cat((source_domain[0], target_domain[0])), torch.cat((source_data.domain, target_data.domain)))
        return loss + self.target_weight * target_loss + domain_loss

    def _evaluate(self, graph, x, step, target=False, coefficient=1):
        self.model.eval()
        with torch.no_grad():
            result = self.model(graph, x, coefficient, target, step)
            embedding, logits = result[:2]
            return embedding, logits.argmax(1), logits.softmax(1)[:, -1]

    def _pseudo_labels(self, source_features, source_y, target_features, predictions, scores):
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
        """Train fixed phases; callback(model, phase, epoch) runs after each epoch.

        Source/private filtering uses only source labels. Target labels, target
        validation scores and target masks never enter training or model selection.
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
        gs, gt = self._graph(source), self._graph(target)
        self.model = self.init_model()
        maximum = max(len(source.x), len(target.x))
        size = min(self.batch_size or maximum, len(source.x), len(target.x))
        self.history_ = []
        source_features = target_features = predictions = scores = None
        for step, length in ((0, self.separation_epochs), (1, self.epoch)):
            parameters = ([{'params': self.model.generator.parameters()},
                           {'params': self.model.classifier.parameters()}] if step == 0 else self.model.parameters())
            optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
            for epoch in range(1, length + 1):
                if step == 1:
                    pseudo, coefficients = self._pseudo_labels(source_features, source.y,
                                                               target_features, predictions, scores)
                self.model.train()
                total, count = 0.0, 0
                for (si, so, sb), (ti, to, tb) in zip(self._batch(gs, maximum, size), self._batch(gt, maximum, size)):
                    source_batch = SimpleNamespace(blocks=sb, x=source.x[si], y=source.y[so])
                    target_batch = SimpleNamespace(blocks=tb, x=target.x[ti])
                    if step == 1:
                        source_batch.domain = torch.zeros(len(sb[-1].srcdata['_ID']), device=self.device)
                        target_batch.domain = torch.ones(len(tb[-1].srcdata['_ID']), device=self.device)
                        target_batch.grl = coefficients[tb[-1].srcdata['_ID']].reshape(-1, 1)
                        target_batch.pseudo_label = pseudo[to]
                    loss = self.forward_model(source_batch, target_batch, step)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Non-finite UAGA loss.')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())
                    count += 1
                source_features = self._evaluate(gs, source.x, step)[0]
                target_features, predictions, scores = self._evaluate(
                    gt, target.x, step, target=True, coefficient=coefficients if step else 1)
                self.history_.append(dict(phase=step, epoch=epoch, loss=total/count, steps=count))
                if self.verbose:
                    print('UAGA phase={} epoch={} loss={:.6f}'.format(step, epoch, total/count), flush=True)
                if epoch_callback is not None:
                    epoch_callback(self, step, epoch)
        self.is_fitted_ = True
        return self

    def predict(self, data, source=False):
        """Return logits and optional mapped labels for the supplied graph.

        Default logits have K+1 columns. source=True filters private source
        nodes and returns only K known-class columns, as in official evaluation.
        """
        if not self.is_fitted_:
            raise RuntimeError('Call fit before predict.')
        prepared = self.process_graph(data, include_labels=source)
        graph = self._graph(prepared)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(graph, prepared.x, 1, not source, 1)[1]
        if source:
            return logits[:, :self.num_classes], prepared.y
        labels = getattr(data, 'y', None)
        if labels is not None:
            if labels.dtype != torch.long or labels.ndim != 1 or len(labels) != len(prepared.x) or labels.min() < 0:
                raise ValueError('Prediction y must be a nonnegative long vector.')
            labels = labels.detach().to(self.device).clone().clamp(max=self.num_classes)
        return logits, labels
