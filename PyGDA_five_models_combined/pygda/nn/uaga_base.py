from torch import nn

from .reverse_layer import GradReverse


def _gat(*args, **kwargs):
    """Create a GATConv without requiring DGL when importing PyGDA."""
    try:
        from dgl.nn.pytorch import GATConv
    except ImportError as exc:
        raise ImportError('UAGA requires the optional DGL PyTorch backend. See docs/models/UAGA.md.') from exc
    return GATConv(*args, **kwargs)


class DomainDiscriminator(nn.Module):
    """
    Predict whether a node embedding comes from the target domain.

    Parameters
    ----------
    in_features : int
        Input embedding dimension.
    """

    def __init__(self, in_features):
        super().__init__()
        self.linear0 = nn.Linear(in_features, 128)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Compute domain probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Node embeddings.

        Returns
        -------
        torch.Tensor
            One target-domain probability per input node.
        """
        return self.sigmoid(
            self.linear2(self.relu(self.linear1(self.relu(self.linear0(x)))))
        ).reshape(-1)


class GAT(nn.Module):
    """
    Encode nodes with multi-head graph attention.

    Parameters
    ----------
    num_layers : int
        Number of encoder layers.
    in_feats : int
        Input feature dimension.
    num_hidden : int
        Output dimension per attention head.
    heads : list of int
        Attention head counts. The encoder uses the first ``num_layers`` entries.
    feat_drop : float
        Feature dropout rate.
    attn_drop : float
        Attention dropout rate.
    activation : callable or None
        Activation applied by each attention layer.
    """

    def __init__(self,
                 num_layers,
                 in_feats,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 activation):
        super().__init__()
        self.gat_layers = nn.ModuleList((
            _gat(in_feats, num_hidden, heads[0], feat_drop, attn_drop,
                 activation=activation),
        ))
        for layer in range(1, num_layers):
            self.gat_layers.append(
                _gat(num_hidden * heads[layer - 1], num_hidden, heads[layer],
                     feat_drop, attn_drop, activation=activation)
            )

    def forward(self, blocks_or_graph, x):
        """
        Extract node embeddings.

        Parameters
        ----------
        blocks_or_graph : list of dgl.DGLBlock or dgl.DGLGraph
            Encoder blocks during training or the full graph during evaluation.
        x : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            Node embeddings with the attention heads concatenated.
        """
        for layer, conv in enumerate(self.gat_layers):
            graph = blocks_or_graph[layer] if self.training else blocks_or_graph
            x = conv(graph, x).flatten(1)
        return x


class Classifier(nn.Module):
    """
    Classify nodes with a graph attention layer.

    Parameters
    ----------
    in_feats : int
        Input embedding dimension.
    num_classes : int
        Number of outputs, including the unknown class.
    heads : int
        Number of classifier attention heads.
    feat_drop : float
        Feature dropout rate.
    attn_drop : float
        Attention dropout rate.
    """

    def __init__(self, in_feats, num_classes, heads, feat_drop, attn_drop):
        super().__init__()
        self.gat_conv = _gat(in_feats, num_classes, heads, feat_drop, attn_drop)

    def forward(self, graph, x):
        """
        Compute class logits.

        Parameters
        ----------
        graph : dgl.DGLBlock or dgl.DGLGraph
            Classifier block during training or the full graph during evaluation.
        x : torch.Tensor
            Input node embeddings.

        Returns
        -------
        torch.Tensor
            Class logits averaged over the classifier attention heads.
        """
        return self.gat_conv(graph, x).mean(1)


class UAGABase(nn.Module):
    """
    Open-Set Cross-Network Node Classification via Unknown-Excluded
    Adversarial Graph Domain Alignment (AAAI-25).

    Parameters
    ----------
    num_layers : int
        Number of GAT encoder layers.
    in_feats : int
        Input feature dimension.
    num_hidden : int
        Hidden dimension per encoder attention head.
    heads : list of int
        Attention head counts for the encoder layers followed by the classifier.
    feat_drop : float
        Feature dropout rate.
    attn_drop : float
        Attention dropout rate.
    activation : callable or None
        Activation applied by the encoder attention layers.
    np1class : int
        Number of known classes plus one output for the unknown class.
    """

    def __init__(self,
                 num_layers,
                 in_feats,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 activation,
                 np1class):
        super().__init__()
        hidden_size = num_hidden * heads[-2]
        self.classifier = Classifier(hidden_size, np1class, heads[-1], feat_drop, attn_drop)
        self.generator = GAT(num_layers, in_feats, num_hidden, heads, feat_drop, attn_drop, activation)
        self.domain_discriminator = DomainDiscriminator(hidden_size)

    def forward(self, blocks_or_graph, x, grl_lambda, is_target, step):
        """
        Compute node embeddings, class logits, and optional domain probabilities.

        Parameters
        ----------
        blocks_or_graph : list of dgl.DGLBlock or dgl.DGLGraph
            During training, supply one block per encoder layer and a final
            classifier block. During evaluation, supply the full graph.
        x : torch.Tensor
            Node features for the first block or full graph.
        grl_lambda : float or torch.Tensor
            Gradient reversal coefficient, scalar or broadcastable to the node
            embeddings.
        is_target : bool
            Whether the input is from the target domain.
        step : int
            Training stage: ``0`` for separation or ``1`` for adaptation.

        Returns
        -------
        tuple
            Node embeddings and class logits. In adaptation, also returns
            domain probabilities for the input nodes of the classifier block
            during training, or all nodes during evaluation.

        Notes
        -----
        Separation reverses target gradients before classification.
        Adaptation applies gradient reversal only on the domain branch.
        """
        if step not in (0, 1):
            raise ValueError('step must be 0 (separation) or 1 (adaptation).')
        encoder_graph = blocks_or_graph[:-1] if self.training else blocks_or_graph
        classifier_graph = blocks_or_graph[-1] if self.training else blocks_or_graph
        x = self.generator(encoder_graph, x)
        if step == 0:
            if is_target:
                x = GradReverse.apply(x, grl_lambda)
            return x, self.classifier(classifier_graph, x)
        return (
            x,
            self.classifier(classifier_graph, x),
            self.domain_discriminator(GradReverse.apply(x, grl_lambda)),
        )
