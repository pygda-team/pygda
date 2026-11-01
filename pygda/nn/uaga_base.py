"""DGL backbone for UAGA, adapted from 3480430977/UAGA (4ef2f8f).

The original parameter names and initialization order are retained. DGL is
imported only when constructing the network. Gradient reversal coefficients
belong to each autograd invocation rather than mutable class attributes.
"""
from torch import nn
from torch.autograd import Function


def _gat(*args, **kwargs):
    try:
        from dgl.nn.pytorch import GATConv
    except ImportError as exc:
        raise ImportError('UAGA requires the optional DGL PyTorch backend. See docs/models/UAGA.md.') from exc
    return GATConv(*args, **kwargs)


class _Reverse(Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class DomainDiscriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear0 = nn.Linear(in_features, 128)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear2(self.relu(self.linear1(self.relu(self.linear0(x)))))).reshape(-1)


class GAT(nn.Module):
    def __init__(self, num_layers, in_feats, num_hidden, heads, feat_drop, attn_drop, activation):
        super().__init__()
        self.gat_layers = nn.ModuleList((_gat(in_feats, num_hidden, heads[0], feat_drop,
                                              attn_drop, activation=activation),))
        for layer in range(1, num_layers):
            self.gat_layers.append(_gat(num_hidden * heads[layer-1], num_hidden, heads[layer],
                                        feat_drop, attn_drop, activation=activation))

    def forward(self, blocks_or_graph, x):
        for layer, conv in enumerate(self.gat_layers):
            graph = blocks_or_graph[layer] if self.training else blocks_or_graph
            x = conv(graph, x).flatten(1)
        return x


class Classifier(nn.Module):
    def __init__(self, in_feats, num_classes, heads, feat_drop, attn_drop):
        super().__init__()
        self.gat_conv = _gat(in_feats, num_classes, heads, feat_drop, attn_drop)

    def forward(self, graph, x):
        return self.gat_conv(graph, x).mean(1)


class UAGABase(nn.Module):
    """Official GAT encoder, graph classifier and domain discriminator.

    ``np1class`` counts known classes plus one unknown output. In training,
    supply a list of ``num_layers + 1`` DGL blocks; in evaluation supply a graph.
    Domain predictions cover the last block's source nodes, including neighbors.
    """
    def __init__(self, num_layers, in_feats, num_hidden, heads, feat_drop,
                 attn_drop, activation, np1class):
        super().__init__()
        hidden_size = num_hidden * heads[-2]
        self.classifier = Classifier(hidden_size, np1class, heads[-1], feat_drop, attn_drop)
        self.generator = GAT(num_layers, in_feats, num_hidden, heads, feat_drop, attn_drop, activation)
        self.domain_discriminator = DomainDiscriminator(hidden_size)

    def forward(self, blocks_or_graph, x, grl_lambda, is_target, step):
        if step not in (0, 1):
            raise ValueError('step must be 0 (separation) or 1 (adaptation).')
        encoder_graph = blocks_or_graph[:-1] if self.training else blocks_or_graph
        classifier_graph = blocks_or_graph[-1] if self.training else blocks_or_graph
        x = self.generator(encoder_graph, x)
        if step == 0:
            if is_target:
                x = _Reverse.apply(x, grl_lambda)
            return x, self.classifier(classifier_graph, x)
        return x, self.classifier(classifier_graph, x), self.domain_discriminator(_Reverse.apply(x, grl_lambda))
