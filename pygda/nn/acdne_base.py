import torch
import torch.nn as nn
import torch.nn.functional as F

from .reverse_layer import GradReverse


class FE1(nn.Module):
    """
    First Feature Encoder for self-features.

    Parameters
    ----------
    n_input : int
        Input feature dimension.
    n_hidden : list
        Hidden layer dimensions [h1_dim, h2_dim].
    drop : float
        Dropout rate.

    Notes
    -----
    - Two-layer neural network for self-feature encoding
    - Uses truncated normal initialization
    - Applies ReLU activation and dropout
    """

    def __init__(self, n_input, n_hidden, drop):
        super(FE1, self).__init__()
        self.drop = drop
        self.h1_self = nn.Linear(n_input, n_hidden[0])
        self.h2_self = nn.Linear(n_hidden[0], n_hidden[1])
        std = 1/(n_input/2)**0.5
        nn.init.trunc_normal_(self.h1_self.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.h1_self.bias, 0.1)
        std = 1/(n_hidden[0]/2)**0.5
        nn.init.trunc_normal_(self.h2_self.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.h2_self.bias, 0.1)

    def forward(self, x):
        """
        Forward pass of self-feature encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.

        Returns
        -------
        torch.Tensor
            Encoded self-features.

        Notes
        -----
        - Two-layer transformation with ReLU
        - Dropout after first layer
        """
        x = F.dropout(F.relu(self.h1_self(x)), self.drop)
        return F.relu(self.h2_self(x))


class FE2(nn.Module):
    """
    Second Feature Encoder for neighbor-features.

    Parameters
    ----------
    n_input : int
        Input feature dimension.
    n_hidden : list
        Hidden layer dimensions [h1_dim, h2_dim].
    drop : float
        Dropout rate.

    Notes
    -----
    - Parallel to FE1 but processes neighbor features
    - Identical architecture to FE1
    - Separate parameters for neighbor processing
    """

    def __init__(self, n_input, n_hidden, drop):
        super(FE2, self).__init__()
        self.drop = drop
        self.h1_nei = nn.Linear(n_input, n_hidden[0])
        self.h2_nei = nn.Linear(n_hidden[0], n_hidden[1])
        std = 1/(n_input/2)**0.5
        nn.init.trunc_normal_(self.h1_nei.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.h1_nei.bias, 0.1)
        std = 1/(n_hidden[0]/2)**0.5
        nn.init.trunc_normal_(self.h2_nei.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.h2_nei.bias, 0.1)

    def forward(self, x_nei):
        """
        Forward pass of neighbor-feature encoder.

        Parameters
        ----------
        x_nei : torch.Tensor
            Input neighbor features.

        Returns
        -------
        torch.Tensor
            Encoded neighbor features.
        """
        x_nei = F.dropout(F.relu(self.h1_nei(x_nei)), self.drop)
        return F.relu(self.h2_nei(x_nei))


class NetworkEmbedding(nn.Module):
    """
    Network Embedding module combining self and neighbor features.

    Parameters
    ----------
    n_input : int
        Input feature dimension.
    n_hidden : list
        Hidden layer dimensions.
    n_emb : int
        Final embedding dimension.
    drop : float
        Dropout rate.
    batch_size : int
        Size of mini-batches.

    Notes
    -----
    - Combines FE1 and FE2 outputs
    - Projects combined features to embedding space
    - Supports pairwise constraints for domain adaptation
    """

    def __init__(self, n_input, n_hidden, n_emb, drop, batch_size):
        super(NetworkEmbedding, self).__init__()
        self.drop = drop
        self.batch_size = batch_size
        self.fe1 = FE1(n_input, n_hidden, drop)
        self.fe2 = FE2(n_input, n_hidden, drop)
        self.emb = nn.Linear(n_hidden[-1]*2, n_emb)
        std = 1/(n_hidden[-1]*2)**0.5
        nn.init.trunc_normal_(self.emb.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.emb.bias, 0.1)

    def forward(self, x, x_nei):
        """
        Forward pass of network embedding.

        Parameters
        ----------
        x : torch.Tensor
            Self features.
        x_nei : torch.Tensor
            Neighbor features.

        Returns
        -------
        torch.Tensor
            Combined network embedding.
        """
        h2_self = self.fe1(x)
        h2_nei = self.fe2(x_nei)
        return F.relu(self.emb(torch.cat((h2_self, h2_nei), 1)))

    def pairwise_constraint(self, emb):
        """
        Split embeddings into source and target domains.

        Parameters
        ----------
        emb : torch.Tensor
            Combined embeddings.

        Returns
        -------
        tuple
            Source and target embeddings.
        """
        emb_s = emb[:int(self.batch_size/2), :]
        emb_t = emb[int(self.batch_size/2):, :]
        return emb_s, emb_t

    @staticmethod
    def net_pro_loss(emb, a):
        """
        Network proximity loss computation.

        Parameters
        ----------
        emb : torch.Tensor
            Network embeddings.
        a : torch.Tensor
            Adjacency matrix.

        Returns
        -------
        torch.Tensor
            Network proximity loss.

        Notes
        -----
        - Computes pairwise distances in embedding space
        - Weighted by adjacency matrix
        """
        r = torch.sum(emb*emb, 1)
        r = torch.reshape(r, (-1, 1))
        dis = r-2*torch.matmul(emb, emb.T)+r.T
        return torch.mean(torch.sum(a.clone().detach().__mul__(dis), 1))


class NodeClassifier(nn.Module):
    """
    Node classification layer.

    Parameters
    ----------
    n_emb : int
        Input embedding dimension.
    num_class : int
        Number of classes.

    Notes
    -----
    - Single linear layer classifier
    - Uses truncated normal initialization
    """

    def __init__(self, n_emb, num_class):
        super(NodeClassifier, self).__init__()
        self.layer = nn.Linear(n_emb, num_class)
        std = 1/(n_emb/2)**0.5
        nn.init.trunc_normal_(self.layer.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.layer.bias, 0.1)

    def forward(self, emb):
        """
        Forward pass of classifier.

        Parameters
        ----------
        emb : torch.Tensor
            Input embeddings.

        Returns
        -------
        torch.Tensor
            Classification logits.
        """
        pred_logit = self.layer(emb)
        return pred_logit


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial training.

    Parameters
    ----------
    n_emb : int
        Input embedding dimension.

    Notes
    -----
    - Three-layer neural network
    - Binary domain classification
    - Used with gradient reversal
    """

    def __init__(self, n_emb):
        super(DomainDiscriminator, self).__init__()
        self.h_dann_1 = nn.Linear(n_emb, 128)
        self.h_dann_2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 2)
        std = 1/(n_emb/2)**0.5
        nn.init.trunc_normal_(self.h_dann_1.weight, std=std, a=-2*std, b=2*std)
        nn.init.constant_(self.h_dann_1.bias, 0.1)
        nn.init.trunc_normal_(self.h_dann_2.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.constant_(self.h_dann_2.bias, 0.1)
        nn.init.trunc_normal_(self.output_layer.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, h_grl):
        """
        Forward pass of domain discriminator.

        Parameters
        ----------
        h_grl : torch.Tensor
            Input features after gradient reversal.

        Returns
        -------
        torch.Tensor
            Domain classification logits.
        """
        h_grl = F.relu(self.h_dann_1(h_grl))
        h_grl = F.relu(self.h_dann_2(h_grl))
        d_logit = self.output_layer(h_grl)
        return d_logit


class ACDNEBase(nn.Module):
    """
    Base class for ACDNE.

    Parameters
    ----------
    n_input : int
        Input feature dimension.
    n_hidden : list
        Hidden layer dimensions.
    n_emb : int
        Embedding dimension.
    num_class : int
        Number of classes.
    batch_size : int
        Size of mini-batches.
    drop : float
        Dropout rate.

    Notes
    -----
    Architecture components:

    1. Network embedding module
    2. Node classifier
    3. Domain discriminator
    """

    def __init__(self, n_input, n_hidden, n_emb, num_class, batch_size, drop):
        super(ACDNEBase, self).__init__()
        self.network_embedding = NetworkEmbedding(n_input, n_hidden, n_emb, drop, batch_size)
        self.node_classifier = NodeClassifier(n_emb, num_class)
        self.domain_discriminator = DomainDiscriminator(n_emb)

    def forward(self, x, x_nei, alpha):
        """
        Forward pass of ACDNE model.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        x_nei : torch.Tensor
            Neighbor features.
        alpha : float
            Gradient reversal scaling parameter.

        Returns
        -------
        tuple
            Contains:
            - emb : Network embeddings
            - pred_logit : Classification logits
            - d_logit : Domain classification logits

        Notes
        -----
        Three-stage process:
        1. Network embedding
        2. Node classification
        3. Domain discrimination with gradient reversal
        """
        # Network_Embedding
        emb = self.network_embedding(x, x_nei)
        # Node_Classifier
        pred_logit = self.node_classifier(emb)
        # Domain_Discriminator
        d_logit = self.domain_discriminator(GradReverse.apply(emb, alpha))
        return emb, pred_logit, d_logit
