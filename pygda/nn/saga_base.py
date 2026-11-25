import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGABase(nn.Module):
    """
    Network and training objectives from the public SAGA implementation.

    Parameters
    ----------
    feats_dim : int
        Width of raw and propagated node features.
    sub_num : int
        Number of ordered graph views.
    hidden_dim : int
        Hidden width of the encoder and decoder.
    embed_dim : int
        Embedding width for each view.
    tau : float
        Temperature for the cross-alignment objective.
    dropout : float
        Dropout probability in the encoder and decoder.
    nnodes : int
        Retained author argument for the node count. Unused by this backbone.
    dominant_index : int
        Initial view index retained for author compatibility. Forward mines
        dominant views from the current batch.
    device : str or torch.device
        Device for loss tensors created during forward.
    nb_classes : int
        Number of class logits per node.
    config : object
        Configuration exposing alpha, beta and delta in its loss_weights mapping.
    """

    def __init__(self,
                 feats_dim,
                 sub_num,
                 hidden_dim,
                 embed_dim,
                 tau,
                 dropout,
                 nnodes,
                 dominant_index,
                 device,
                 nb_classes,
                 config):
        super(SAGABase, self).__init__()
        self.feats_dim = feats_dim
        self.embed_dim = embed_dim
        self.sub_num = sub_num
        self.tau = tau
        self.device = device
        self.dominant_index = dominant_index
        self.dropout = dropout
        self.lam_sad = 1
        self.config = config

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feats_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feats_dim)
        )

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(self.sub_num)
        ])

        self.classifier = nn.Linear(embed_dim * sub_num, nb_classes)

    def forward(self, feat_t, f_list_t, feat_s, f_list_s, label_s, re_computing=False):
        """
        Compute the training loss for the provided domain batches.

        Parameters
        ----------
        feat_t : torch.Tensor or None
            Raw target node features of shape (n_target, feats_dim).
        f_list_t : list of torch.Tensor or None
            Propagated target features in view order, each of shape
            (n_target, feats_dim). Use None when the target batch is absent.
        feat_s : torch.Tensor or None
            Raw source node features of shape (n_source, feats_dim).
        f_list_s : list of torch.Tensor or None
            Propagated source features in view order, each of shape
            (n_source, feats_dim). Use None when the source batch is absent.
        label_s : torch.Tensor or None
            One-hot labels of shape (n_source, nb_classes) for a source batch.
        re_computing : bool, optional
            Store the dominant view indices. Requires both domain batches.
            Default: False.

        Returns
        -------
        torch.Tensor
            Scalar loss combining reconstruction, source classification, target
            prediction entropy, and alignment terms.

        Notes
        -----
        Each available batch contributes reconstruction loss. The cross-domain
        and within-domain alignment terms require both batches and their raw
        features. To obtain class logits, pass concatenated view embeddings
        from the encoder to the classifier.
        """
        z_list_t = None
        z_list_s = None

        if f_list_t is not None:
            z_list_t = self.encoder(f_list_t)

            dominant_index_t = self.dominant_view_mining(z_list_t, feat_t)
            ae_loss_t = self.ae_loss(z_list_t, f_list_t)
        else:
            ae_loss_t = 0.0

        if f_list_s is not None:
            z_list_s = self.encoder(f_list_s)
            dominant_index_s = self.dominant_view_mining(z_list_s, feat_s)
            ae_loss_s = self.ae_loss(z_list_s, f_list_s)
        else:
            ae_loss_s = 0.0

        if re_computing:
            self.dominant_index_t = dominant_index_t
            self.dominant_index_s = dominant_index_s

        clf_loss_s = torch.tensor(0.0, device=self.device)
        embedding_s = torch.zeros(self.feats_dim, self.embed_dim * self.sub_num).to(self.device)
        embedding_t = torch.zeros(self.feats_dim, self.embed_dim * self.sub_num).to(self.device)
        clf_loss_t = torch.tensor(0.0, device=self.device)

        if z_list_s is not None:
            embedding_s = torch.cat(z_list_s, dim=1)
            logits_s = self.classifier(embedding_s)
            label_indices_s = label_s.argmax(dim=-1)

            clf_loss_s = F.cross_entropy(logits_s, label_indices_s)
        if z_list_t is not None:
            embedding_t = torch.cat(z_list_t, dim=1)
            logits_t = self.classifier(embedding_t)

            clf_loss_t = torch.mean(Entropy(F.softmax(logits_t, dim=-1)))

        ca_loss = torch.tensor(0.0, device=self.device)
        if z_list_t is not None and z_list_s is not None:
            ca_loss = self.ca_loss(z_list_t, z_list_s, dominant_index_t, dominant_index_s)

        ia_loss = torch.tensor(0.0, device=self.device)
        if z_list_t is not None and z_list_s is not None:
            ia_loss = self.ia_loss(z_list_t, z_list_s, dominant_index_t, dominant_index_s)

        weights = self.config.loss_weights

        loss = (ae_loss_s + ae_loss_t
                + weights['beta'] * clf_loss_s
                + weights['delta'] * clf_loss_t
                + (1 - weights['alpha']) * ca_loss
                + weights['alpha'] * ia_loss
                )

        return loss

    def encoder(self, x_list):
        """
        Encode each view with the shared encoder.

        Parameters
        ----------
        x_list : list of torch.Tensor
            Propagated feature matrices in view order, each of shape
            (n_nodes, feats_dim).

        Returns
        -------
        list of torch.Tensor
            Embedding matrices in the same order, each of shape
            (n_nodes, embed_dim).
        """
        z_list = []
        for i in range(len(x_list)):
            z = self.fc(x_list[i])
            z_list.append(z)
        return z_list

    def ae_loss(self, z_list, f_list):
        """
        Compute the mean reconstruction loss across views.

        Parameters
        ----------
        z_list : list of torch.Tensor
            Encoded views of shape (n_nodes, embed_dim).
        f_list : list of torch.Tensor
            Propagated features of shape (n_nodes, feats_dim) in the same order.

        Returns
        -------
        torch.Tensor
            Scalar mean of the per-view mean squared errors.
        """
        loss_rec = 0.0
        for i in range(self.sub_num):
            fea_rec = self.decoder(z_list[i])
            loss_rec += F.mse_loss(fea_rec, f_list[i])
        loss_ae = loss_rec / self.sub_num
        return loss_ae

    def ca_loss(self, z_list_t, z_list_s, dominant_index_t, dominant_index_s):
        """
        Align the projected dominant views across domains.

        Parameters
        ----------
        z_list_t : list of torch.Tensor or None
            Target embeddings in view order.
        z_list_s : list of torch.Tensor or None
            Source embeddings in view order.
        dominant_index_t : int or torch.Tensor
            Index of the target view to project.
        dominant_index_s : int or torch.Tensor
            Index of the source view to project.

        Returns
        -------
        torch.Tensor
            Sum of the source-to-target and target-to-source contrastive losses.
            Returns zero if either domain is absent.
        """
        if z_list_t is None or z_list_s is None:
            return torch.tensor(0.0, device=self.device)

        proj_t = self.proj[dominant_index_t](z_list_t[dominant_index_t])
        proj_s = self.proj[dominant_index_s](z_list_s[dominant_index_s])

        loss_s2t = self._contrastive_loss(proj_s, proj_t)
        loss_t2s = self._contrastive_loss(proj_t, proj_s)

        ca_loss = loss_s2t + loss_t2s

        return ca_loss

    def _contrastive_loss(self, pos_embed, neg_embed):
        """
        Compute one direction of the contrastive loss.

        Parameters
        ----------
        pos_embed : torch.Tensor
            Projected embeddings of shape (n_pos, embed_dim).
        neg_embed : torch.Tensor
            Projected embeddings from the other domain of shape
            (n_neg, embed_dim).

        Returns
        -------
        torch.Tensor
            Scalar loss from the row-wise log sums of exponentiated
            temperature-scaled cosine similarities.
        """
        pos_embed_norm = F.normalize(pos_embed, dim=1, p=2)
        neg_embed_norm = F.normalize(neg_embed, dim=1, p=2)

        sim_matrix = torch.mm(pos_embed_norm, neg_embed_norm.t())

        sim_matrix = sim_matrix / self.tau

        exp_sim = torch.exp(sim_matrix)
        sum_exp = torch.sum(exp_sim, dim=1)

        N_pos = pos_embed.size(0)
        N_neg = neg_embed.size(0)
        log_sum = torch.log(sum_exp + 1e-8)
        loss = -torch.mean(log_sum) + torch.log(torch.tensor(N_pos * N_neg, dtype=torch.float32, device=self.device))

        return loss

    def dominant_view_mining(self, z_list, feat):
        """
        Find the view that best preserves raw feature similarities.

        Parameters
        ----------
        z_list : list of torch.Tensor or None
            Embeddings for the views of one domain.
        feat : torch.Tensor or None
            Raw features for the same nodes in the same order.

        Returns
        -------
        torch.Tensor or tuple
            Scalar index of the view whose pairwise cosine similarities best
            match those of the raw features by mean squared error. Returns
            ``(0, 0.0)`` if either input is ``None``.
        """
        if z_list is None or feat is None:
            return 0, 0.0

        err_list = []
        feat = F.normalize(feat, dim=1, p=2)
        z_list = [F.normalize(z, dim=1, p=2) for z in z_list]
        feat_sim = torch.mm(feat, feat.t())

        for i in range(len(z_list)):
            embed = z_list[i]
            z_sim = torch.mm(embed, embed.t())
            err = F.mse_loss(z_sim, feat_sim)
            err_list.append(err)

        dominant_index = torch.argmin(torch.tensor(err_list))
        err_loss = sum(err_list) / len(err_list)

        return dominant_index

    def get_embeds(self, f_list):
        """
        Compute the normalized node embeddings for a domain.

        Parameters
        ----------
        f_list : list of torch.Tensor
            Propagated node features in view order.

        Returns
        -------
        torch.Tensor
            Detached, row-normalized embeddings of shape
            (n_nodes, sub_num * embed_dim), concatenated in view order.
        """
        z_list = self.encoder(f_list)
        z = torch.cat(z_list, dim=1)
        z = F.normalize(z, dim=1, p=2)
        return z.detach()

    def compute_sad(self, z_list_t, z_list_s):
        """
        Compute the structure-aware discrepancy for each view.

        Parameters
        ----------
        z_list_t : list of torch.Tensor or None
            Target embeddings in view order.
        z_list_s : list of torch.Tensor or None
            Source embeddings in the same view order.

        Returns
        -------
        torch.Tensor or None
            Squared Frobenius distances between the uncentered second moments
            of the two domains, one per view. Returns None if a domain is absent.
        """
        if z_list_t is None or z_list_s is None:
            return None

        sad_v_k = []

        for v in range(self.sub_num):

            cov_t = torch.mm(z_list_t[v].t(), z_list_t[v]) / z_list_t[v].size(0)
            cov_s = torch.mm(z_list_s[v].t(), z_list_s[v]) / z_list_s[v].size(0)

            sad = torch.norm(cov_t - cov_s, p='fro') ** 2
            sad_v_k.append(sad)

        return torch.stack(sad_v_k)

    def compute_sad_weights(self, sad_v_k_t, sad_v_k_s):
        """
        Convert view discrepancies to alignment weights.

        Parameters
        ----------
        sad_v_k_t : torch.Tensor
            Target discrepancies, one per view.
        sad_v_k_s : torch.Tensor
            Source discrepancies, one per view.

        Returns
        -------
        tuple of torch.Tensor
            Target and source weight vectors, in that order. Each is normalized
            over the views of its domain.
        """

        exp_t = torch.exp(-self.lam_sad * sad_v_k_t)
        omega_t = exp_t / (torch.sum(exp_t) + 1e-8)

        exp_s = torch.exp(-self.lam_sad * sad_v_k_s)
        omega_s = exp_s / (torch.sum(exp_s) + 1e-8)

        return omega_t, omega_s

    def ia_loss(self, z_list_t, z_list_s, dominant_index_t, dominant_index_s):
        """
        Align each domain's views to its first view.

        Parameters
        ----------
        z_list_t : list of torch.Tensor or None
            Target embeddings in view order.
        z_list_s : list of torch.Tensor or None
            Source embeddings in the same view order.
        dominant_index_t : int or torch.Tensor
            Target dominant view index retained in the method interface.
        dominant_index_s : int or torch.Tensor
            Source dominant view index retained in the method interface.

        Returns
        -------
        torch.Tensor
            Sum of the domains' discrepancy-weighted alignment losses.
            Returns zero if either domain is absent.

        Notes
        -----
        The first view anchors the alignment in each domain regardless of the
        supplied dominant view indices.
        """
        if z_list_t is None or z_list_s is None:
            return torch.tensor(0.0, device=self.device)

        sad_v_k_t = self.compute_sad(z_list_t, z_list_s)
        sad_v_k_s = self.compute_sad(z_list_s, z_list_t)

        omega_t, omega_s = self.compute_sad_weights(sad_v_k_t, sad_v_k_s)

        batch_size_t = z_list_t[0].size(0)
        batch_size_s = z_list_s[0].size(0)

        ia_loss_t = torch.tensor(0.0, device=self.device)
        cov_hop1_t = torch.mm(z_list_t[0].t(), z_list_t[0]) / batch_size_t

        for v in range(self.sub_num):
            cov_v_t = torch.mm(z_list_t[v].t(), z_list_t[v]) / batch_size_t
            cov_diff_t = torch.norm(cov_v_t - cov_hop1_t, p='fro') ** 2
            ia_loss_t += omega_t[v] * cov_diff_t

        ia_loss_t = ia_loss_t / self.sub_num

        ia_loss_s = torch.tensor(0.0, device=self.device)
        cov_hop1_s = torch.mm(z_list_s[0].t(), z_list_s[0]) / batch_size_s

        for v in range(self.sub_num):
            cov_v_s = torch.mm(z_list_s[v].t(), z_list_s[v]) / batch_size_s
            cov_diff_s = torch.norm(cov_v_s - cov_hop1_s, p='fro') ** 2
            ia_loss_s += omega_s[v] * cov_diff_s

        ia_loss_s = ia_loss_s / self.sub_num

        return ia_loss_t + ia_loss_s


def Entropy(input):
    """
    Compute one entropy value per node.

    Parameters
    ----------
    input : torch.Tensor
        Class probabilities of shape (n_nodes, n_classes).

    Returns
    -------
    torch.Tensor
        Entropy per row, using an offset of ``1e-4`` inside the logarithm.
    """
    epsilon = 1e-4
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


__all__ = ["SAGABase", "Entropy"]
