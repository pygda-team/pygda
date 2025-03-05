import torch
import warnings
import torch.nn.functional as F
import itertools
import time

import torch.nn as nn

from scipy import sparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse import vstack

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_dense_adj

from . import BaseGDA
from ..nn import ACDNEBase
from ..utils import logger
from ..metrics import eval_macro_f1, eval_micro_f1


class DMGNN(BaseGDA):
    """
    Domain-adaptive message passing graph neural network (NN-23).

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hid_dim : int
        Hidden dimension of model.
    num_classes : int
        Total number of classes.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    n_emb : int, optional
        Adversarial learning module hidden dimension. Default: ``128``.
    pair_weight : float, optional
        Trade-off hyper-parameter for pairwise constraint. Default: ``0.1``.
    step: int, optional
        Propagation steps in PPMI matrix. Default: ``3``.
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
        Minibatch size, 0 for full batch training. Default: ``100``.
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
        num_layers=0,
        n_emb=128,
        pair_weight=0.1,
        step=3,
        dropout=0.,
        act=F.relu,
        weight_decay=0.,
        lr=4e-3,
        epoch=200,
        device='cuda:0',
        batch_size=100,
        num_neigh=-1,
        verbose=2,
        **kwargs):
        
        super(DMGNN, self).__init__(
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
        
        self.n_emb=n_emb
        self.pair_weight=pair_weight
        self.step=step

    def init_model(self, **kwargs):
        """
        Initialize the DMGNN base model.

        Parameters
        ----------
        **kwargs
            Additional parameters for model initialization.

        Returns
        -------
        ACDNEBase
            Initialized model with specified architecture parameters.

        Notes
        -----
        Configures a two-layer network with:

        - Input dimension handling
        - Hidden layer configuration
        - Embedding dimension for adversarial learning
        - Dropout regularization
        - Batch processing settings
        """

        return ACDNEBase(
            n_input=self.in_dim,
            n_hidden=[self.hid_dim]*2,
            n_emb=self.n_emb,
            num_class=self.num_classes,
            drop=self.dropout,
            batch_size=self.batch_size,
            **kwargs
        ).to(self.device)

    def forward_model(self, **kwargs):
        """
        Forward pass placeholder.

        Parameters
        ----------
        **kwargs
            Arbitrary keyword arguments.

        Notes
        -----
        Main forward logic is implemented in fit method
        due to complex batch processing requirements.
        """
        pass

    def fit(self, source_data, target_data):
        """
        Train the DMGNN model on source and target domain data.

        Parameters
        ----------
        source_data : torch_geometric.data.Data
            Source domain graph data.
        target_data : torch_geometric.data.Data
            Target domain graph data.

        Notes
        -----
        Training process consists of several key components:

        Data Preprocessing

        - Computes PPMI matrices for both domains
        - Generates neighborhood-aware features
        - Prepares one-hot encoded labels
        - Concatenates direct and neighborhood features

        Model Setup

        - Initializes DMGNN model
        - Configures Adam optimizer
        - Sets up classification and domain loss functions

        Training Loop

        - Generates balanced source/target batches
        - Updates domain adaptation parameter
        - Computes multiple loss components:
            
            * Classification loss on labeled data
            * Domain adversarial loss
            * Network proximity loss

        - Enhances predictions with neighborhood information
        - Tracks and logs training progress

        Batch Processing

        - Handles source and target domains separately
        - Maintains balanced sampling
        - Applies PPMI-based structural learning
        - Implements dynamic batch generation

        Loss Computation

        - Classification loss on source domain
        - Domain adaptation through adversarial training
        - Neighborhood proximity preservation
        - Combined loss optimization

        Monitoring

        - Tracks epoch-wise loss
        - Computes micro-F1 score
        - Logs training progress
        - Measures computation time
        """
        self.ppmi_s, x_n_s = self.process_graph(source_data)
        self.ppmi_t, x_n_t = self.process_graph(target_data)
        x_s = source_data.x.detach().cpu().numpy()
        x_t = target_data.x.detach().cpu().numpy()
        y_s = F.one_hot(source_data.y).detach().cpu().numpy()
        y_t = F.one_hot(target_data.y).detach().cpu().numpy()
        y_t_o = np.zeros(np.shape(y_t))

        num_nodes_s = x_s.shape[0]
        num_nodes_t = x_t.shape[0]

        x_s_new = np.concatenate((x_s, x_n_s), axis=1)
        x_t_new = np.concatenate((x_t, x_n_t), axis=1)

        self.whole_xs_xt_stt = torch.tensor(vstack([x_s, x_t]).toarray()).to(self.device)
        self.whole_xs_xt_stt_nei = torch.tensor(vstack([x_n_s, x_n_t]).toarray()).to(self.device)

        self.dmgnn = self.init_model(**self.kwargs)

        optimizer = torch.optim.Adam(
            self.dmgnn.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.clf_loss_func = nn.CrossEntropyLoss()
        self.domain_loss_func = nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_source_logits = None
            epoch_source_labels = None

            s_batches = self.batch_generator([x_s_new, y_s], int(self.batch_size / 2), shuffle=True)
            t_batches = self.batch_generator([x_t_new, y_t_o], int(self.batch_size / 2), shuffle=True)
            num_batch = round(max(num_nodes_s / (self.batch_size / 2), num_nodes_t / (self.batch_size / 2)))

            p = float(epoch) / self.epoch
            alpha = 2. / (1. + np.exp(-10. * p)) - 1

            for idx in range(num_batch):
                xs_ys_batch, shuffle_index_s = next(s_batches)
                xs_batch = xs_ys_batch[0]
                ys_batch = xs_ys_batch[1]
                xt_yt_batch, shuffle_index_t = next(t_batches)
                xt_batch = xt_yt_batch[0]
                yt_batch = xt_yt_batch[1]

                x_batch = np.vstack([xs_batch, xt_batch])
                xb = torch.FloatTensor(x_batch[:, 0:self.in_dim])
                xb_nei = torch.FloatTensor(x_batch[:, -self.in_dim:])
                yb = np.vstack([ys_batch, yt_batch])

                mask_l = np.sum(yb, axis=1) > 0

                domain_label = np.vstack([np.tile([1., 0.], [self.batch_size // 2, 1]), np.tile([0., 1.], [self.batch_size // 2, 1])]) 
                # [1,0] for source, [0,1] for target
                # #topological proximity matrix between nodes in each mini-batch
                a_s, a_t = self.batch_ppmi(self.batch_size, shuffle_index_s, shuffle_index_t, self.ppmi_s, self.ppmi_t)

                self.dmgnn.train()
                optimizer.zero_grad()
                emb, pred_logit, d_logit = self.dmgnn(xb.to(self.device), xb_nei.to(self.device), alpha)
                emb_s, emb_t = self.dmgnn.network_embedding.pairwise_constraint(emb)
                net_pro_loss_s = self.nei_prox_loss(emb_s, torch.FloatTensor(a_s).to(self.device))
                net_pro_loss_t = self.nei_prox_loss(emb_t, torch.FloatTensor(a_t).to(self.device))
                net_pro_loss = self.pair_weight * (net_pro_loss_s + net_pro_loss_t)

                pred_logit_nei_s = torch.mm(torch.FloatTensor(a_s).to(self.device), pred_logit[:int(self.batch_size/2), :])
                pred_logit_nei_t = torch.mm(torch.FloatTensor(a_t).to(self.device), pred_logit[int(self.batch_size/2):, :])
                pred_logit_nei = torch.cat((pred_logit_nei_s, pred_logit_nei_t), dim=0)
                pred_logit = pred_logit + pred_logit_nei

                clf_loss = self.clf_loss_func(pred_logit[mask_l], torch.tensor(yb[mask_l]).to(self.device))
                domain_loss = self.domain_loss_func(d_logit, torch.argmax(torch.tensor(domain_label).to(self.device), 1))
                total_loss = clf_loss + domain_loss + net_pro_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            epoch_source_logits, epoch_source_labels = self.predict(source_data, train=True)
            
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
        Process input graph data to compute PPMI matrices and neighborhood features.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.

        Returns
        -------
        tuple
            Contains:
            - A_ppmi : numpy.ndarray
                PPMI matrix for structural information.
            - X_nei : numpy.ndarray
                Neighborhood-aware node features.

        Notes
        -----
        Processing steps:

        1. Converts edge index to dense adjacency
        2. Computes k-step transition probabilities
        3. Generates PPMI matrix
        4. Creates neighborhood feature aggregation
        """
        adj = to_dense_adj(data.edge_index).squeeze()
        g = sparse.csc_matrix(adj.detach().cpu().numpy())
        A_k = self.agg_tran_prob_mat(g, self.step)
        A_ppmi = self.compute_ppmi(A_k)
        n_A_ppmi = self.my_scale_sim_mat(A_ppmi)
        X = data.x.detach().cpu().numpy()
        X_nei = np.matmul(n_A_ppmi, X)

        return A_ppmi, X_nei
    
    def nei_prox_loss(self, emb, a):
        """
        Calculate neighborhood proximity loss.

        Parameters
        ----------
        emb : torch.Tensor
            Node embeddings.
        a : torch.Tensor
            Adjacency/PPMI matrix.

        Returns
        -------
        torch.Tensor
            Normalized proximity loss between nodes and their neighbors.

        Notes
        -----
        Computes average L2 distance between node embeddings
        and their neighborhood representations.
        """
        nei_emb = torch.mm(a, emb)
        r = torch.norm(emb - nei_emb)
        r = r / emb.shape[0]

        return r 

    def agg_tran_prob_mat(self, g, step):
        """
        Compute aggregated k-step transition probability matrix.

        Parameters
        ----------
        g : scipy.sparse.csc_matrix
            Input graph adjacency matrix.
        step : int
            Number of propagation steps.

        Returns
        -------
        numpy.ndarray
            Aggregated transition probability matrix.

        Notes
        -----
        Implements iterative computation of transition probabilities
        up to k steps, with step-wise normalization.
        """
        g = self.my_scale_sim_mat(g)
        g = csc_matrix.toarray(g)
        a_k = g
        a = g
        for k in np.arange(2, step+1):
            a_k = np.matmul(a_k, g)
            a = a+a_k/k
        
        return a
    
    def my_scale_sim_mat(self, w):
        """
        Compute L1 row normalization of a matrix.

        Parameters
        ----------
        w : numpy.ndarray or scipy.sparse.csc_matrix
            Input similarity/adjacency matrix.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csc_matrix
            Row-normalized matrix.

        Notes
        -----
        Implementation details:

        1. Computes row sums
        2. Handles numerical stability with epsilon
        3. Prevents infinite values
        4. Applies row-wise normalization
        """
        rowsum = np.array(np.sum(w, axis=1), dtype=np.float32)
        r_inv = np.power(rowsum + 1e-12, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        w = r_mat_inv.dot(w)
        
        return w
    
    def compute_ppmi(self, a):
        """
        Compute Positive Pointwise Mutual Information matrix.

        Parameters
        ----------
        a : numpy.ndarray
            Aggregated transition probability matrix.

        Returns
        -------
        numpy.ndarray
            PPMI matrix with non-negative entries.

        Notes
        -----
        1. Removes self-loops
        2. Normalizes transition probabilities
        3. Computes log-based PPMI values
        4. Handles numerical stability
        """
        np.fill_diagonal(a, 0)
        a = self.my_scale_sim_mat(a)
        (p, q) = np.shape(a)
        col = np.sum(a, axis=0)
        col[col == 0] = 1
        ppmi = np.log((float(p)*a) / (col[None, :]) + 1e-12)
        idx_nan = np.isnan(ppmi)
        ppmi[idx_nan] = 0
        ppmi[ppmi < 0] = 0
        
        return ppmi

    def batch_ppmi(self, batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t):
        """
        Generate batch-wise PPMI matrices for source and target domains.

        Parameters
        ----------
        batch_size : int
            Size of mini-batch.
        shuffle_index_s : numpy.ndarray
            Shuffled indices for source domain.
        shuffle_index_t : numpy.ndarray
            Shuffled indices for target domain.
        ppmi_s : numpy.ndarray
            Source domain PPMI matrix.
        ppmi_t : numpy.ndarray
            Target domain PPMI matrix.

        Returns
        -------
        tuple
            Contains:
            - a_s : numpy.ndarray
                Normalized source batch PPMI matrix.
            - a_t : numpy.ndarray
                Normalized target batch PPMI matrix.
        """
        # #proximity matrix between source network nodes in each mini-batch
        # noinspection DuplicatedCode
        a_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
        for ii in range(int(batch_size / 2)):
            for jj in range(int(batch_size / 2)):
                if ii != jj:
                    a_s[ii, jj] = ppmi_s[shuffle_index_s[ii], shuffle_index_s[jj]]
        # #proximity matrix between target network nodes in each mini-batch
        # noinspection DuplicatedCode
        a_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
        for ii in range(int(batch_size / 2)):
            for jj in range(int(batch_size / 2)):
                if ii != jj:
                    a_t[ii, jj] = ppmi_t[shuffle_index_t[ii], shuffle_index_t[jj]]
        return self.my_scale_sim_mat(a_s), self.my_scale_sim_mat(a_t)

    def predict(self, data, train=False):
        """
        Make predictions on input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data.
        train : bool, optional
            Whether in training mode. Default: ``False``.

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
        1. Uses stored whole-graph representations
        2. Combines direct and neighborhood predictions
        3. Handles source/target domains differently
        """
        self.dmgnn.eval()

        with torch.no_grad():
            emb, pred_logit_xs_xt, _ = self.dmgnn(self.whole_xs_xt_stt, self.whole_xs_xt_stt_nei, 1)
            pred_prob_xs_xt = F.softmax(pred_logit_xs_xt, dim=1)
            if train:
                logits = pred_prob_xs_xt[0:data.x.shape[0], :]
                logits_nei = torch.mm(torch.FloatTensor(self.ppmi_s).to(self.device), logits)
                logits = logits + logits_nei
            else:
                logits = pred_prob_xs_xt[-data.x.shape[0]:, :]
                logits_nei = torch.mm(torch.FloatTensor(self.ppmi_t).to(self.device), logits)
                logits = logits + logits_nei

        return logits, data.y
    
    def shuffle_aligned_list(self, data):
        """
        Shuffle multiple data arrays while maintaining alignment.

        Parameters
        ----------
        data : list
            List of numpy arrays to be shuffled.

        Returns
        -------
        tuple
            Contains:
            - shuffle_index : numpy.ndarray
                Generated permutation indices.
            - shuffled_data : list
                List of shuffled arrays maintaining alignment.

        Notes
        -----
        Ensures consistent shuffling across multiple data arrays,
        particularly useful for maintaining correspondence between
        features and labels during batch generation.
        """
        num = data[0].shape[0]
        shuffle_index = np.random.permutation(num)
        return shuffle_index, [d[shuffle_index] for d in data]

    def batch_generator(self, data, batch_size, shuffle=True):
        """
        Generate mini-batches of data.

        Parameters
        ----------
        data : list
            List of data arrays to be batched.
        batch_size : int
            Size of each batch.
        shuffle : bool, optional
            Whether to shuffle data. Default: ``True``.

        Yields
        ------
        tuple
            Contains:
            - batch_data : list
                List of batched data arrays.
            - shuffle_index : numpy.ndarray
                Indices for current batch.

        Notes
        -----
        Implements infinite batch generation with optional
        shuffling and aligned data handling.
        """
        shuffle_index = None
        if shuffle:
            shuffle_index, data = self.shuffle_aligned_list(data)
        batch_count = 0
        while True:
            if batch_count * batch_size + batch_size >= data[0].shape[0]:
                batch_count = 0
                if shuffle:
                    shuffle_index, data = self.shuffle_aligned_list(data)
            start = batch_count * batch_size
            end = start + batch_size
            batch_count += 1
            yield [d[start:end] for d in data], shuffle_index[start:end]
