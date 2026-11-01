"""PALA core adapted from maxin88scu/PALA, commit 612b35d.

Original Copyright (c) 2025 Machine-xy, MIT License.
The contrastive reduction uses an equivalent log-domain evaluation to avoid
the observed float32 overflow. Other published-code conventions are retained.
"""
import random

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add

__all__ = ['PALABase']


class PALAConv(MessagePassing):
    """Author message passing with explicit, caller-supplied cached weights.

    This layer neither generates random-walk caches nor reads/writes files.
    Cached node order is part of the input contract. Signed weights are kept.
    """

    def __init__(self, in_channels, out_channels, weight=None, bias=None, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.cache_dict = {}
        if weight is None:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels).to(torch.float32))
            glorot(self.weight)
        else:
            self.weight = weight
        if bias is None:
            self.bias = Parameter(torch.Tensor(out_channels).to(torch.float32))
            zeros(self.bias)
        else:
            self.bias = bias

    def forward(self, x, edge_index, use_ppmi=True, cache_name='default_cache', edge_weight=None):
        if not use_ppmi or edge_weight is not None:
            raise ValueError('PALA requires explicit cached edge indices and normalization weights.')
        if cache_name not in self.cache_dict:
            raise ValueError('Missing PALA cache: ' + cache_name)
        cached_edges, norm = self.cache_dict[cache_name]
        x = torch.matmul(x, self.weight)
        return self.propagate(cached_edges.to(x.device), x=x, norm=norm.to(x.device))

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out + self.bias


class PALACompensation(nn.Module):
    def __init__(self, nodes_num, encoder_dim, device):
        super(PALACompensation, self).__init__()
        self.compensation = nn.Parameter(torch.FloatTensor(nodes_num, encoder_dim).normal_(-1,1).to(device))
        self.compensation.requires_grad_(True)

    def forward(self, input):
        return input + self.compensation

class PALAGNN(nn.Module):
    def __init__(self, source_data, encoder_dim, drop_out, feature_num, device, **kwargs):
        super(PALAGNN, self).__init__()
        weights = [None, None]
        biases = [None, None]

        self.dropout_layers = [nn.Dropout(drop_out) for _ in weights]

        self.compensation_layers = nn.ModuleList(
            [PALACompensation(source_data.size(0), encoder_dim, device),
             PALACompensation(source_data.size(0), encoder_dim, device),]
        )

        self.conv_layers = nn.ModuleList([
            PALAConv(feature_num, encoder_dim, weight=weights[0], bias = biases[0], **kwargs),
            PALAConv(encoder_dim, encoder_dim, weight=weights[1], bias = biases[1], **kwargs),
        ]).to(device)

    def forward(self, x, edge_index, cache_name, compensate, device, use_ppmi = True):
        x = x.to(device)
        edge_index = edge_index.to(device)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, use_ppmi = use_ppmi, cache_name= cache_name)
            if compensate:
                x = self.compensation_layers[i](x)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x

class PALAEncoder(nn.Module):
    def __init__(self,source_data, encoder_dim,drop_out,feature_num,device,**kwargs):
        super(PALAEncoder,self).__init__()

        self.GCNConv = PALAGNN(source_data,encoder_dim,drop_out,feature_num,device,**kwargs)
        self.num_prototypes = source_data.y.max().item()+1
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes,encoder_dim))
        self.classifier = nn.Linear(encoder_dim,self.num_prototypes)


    def forward(self, x, edge_index, cache_name, compensate,device,use_ppmi=True):
        embedding = self.GCNConv(x, edge_index, cache_name, compensate,device,use_ppmi=use_ppmi)
        logits = self.classifier(embedding)

        return embedding,logits

class PALAMemory(nn.Module):
    def __init__(self,num_nodes,num_class,device,beta = 0.9,temperature=0.5):
        super(PALAMemory,self).__init__()

        self.register_buffer('memory_bank', torch.zeros((num_nodes,num_class)).to(device))
        self.beta = beta
        self.temperature = temperature

    def forward(self,target_feature,prototypes):

        target_feature_nor = F.normalize(target_feature,dim=1)
        prototypes_nor = F.normalize(prototypes,dim=1)

        similarity = torch.einsum('nd,md->nm',target_feature_nor,prototypes_nor) / self.temperature
        coefficient = F.softmax(similarity,dim=1)

        # update memory bank
        with torch.no_grad():
            self.memory_bank = self.beta * self.memory_bank + (1 - self.beta) * coefficient

        elr_loss = -torch.mean(torch.log(1-torch.sum(self.memory_bank * coefficient,dim=1)))

        return elr_loss

def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    """
    make long-tailed data by removing nodes sequentially
    method comes form paper 'GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification'
    code:https://github.com/JoonHyung-Park/GraphENS
    """
    # sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    #compute number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu,i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu,i), sorted_n_data[i])))
        """
        remove low degree nodes sequentially(10 steps)
        since degrees of remaining nodes are changed when some nodes are moved
        """
        #we do not remove the nodes of magjor
        if i <1 :
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    #compute the removing nodes for each class
    remove_class_num_list = [n_data[i].item() - class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()

    device = label.device
    index_list = index_list.to(device)

    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])
    # compute remove nodes index
    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # find already removed nodes
            node_mask = label.new_ones(label.size(), dtype = torch.bool)
            node_mask[sum(remove_idx_list, [])] = False

            # remove connection with removed nodes in edge
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size = label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            # remove low degree nodes first
            _, remove_idx = torch.topk(degree, (r * remove_class_num_list[i])//n_round[i], largest = False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.cpu().numpy())

    # find removed nodes
    node_mask = label.new_ones(label.size(), dtype = torch.bool)
    node_mask[sum(remove_idx_list, [])] = False

    # remove connection with removed nodes in edge
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask

def sampling_anchor_nodes(data, k, use_ppmi = True):
    """
    sampling anchor nodes for each class according to degrees
    """
    labels = data.y
    anchor_nodes = []
    label_class = torch.unique(labels)

    # sampling by using ppmi matrix
    if use_ppmi:
        for label in label_class:
            class_nodes = (labels == label).nonzero().squeeze()
            degrees = torch_geometric.utils.degree(data.global_adj[0], num_nodes = data.num_nodes)
            _, sorted_indices = torch.sort(degrees[class_nodes], descending=True)
            anchor_nodes.extend(class_nodes[sorted_indices[:k]].tolist())

    # sampling by using adjacent matrix
    else:
        for label in label_class:
            class_nodes = (labels == label).nonzero().squeeze()
            degrees = torch_geometric.utils.degree(data.edge_index[0], num_nodes = data.num_nodes)
            _, sorted_indices = torch.sort(degrees[class_nodes], descending=True)
            anchor_nodes.extend(class_nodes[sorted_indices[:k]].tolist())

    return anchor_nodes

def get_new_cl_samples(distance,gamma,encoded_q,prototype,epsilon,positive=True):
    """
    get contrastive learning samples for each node
    """
    n = distance.shape[0]
    m,k=encoded_q.size()
    new_cl_samples = torch.zeros(n,gamma*2,k).to(encoded_q.device)
    z = encoded_q.size(0) - 1
    epsilon1=random.uniform(0,epsilon)

    # get positive samples
    if positive:
        for i in range(n):
            if (z+1) >= gamma*2:
                new_cl_samples[i][:gamma*2] =  (1-epsilon1) * encoded_q[:gamma*2] + epsilon1 * prototype.unsqueeze(0)
            else:
                new_cl_samples[i][:z] =  (1-epsilon1) * encoded_q[:z] + epsilon1 * prototype.unsqueeze(0)
                m=gamma*2-(z+1)
                n=z
                while m > 0:
                    index = min(z,m)
                    new_cl_samples[i][n:n+index] = (1-epsilon1) * encoded_q[:index] + epsilon1 * prototype.unsqueeze(0)
                    n = n+index
                    m = m-index

    # get negative samples
    else:
        for i in range(n):
            _,sorted_indices = torch.topk(distance[i],gamma*2)
            new_cl_samples[i][:gamma] = encoded_q[sorted_indices[:gamma]]
            remain_indices = sorted_indices[gamma:]
            new_cl_samples[i][gamma:] =  (1-epsilon1) * encoded_q[remain_indices] + epsilon1 * prototype.unsqueeze(0)

    return new_cl_samples

def compute_cluster_centers(logits, features):
    """
    calculate cluster centers in a non-parametric way
    """
    n, num_classes = logits.shape
    _, d = features.shape

    cluster_centers = torch.zeros((num_classes, d), device=logits.device)

    # calculate centers for each class
    for i in range(num_classes):

        class_probs = logits[:, i]

        weighted_features =  features * class_probs.unsqueeze(1)
        weighted_sum = weighted_features.sum(dim=0)

        total_prob_sum = class_probs.sum()

        cluster_centers[i] = weighted_sum / (total_prob_sum + 1e-8)

    return cluster_centers

def align_loss(encoded_target,prototypes, logits_t,labels,temperature=0.5):
    """
    contrastive alignment loss
    """
    num_class = labels.max().item() + 1
    centers = compute_cluster_centers(logits_t, encoded_target)

    centers_nor = F.normalize(centers,dim=1)
    encoded_target_nor = F.normalize(encoded_target,dim=1)
    prototypes_nor = F.normalize(prototypes,dim=1)

    # calculate pseudo label
    similarity = torch.matmul(encoded_target_nor,centers_nor.T)
    target_labels = torch.argmax(similarity,dim=1)

    align_loss = 0.0
    for label in range(num_class):
        target_class_nodes =(target_labels == label).nonzero().squeeze(1)
        encoded_target_class = encoded_target_nor[target_class_nodes]

        # compute prototype-related weight
        class_similarity = torch.exp(torch.matmul(centers_nor[label],encoded_target_class.T) / temperature)
        class_similarity_sum = class_similarity.sum()
        weight = class_similarity / class_similarity_sum

        class_pos_similarity = torch.exp(torch.matmul(prototypes_nor[label],encoded_target_class.T) / temperature)
        class_all_similarity = torch.exp(torch.matmul(prototypes_nor,encoded_target_class.T) / temperature)
        class_all_similarity_sum = class_all_similarity.sum(dim=0)
        similarities = torch.log(class_pos_similarity / class_all_similarity_sum)
        align_loss += -torch.matmul(weight,similarities.unsqueeze(1))

    return align_loss / num_class

def prototype_contrastive_loss(encoded_f,encoded_g,anchor_nodes,query_nodes,prototypes,epsilon,labels,gamma):
    """
    prototype-anchored learning
    """
    num_class = labels.max().item() + 1

    encoded_f_nor = F.normalize(encoded_f,dim=1)
    encoded_g_nor = F.normalize(encoded_g,dim=1)
    prototypes_nor = F.normalize(prototypes,dim=1)


    # update prototypes(prototype recalibration)
    class_masks = torch.stack([(labels == label) for label in range(num_class)])
    weights = 1 / (1 + torch.exp(-torch.einsum('cd,nd->cn', prototypes_nor, encoded_f_nor)))
    weights = torch.einsum('cn,cn->c', class_masks.float(), weights) / class_masks.sum(dim=1)
    new_prototypes = torch.log(weights).unsqueeze(1) + prototypes
    new_prototypes_nor = F.normalize(new_prototypes, dim=1)

    p_loss = 0.0
    for label in range(num_class):
        class_nodes = [node for node in anchor_nodes if labels[node] == label]
        pos_nodes_q = [node for node in query_nodes if labels[node] == label]
        neg_nodes_q = [node for node in query_nodes if labels[node] != label]

        encoded_cl_g = encoded_g_nor[class_nodes]
        encoded_cl_f = encoded_f_nor[class_nodes]

        neg_prototype = new_prototypes[torch.arange(new_prototypes.size(0)) != label]
        neg_prototype_mean = neg_prototype.mean(dim=0)

        neg_l2_norm = torch.matmul((torch.norm(encoded_cl_g, p=2, dim=1,keepdim = True)), (torch.norm(encoded_f_nor[neg_nodes_q],p=2,dim=1,keepdim=True)).T)
        neg_distance = 1-torch.matmul(encoded_cl_g,encoded_f_nor[neg_nodes_q].T) / neg_l2_norm
        pos_distance = neg_distance.clone()

        # get negative/positive CL samples
        new_pos_samples = get_new_cl_samples(pos_distance,gamma,encoded_f[pos_nodes_q],new_prototypes[label],epsilon,positive=True)
        new_neg_samples = get_new_cl_samples(neg_distance,gamma,encoded_f[neg_nodes_q],neg_prototype_mean,epsilon,positive=False)

        new_pos_samples_nor = F.normalize(new_pos_samples,dim=1)
        new_neg_samples_nor = F.normalize(new_neg_samples,dim=1)

        # calculate prototype-anchored learning loss
        log_sigma1 = torch.logsumexp((torch.einsum('nd,nmd->nm',encoded_cl_g,new_neg_samples_nor)).reshape(-1), dim=0)
        log_sigma2 = torch.logsumexp((torch.einsum('nd,md->nm',encoded_cl_f,neg_prototype)).reshape(-1), dim=0)
        log_sigma3 = torch.logsumexp((-torch.einsum('nd,nmd->nm',encoded_cl_g,new_pos_samples_nor)).reshape(-1), dim=0)
        log_sigma4 = torch.logsumexp((-torch.einsum('nd,md->nm',encoded_cl_g,prototypes[label].unsqueeze(1))).reshape(-1), dim=0)

        log_product = torch.logaddexp(log_sigma1, log_sigma2) + torch.logaddexp(log_sigma3, log_sigma4)
        loss = torch.logaddexp(torch.zeros_like(log_product), log_product)
        p_loss += loss
    return p_loss / num_class


def update_sample_queue(data, cache_name, models,last_sample_queue,gamma):
    """
    update sample queue each epoch
    """
    for model in models:
        model.eval()
    encoder = models[0]
    _, logits = encoder(data.x, data.edge_index, cache_name, False, data.x.device)
    preds = logits.argmax(dim=1)
    labels = data.y

    # misclassified nodes
    sample_queue = (preds != labels).nonzero().squeeze(1)
    sample_queue = sample_queue.tolist()
    num_class=data.y.max().item()+1

    sample_num = gamma * 2
    for label in range(num_class):
        class_nodes = [node for node in sample_queue if labels[node].item() == label]
        if len(last_sample_queue) >= 0 :
            last_class_nodes = [node for node in last_sample_queue if labels[node] == label]
        if len(class_nodes) < sample_num:
            if len(last_class_nodes) >= sample_num:
                nodes = [node for node in last_class_nodes if node not in class_nodes]
            else:
                all_class_nodes = (labels == label).nonzero().squeeze(1)
                all_class_nodes = all_class_nodes.tolist()
                nodes = random.sample(all_class_nodes, sample_num)
            sample_queue.extend(nodes)

    return sample_queue

class PALABase(nn.Module):
    """Two author encoders and target memory for prepared, full-batch graphs.

    Source input contains the already selected long-tail training nodes.
    Target input needs only features and edges. Class count comes from source.
    Dropout deliberately follows the author's ordinary-list behavior: eval()
    leaves it active, so evaluation calls affect the subsequent RNG trajectory.
    """

    def __init__(self, source_data, num_target_nodes, hid_dim=512, dropout=0.1):
        super().__init__()
        device = source_data.x.device
        classes = int(source_data.y.max()) + 1
        self.encoder_f = PALAEncoder(source_data, hid_dim, dropout, source_data.x.shape[1], device).to(device)
        self.encoder_g = PALAEncoder(source_data, hid_dim, dropout, source_data.x.shape[1], device).to(device)
        self.memory = PALAMemory(num_target_nodes, classes, device)

    def set_cache(self, name, edge_index, norm, num_nodes):
        """Copy a validated cache, preserving its edge order and signed weights."""
        if not isinstance(name, str) or not name:
            raise ValueError('Cache names must be nonempty strings.')
        if (edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2 or
                norm.dtype != torch.float32 or norm.ndim != 1 or norm.numel() != edge_index.shape[1]):
            raise ValueError('Expected long [2,E] cached edges and float32 [E] weights.')
        if not bool(torch.isfinite(norm).all()) or (edge_index.numel() and
                (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes)):
            raise ValueError('Cache has non-finite weights or invalid endpoints.')
        cache = (edge_index.detach().clone(), norm.detach().clone())
        for encoder in [self.encoder_f, self.encoder_g]:
            for layer in encoder.GCNConv.conv_layers:
                layer.cache_dict[name] = cache

    def update_queue(self, source_data, previous_queue, gamma=5, cache_name='source'):
        """Run the original queue update, including its stochastic eval forward."""
        return update_sample_queue(source_data, cache_name,
                                   [self.encoder_f, self.encoder_g], previous_queue, gamma)

    def forward(self, source_data, target_data, sample_queue, epoch,
                gamma=5, epsilon=0.2, anchor_num=200,
                source_cache='source', target_cache='target'):
        self.encoder_f.train()
        self.encoder_g.train()
        device = source_data.x.device
        fs, logits_s = self.encoder_f(source_data.x, source_data.edge_index, source_cache, True, device)
        gs, _ = self.encoder_g(source_data.x, source_data.edge_index, source_cache, True, device)
        ft, logits_t = self.encoder_f(target_data.x, target_data.edge_index, target_cache, False, device)
        anchors = sampling_anchor_nodes(source_data, anchor_num, use_ppmi=True)
        prototypes = self.encoder_f.prototypes
        p_loss = prototype_contrastive_loss(fs, gs, anchors, sample_queue, prototypes,
                                            epsilon, source_data.y, gamma)
        alignment = align_loss(ft, prototypes, logits_t, source_data.y)
        elr = self.memory(ft, prototypes)
        ce = F.cross_entropy(logits_s, source_data.y)
        loss = ce + p_loss + alignment
        if epoch > 50:
            loss = loss + elr
        return loss, dict(classification=ce, prototype=p_loss, alignment=alignment, elr=elr)

    def predict_logits(self, data, cache_name='target'):
        self.encoder_f.eval()
        self.encoder_g.eval()
        return self.encoder_f(data.x, data.edge_index, cache_name, False, data.x.device)[1]
