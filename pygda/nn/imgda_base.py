"""ImGDA numerical core adapted from maxin88scu/ImGDA, commit d4237cd.

Copyright (c) 2026 maxin88scu. Licensed under MIT; see the accompanying license.

This candidate preserves the released objectives and ordinary-list Dropout
behavior. Precomputed global edges and normalized weights must be supplied
explicitly. Cache file I/O and random-walk fallback are excluded. No normalized
cache weights are clamped or normalized again.
"""
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import torch_geometric
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops

class ImGDAConv(MessagePassing):

    def __init__(self, in_channels, out_channels, weight=None, bias=None, improved=False, use_bias=True, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cache_dict = {}
        if weight is None:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels).to(torch.float32))
            glorot(self.weight)
        else:
            self.weight = weight
            print('use shared weight')
        if bias is None:
            if use_bias:
                self.bias = Parameter(torch.Tensor(out_channels).to(torch.float32))
            else:
                self.register_parameter('bias', None)
            zeros(self.bias)
        else:
            self.bias = bias
            print('use shared bias')

    @staticmethod
    def norm2(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return (edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col])

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def set_global_cache(self, name, edge_index, norm):
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError('Global cache edge_index must be a [2, E] long tensor.')
        if norm.ndim != 1 or norm.numel() != edge_index.size(1) or not torch.isfinite(norm).all():
            raise ValueError('Global cache requires one finite normalized weight per edge.')
        self.cache_dict[name] = (edge_index, norm)

    def forward(self, x, edge_index, use_ppmi=True, cache_name='default_cache', edge_weight=None):
        x = torch.matmul(x, self.weight)
        if use_ppmi:
            if cache_name not in self.cache_dict:
                raise ValueError('An explicit precomputed global cache is required: ' + cache_name)
            edge_index, norm = self.cache_dict[cache_name]
        else:
            edge_index, norm = self.norm2(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
        edge_index = edge_index.to(x.device)
        norm = norm.to(x.device)
        return self.propagate(edge_index, x=x, norm=norm)


class ImGDACompensation(nn.Module):

    def __init__(self, nodes_num, encoder_dim, device):
        super(ImGDACompensation, self).__init__()
        self.compensation = nn.Parameter(torch.FloatTensor(nodes_num, encoder_dim).normal_(-1, 1).to(device))
        self.compensation.requires_grad_(True)

    def forward(self, input):
        return input + self.compensation

class ImGDAEncoder(nn.Module):

    def __init__(self, source_data, encoder_dim, drop_out, feature_num, device, **kwargs):
        super(ImGDAEncoder, self).__init__()
        weights = [None, None]
        biases = [None, None]
        self.dropout_layers = [nn.Dropout(drop_out) for _ in weights]
        self.compensation_layers = nn.ModuleList([ImGDACompensation(source_data.size(0), encoder_dim, device), ImGDACompensation(source_data.size(0), encoder_dim, device)])
        self.conv_layers = nn.ModuleList([ImGDAConv(feature_num, encoder_dim, weight=weights[0], bias=biases[0], **kwargs), ImGDAConv(encoder_dim, encoder_dim, weight=weights[1], bias=biases[1], **kwargs)]).to(device)

    def forward(self, x, edge_index, cache_name, compensate, device, use_ppmi=True):
        x = x.to(device)
        edge_index = edge_index.to(device)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, use_ppmi=use_ppmi, cache_name=cache_name)
            if compensate:
                x = self.compensation_layers[i](x)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x

def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    """
    make long-tailed data by removing nodes sequentially
    method comes form paper 'GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification'
    code:https://github.com/JoonHyung-Park/GraphENS
    """
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12
    mu = np.power(1 / ratio, 1 / (n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        '\n        remove low degree nodes sequentially(10 steps)\n        since degrees of remaining nodes are changed when some nodes are moved\n        '
        if i < 1:
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]
    remove_class_num_list = [n_data[i].item() - class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    device = label.device
    index_list = index_list.to(device)
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])
    for i in indices.numpy():
        for r in range(1, n_round[i] + 1):
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list, [])] = False
            row, col = (edge_index[0], edge_index[1])
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]
            _, remove_idx = torch.topk(degree, r * remove_class_num_list[i] // n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.cpu().numpy())
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list, [])] = False
    row, col = (edge_index[0], edge_index[1])
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask
    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)
    return (list(class_num_list), train_mask, idx_info, node_mask, edge_mask)

def sampling_anchor_nodes(data, k, use_ppmi=True):
    """
    sampling anchor nodes for each class according to degrees

    parameters:
    k: number of anchor nodes for each class
    use_ppmi: sampling for local or global branch
    return: a list of anchor nodes
    """
    labels = data.y
    anchor_nodes = []
    label_class = torch.unique(labels)
    if use_ppmi:
        for label in label_class:
            class_nodes = (labels == label).nonzero().squeeze()
            degrees = torch_geometric.utils.degree(data.global_adj[0], num_nodes=data.num_nodes)
            _, sorted_indices = torch.sort(degrees[class_nodes], descending=True)
            anchor_nodes.extend(class_nodes[sorted_indices[:k]].tolist())
    else:
        for label in label_class:
            class_nodes = (labels == label).nonzero().squeeze()
            degrees = torch_geometric.utils.degree(data.edge_index[0], num_nodes=data.num_nodes)
            _, sorted_indices = torch.sort(degrees[class_nodes], descending=True)
            anchor_nodes.extend(class_nodes[sorted_indices[:k]].tolist())
    return anchor_nodes

def calculate_prototypes(anchor_nodes, node_embeddings, labels):
    """
    calculate prototypes for each class
    return: (tensor) prototypes
    """
    label_class = torch.unique(labels)
    prototypes = []
    for label in label_class:
        class_nodes = (labels == label).nonzero().squeeze()
        class_nodes_embeddings = node_embeddings[class_nodes]
        prototype = torch.mean(class_nodes_embeddings, dim=0)
        prototypes.append(prototype.unsqueeze(0))
    prototypes = torch.cat(prototypes, dim=0)
    return prototypes

def cross_branch_prototype_contrastive_loss(local_anchor_nodes, global_anchor_nodes, local_nodes_embeddings, global_nodes_embeddings, labels, counts, dynamic_temperature=0.5):
    """
    calculate cross branch CL loss
    """
    local_prototypes = calculate_prototypes(local_anchor_nodes, local_nodes_embeddings, labels)
    global_prototypes = calculate_prototypes(global_anchor_nodes, global_nodes_embeddings, labels)
    local_prototypes = F.normalize(local_prototypes, dim=1)
    local_nodes_embeddings = F.normalize(local_nodes_embeddings, dim=1)
    global_prototypes = F.normalize(global_prototypes, dim=1)
    global_nodes_embeddings = F.normalize(global_nodes_embeddings, dim=1)
    prototypes = [local_prototypes, global_prototypes]
    nodes_embeddings = [local_nodes_embeddings, global_nodes_embeddings]
    anchor_nodes = [local_anchor_nodes, global_anchor_nodes]
    max_count = counts.max()
    label_class = torch.unique(labels)
    loss = 0
    for i in range(len(prototypes)):
        branch_prototype = prototypes[i]
        for label in label_class:
            count = counts[label]
            temp = dynamic_temperature + (1 - dynamic_temperature) * (count / max_count)
            prototype = branch_prototype[label]
            for j in range(len(anchor_nodes)):
                class_nodes = [node for node in anchor_nodes[j] if labels[node] == label]
                branch_similarity = torch.exp(torch.matmul(nodes_embeddings[j], prototype.unsqueeze(1)).t() / temp)
                sum_branch_similarity = torch.sum(branch_similarity, dim=1)
                class_similarity = torch.exp(torch.matmul(nodes_embeddings[j][class_nodes], prototype.unsqueeze(1)).t() / temp)
                loss += -torch.mean(torch.log(class_similarity / sum_branch_similarity))
    N = (len(prototypes) + len(nodes_embeddings)) * (label_class.max().item() + 1)
    return loss / N

def cross_domain_contrastive_loss(local_anchor_nodes, global_anchor_nodes, local_nodes_embeddings, global_nodes_embeddings, target_embedding, labels, temperature=1):
    """
    calculate cross domain CL loss
    """
    anchor_nodes = [local_anchor_nodes, global_anchor_nodes]
    source_embedding = [local_nodes_embeddings, global_nodes_embeddings]
    loss = 0.0
    for i in range(len(anchor_nodes)):
        target_embedding_nor = F.normalize(target_embedding, dim=1)
        branch_source_embedding_nor = F.normalize(source_embedding[i][anchor_nodes[i]], dim=1)
        similarities = torch.exp(torch.matmul(target_embedding_nor, branch_source_embedding_nor.T) / temperature)
        similarities_sums = similarities.sum(dim=1, keepdim=True)
        normalized_similarities = similarities / similarities_sums
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels[anchor_nodes[i]], num_classes=num_classes).float().to(labels.device)
        pseudo_labels = torch.matmul(normalized_similarities, one_hot_labels)
        target_pseudo_labels = pseudo_labels.argmax(dim=1)
        for class_idx in range(num_classes):
            target_class_nodes = (target_pseudo_labels == class_idx).nonzero().squeeze(1)
            if target_class_nodes.size(0) == 0:
                continue
            target_class_embedding = target_embedding[target_class_nodes]
            source_class_nodes = [node for node in anchor_nodes[i] if labels[node] == class_idx]
            source_class_embedding = source_embedding[i][source_class_nodes]
            target_class_embedding_nor = F.normalize(target_class_embedding, dim=1)
            source_class_embedding_nor = F.normalize(source_class_embedding, dim=1)
            similarity = torch.exp(torch.matmul(target_class_embedding_nor, source_class_embedding_nor.T) / temperature)
            similarity_sums = similarity.sum(dim=1, keepdim=True)
            nor_similarity = similarity / similarity_sums
            target_loss = -torch.log(nor_similarity).mean(dim=1)
            loss += target_loss.mean() / target_loss.size(0)
    return loss / len(anchor_nodes)

def imgda_objective(encoder, classifier, source, target, original_source_counts,
                    source_cache_name, target_cache_name, k=200, dynamic_temperature=0.5):
    """Released three-loss objective; target labels are never read.

    Counts come from the original complete source labels, as in official main.py.
    Source classification uses the global branch. Target contrastive features
    sum both branches. The source population must contain at least two nodes
    per class, matching the released prototype/anchor implementation's scope.
    """
    device = source.x.device
    local_source = encoder(source.x, source.edge_index, source_cache_name, True, device, use_ppmi=False)
    local_target = encoder(target.x, target.edge_index, target_cache_name, False, device, use_ppmi=False)
    global_source = encoder(source.x, source.edge_index, source_cache_name, True, device, use_ppmi=True)
    global_target = encoder(target.x, target.edge_index, target_cache_name, False, device, use_ppmi=True)
    target_embedding = local_target + global_target
    local_anchors = sampling_anchor_nodes(source, k, use_ppmi=False)
    global_anchors = sampling_anchor_nodes(source, k, use_ppmi=True)
    branch_loss = cross_branch_prototype_contrastive_loss(local_anchors, global_anchors,
        local_source, global_source, source.y, original_source_counts, dynamic_temperature)
    domain_loss = cross_domain_contrastive_loss(local_anchors, global_anchors,
        local_source, global_source, target_embedding, source.y)
    classification_loss = F.cross_entropy(classifier(global_source), source.y)
    return dict(loss=branch_loss + domain_loss + classification_loss,
                cb_cL_loss=branch_loss, cd_cl_loss=domain_loss, lc_loss=classification_loss,
                local_anchor_nodes=local_anchors, global_anchor_nodes=global_anchors)
