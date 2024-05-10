import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from math import ceil
from sklearn.mixture import GaussianMixture
from torch_sparse import SparseTensor, spmm

from torch_geometric.utils import degree, add_self_loops
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops


class DIST(object):
    def __init__(self, dist_type):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            NA = pointA.size(0)
            NB = pointB.size(0)
            assert(pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))

    def euc(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)
        if not cross:
            return torch.norm(pointA-pointB)
        else:
            return torch.cdist(pointA,pointB,p=2)


def onehot(label_matrix, num_classes):
    device = label_matrix.device
    identity = torch.eye(num_classes).to(device)
    onehot = torch.index_select(identity, 0, label_matrix)
    
    return onehot

def get_emb_centers(embeddings, label_array, n_label):
    device = embeddings.device

    if len(label_array.shape) == 1:
        label_array = onehot(label_array, n_label).to(device)

    norm = 1 / torch.sum(label_array,dim=0).reshape(-1,1)

    return torch.matmul(label_array.T, embeddings)

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        
        return adj_t
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GMMClustering(object):
    def __init__(self, num_class, device, dist_type='cos'):
        self.Dist = DIST(dist_type)
        self.num_class = num_class
        self.device = device

    def forward(self, src_centers, emb_t, y_t, target_edge_index, target_edge_attr, smooth, smooth_r):
        n_label = src_centers.shape[0]
        self.gmm = GaussianMixture(n_components=self.num_class, n_init=5)
        self.gt = y_t
        model = self.gmm.fit(emb_t.cpu().numpy())
        self.target_pred = torch.IntTensor(model.predict(emb_t.cpu().numpy())).to(self.device)
        self.target_pred_prob = torch.FloatTensor(model.predict_proba(emb_t.cpu().numpy())).to(self.device)
        tgt_centers = get_emb_centers(emb_t, self.target_pred, n_label)
        cluster2label = self.align_centers(src_centers, tgt_centers)
        shuffle_idx = [x[1] for x in sorted({idx:i for i, idx in enumerate(cluster2label)}.items())]
        self.num_nodes = emb_t.size(0)
        labels = []
        for i in range(self.num_nodes):
            labels.append(cluster2label[self.target_pred[i]])
        self.labels = torch.LongTensor(labels).to(self.device)
        # re-arrange
        self.target_pred_prob = self.target_pred_prob[:,shuffle_idx]
        # assert torch.argmax(self.target_pred_prob, dim=1).equal(self.labels)
        if smooth:
            # smooth
            self.smooth_labels, p = self.smooth(target_edge_index.cpu(), target_edge_attr.cpu(), smooth_r)
            # summary
            self.samples = {"data":list(range(emb_t.size(0))), "label":self.smooth_labels, "dist2center":p, "gt":self.gt}
        else:
            self.samples = {"data":list(range(emb_t.size(0))), "label":self.labels, "dist2center":self.target_pred_prob, "gt":self.gt}
        
        return self.samples

    def smooth(self, edge_index, v, smooth_r):
        edge_index_sp = SparseTensor(row=edge_index[0], col=edge_index[1], value=v, sparse_sizes=(self.num_nodes,self.num_nodes))
        A_norm = gcn_norm(edge_index=edge_index_sp)
        row, col, v = A_norm.storage._row, A_norm.storage._col, A_norm.storage._value
        p = self.target_pred_prob
        for i in range(20):
            pred_prob_smooth = spmm(index=torch.vstack([row,col]), value=v, m=self.num_nodes, n=self.num_nodes, matrix=p)
            label = smooth_r*p+(1-smooth_r)*pred_prob_smooth
            p = label

        return torch.argmax(p, dim=1), p

    def align_centers(self, src_center, tgt_center):
        cost = self.Dist.get_dist(tgt_center, src_center, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)

        return col_ind


class Clustering(object):
    def __init__(self, eps, device,max_len=1000, dist_type='cos'):
        self.eps = eps
        self.device = device
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.max_len = max_len

    def set_init_centers(self, init_centers):
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            print('dist %.4f' % dist.item())
            self.stop = dist.item() < self.eps

    def assign_fake_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)

        return dists, labels

    def align_centers(self):

        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, feat,label):
        self.samples['gt'] = label
        self.samples['feature'] = feat
        self.samples['data'] = list(range(feat.size(0)))

    def feature_clustering(self,feat,label):
        centers = None
        self.stop = False

        self.collect_samples(feat,label)
        feature = self.samples['feature']

        refs = torch.LongTensor(range(self.num_classes)).unsqueeze(1).to(self.device)
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop: break

            centers = 0
            count = 0

            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_fake_labels(cur_feature)
                labels_onehot = onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.FloatTensor).to(self.device)
                reshaped_feature = cur_feature.unsqueeze(0)
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

            mask = (count.unsqueeze(1) > 0).type(torch.FloatTensor).to(self.device)
            centers = mask * centers + (1 - mask) * self.init_centers

        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_fake_labels(cur_feature)

            labels_onehot = onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        self.centers = self.centers[cluster2label, :]
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['label'][k] = cluster2label[self.samples['label'][k]].item()

        self.center_change = torch.mean(self.Dist.get_dist(self.centers, self.init_centers))

        del self.samples['feature']
        self.samples['label'].cpu()
