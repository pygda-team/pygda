import sys
sys.path.append('..')

import torch
import argparse
import time
import os.path as osp
import numpy as np

from pygda.datasets import ArxivDataset

from pygda.models import AdaGCN

from pygda.metrics import eval_micro_f1, eval_macro_f1, eval_roc_auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree, is_undirected, to_undirected
from torch_geometric.transforms import OneHotDegree

parser = argparse.ArgumentParser()

# model agnostic params
parser.add_argument('--seed', type=int, default=200, help='random seed')
parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:3', help='specify cuda devices')
parser.add_argument('--source', type=str, default='arxiv-1950-2016', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='arxiv-2016-2018', help='target domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--filename', type=str, default='test.txt', help='store results into file')

# model specific params
parser.add_argument('--gnn_type', type=str, default='gcn', help='use GCN or PPMIConv')
parser.add_argument('--adv_dim', type=int, default=40, help='hidden dimension of adversarial module')
parser.add_argument('--gp_weight', type=float, default=5.0, help='trade off parameter for gradient penalty')
parser.add_argument('--domain_weight', type=float, default=1.0, help='trade off parameter for domain loss')


args = parser.parse_args()

# load data 
if args.source in {'arxiv-1950-2016', 'arxiv-2016-2018', 'arxiv-2018-2020'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/arxiv', args.source)
    source_dataset = ArxivDataset(path, args.source)
elif args.source in {'llm-arxiv-1950-2016', 'llm-arxiv-2016-2018', 'llm-arxiv-2018-2020'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/llm-arxiv', args.source)
    source_dataset = ArxivDataset(path, args.source)
elif args.source in {'llm-bert-arxiv-1950-2016', 'llm-bert-arxiv-2016-2018', 'llm-bert-arxiv-2018-2020'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/bert-arxiv', args.source)
    source_dataset = ArxivDataset(path, args.source)

if args.target in {'arxiv-1950-2016', 'arxiv-2016-2018', 'arxiv-2018-2020'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/arxiv', args.target)
    target_dataset = ArxivDataset(path, args.target)
elif args.target in {'llm-arxiv-1950-2016', 'llm-arxiv-2016-2018', 'llm-arxiv-2018-2020'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/llm-arxiv', args.target)
    target_dataset = ArxivDataset(path, args.target)
elif args.target in {'llm-bert-arxiv-1950-2016', 'llm-bert-arxiv-2016-2018', 'llm-bert-arxiv-2018-2020'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/bert-arxiv', args.target)
    target_dataset = ArxivDataset(path, args.target)

source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)

if args.source not in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
    if not is_undirected(source_data.edge_index):
        source_data.edge_index = to_undirected(source_data.edge_index)
    
    if not is_undirected(target_data.edge_index):
        target_data.edge_index = to_undirected(target_data.edge_index)

num_features = source_data.x.size(1)
num_classes = len(np.unique(source_data.y.cpu().numpy()))

# choose a graph domain adaptation model
model = AdaGCN(
    in_dim=num_features,
    hid_dim=args.nhid,
    num_classes=num_classes,
    num_layers=args.num_layers,
    weight_decay=args.weight_decay,
    lr=args.lr,
    dropout=args.dropout_ratio,
    epoch=args.epochs,
    device=args.device,
    gnn_type=args.gnn_type,
    adv_dim=args.adv_dim,
    gp_weight=args.gp_weight,
    domain_weight=args.domain_weight
    )

# train the model
model.fit(source_data, target_data)

# evaluate the performance
logits, labels = model.predict(target_data)

maxvalue, maxindex = torch.max(logits, dim=1)

preds = logits.argmax(dim=1)

mi_f1 = eval_micro_f1(labels, preds)
ma_f1 = eval_macro_f1(labels, preds)

if args.source in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
    auc = eval_roc_auc(labels, maxvalue)
else:
    auc = 0.0

results = 'adagcn,source,' + args.source + ',target,' + args.target + ',micro-f1,' + str(mi_f1) + ',macro-f1,' + str(ma_f1) + ',auc,' + str(auc)

with open(args.filename, 'a+') as f:
    f.write(results + '\n')

print(results)