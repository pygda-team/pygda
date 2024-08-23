import sys
sys.path.append('..')

import torch
import argparse
import time
import os.path as osp
import numpy as np

from pygda.datasets import GraphTUDataset

from pygda.models import UDAGCN

from pygda.metrics import eval_micro_f1, eval_macro_f1, eval_roc_auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree, is_undirected, to_undirected
from torch_geometric.transforms import OneHotDegree

parser = argparse.ArgumentParser()

# model agnostic params
parser.add_argument('--seed', type=int, default=200, help='random seed')
parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
parser.add_argument('--device', type=str, default='cuda:1', help='specify cuda devices')
parser.add_argument('--source', type=str, default='FRANKENSTEIN_F1', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='FRANKENSTEIN_F2', help='target domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--filename', type=str, default='test.txt', help='store results into file')

# model specific params
parser.add_argument('--ppmi', type=bool, default=True, help='use PPMI matrix or not')
parser.add_argument('--adv_dim', type=int, default=40, help='hidden dimension of adversarial module')
parser.add_argument('--mode', type=str, default='graph', help='node or graph tasks')

args = parser.parse_args()

# load data 
if args.source in {'FRANKENSTEIN_F1', 'FRANKENSTEIN_F2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/FRANKENSTEIN', args.source)
    source_dataset = GraphTUDataset(path, args.source)
elif args.source in {'PROTEINS_P1', 'PROTEINS_P2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/PROTEINS', args.source)
    source_dataset = GraphTUDataset(path, args.source)
elif args.source in {'Mutagenicity_M1', 'Mutagenicity_M2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Mutagenicity', args.source)
    source_dataset = GraphTUDataset(path, args.source)

if args.target in {'FRANKENSTEIN_F1', 'FRANKENSTEIN_F2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/FRANKENSTEIN', args.target)
    target_dataset = GraphTUDataset(path, args.target)
elif args.target in {'PROTEINS_P1', 'PROTEINS_P2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/PROTEINS', args.target)
    target_dataset = GraphTUDataset(path, args.target)
elif args.target in {'Mutagenicity_M1', 'Mutagenicity_M2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Mutagenicity', args.target)
    target_dataset = GraphTUDataset(path, args.target)

if args.mode == 'node':
    source_data = source_dataset[0].to(args.device)
    target_data = target_dataset[0].to(args.device)

    num_features = source_data.x.size(1)
    num_classes = len(np.unique(source_data.y.cpu().numpy()))

    if args.source not in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
        if not is_undirected(source_data.edge_index):
            source_data.edge_index = to_undirected(source_data.edge_index)
    
        if not is_undirected(target_data.edge_index):
            target_data.edge_index = to_undirected(target_data.edge_index)
elif args.mode == 'graph':
    source_data = source_dataset
    target_data = target_dataset

    num_features = source_data.num_features
    num_classes = source_data.num_classes

# choose a graph domain adaptation model
model = UDAGCN(
    in_dim=num_features,
    hid_dim=args.nhid,
    num_classes=num_classes,
    mode=args.mode,
    num_layers=args.num_layers,
    weight_decay=args.weight_decay,
    lr=args.lr,
    dropout=args.dropout_ratio,
    epoch=args.epochs,
    device=args.device,
    ppmi=args.ppmi,
    adv_dim=args.adv_dim
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

results = 'udagcn,source,' + args.source + ',target,' + args.target + ',micro-f1,' + str(mi_f1) + ',macro-f1,' + str(ma_f1) + ',auc,' + str(auc)

with open(args.filename, 'a+') as f:
    f.write(results + '\n')

print(results)