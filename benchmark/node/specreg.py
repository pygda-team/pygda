import sys
sys.path.append('..')

import torch
import argparse
import time
import os.path as osp
import numpy as np

from pygda.datasets import CitationDataset, TwitchDataset
from pygda.datasets import AirportDataset, MAGDataset, BlogDataset

from pygda.models import SpecReg
from pygda.utils import svd_transform

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
parser.add_argument('--source', type=str, default='Blog1', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='Blog2', help='target domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--epochs', type=int, default=800, help='maximum number of epochs')
parser.add_argument('--filename', type=str, default='test.txt', help='store results into file')

# model specific params
parser.add_argument('--ppmi', type=bool, default=True, help='use PPMI matrix or not')
parser.add_argument('--adv_dim', type=int, default=40, help='hidden dimension of adversarial module')
parser.add_argument('--reg_mode', type=bool, default=True, help='use reg mode or adv mode')
parser.add_argument('--gamma_adv', type=float, default=0.1, help='trade off parameter for adv')
parser.add_argument('--thr_smooth', type=float, default=-1, help='spectral smoothness threshold')
parser.add_argument('--gamma_smooth', type=float, default=0.01, help='trade off parameter for spectral smoothness')
parser.add_argument('--thr_mfr', type=float, default=-1, help='maximum frequency response threshold')
parser.add_argument('--gamma_mfr', type=float, default=0.01, help='trade off parameter for maximum frequency response')

args = parser.parse_args()

# load data 
if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Citation', args.source)
    source_dataset = CitationDataset(path, args.source, pre_transform=svd_transform)
elif args.source in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Twitch', args.source)
    source_dataset = TwitchDataset(path, args.source, pre_transform=svd_transform)
elif args.source in {'BRAZIL', 'USA', 'EUROPE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Airport', args.source)
    source_dataset = AirportDataset(path, args.source, pre_transform=svd_transform)
elif args.source in {'Blog1', 'Blog2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Blog', args.source)
    source_dataset = BlogDataset(path, args.source, pre_transform=svd_transform)
elif args.source in {'MAG_CN', 'MAG_DE', 'MAG_FR', 'MAG_JP', 'MAG_RU', 'MAG_US'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/MAG', args.source)
    source_dataset = MAGDataset(path, args.source, pre_transform=svd_transform)

if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target, pre_transform=svd_transform)
elif args.target in {'DE', 'EN', 'ES', 'FR', 'PT', 'RU'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Twitch', args.target)
    target_dataset = TwitchDataset(path, args.target, pre_transform=svd_transform)
elif args.target in {'BRAZIL', 'USA', 'EUROPE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Airport', args.target)
    target_dataset = AirportDataset(path, args.target, pre_transform=svd_transform)
elif args.target in {'Blog1', 'Blog2'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Blog', args.target)
    target_dataset = BlogDataset(path, args.target, pre_transform=svd_transform)
elif args.target in {'MAG_CN', 'MAG_DE', 'MAG_FR', 'MAG_JP', 'MAG_RU', 'MAG_US'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/MAG', args.target)
    target_dataset = MAGDataset(path, args.target, pre_transform=svd_transform)

# construct node attributes for Airport dataset
max_degree = 0
for name in {'BRAZIL', 'USA', 'EUROPE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/Airport', name)
    dataset = AirportDataset(path, name)
    data_degree = max(degree(dataset[0].edge_index[0,:]))
    if data_degree > max_degree:
        max_degree = data_degree

if args.source in {'BRAZIL', 'USA', 'EUROPE'}:
    target_dataset.transform = OneHotDegree(int(max_degree))
    source_dataset.transform = OneHotDegree(int(max_degree))

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
model = SpecReg(
    in_dim=num_features,
    hid_dim=args.nhid,
    num_classes=num_classes,
    num_layers=args.num_layers,
    weight_decay=args.weight_decay,
    lr=args.lr,
    dropout=args.dropout_ratio,
    epoch=args.epochs,
    device=args.device,
    ppmi=args.ppmi,
    adv_dim=args.adv_dim,
    reg_mode=args.reg_mode,
    gamma_adv=args.gamma_adv,
    thr_smooth=args.thr_smooth,
    gamma_smooth=args.gamma_smooth,
    thr_mfr=args.thr_mfr,
    gamma_mfr=args.gamma_mfr
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

results = 'specreg,source,' + args.source + ',target,' + args.target + ',micro-f1,' + str(mi_f1) + ',macro-f1,' + str(ma_f1) + ',auc,' + str(auc)

with open(args.filename, 'a+') as f:
    f.write(results + '\n')

print(results)