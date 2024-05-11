import argparse
import os.path as osp

from pygda.datasets import CitationDataset

from pygda.models import UDAGCN, A2GNN, GRADE
from pygda.models import ASN, SpecReg, GNN
from pygda.models import StruRW, ACDNE, DANE
from pygda.models import AdaGCN, JHGDA, KBL
from pygda.models import DGDA, SAGDA, CWGCN
from pygda.models import DMGNN, PairAlign

from pygda.metrics import eval_micro_f1, eval_macro_f1
from pygda.utils import svd_transform

parser = argparse.ArgumentParser()

parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--device', type=str, default='cuda:3', help='specify cuda devices')
parser.add_argument('--source', type=str, default='DBLPv7', help='source domain data, DBLPv7/ACMv9/Citationv1')
parser.add_argument('--target', type=str, default='ACMv9', help='target domain data, DBLPv7/ACMv9/Citationv1')

args = parser.parse_args()

# load data 
if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.source)
    source_dataset = CitationDataset(path, args.source)

if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)

source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)

num_features = source_data.x.size(1)
num_classes = len(np.unique(source_data.y.cpu().numpy()))

# choose a graph domain adaptation model
model = A2GNN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = UDAGCN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = GRADE(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = ASN(in_dim=num_features, hid_dim=args.nhid, hid_dim_vae=args.nhid, num_classes=num_classes, device=args.device)
# model = SpecReg(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device, reg_mode=True)
# model = GNN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = StruRW(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = ACDNE(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = DANE(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = AdaGCN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = JHGDA(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = KBL(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = DGDA(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = SAGDA(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = CWGCN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = DMGNN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
# model = PairAlign(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)

# train the model
model.fit(source_data, target_data)

# evaluate the performance
logits, labels = model.predict(target_data)

preds = logits.argmax(dim=1)

mi_f1 = eval_micro_f1(labels, preds)
ma_f1 = eval_macro_f1(labels, preds)

print('micro-f1: ' + str(mi_f1))
print('macro-f1: ' + str(ma_f1))
