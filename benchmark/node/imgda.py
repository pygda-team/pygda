"""Released ImGDA protocol: all-target evaluation and best-Micro epoch selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from pygda.datasets import load_imgda_pair
from pygda.models import ImGDA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Trusted released data/ and tmp/ parent.')
    parser.add_argument('--source', default='ACMv9')
    parser.add_argument('--target', default='Citationv1')
    parser.add_argument('--imbalance-factor',type=int,default=20)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--runs',type=int,default=5)
    parser.add_argument('--hid-dim',type=int,default=512)
    parser.add_argument('--dropout',type=float,default=.1)
    parser.add_argument('--lr',type=float,default=.01)
    parser.add_argument('--weight-decay',type=float,default=.001)
    parser.add_argument('--k',type=int,default=200)
    parser.add_argument('--dynamic-temperature',type=float,default=.5)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error('--runs must be positive')
    source,target = load_imgda_pair(args.root,args.source,args.target,args.imbalance_factor,args.device)
    # Target labels are used only here, outside model.fit, for released evaluation.
    graph,_ = torch.load(args.root/'data'/args.target/'processed/data.pt',map_location='cpu',weights_only=False)
    labels = graph.y.numpy()
    del graph
    results = []
    for seed in range(args.runs):
        best = dict(seed=seed,micro_f1=-1.,macro_f1=None,epoch=None)
        def evaluate(epoch,losses,logits):
            predictions = logits.argmax(dim=1).cpu().numpy()
            micro = float(f1_score(labels,predictions,average='micro'))
            macro = float(f1_score(labels,predictions,average='macro'))
            if micro > best['micro_f1']:
                best.update(micro_f1=micro,macro_f1=macro,epoch=epoch)
            if epoch % 10 == 0:
                print(f'seed={seed} epoch={epoch} micro_f1={micro:.6f} macro_f1={macro:.6f}',flush=True)
        model = ImGDA(in_dim=source.x.size(1),hid_dim=args.hid_dim,
            num_classes=source.imgda_class_counts.numel(),dropout=args.dropout,
            lr=args.lr,weight_decay=args.weight_decay,epoch=args.epochs,k=args.k,
            dynamic_temperature=args.dynamic_temperature,seed=seed,device=args.device,verbose=0)
        model.fit(source,target,callback=evaluate)
        results.append(best.copy())
        print('RUN:',json.dumps(best),flush=True)
        del model
    result = dict(source=args.source,target=args.target,imbalance_factor=args.imbalance_factor,
        epochs=args.epochs,runs=results,std_ddof=0,
        protocol='All target nodes, target-label best-Micro epoch selection and paired Macro, outside fit.',
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda))
    for metric in ('micro_f1','macro_f1'):
        values = [r[metric] for r in results]
        result[metric+'_mean'] = float(np.mean(values))
        result[metric+'_std'] = float(np.std(values))
    text = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output is not None:
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(text)
    print(text,flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
