#!/usr/bin/env python3
"""UAGA open-set node benchmark using fixed official task configurations."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
DOMAINS = ('citation-v1', 'dblp-v4', 'acm-v8')
CONFIGS = {
    ('citation-v1', 'dblp-v4'): dict(dropout=.4, weight_decay=1e-4),
    ('citation-v1', 'acm-v8'): dict(dropout=.5, weight_decay=1e-4),
    ('dblp-v4', 'citation-v1'): dict(dropout=.5, weight_decay=1e-4),
    ('dblp-v4', 'acm-v8'): dict(dropout=.5, weight_decay=1e-5),
    ('acm-v8', 'citation-v1'): dict(dropout=.6, weight_decay=1e-4),
    ('acm-v8', 'dblp-v4'): dict(dropout=.5, weight_decay=1e-5),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source', choices=DOMAINS, default='citation-v1')
    parser.add_argument('--target', choices=DOMAINS, default='dblp-v4')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=200, help='Adaptation epochs; official default 200.')
    parser.add_argument('--separation-epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--output', type=Path, help='New JSON file; an existing path is never overwritten.')
    args = parser.parse_args(argv)
    if args.source == args.target:
        parser.error('Source and target must differ.')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('Seeds must be distinct integers in [0, 2**32).')
    if min(args.epochs, args.separation_epochs, args.batch_size) < 1:
        parser.error('Epoch counts and batch size must be positive.')
    return args


def summarize(rows):
    if not rows:
        raise ValueError('At least one completed seed is required.')
    return {key: dict(mean=statistics.mean(r['metrics'][key] for r in rows),
                      std=statistics.pstdev(r['metrics'][key] for r in rows))
            for key in ('OS', 'OS*', 'UNK_AUC', 'HS')}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_one(args, seed):
    import torch
    from torch_geometric.data import Data
    from pygda.models import UAGA
    from pygda.datasets.uaga import load_uaga_graph
    from pygda.metrics.uaga import evaluate_uaga
    source = load_uaga_graph(args.data_root, args.source)
    target = load_uaga_graph(args.data_root, args.target)
    unlabelled = Data(x=target.x, edge_index=target.edge_index)
    settings = dict(in_dim=source.x.shape[1], hid_dim=32, num_classes=5,
                    epoch=args.epochs, separation_epochs=args.separation_epochs,
                    batch_size=args.batch_size, device=args.device, seed=seed, verbose=0,
                    **CONFIGS[(args.source, args.target)])
    model = UAGA(**settings)
    cuda = torch.device(args.device).type == 'cuda'
    if cuda:
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    def progress(adapter, phase, epoch):
        if epoch == 1 or epoch % 25 == 0 or (phase == 0 and epoch == args.separation_epochs):
            print('UAGA seed={} phase={} epoch={}'.format(seed, phase, epoch), flush=True)
    model.fit(source, unlabelled, epoch_callback=progress)
    logits, _ = model.predict(unlabelled)
    metrics = evaluate_uaga(target.y.numpy(), logits.argmax(1).cpu().numpy(),
                            logits.softmax(1)[:, -1].cpu().numpy(), num_classes=5)
    if cuda:
        torch.cuda.synchronize()
    result = dict(seed=seed, settings=settings, metrics=metrics, epochs_completed=len(model.history_),
                  elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if cuda else None)
    del model, logits, source, target, unlabelled
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return result


def main(argv=None):
    args = parse_args(argv)
    # Reserve output before expensive work; partial seed results remain inspectable.
    output = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open('x', encoding='utf-8')
    report = dict(source=args.source, target=args.target, seeds=args.seeds, runs=[], checks_passed=False,
                  protocol='Last adaptation epoch; target labels used only for external final evaluation.',
                  metric_units='fractions', std_ddof=0,
                  full_official_schedule=args.epochs == 200 and args.separation_epochs == 30)
    def save():
        if output:
            output.seek(0)
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
            output.truncate()
            output.flush()
    try:
        fingerprints = {n: sha(args.data_root/(n+'.mat')) for n in (args.source,args.target)}
        report['data_sha256'] = fingerprints
        save()
        for seed in args.seeds:
            report['runs'].append(run_one(args, seed))
            report['summary'] = summarize(report['runs'])
            save()
        report['data_unchanged'] = fingerprints == {n:sha(args.data_root/(n+'.mat')) for n in fingerprints}
        report['checks_passed'] = report['data_unchanged'] and len(report['runs']) == len(args.seeds)
        if not report['checks_passed']:
            raise RuntimeError('Dataset files changed during the benchmark.')
        print(json.dumps(report['summary'], indent=2), flush=True)
        return report
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        save()
        if output:
            output.close()


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    main()
