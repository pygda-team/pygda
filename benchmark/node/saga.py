#!/usr/bin/env python3
"""SAGA ACM benchmark using fixed public configurations and final-epoch evaluation."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


REPO = Path(__file__).resolve().parents[2]
CONFIGS = {
    'ACM1-ACM2': dict(epoch=50, dropout=0.0, delta=0.01),
    'ACM2-ACM1': dict(epoch=100, dropout=0.05, delta=0.0),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Trusted author tensor directory containing ACM1/ and ACM2/.')
    parser.add_argument('--direction', choices=(*CONFIGS, 'both'), default='both')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, help='Explicit override, e.g. 1 for a smoke run.')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--output', type=Path, help='New JSON path; existing files are never replaced.')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('Seeds must be distinct integers in [0, 2**32).')
    if args.batch_size <= 0 or (args.epochs is not None and args.epochs <= 0):
        parser.error('Batch size and epoch override must be positive.')
    return args


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows):
    summaries = []
    for direction in CONFIGS:
        group = [r for r in rows if r['direction'] == direction]
        if not group:
            continue
        result = dict(direction=direction, seeds=[r['seed'] for r in group], n=len(group),
                      std_convention='sample standard deviation (ddof=1); null for n<2')
        for metric in ('accuracy', 'micro_f1', 'macro_f1'):
            values = [r['evaluation'][metric] for r in group]
            result[metric] = dict(mean=statistics.mean(values),
                                  std=statistics.stdev(values) if len(values) > 1 else None)
        summaries.append(result)
    return summaries


def run_one(data_root, direction, seed, device, epochs=None, batch_size=1000):
    import torch
    from sklearn.metrics import confusion_matrix, f1_score
    from pygda.datasets.saga import load_saga_graph
    from pygda.models import SAGA

    source_name, target_name = direction.split('-')
    source = load_saga_graph(data_root, source_name, include_labels=True)
    target = load_saga_graph(data_root, target_name, include_labels=False)
    if target.y is not None:
        raise RuntimeError('Target training graph must be unlabeled.')
    params = dict(in_dim=source.x.size(1), hid_dim=256, num_classes=source.saga_num_classes,
                  embed_dim=128, num_views=2, k_hop=2, filter_alpha=0.0,
                  alpha=0.95, beta=1.0, tau=1.0, lr=0.001, weight_decay=0.0001,
                  batch_size=batch_size, seed=seed, device=device, verbose=0, **CONFIGS[direction])
    if epochs is not None:
        params['epoch'] = epochs
    model = SAGA(**params)
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    model.fit(source, target)
    if cuda:
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - started
    if len(model.history_) != params['epoch']:
        raise RuntimeError('Incomplete fixed training budget.')
    logits, labels_returned = model.predict(target)
    if labels_returned is not None or not bool(torch.isfinite(logits).all()):
        raise RuntimeError('Invalid unlabeled target prediction.')
    # Target labels are decoded only after fixed-budget training and prediction.
    raw = torch.load(data_root / target_name / 'label.pt', map_location='cpu', weights_only=False)
    labels = raw.argmax(1) if raw.ndim == 2 else raw.long()
    predictions = logits.argmax(1).detach().cpu()
    if labels.shape != predictions.shape or (labels < 0).any() or (labels >= params['num_classes']).any():
        raise ValueError('Unexpected target evaluation labels.')
    classes = list(range(params['num_classes']))
    evaluation = dict(accuracy=float((predictions == labels).double().mean()),
                      micro_f1=float(f1_score(labels, predictions, average='micro', zero_division=0)),
                      macro_f1=float(f1_score(labels, predictions, labels=classes, average='macro', zero_division=0)),
                      prediction_class_counts=torch.bincount(predictions, minlength=len(classes)).tolist(),
                      confusion_matrix=confusion_matrix(labels, predictions, labels=classes).tolist())
    return dict(direction=direction, seed=seed, parameters=params, evaluation=evaluation,
                fit_seconds=fit_seconds, history=list(model.history_))


def main(argv=None):
    args = parse_args(argv)
    # Reserve the requested path before spending time training, protecting user files.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(dict(status='started'), stream)
    report = dict(status='started', protocol='All target nodes, final fixed epoch, no target-based selection.',
                  author_commit='55ff00a4e426a2d0fbeb913f11330ab69b68a506',
                  parameter_source='Pinned author ACM CLI/config defaults; explicit overrides recorded in parameters.',
                  comparison_note='Author main reports maximum target accuracy over epochs; this runner reports the final epoch.',
                  runs=[])
    inputs = {}
    started = time.perf_counter()
    try:
        sys.path.insert(0, str(REPO))
        import torch
        import torch_geometric
        report['environment'] = dict(torch=torch.__version__, pyg=torch_geometric.__version__, device=args.device)
        directions = tuple(CONFIGS) if args.direction == 'both' else (args.direction,)
        domains = {name for direction in directions for name in direction.split('-')}
        inputs = {str(args.data_root / domain / name): sha(args.data_root / domain / name)
                  for domain in sorted(domains) for name in ('feat.pt', 'label.pt', 'pap.pt', 'psp.pt')}
        report['input_sha256'] = inputs
        for direction in directions:
            for seed in args.seeds:
                row = run_one(args.data_root, direction, seed, args.device, args.epochs, args.batch_size)
                report['runs'].append(row)
                print(json.dumps(dict(direction=direction, seed=seed, evaluation=row['evaluation'])), flush=True)
                gc.collect()
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
        report['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)))
    finally:
        report['summary'] = summarize(report['runs'])
        report['elapsed_seconds'] = time.perf_counter() - started
        if inputs:
            report['input_files_unchanged'] = all(Path(p).is_file() and sha(Path(p)) == digest for p, digest in inputs.items())
            if not report['input_files_unchanged']:
                report.update(status='failed', input_error='Input fingerprints changed.')
        if args.output is not None:
            # This path was exclusively reserved by this invocation above.
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            print('REPORT:', args.output, flush=True)
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
