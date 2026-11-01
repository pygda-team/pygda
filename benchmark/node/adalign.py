#!/usr/bin/env python3
"""ADAlign full-batch Citation benchmark with final-epoch evaluation.

Read existing trusted PyG caches only. No downloads, cache regeneration,
target-label selection, dependency installation, or model checkpoints.
"""
import argparse
import csv
import datetime
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from contextlib import contextmanager, nullcontext
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
DOMAINS = ('ACMv9', 'Citationv1', 'DBLPv7')
NODES = {'ACMv9': 9360, 'Citationv1': 8935, 'DBLPv7': 5484}
SEEDS = (42, 43, 44, 45, 46)
DATA_ROOT = None
PARAMS = {}
AUDIT_TRAINING = False


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_forward_monitor(forward, phase_fn, report, params,
                         source_nodes, target_nodes, finite):
    """Wrap a training forward and retain only scalar diagnostic records."""
    def monitored_forward(src, tgt, *args, **kwargs):
        phase = phase_fn()
        epoch = len(report['forward_checks']) // 2 + 1
        expected_phase = 'sampler_ascent' if len(report['forward_checks']) % 2 == 0 else 'main_descent'
        require(phase == expected_phase and epoch <= params['epoch'], 'Unexpected forward order or epoch count.')
        record = {'epoch': epoch, 'phase': phase, 'source_nodes': src.x.size(0),
                  'target_nodes': tgt.x.size(0), 'target_y_absent': tgt.y is None,
                  'passed': False}
        report['forward_checks'].append(record)
        require(record['target_y_absent'], 'Target labels reached the training forward.')
        require(src.x.size(0) == source_nodes and tgt.x.size(0) == target_nodes,
                'Training is not using both full graphs.')
        values = forward(src, tgt, *args, **kwargs)
        for name, value in zip(('total_loss', 'source_logits', 'target_logits', 'alignment_loss'), values):
            finite(value, phase + '/' + name)
        record.update(total_loss=float(values[0].detach()),
                      alignment_loss=float(values[3].detach()), passed=True)
        record['source_ce_loss'] = record['total_loss'] - params['weight'] * record['alignment_loss']
        if params.get('verbose', 0) and (epoch == 1 or epoch % 10 == 0):
            print('FORWARD PASS: epoch=', epoch, phase, 'loss=', record['total_loss'],
                  'alignment=', record['alignment_loss'], flush=True)
        return values

    return monitored_forward


@contextmanager
def training_monitor(model, report, journal, params, source_nodes, target_nodes, cuda):
    """Audit one isolated fit; restore patched methods even if training fails."""
    import torch

    original_step = torch.optim.Adam.step

    def finite(tensor, name):
        require(bool(torch.isfinite(tensor).all()), 'Non-finite tensor: ' + name)

    def monitored_step(optimizer, *args, **kwargs):
        pairs = [(name, p) for name, p in model.a2gnn.named_parameters()]
        main_ids = {id(p) for _, p in pairs}
        sample_pairs = list(model.sample_net.named_parameters())
        sample_ids = {id(p) for _, p in sample_pairs}
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        require(not main_ids & sample_ids, 'Optimizer parameter groups overlap.')
        if optimizer_ids == sample_ids:
            phase, pairs = 'sampler_ascent', sample_pairs
        elif optimizer_ids == main_ids:
            phase = 'main_descent'
        else:
            raise RuntimeError('Unexpected optimizer parameter set.')
        epoch = len(report['optimizer_checks']) // 2 + 1
        expected_phase = 'sampler_ascent' if len(report['optimizer_checks']) % 2 == 0 else 'main_descent'
        require(phase == expected_phase and epoch <= params['epoch'], 'Unexpected optimizer order or epoch count.')
        record = {'epoch': epoch, 'phase': phase, 'passed': False, 'gradient_tensors': 0}
        report['optimizer_checks'].append(record)
        grad_norms = []
        for name, p in pairs:
            finite(p.detach(), phase + '/parameter/' + name)
            if p.grad is not None:
                finite(p.grad, phase + '/gradient/' + name)
                grad_norms.append(float(torch.linalg.vector_norm(p.grad.detach().double())))
        record['gradient_tensors'] = len(grad_norms)
        norm = math.sqrt(sum(value * value for value in grad_norms))
        record['gradient_l2'] = norm if math.isfinite(norm) else str(norm)
        require(grad_norms and math.isfinite(norm), phase + ': missing or non-finite gradients.')
        record['zero_gradient_norm'] = norm == 0
        # Upstream also computes/accumulates sampler gradients in the main
        # phase. Check those too, even though that optimizer is not stepped.
        other_pairs = sample_pairs if phase == 'main_descent' else list(model.a2gnn.named_parameters())
        for name, p in other_pairs:
            if p.grad is not None:
                finite(p.grad, phase + '/unstepped_group_gradient/' + name)
        record['unstepped_group_gradients_finite'] = True
        before = [p.detach().clone() for _, p in pairs]
        result = original_step(optimizer, *args, **kwargs)
        changed = 0
        for (name, p), previous in zip(pairs, before):
            finite(p.detach(), phase + '/updated_parameter/' + name)
            changed += not torch.equal(previous, p.detach())
        for name, buffer in list(model.a2gnn.named_buffers()) + list(model.sample_net.named_buffers()):
            if buffer.is_floating_point() or buffer.is_complex():
                finite(buffer, phase + '/buffer/' + name)
        # An exactly zero finite gradient or unchanged parameters can occur at
        # a stationary point. Record them without changing the training rule.
        record.update(updated_parameter_tensors=changed, passed=True)
        if params.get('verbose', 0) and (epoch == 1 or epoch % 10 == 0):
            print('OPTIMIZER PASS: epoch=', epoch, phase, 'gradient_l2=', record['gradient_l2'],
                  'updated_tensors=', changed, flush=True)
        if phase == 'main_descent':
            current = report['forward_checks'][-1]
            event = {'epoch': epoch, 'loss': current['total_loss'],
                     'alignment_loss': current['alignment_loss'],
                     'source_ce_loss': current['source_ce_loss'],
                     'gnn_gradient_l2': record['gradient_l2'],
                     'sampler_gradient_l2': report['optimizer_checks'][-2]['gradient_l2'],
                     'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                     'current_allocated_mib': torch.cuda.memory_allocated() / 2**20 if cuda else None,
                     'current_reserved_mib': torch.cuda.memory_reserved() / 2**20 if cuda else None}
            journal.write(json.dumps(event, allow_nan=False) + '\n')
            journal.flush()
            report['completed_optimizer_epoch_pairs'] = epoch
        return result

    monitored_forward = make_forward_monitor(
        model.forward_model,
        lambda: 'sampler_ascent' if not any(
            p.requires_grad for p in model.a2gnn.parameters()) else 'main_descent',
        report, params, source_nodes, target_nodes, finite)
    with patch.object(model, 'forward_model', new=monitored_forward), \
         patch.object(torch.optim.Adam, 'step', new=monitored_step):
        yield


def train_epochs(report, journal, source_name, target_name, seed):
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    import torch_geometric
    import pygda
    from pygda.models import ADAlign
    from torch_geometric.data import Data
    from sklearn.metrics import f1_score
    require(Path(pygda.__file__).resolve() == REPO / 'pygda/__init__.py',
            'Imported PyGDA from the wrong checkout.')
    cuda = torch.device(PARAMS['device']).type == 'cuda'
    if cuda:
        require(torch.cuda.is_available(), 'Requested CUDA device unavailable.')
        torch.cuda.set_device(torch.device(PARAMS['device']))
    report['environment'] = {'python': sys.version, 'torch': torch.__version__,
        'pyg': torch_geometric.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if cuda else None,
        'gpu_total_mib': torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**20 if cuda else None,
        'pygda_path': pygda.__file__}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if cuda else None
    report['seed'] = seed
    report['deterministic_algorithms_enabled'] = torch.are_deterministic_algorithms_enabled()
    report['parameters'] = PARAMS
    source_cache = torch.load(DATA_ROOT / source_name / 'processed/data.pt',
                              map_location='cpu', weights_only=False)
    target_cache = torch.load(DATA_ROOT / target_name / 'processed/data.pt',
                              map_location='cpu', weights_only=False)
    source_data, source_slices = source_cache
    target_data, target_slices = target_cache
    require(source_slices is None and target_slices is None, 'Expected single-graph caches.')
    source_nodes, target_nodes = NODES[source_name], NODES[target_name]
    require(tuple(source_data.x.shape) == (source_nodes, 6775) and
            tuple(target_data.x.shape) == (target_nodes, 6775), 'Unexpected graph feature shapes.')
    evaluation_y = target_data.y.detach().cpu().clone()
    require(source_data.y.dtype == torch.long and evaluation_y.dtype == torch.long,
            'Label tensors must be int64.')
    require(set(source_data.y.tolist()) == set(range(5)) and
            set(evaluation_y.tolist()) == set(range(5)), 'Expected label IDs 0 through 4.')
    # Deliberately construct minimal graphs: no split masks, eival, eivec,
    # or other TDSS-specific attributes. Keep original target labels outside fit.
    source = Data(x=source_data.x, edge_index=source_data.edge_index, y=source_data.y)
    target = Data(x=target_data.x, edge_index=target_data.edge_index,
                  y=torch.full((target_nodes,), -999, dtype=torch.long))
    predict_graph = Data(x=target_data.x, edge_index=target_data.edge_index)
    del source_cache, target_cache, source_data, target_data
    gc.collect()
    model = ADAlign(**PARAMS)
    report['forward_checks'] = []
    report['optimizer_checks'] = []
    report['target_label_protocol'] = {
        'fit_input': 'Synthetic invalid label -999 on every target node',
        'actual_target_labels': 'Held separately for evaluation after fit',
        'split_masks': 'Excluded from both training graphs',
        'selection': 'Fixed configured epochs; no target-label early stopping, checkpoint or hyperparameter selection'}
    def finite(tensor, name):
        require(bool(torch.isfinite(tensor).all()), 'Non-finite tensor: ' + name)

    torch.cuda.reset_peak_memory_stats() if cuda else None
    started = time.perf_counter()
    try:
        monitor = training_monitor(
            model, report, journal, PARAMS, source_nodes, target_nodes, cuda
        ) if AUDIT_TRAINING else nullcontext()
        with monitor:
            fitted = model.fit(source, target)
        require(fitted is model and len(model.history_) == PARAMS['epoch'], 'Fit did not finish the configured epochs.')
        require(bool((target.y == -999).all()), 'Fit mutated its target input labels.')
        require(model.target_loader.data.y is None, 'Target loader retained labels.')
        require(all(set(h) == {'epoch', 'loss', 'alignment_loss'} for h in model.history_),
                'Unexpected training-history metric fields.')
        report['history'] = model.history_
        report['training_audit_enabled'] = AUDIT_TRAINING
        if AUDIT_TRAINING:
            require([r['phase'] for r in report['forward_checks']] == ['sampler_ascent', 'main_descent'] * PARAMS['epoch'],
                    'Unexpected training forward sequence.')
            require([r['phase'] for r in report['optimizer_checks']] == ['sampler_ascent', 'main_descent'] * PARAMS['epoch'],
                    'Unexpected optimizer sequence.')
            for phase in ('sampler_ascent', 'main_descent'):
                require(any(r['updated_parameter_tensors'] > 0 for r in report['optimizer_checks']
                            if r['phase'] == phase), 'No parameter update in ' + phase)
            report['monitor_summary'] = {
                'forward_checks_passed': sum(r['passed'] for r in report['forward_checks']),
                'optimizer_checks_passed': sum(r['passed'] for r in report['optimizer_checks']),
                'zero_gradient_steps': sum(r['zero_gradient_norm'] for r in report['optimizer_checks']),
                'unchanged_parameter_steps': sum(r['updated_parameter_tensors'] == 0
                                                 for r in report['optimizer_checks']),
                'target_labels_absent_all_forwards': all(r['target_y_absent'] for r in report['forward_checks'])}
        logits, returned_y = model.predict(predict_graph)
        require(returned_y is None and tuple(logits.shape) == (target_nodes, 5), 'Unexpected prediction output.')
        finite(logits, 'prediction_logits')
        predictions = logits.argmax(dim=1).detach().cpu().numpy()
        report['evaluation'] = {
            'protocol': 'Final epoch, all target nodes, external evaluation after fit',
            'micro_f1': float(f1_score(evaluation_y.numpy(), predictions, average='micro')),
            'macro_f1': float(f1_score(evaluation_y.numpy(), predictions, average='macro')),
            'prediction_class_counts': {str(k): int((predictions == k).sum()) for k in range(5)},
            'interpretation': 'Single-seed final-epoch validation with audited CLI defaults. Not the author target-selected maximum-over-epoch protocol; not a full paper reproduction.'}
        report['checks_passed'] = True
    finally:
        # Preserve completed history and partial phase records on numerical
        # failure or Ctrl+C. The JSONL journal is flushed after each main step.
        report['history'] = list(model.history_)
        report['completed_history_epochs'] = len(model.history_)
        torch.cuda.synchronize() if cuda else None
        report['elapsed_seconds'] = time.perf_counter() - started
        report['gpu_memory_mib'] = {
            'peak_allocated': torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
            'peak_reserved': torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
            'current_allocated': torch.cuda.memory_allocated() / 2**20 if cuda else None,
            'note': 'PyTorch allocator statistics; optional training audit adds parameter snapshots.'}


def summarize(runs, tasks):
    summary = []
    for source, target in tasks:
        group = [r for r in runs if r['source'] == source and r['target'] == target
                 and r.get('checks_passed') and r.get('evaluation')]
        micro = [r['evaluation']['micro_f1'] for r in group]
        macro = [r['evaluation']['macro_f1'] for r in group]
        summary.append({'source': source, 'target': target, 'n_runs': len(group),
            'complete': len(group) == len(SEEDS), 'seeds': [r['seed'] for r in group],
            'micro_f1_mean': statistics.mean(micro) if micro else None,
            'micro_f1_std': statistics.pstdev(micro) if micro else None,
            'macro_f1_mean': statistics.mean(macro) if macro else None,
            'macro_f1_std': statistics.pstdev(macro) if macro else None})
    return summary


def write_summary(batch_dir, batch):
    batch['summary'] = summarize(batch['runs'], batch['tasks'])
    target = batch_dir / 'batch.json'
    temporary = batch_dir / 'batch.json.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(batch, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(target)
    with (batch_dir / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['source', 'target', 'n_runs', 'complete', 'seeds',
                  'micro_f1_mean_percent', 'micro_f1_std_percent',
                  'macro_f1_mean_percent', 'macro_f1_std_percent']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in batch['summary']:
            formatted = {k: row[k] for k in fields[:4]}
            formatted['seeds'] = ','.join(map(str, row['seeds']))
            for metric in ('micro_f1_mean', 'micro_f1_std', 'macro_f1_mean', 'macro_f1_std'):
                formatted[metric + '_percent'] = '' if row[metric] is None else row[metric] * 100
            writer.writerow(formatted)



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Existing Citation domain directories; caches are read-only')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audit-report', type=Path,
                        help='Optional stage-5 raw/cache audit; fingerprints must still match')
    parser.add_argument('--source', choices=DOMAINS, default='ACMv9')
    parser.add_argument('--target', choices=DOMAINS, default='Citationv1')
    parser.add_argument('--all-transfers', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--hid-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--weight', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--s-pnums', type=int, default=0)
    parser.add_argument('--t-pnums', type=int, default=10)
    parser.add_argument('--t-batchsize', type=int, default=2048)
    parser.add_argument('--verbose', type=int, choices=range(4), default=1)
    parser.add_argument('--audit-training', action='store_true',
                        help='Record per-phase gradients, updates and finite-value checks')
    args = parser.parse_args(argv)
    if not args.all_transfers and args.source == args.target:
        parser.error('Source and target must differ.')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('Seeds must be unique integers in [0, 2**32).')
    if min(args.epochs, args.hid_dim, args.num_layers) < 1:
        parser.error('Epochs and dimensions must be positive.')
    if args.t_batchsize < 32 or args.t_batchsize % 16:
        parser.error('t-batchsize must be a multiple of 16 and at least 32.')
    numeric = (args.dropout, args.lr, args.weight_decay, args.weight, args.alpha)
    if not all(math.isfinite(x) for x in numeric):
        parser.error('Numeric parameters must be finite.')
    if not 0 <= args.dropout < 1 or not 0 <= args.alpha <= 1 or args.lr <= 0:
        parser.error('Invalid dropout, alpha, or learning rate.')
    if min(args.weight_decay, args.weight, args.s_pnums, args.t_pnums, args.verbose) < 0:
        parser.error('Weights, propagation counts and verbosity must be nonnegative.')
    return args


def planned_tasks(args):
    return [(s, t) for s in DOMAINS for t in DOMAINS if s != t] if args.all_transfers else [(args.source, args.target)]


def verify_data(report, tasks, audit_path):
    selected = sorted({name for pair in tasks for name in pair})
    tracked = {}
    for name in selected:
        domain = DATA_ROOT / name
        cache = domain / 'processed/data.pt'
        require(cache.is_file(), 'Existing processed cache required: ' + str(cache))
        paths = [cache] + sorted((domain / 'raw').glob('*.txt'))
        for path in paths:
            require(path.resolve().is_relative_to(DATA_ROOT), 'Data path leaves data root.')
            tracked[str(path)] = sha(path)
    if audit_path is not None:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        require(Path(audit['dataset_root']).resolve() == DATA_ROOT, 'Audit dataset root differs.')
        entries = {row['domain']: row for row in audit['domains']}
        for name in selected:
            row = entries[name]
            require(row.get('status') == 'audited' and row.get('input_files_unchanged') and
                    row.get('untransformed_cache_match'), 'Raw/cache audit failed: ' + name)
            for text, prior in row['files_after'].items():
                path = Path(text)
                require(path.resolve().is_relative_to(DATA_ROOT / name), 'Audit path leaves domain root.')
                require(path.stat().st_size == prior['bytes'] and sha(path) == prior['sha256'],
                        'Audited fingerprint changed: ' + str(path))
                tracked[str(path)] = prior['sha256']
        report['data_audit_report'] = {'path': str(audit_path), 'sha256': sha(audit_path)}
    report['data_files_before'] = tracked
    return tracked


def main(argv=None):
    global DATA_ROOT, SEEDS, PARAMS, AUDIT_TRAINING
    args = parse_args(argv)
    AUDIT_TRAINING = args.audit_training
    DATA_ROOT = args.data_root.resolve()
    SEEDS = tuple(args.seeds)
    PARAMS = dict(in_dim=6775, hid_dim=args.hid_dim, num_classes=5,
        num_layers=args.num_layers, dropout=args.dropout, s_pnums=args.s_pnums,
        t_pnums=args.t_pnums, weight=args.weight, weight_decay=args.weight_decay,
        lr=args.lr, epoch=args.epochs, device=args.device, batch_size=0,
        num_neigh=-1, verbose=args.verbose, t_batchsize=args.t_batchsize,
        alpha=args.alpha, mode='node', adv=False)
    tasks = planned_tasks(args)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    batch_dir = output_root / ('adalign-citation-' + stamp)
    batch_dir.mkdir()
    batch = {'scope': 'Citation full-batch benchmark', 'tasks': tasks,
        'planned_runs': len(tasks)*len(SEEDS), 'epochs_per_run': args.epochs,
        'seeds': SEEDS, 'parameters': PARAMS, 'monitor_revision': 3, 'training_audit_enabled': AUDIT_TRAINING,
        'status': 'started', 'runs': [], 'runner_sha256': sha(Path(__file__).resolve()),
        'evaluation_protocol': 'Final epoch only; all target nodes; no target-label model selection',
        'std_convention': 'Population standard deviation (ddof=0); JSON scores in [0,1], CSV in percent.',
        'interpretation': 'Fixed configuration; not task-specific paper-best or author maximum-over-epoch results.'}
    start, tracked = time.perf_counter(), None
    try:
        tracked = verify_data(batch, tasks, args.audit_report)
        write_summary(batch_dir, batch)
        stop = False
        for source, target in tasks:
            if stop:
                break
            for seed in SEEDS:
                prefix = f'{source}-to-{target}-seed{seed}'
                report_path, journal_path = batch_dir/(prefix+'.json'), batch_dir/(prefix+'.jsonl')
                run = {'source': source, 'target': target, 'seed': seed,
                    'task': source+' -> '+target, 'epochs': args.epochs,
                    'parameters': PARAMS, 'checks_passed': False, 'status': 'started',
                    'monitor_revision': 3, 'training_audit_enabled': AUDIT_TRAINING, 'progress_journal': str(journal_path),
                    'report_path': str(report_path)}
                if args.verbose:
                    print(f'RUN {len(batch["runs"])+1}/{batch["planned_runs"]}: {source} -> {target}, seed={seed}', flush=True)
                try:
                    with journal_path.open('x', encoding='utf-8', buffering=1) as journal:
                        train_epochs(run, journal, source, target, seed)
                    require(run.get('checks_passed'), 'Training did not pass all checks.')
                    run['status'] = 'completed'
                except (Exception, KeyboardInterrupt) as exc:
                    run.update(checks_passed=False, status='stopped_with_error',
                        error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                    stop = True
                with report_path.open('x', encoding='utf-8') as stream:
                    json.dump(run, stream, ensure_ascii=False, indent=2, allow_nan=False)
                batch['runs'].append({k: run[k] for k in
                    ('source','target','seed','checks_passed','status','report_path')})
                for key in ('evaluation','elapsed_seconds','gpu_memory_mib','error'):
                    if key in run:
                        batch['runs'][-1][key] = run[key]
                for key in ('environment','deterministic_algorithms_enabled'):
                    if key in run:
                        batch.setdefault(key, run[key])
                write_summary(batch_dir, batch)
                if args.verbose:
                    print('RUN RESULT:', run['status'], run.get('evaluation'), run.get('error'), flush=True)
                gc.collect()
                import torch
                if str(args.device).startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if stop:
                    break
        batch['status'] = 'all_runs_passed' if len(batch['runs']) == batch['planned_runs'] and not stop else 'stopped_with_error'
    except (Exception, KeyboardInterrupt) as exc:
        batch.update(status='stopped_with_error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        try:
            if tracked is not None:
                batch['data_files_after'] = {p: sha(Path(p)) for p in tracked}
                batch['data_files_unchanged'] = tracked == batch['data_files_after']
                if not batch['data_files_unchanged']:
                    batch['status'] = 'input_state_changed'
        except Exception as exc:
            batch.update(status='postcheck_failed', postcheck_error=f'{type(exc).__name__}: {exc}')
        batch['total_elapsed_seconds'] = time.perf_counter()-start
        batch['successful_runs'] = sum(r['checks_passed'] for r in batch['runs'])
        write_summary(batch_dir, batch)
        print('BATCH STATUS:', batch['status'], flush=True)
        print('BATCH JSON:', batch_dir/'batch.json', flush=True)
        print('SUMMARY CSV:', batch_dir/'summary.csv', flush=True)
    return 0 if batch['status'] == 'all_runs_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
