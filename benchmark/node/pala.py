"""PALA with explicit author data/cache paths and external target evaluation."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--source', default='ACMv9')
    parser.add_argument('--target', default='Citationv1')
    parser.add_argument('--imbalance-factor', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.epochs < 1 or args.imbalance_factor < 1 or args.source == args.target:
        parser.error('Use positive runs/epochs/IF and distinct source/target domains.')
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from sklearn.metrics import f1_score
    from pygda.models import PALA
    from pygda.datasets import load_pala_graph, load_pala_cache, prepare_pala_source, prepare_pala_target

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
    source_raw = load_pala_graph(args.data_root, args.source)
    target_raw = load_pala_graph(args.data_root, args.target)
    labels = target_raw.y.cpu().numpy()
    source = prepare_pala_source(source_raw, load_pala_cache(args.cache_dir / f'{args.source}{args.imbalance_factor}.pkl'),
                                 args.imbalance_factor, args.device)
    target = prepare_pala_target(target_raw, load_pala_cache(args.cache_dir / f'{args.target}.pkl'), args.device)
    report = dict(source=args.source, target=args.target, imbalance_factor=args.imbalance_factor,
                  epochs=args.epochs, seeds=list(range(args.runs)), deterministic=args.deterministic,
                  baseline='PALA with documented log-domain contrastive reduction', runs=[],
                  evaluation_protocol='External target oracle best Micro-F1 epoch and paired Macro-F1; final epoch also reported. Population std ddof=0.',
                  environment={key: importlib.metadata.version(key) for key in ['torch', 'torch-geometric', 'numpy', 'scikit-learn']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in range(args.runs):
        evaluations = []

        def evaluate(epoch, logits, record):
            predicted = logits.argmax(dim=1).cpu().numpy()
            evaluations.append(dict(epoch=epoch, micro_f1=float(f1_score(labels, predicted, average='micro')),
                                    macro_f1=float(f1_score(labels, predicted, average='macro'))))

        model = PALA(source.x.shape[1], 512, int(source.y.max()) + 1, epoch=args.epochs,
                     seed=seed, device=args.device, verbose=0)
        model.fit(source, target, epoch_callback=evaluate)
        best = max(evaluations, key=lambda row: row['micro_f1'])
        report['runs'].append(dict(seed=seed, best=best, final=evaluations[-1], evaluations=evaluations,
                                   training_history=model.history_))
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('seed', seed, 'best', best, 'final', evaluations[-1], flush=True)
    micro = [row['best']['micro_f1'] for row in report['runs']]
    macro = [row['best']['macro_f1'] for row in report['runs']]
    report['aggregate'] = dict(micro_f1_mean=statistics.mean(micro), micro_f1_std=statistics.pstdev(micro),
                               macro_f1_mean=statistics.mean(macro), macro_f1_std=statistics.pstdev(macro), std_ddof=0)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['aggregate'], indent=2))


if __name__ == '__main__':
    main()
