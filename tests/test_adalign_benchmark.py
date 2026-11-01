"""Benchmark planning/aggregation/monitor regressions; no real training."""
import contextlib
import gc
import importlib.util
import io
import math
from pathlib import Path
import unittest
import weakref
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'benchmark/node/adalign.py'
spec = importlib.util.spec_from_file_location('adalign_benchmark', PATH)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def args(self, extra=()):
        return benchmark.parse_args(['--data-root', '/unused/data', '--output-dir', '/unused/results', *extra])

    def test_default_plan_and_all_transfers(self):
        args = self.args()
        self.assertEqual(args.epochs, 150)
        self.assertFalse(args.audit_training)
        self.assertTrue(self.args(['--audit-training']).audit_training)
        self.assertEqual(args.seeds, [42, 43, 44, 45, 46])
        self.assertEqual(benchmark.planned_tasks(args), [('ACMv9', 'Citationv1')])
        tasks = benchmark.planned_tasks(self.args(['--all-transfers']))
        self.assertEqual(len(set(tasks)), 6)
        self.assertTrue(all(s != t for s, t in tasks))

    def test_invalid_arguments(self):
        for extra in (['--source', 'ACMv9', '--target', 'ACMv9'],
                      ['--seeds', '42', '42'], ['--seeds', '-1'],
                      ['--epochs', '0'], ['--t-batchsize', '16'],
                      ['--t-batchsize', '33'], ['--alpha', 'nan'],
                      ['--weight', '-1'], ['--dropout', '1']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(extra)

    def test_summary_population_std_and_partial_runs(self):
        runs = [{'source': 'ACMv9', 'target': 'Citationv1', 'seed': seed,
                 'checks_passed': True, 'evaluation': {'micro_f1': value, 'macro_f1': value}}
                for seed, value in zip([42, 43], [0.6, 0.8])]
        failed = dict(runs[0], seed=44, checks_passed=False)
        with patch.object(benchmark, 'SEEDS', (42, 43, 44, 45, 46)):
            summary = benchmark.summarize(runs + [failed], [('ACMv9', 'Citationv1')])
        self.assertEqual(summary[0]['n_runs'], 2)
        self.assertFalse(summary[0]['complete'])
        self.assertAlmostEqual(summary[0]['micro_f1_mean'], 0.7)
        self.assertAlmostEqual(summary[0]['micro_f1_std'], 0.1)
        empty = benchmark.summarize([], [('DBLPv7', 'ACMv9')])[0]
        self.assertIsNone(empty['micro_f1_mean'])

    def test_real_forward_monitor_releases_inputs(self):
        class Features:
            def size(self, dimension):
                return 8
        class Graph:
            x = Features()
            y = None
        class Scalar(float):
            def detach(self):
                return self
        records = {'forward_checks': []}
        params = {'epoch': 150, 'weight': 0.4, 'verbose': 0}
        refs = []
        def forward(src, tgt):
            return Scalar(1), Scalar(2), Scalar(3), Scalar(0.5)
        def phase():
            return 'sampler_ascent' if len(records['forward_checks']) % 2 == 0 else 'main_descent'
        def finite(value, name):
            self.assertTrue(math.isfinite(value), name)
        monitor = benchmark.make_forward_monitor(
            forward, phase, records, params, 8, 8, finite)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            for _ in range(300):
                src, tgt = Graph(), Graph()
                refs.extend((weakref.ref(src), weakref.ref(tgt)))
                monitor(src, tgt)
                del src, tgt
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(len(records['forward_checks']), 300)
        self.assertEqual(captured.getvalue(), '')
        __import__('json').dumps(records, allow_nan=False)

    def test_real_forward_monitor_rejects_target_labels(self):
        class Graph:
            y = 1
            x = type('Features', (), {'size': lambda self, dim: 8})()
        monitor = benchmark.make_forward_monitor(
            lambda src, tgt: (), lambda: 'sampler_ascent',
            {'forward_checks': []}, {'epoch': 1}, 8, 8, lambda value, name: None)
        with self.assertRaisesRegex(RuntimeError, 'Target labels'):
            monitor(Graph(), Graph())


if __name__ == '__main__':
    unittest.main(verbosity=2)
