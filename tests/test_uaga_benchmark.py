"""Fixed benchmark configuration and official open-set metric conventions."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


class UAGABenchmarkTests(unittest.TestCase):
    def test_author_metric_conventions(self):
        from pygda.metrics.uaga import evaluate_uaga
        # Known micro accuracy 2/3 differs from macro known recall 3/4.
        y = np.array([0, 0, 1, 2, 2])
        pred = np.array([0, 1, 1, 2, 0])
        score = np.array([.1, .2, .3, .8, .9])
        result = evaluate_uaga(y, pred, score, 2)
        self.assertAlmostEqual(result['OS'], 2/3)
        self.assertAlmostEqual(result['OS*'], .75)
        self.assertAlmostEqual(result['HS'], 4/7)
        self.assertEqual(result['UNK_AUC'], 1.)
        zero = evaluate_uaga(y, np.array([1, 1, 0, 0, 0]), score, 2)
        self.assertEqual(zero['HS'], 0.)

    def test_fixed_configs_and_population_standard_deviation(self):
        path = Path(__file__).resolve().parents[1]/'benchmark/node/uaga.py'
        spec = importlib.util.spec_from_file_location('uaga_benchmark', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(len(module.CONFIGS), 6)
        args = module.parse_args(['--data-root','/unused'])
        self.assertEqual((args.seeds,args.epochs,args.separation_epochs), (list(range(5)),200,30))
        rows = [dict(metrics={k:v for k in ('OS','OS*','UNK_AUC','HS')}) for v in (.2,.4)]
        self.assertAlmostEqual(module.summarize(rows)['HS']['std'], .1)


if __name__ == '__main__':
    unittest.main()
