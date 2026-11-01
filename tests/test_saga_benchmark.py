"""CLI contracts and target-label separation in the SAGA benchmark."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('saga_benchmark', Path(__file__).resolve().parents[1] / 'benchmark/node/saga.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class SAGABenchmarkTests(unittest.TestCase):
    def test_configuration_and_invalid_cli(self):
        self.assertEqual(benchmark.CONFIGS['ACM2-ACM1'], dict(epoch=100, dropout=0.05, delta=0.0))
        self.assertEqual(benchmark.parse_args(['--data-root', '/data']).seeds, [0, 1, 2, 3, 4])
        for extra in (['--seeds', '0', '0'], ['--epochs', '0'], ['--batch-size', '-1']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                benchmark.parse_args(['--data-root', '/data'] + extra)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'result.json'
            destination.write_text('user content', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                benchmark.main(['--data-root', folder, '--output', str(destination)])
            self.assertEqual(destination.read_text(encoding='utf-8'), 'user content')

    def test_real_fit_evaluates_target_labels_only_after_training(self):
        import torch
        from torch_geometric.data import Data
        from pygda.models import SAGA

        def graph(nodes, labeled):
            index = torch.stack((torch.arange(nodes), (torch.arange(nodes) + 1) % nodes))
            value = Data(x=torch.arange(nodes * 8, dtype=torch.float32).reshape(nodes, 8) / 100,
                         view_names=('PAP', 'PSP'), saga_num_classes=3,
                         edge_index_dict={v: index.clone() for v in ('PAP', 'PSP')},
                         edge_weight_dict={v: torch.ones(nodes) for v in ('PAP', 'PSP')})
            if labeled:
                value.y = torch.arange(nodes) % 3
            return value

        source, target = graph(6, True), graph(10, False)
        completed = []
        original = SAGA.fit

        def checked_fit(model, src, tgt):
            self.assertIsNone(tgt.y)
            result = original(model, src, tgt)
            completed.append(True)
            return result

        def read_labels(path, **kwargs):
            self.assertEqual(Path(path), Path('/fixture/ACM2/label.pt'))
            self.assertEqual(completed, [True])
            return torch.arange(10) % 3

        with patch('pygda.datasets.saga.load_saga_graph', side_effect=[source, target]) as loader, \
             patch.object(SAGA, 'fit', new=checked_fit), patch.object(torch, 'load', new=read_labels):
            row = benchmark.run_one(Path('/fixture'), 'ACM1-ACM2', 0, 'cpu', epochs=1, batch_size=4)
        self.assertEqual([call.kwargs['include_labels'] for call in loader.call_args_list], [True, False])
        self.assertEqual(len(row['history']), 1)
        self.assertEqual(sum(row['evaluation']['prediction_class_counts']), 10)
        self.assertIsNone(target.y)


if __name__ == '__main__':
    unittest.main()
