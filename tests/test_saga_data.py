"""Multi-view contract tests; author preprocessing comparisons are optional."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from torch_geometric.data import Data

from pygda.datasets.saga import (SAGADataset, load_saga_graph, prepare_saga_views,
                                saga_adjacencies, validate_saga_graph, validate_saga_pair)


def graph(nodes=5):
    # Asymmetric weighted matrices expose accidental transposes or weight loss.
    indices = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 0]])
    weights = torch.tensor([0.5, 2.0, 3.0, 0.75, 1.0, 9.0])
    return Data(x=torch.arange(nodes * 3, dtype=torch.float32).reshape(nodes, 3),
                view_names=('PAP', 'PSP'),
                edge_index_dict={'PAP': indices.flip(0), 'PSP': indices},
                edge_weight_dict={'PAP': weights.clone(), 'PSP': weights.clone()})


class SAGADataTests(unittest.TestCase):
    def test_integer_and_boolean_weights_are_preserved(self):
        for dtype in (torch.int64, torch.int32, torch.uint8, torch.bool):
            with self.subTest(dtype=dtype):
                data = graph()
                for view in data.view_names:
                    data.edge_weight_dict[view] = data.edge_weight_dict[view].to(dtype)
                self.assertEqual(validate_saga_graph(data), ('PAP', 'PSP'))
                matrices = saga_adjacencies(data)
                self.assertEqual(matrices[0].dtype, dtype)
                normalized, features = prepare_saga_views(data)
                self.assertEqual(normalized[0].dtype, torch.float32)
                self.assertTrue(all(torch.isfinite(value).all() for value in features))
        for value, message in ((-1.0, 'negative'), (float('nan'), 'non-finite'),
                               (float('inf'), 'non-finite')):
            data = graph()
            data.edge_weight_dict['PAP'][0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                validate_saga_graph(data)

    def test_weighted_matrix_round_trip_and_no_mutation(self):
        data = graph()
        saved = data.clone()
        raw = saga_adjacencies(data)
        expected = torch.sparse_coo_tensor(data.edge_index_dict['PAP'].flip(0),
                                           data.edge_weight_dict['PAP'], (5, 5)).coalesce()
        torch.testing.assert_close(raw[0].to_dense(), expected.to_dense())
        self.assertAlmostEqual(raw[0].to_dense()[1, 2].item(), 2.0)
        normalized, propagated = prepare_saga_views(data, k_hop=2, filter_alpha=0.2)
        self.assertEqual(len(normalized), 2)
        for name in data.view_names:
            torch.testing.assert_close(data.edge_index_dict[name], saved.edge_index_dict[name])
            torch.testing.assert_close(data.edge_weight_dict[name], saved.edge_weight_dict[name])
        torch.testing.assert_close(data.x, saved.x)
        for result in propagated:
            self.assertEqual(tuple(result.shape), (5, 3))
            self.assertTrue(torch.isfinite(result).all())

    def test_invalid_contract_and_different_node_counts(self):
        source, target = graph(), graph()
        target.x = torch.ones(6, 3)
        target.y = torch.full((6,), -999)
        self.assertEqual(validate_saga_pair(source, target), ('PAP', 'PSP'))
        target.view_names = ('PSP', 'PAP')
        with self.assertRaises(ValueError):
            validate_saga_pair(source, target)
        invalid = graph()
        invalid.edge_weight_dict['PAP'][0] = float('nan')
        with self.assertRaises(ValueError):
            validate_saga_graph(invalid)
        for kwargs in ({'k_hop': 0}, {'filter_alpha': float('nan')}, {'filter_alpha': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_saga_views(source, **kwargs)

    def test_target_loader_does_not_open_label_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'ACM1'
            folder.mkdir()
            torch.save(torch.ones(5, 512), folder / 'feat.pt')
            matrix = saga_adjacencies(graph())[0]
            for name in ('pap', 'psp'):
                torch.save(matrix, folder / (name + '.pt'))
            target = load_saga_graph(temporary, 'ACM1', include_labels=False)
            self.assertIsNone(target.y)
            dataset = SAGADataset(temporary, 'ACM1', include_labels=False)
            clone = dataset[0]
            clone.x.zero_()
            self.assertEqual(dataset[0].x.sum().item(), 5 * 512)
            with self.assertRaises(FileNotFoundError):
                load_saga_graph(temporary, 'ACM1', include_labels=True)
            torch.save(torch.eye(3)[torch.tensor([0, 1, 2, 0, 1])], folder / 'label.pt')
            labeled = load_saga_graph(temporary, 'ACM1')
            torch.testing.assert_close(labeled.y, torch.tensor([0, 1, 2, 0, 1]))
            self.assertFalse((Path(temporary) / 'processed').exists())

    def test_author_preprocessing_parity(self):
        location = os.environ.get('SAGA_OFFICIAL_DIR')
        if not location:
            self.skipTest('Set SAGA_OFFICIAL_DIR to enable the pinned author comparison.')
        folder = Path(location)
        commit = subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()
        self.assertEqual(commit, '55ff00a4e426a2d0fbeb913f11330ab69b68a506')
        self.assertFalse(subprocess.check_output(
            ['git', '-C', str(folder), 'status', '--porcelain'], text=True).strip())
        spec = importlib.util.spec_from_file_location('saga_author_preprocess', folder / 'module/preprocess.py')
        author = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(author)
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            for alpha, dtype in ((0.0, torch.float32), (0.2, torch.float32),
                                 (1.0, torch.float32), (0.0, torch.int64),
                                 (0.0, torch.bool)):
                with self.subTest(device=device, alpha=alpha, dtype=dtype):
                    data = graph()
                    for view in data.view_names:
                        data.edge_weight_dict[view] = data.edge_weight_dict[view].to(dtype)
                    actual_adjs, actual_features = prepare_saga_views(data, 2, alpha, device)
                    x = data.x.to(device)
                    expected_adjs = author.graph_process_large(saga_adjacencies(data, device), x, None)
                    expected_features = author.APPNP([x, x], expected_adjs, 2, alpha)
                    for actual, expected in zip(actual_adjs, expected_adjs):
                        torch.testing.assert_close(actual.to_dense(), expected.to_dense())
                    for actual, expected in zip(actual_features, expected_features):
                        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
