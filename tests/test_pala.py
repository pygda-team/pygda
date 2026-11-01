import unittest

import torch
from torch_geometric.data import Data

from pygda.models import PALA


class PALATests(unittest.TestCase):
    def pair(self):
        torch.manual_seed(12)
        n = 40
        edges = torch.stack([torch.arange(n), torch.arange(n)])
        source = Data(x=torch.randn(n, 4), y=torch.arange(n) // 20, edge_index=edges,
                      ppmi_edge_index=edges, ppmi_edge_weight=torch.ones(n))
        target = Data(x=torch.randn(n, 4), edge_index=edges,
                      ppmi_edge_index=edges, ppmi_edge_weight=torch.ones(n))
        return source, target

    def model(self, **kwargs):
        return PALA(4, 8, 2, epoch=2, gamma=2, anchor_num=5, device='cpu', verbose=0, seed=7, **kwargs)

    def test_training_ignores_target_labels(self):
        source, target = self.pair()
        first = self.model().fit(source, target)
        target.y = torch.full((target.num_nodes,), 999, dtype=torch.long)
        second = self.model().fit(source, target)
        self.assertEqual(first.history_, second.history_)
        for key, value in first.model.state_dict().items():
            torch.testing.assert_close(value, second.model.state_dict()[key], rtol=0, atol=0)

    def test_missing_cache_and_invalid_classes_fail_before_training(self):
        source, target = self.pair()
        del target.ppmi_edge_weight
        with self.assertRaises((ValueError, AttributeError)):
            self.model().fit(source, target)
        source, target = self.pair()
        source.y[:] = 0
        with self.assertRaises(ValueError):
            self.model().fit(source, target)

    def test_predict_requires_fit_and_accepts_different_graph(self):
        source, target = self.pair()
        model = self.model()
        with self.assertRaises(RuntimeError):
            model.predict(target)
        model.fit(source, target)
        edge = torch.tensor([[0, 1, 2], [0, 1, 2]])
        new = Data(x=torch.ones(3, 4), edge_index=edge, ppmi_edge_index=edge,
                   ppmi_edge_weight=torch.ones(3))
        scores, labels = model.predict(new, return_labels=False)
        self.assertEqual(tuple(scores.shape), (3, 2))
        self.assertIsNone(labels)

    def test_unsupported_minibatch_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model(batch_size=8)


if __name__ == '__main__':
    unittest.main()
