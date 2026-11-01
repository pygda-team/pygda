"""SAGA interface tests and optional pinned-author loss/gradient/update parity."""
import importlib.util
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch_geometric.data import Data

from pygda.models import SAGA
from pygda.nn import SAGABase
from pygda.nn.saga_base import Entropy
from pygda.datasets.saga import prepare_saga_views


def make_graph(nodes):
    index = torch.arange(nodes)
    edge = torch.stack((index, (index + 1) % nodes))
    return Data(x=torch.randn(nodes, 5), y=index.remainder(3).long(),
                view_names=('PAP', 'PSP'),
                edge_index_dict={'PAP': edge, 'PSP': edge.flip(0)},
                edge_weight_dict={'PAP': torch.ones(nodes), 'PSP': torch.arange(1, nodes + 1).float()})


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)


class SAGAModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])

    def reference(self):
        location = os.environ.get('SAGA_OFFICIAL_DIR')
        if not location:
            self.skipTest('Set SAGA_OFFICIAL_DIR for pinned-author comparisons.')
        folder = Path(location).resolve()
        self.assertEqual(subprocess.check_output(
            ['git', '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip(),
            '55ff00a4e426a2d0fbeb913f11330ab69b68a506')
        self.assertFalse(subprocess.check_output(
            ['git', '-C', str(folder), 'status', '--porcelain'], text=True).strip())
        name = '_pygda_saga_author_reference'
        package = ModuleType(name)
        package.__path__ = [str(folder / 'module')]
        sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(name + '.SAGA', folder / 'module/SAGA.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def wrapper(self, device, **kwargs):
        params = dict(in_dim=5, hid_dim=9, num_classes=3, embed_dim=6,
                      epoch=2, batch_size=4, dropout=0.1, seed=123,
                      device=device, verbose=0)
        params.update(kwargs)
        return SAGA(**params)

    def assert_state_close(self, actual, expected):
        self.assertEqual(set(actual.state_dict()), set(expected.state_dict()))
        for name, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], atol=1e-6, rtol=1e-5,
                                       msg=lambda message: name + ': ' + message)

    def test_public_api_and_invalid_parameters(self):
        self.assertTrue(issubclass(SAGABase, torch.nn.Module))
        for kwargs in ({'batch_size': 1}, {'tau': 0}, {'alpha': float('nan')},
                       {'beta': float('inf')}, {'num_layers': 3}, {'num_neigh': 2},
                       {'seed': -1}, {'filter_alpha': 2}, {'dropout': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.wrapper('cpu', **kwargs)

    def test_fit_target_label_independence_and_new_graph_prediction(self):
        for device in self.devices:
            with self.subTest(device=device):
                seed(42)
                source, target = make_graph(6), make_graph(10)
                source_copy, target_copy = source.clone(), target.clone()
                missing = target.clone()
                missing.y = None
                invalid = target.clone()
                invalid.y = torch.full((10,), -999)
                models = [self.wrapper(device) for _ in range(3)]
                for model, graph in zip(models, (target, missing, invalid)):
                    self.assertIs(model.fit(source, graph), model)
                self.assert_state_close(models[0].model, models[1].model)
                self.assert_state_close(models[0].model, models[2].model)
                torch.testing.assert_close(source.x, source_copy.x)
                torch.testing.assert_close(source.y, source_copy.y)
                torch.testing.assert_close(target.x, target_copy.x)
                torch.testing.assert_close(target.y, target_copy.y)
                for view in source.view_names:
                    torch.testing.assert_close(source.edge_index_dict[view], source_copy.edge_index_dict[view])
                    torch.testing.assert_close(source.edge_weight_dict[view], source_copy.edge_weight_dict[view])
                    torch.testing.assert_close(target.edge_weight_dict[view], target_copy.edge_weight_dict[view])
                self.assertEqual(len(models[0].history_), 2)
                self.assertTrue(any(step['source_nodes'] == 0 for step in models[0].batch_history_))
                self.assertTrue(all(set(row) == {'epoch', 'loss', 'steps'} for row in models[0].history_))
                probe = make_graph(7)
                probe.y = None
                logits, labels = models[0].predict(probe)
                self.assertEqual(tuple(logits.shape), (7, 3))
                self.assertIsNone(labels)
                self.assertTrue(torch.isfinite(logits).all())

    def test_view_order_and_singleton_batch_rejected(self):
        source, target = make_graph(5), make_graph(8)
        model = self.wrapper('cpu')
        with self.assertRaisesRegex(ValueError, 'singleton'):
            model.fit(source, target)
        self.assertIsNone(model.model)
        target.view_names = ('PSP', 'PAP')
        with self.assertRaisesRegex(ValueError, 'ordered'):
            self.wrapper('cpu').fit(make_graph(6), target)

    def test_author_components_and_parameter_gradients(self):
        reference = self.reference()
        for device in self.devices:
            for component in ('ae_loss', 'compute_sad', 'compute_sad_weights',
                              '_contrastive_loss', 'ia_loss', 'ca_loss', 'Entropy', 'forward'):
                with self.subTest(device=device, component=component):
                    seed(42)
                    config = SimpleNamespace(loss_weights=dict(alpha=0.95, beta=1.0, delta=0.01))
                    args = (5, 2, 9, 6, 1.0, 0.1, 6, 0, device, 3, config)
                    actual, expected = SAGABase(*args).to(device), reference.SAGA(*args).to(device)
                    expected.load_state_dict(actual.state_dict())
                    source, target = make_graph(6), make_graph(8)
                    source.x, target.x = source.x.to(device), target.x.to(device)
                    fs = [torch.randn(6, 5, device=device) for _ in range(2)]
                    ft = [torch.randn(8, 5, device=device) for _ in range(2)]
                    actual_features = ([f.clone().requires_grad_() for f in fs],
                                       [f.clone().requires_grad_() for f in ft])
                    expected_features = ([f.clone().requires_grad_() for f in fs],
                                         [f.clone().requires_grad_() for f in ft])
                    labels = F.one_hot(source.y.to(device), 3)

                    def evaluate(model, entropy, feature_sets):
                        fs, ft = feature_sets
                        if component == 'forward':
                            return model(target.x, ft, source.x, fs, labels, False)
                        zs, zt = model.encoder(fs), model.encoder(ft)
                        if component == 'ae_loss':
                            return model.ae_loss(zs, fs) + model.ae_loss(zt, ft)
                        if component == 'compute_sad':
                            return model.compute_sad(zt, zs).sum()
                        if component == 'compute_sad_weights':
                            a = model.compute_sad(zt, zs)
                            weights = model.compute_sad_weights(a, a.flip(0))
                            scale = torch.arange(1, 3, device=device)
                            return sum((weight * scale).sum() for weight in weights)
                        if component == '_contrastive_loss':
                            return model._contrastive_loss(zs[0], zt[1])
                        if component == 'ia_loss':
                            return model.ia_loss(zt, zs, 1, 1)
                        if component == 'ca_loss':
                            ds, dt = model.dominant_view_mining(zs, source.x), model.dominant_view_mining(zt, target.x)
                            return model.ca_loss(zt, zs, dt, ds)
                        return entropy(F.softmax(model.classifier(torch.cat(zs, 1)), -1)).mean()

                    seed(123)
                    actual_loss = evaluate(actual, Entropy, actual_features)
                    seed(123)
                    expected_loss = evaluate(expected, reference.Entropy, expected_features)
                    torch.testing.assert_close(actual_loss, expected_loss)
                    actual_loss.backward()
                    expected_loss.backward()
                    for group, other_group in zip(actual_features, expected_features):
                        for feature, other in zip(group, other_group):
                            self.assertEqual(feature.grad is None, other.grad is None)
                            if feature.grad is not None:
                                self.assertTrue(torch.isfinite(feature.grad).all())
                                torch.testing.assert_close(feature.grad, other.grad, atol=1e-6, rtol=1e-5)
                    reference_parameters = dict(expected.named_parameters())
                    present = 0
                    for name, parameter in actual.named_parameters():
                        other = reference_parameters[name]
                        self.assertEqual(parameter.grad is None, other.grad is None, name)
                        if parameter.grad is not None:
                            present += 1
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-5,
                                                       msg=lambda message: name + ': ' + message)
                    self.assertGreater(present, 0)
                    self.assert_state_close(actual, expected)

    def test_author_node_batches_optimizer_updates_and_predictions(self):
        reference = self.reference()
        for device in self.devices:
            for counts in ((6, 10), (10, 6)):
                with self.subTest(device=device, counts=counts):
                    seed(42)
                    source, target = (make_graph(n) for n in counts)
                    model = self.wrapper(device)
                    model.fit(source, target)
                    x_s, x_t = source.x.to(device), target.x.to(device)
                    _, fs = prepare_saga_views(source, 2, 0.0, device)
                    _, ft = prepare_saga_views(target, 2, 0.0, device)
                    labels = F.one_hot(source.y.to(device), 3)
                    config = SimpleNamespace(loss_weights=dict(alpha=model.alpha, beta=model.beta, delta=model.delta))
                    seed(123)
                    official = reference.SAGA(5, 2, 9, 6, model.tau, model.dropout,
                                              len(source.x), 0, device, 3, config).to(device)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, official.parameters()),
                                                 lr=model.lr, weight_decay=model.weight_decay)
                    history = []
                    for epoch in range(2):
                        official.train()
                        order_s = torch.tensor([i for i in RandomSampler(range(counts[0]), replacement=False)])
                        order_t = torch.tensor([i for i in RandomSampler(range(counts[1]), replacement=False)])
                        for batch in range(max(math.ceil(n / 4) for n in counts)):
                            si, ti = order_s[batch * 4:(batch + 1) * 4], order_t[batch * 4:(batch + 1) * 4]
                            optimizer.zero_grad()
                            loss = official(x_t[ti] if ti.numel() else None,
                                            [f[ti] for f in ft] if ti.numel() else None,
                                            x_s[si] if si.numel() else None,
                                            [f[si] for f in fs] if si.numel() else None,
                                            labels[si] if si.numel() else None, False)
                            loss.backward()
                            optimizer.step()
                            history.append(float(loss.detach()))
                    self.assert_state_close(model.model, official)
                    self.assertEqual(len(model.batch_history_), len(history))
                    for row, value in zip(model.batch_history_, history):
                        self.assertAlmostEqual(row['loss'], value, places=5)
                    official.eval()
                    with torch.no_grad():
                        expected = official.classifier(torch.cat(official.encoder(ft), dim=1))
                    actual, _ = model.predict(target)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
