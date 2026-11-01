"""ADAlign unit tests and optional comparisons with the author implementation."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import io
import contextlib
import unittest

import torch
from torch_geometric.data import Data

from pygda.models import ADAlign
from pygda.nn import ADAlignBase
from pygda.nn.adalign_base import CF, SampleNet


def external_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graphs(device):
    def graph(n):
        index = torch.arange(n)
        edges = torch.stack([index, (index + 1) % n])
        return Data(x=torch.randn(n, 5), edge_index=edges,
                    y=index.remainder(2).long()).to(device)
    return graph(8), graph(11)


class ADAlignIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        location = os.environ.get('ADALIGN_OFFICIAL_DIR')
        cls.official_dir = Path(location).resolve() if location else None

    def official_path(self, relative):
        if self.official_dir is None:
            self.skipTest('Set ADALIGN_OFFICIAL_DIR to enable author comparison tests.')
        try:
            commit = subprocess.check_output(
                ['git', '-C', str(self.official_dir), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.STDOUT).strip()
            dirty = subprocess.check_output(
                ['git', '-C', str(self.official_dir), 'status', '--porcelain'],
                text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            self.fail(f'Invalid ADALIGN_OFFICIAL_DIR: {exc}')
        self.assertEqual(commit, 'd6d7a700049782f9402b2bfa891c60f1cba552c7',
                         'Author comparison requires the documented pinned commit.')
        self.assertFalse(dirty, 'Author comparison requires a clean checkout.')
        path = self.official_dir / relative
        self.assertTrue(path.is_file(), f'Missing author source: {path}')
        return path

    def model(self, device):
        return ADAlign(5, 12, 2, t_batchsize=32, epoch=1,
                       device=device, verbose=0)

    def test_public_api_and_supported_scope(self):
        self.assertTrue(issubclass(ADAlignBase, torch.nn.Module))
        for kwargs in ({'batch_size': 4}, {'mode': 'graph'}, {'adv': True},
                       {'t_batchsize': 16}, {'num_neigh': 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ADAlign(5, 12, 2, **kwargs)

    def test_invalid_numeric_parameters(self):
        for name in ('weight', 'alpha', 'lr', 'weight_decay', 'dropout'):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ADAlign(5, 12, 2, **{name: value})
        for kwargs in ({'lr': 0}, {'weight_decay': -1}, {'dropout': 1},
                       {'verbose': -1}, {'verbose': 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ADAlign(5, 12, 2, **kwargs)

    def test_cf_symmetry_and_gradients_without_author_checkout(self):
        for device in self.devices:
            with self.subTest(device=device):
                torch.manual_seed(42)
                source = torch.randn(8, 12, device=device, requires_grad=True)
                target = torch.randn(11, 12, device=device, requires_grad=True)
                sampler = SampleNet(12, 32).to(device)
                frequencies = sampler(device)
                loss = CF(0.4, 0.6, source, target, frequencies)
                torch.testing.assert_close(loss, CF(0.4, 0.6, target, source, frequencies))
                loss.backward()
                for value in (source, target, *sampler.parameters()):
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all().item())

    def test_audit_monitor_preserves_training_and_releases_graphs(self):
        benchmark = external_module(
            'adalign_benchmark_test',
            Path(__file__).resolve().parents[1] / 'benchmark/node/adalign.py')
        for device in self.devices:
            with self.subTest(device=device):
                torch.manual_seed(42)
                source, target = graphs(device)
                baseline, audited = self.model(device), self.model(device)
                torch.manual_seed(123)
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    baseline.fit(source, target)
                self.assertEqual(captured.getvalue(), '')
                records = {'forward_checks': [], 'optimizer_checks': []}
                params = {'epoch': 1, 'weight': audited.weight, 'verbose': 0}
                torch.manual_seed(123)
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    with benchmark.training_monitor(
                            audited, records, io.StringIO(), params, 8, 11,
                            torch.device(device).type == 'cuda'):
                        audited.fit(source, target)
                self.assertEqual(captured.getvalue(), '')
                self.assertEqual(len(records['forward_checks']), 2)
                self.assertEqual(len(records['optimizer_checks']), 2)
                for plain, monitored in zip(baseline.history_, audited.history_):
                    self.assertEqual(plain['epoch'], monitored['epoch'])
                    for key in ('loss', 'alignment_loss'):
                        self.assertAlmostEqual(plain[key], monitored[key], places=5)
                for original, monitored in ((baseline.a2gnn, audited.a2gnn),
                                            (baseline.sample_net, audited.sample_net)):
                    for name, value in original.state_dict().items():
                        torch.testing.assert_close(value, monitored.state_dict()[name],
                                                   atol=1e-6, rtol=1e-5)
                json_text = __import__('json').dumps(records, allow_nan=False)
                self.assertTrue(json_text)

    def test_official_core_parity_and_gradients(self):
        path = self.official_path('utils/loss.py')
        official = external_module('adalign_official_loss', path)
        for device in self.devices:
            with self.subTest(device=device):
                torch.manual_seed(42)
                source = torch.randn(8, 12, device=device, requires_grad=True)
                target = torch.randn(11, 12, device=device, requires_grad=True)
                net = SampleNet(12, 32).to(device)
                reference = official.SampleNet(12, 32).to(device)
                reference.load_state_dict(net.state_dict())
                torch.manual_seed(123)
                frequencies = net(device)
                torch.manual_seed(123)
                expected = reference(device)
                torch.testing.assert_close(frequencies, expected)
                self.assertEqual(torch.unique(frequencies.detach(), dim=0).size(0), 2)
                reference_source = source.detach().clone().requires_grad_()
                reference_target = target.detach().clone().requires_grad_()
                actual = CF(0.4, 0.6, source, target, frequencies)
                expected_loss = official.CF(
                    0.4, 0.6, reference_source, reference_target, expected)
                torch.testing.assert_close(actual, expected_loss)
                torch.testing.assert_close(actual,
                    CF(0.4, 0.6, target, source, frequencies))
                actual.backward()
                expected_loss.backward()
                for actual_grad, expected_grad in (
                        (source.grad, reference_source.grad),
                        (target.grad, reference_target.grad)):
                    torch.testing.assert_close(actual_grad, expected_grad)
                reference_parameters = dict(reference.named_parameters())
                for name, parameter in net.named_parameters():
                    expected_grad = reference_parameters[name].grad
                    self.assertEqual(parameter.grad is None, expected_grad is None)
                    if parameter.grad is not None:
                        torch.testing.assert_close(parameter.grad, expected_grad)
                for grad in (source.grad, target.grad):
                    self.assertTrue(torch.isfinite(grad).all().item())
                    self.assertGreater(grad.abs().sum().item(), 0)
                grads = [p.grad for p in net.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_official_forward_parity(self):
        path = self.official_path('model/ncfm.py')
        sys.path.insert(0, str(self.official_dir))
        try:
            official = external_module('adalign_official_model', path)
        finally:
            sys.path.pop(0)
        for device in self.devices:
            with self.subTest(device=device):
                torch.manual_seed(42)
                source, target = graphs(device)
                model = self.model(device)
                reference = official.NCFM(
                    5, 12, 2, t_batchsize=32, epoch=1, device=device,
                    weight=model.weight, lr=model.lr, verbose=0)
                model.a2gnn = model.init_model()
                model.sample_net = SampleNet(12, 32).to(device)
                reference.a2gnn = reference.init_model()
                reference.sample_net = SampleNet(12, 32).to(device)
                reference.a2gnn.load_state_dict(model.a2gnn.state_dict())
                reference.sample_net.load_state_dict(model.sample_net.state_dict())
                torch.manual_seed(123)
                actual = model.forward_model(source, target, 0.0)
                torch.manual_seed(123)
                expected = reference.forward_model(source, target, 0.0)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_fit_target_label_independence_and_new_graph_prediction(self):
        for device in self.devices:
            with self.subTest(device=device):
                torch.manual_seed(42)
                source, target = graphs(device)
                source_copy, target_copy = source.clone(), target.clone()
                unlabeled = target.clone()
                unlabeled.y = None
                labeled_model, unlabeled_model = self.model(device), self.model(device)
                torch.manual_seed(123)
                self.assertIs(labeled_model.fit(source, target), labeled_model)
                torch.manual_seed(123)
                unlabeled_model.fit(source, unlabeled)
                for name, actual in labeled_model.a2gnn.state_dict().items():
                    torch.testing.assert_close(
                        actual, unlabeled_model.a2gnn.state_dict()[name],
                        atol=1e-5, rtol=1e-5)
                for name, actual in labeled_model.sample_net.state_dict().items():
                    torch.testing.assert_close(
                        actual, unlabeled_model.sample_net.state_dict()[name],
                        atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(source.x, source_copy.x)
                torch.testing.assert_close(target.x, target_copy.x)
                torch.testing.assert_close(source.y, source_copy.y)
                torch.testing.assert_close(target.y, target_copy.y)
                self.assertEqual(len(labeled_model.history_), 1)
                self.assertNotIn('target_f1', labeled_model.history_[0])
                probe = Data(x=torch.randn(7, 5, device=device),
                             edge_index=torch.empty((2, 0), dtype=torch.long,
                                                    device=device))
                logits, labels = labeled_model.predict(probe)
                self.assertEqual(tuple(logits.shape), (7, 2))
                self.assertIsNone(labels)
                self.assertTrue(torch.isfinite(logits).all().item())
                logits, labels = labeled_model.predict(target)
                self.assertEqual(tuple(logits.shape), (11, 2))
                torch.testing.assert_close(labels, target.y)


if __name__ == '__main__':
    print('CHECK DEVICES:', ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []),
          flush=True)
    unittest.main(verbosity=2)
