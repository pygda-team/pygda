"""UAGA parity tests. Set UAGA_AUTHOR_DIR and UAGA_DATA_ROOT for reference checks."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

RESULTS = {}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cycle(n, width=6):
    i = torch.arange(n)
    return SimpleNamespace(x=torch.arange(n * width).reshape(n, width).float() / 100,
                           edge_index=torch.stack((torch.cat((i, (i+1) % n)),
                                                   torch.cat(((i+1) % n, i)))), y=None)


class UAGATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('GPU reference tests; CPU legacy exit issue remains unresolved.')
        import dgl
        from pygda.models.uaga import UAGA
        from pygda.nn.uaga_base import UAGABase
        cls.UAGA, cls.Core, cls.dgl = UAGA, UAGABase, dgl
        cls.device = 'cuda:0'
        cls.author = Path(os.environ.get('UAGA_AUTHOR_DIR', '/missing-uaga-author'))

    def _states_close(self, a, b, name, atol=2e-6):
        self.assertEqual(list(a), list(b))
        maximum = 0.0
        for key in a:
            self.assertTrue(torch.allclose(a[key], b[key], atol=atol, rtol=1e-5), name + ': ' + key)
            maximum = max(maximum, float((a[key]-b[key]).abs().max()))
        RESULTS[name] = dict(max_abs_difference=maximum, atol=atol, rtol=1e-5)

    def test_official_core_losses_gradients_and_adam(self):
        if not (self.author / 'models.py').is_file():
            self.skipTest('Set UAGA_AUTHOR_DIR to the pinned official checkout.')
        official = _load('_uaga_oracle_models', self.author / 'models.py')
        graph = _cycle(8)
        g = self.dgl.graph(tuple(graph.edge_index), num_nodes=8).to(self.device).add_self_loop()
        collator = self.dgl.dataloading.NodeCollator(g, torch.arange(8, device=self.device),
                           self.dgl.dataloading.MultiLayerFullNeighborSampler(2))
        inputs, outputs, blocks = collator.collate([0, 1, 2, 3])
        for phase in (0, 1):
            torch.manual_seed(19)
            oracle = official.Network(1, 6, 4, [2, 2], .3, .3, torch.nn.ReLU(True), 3).to(self.device)
            torch.manual_seed(19)
            candidate = self.Core(1, 6, 4, [2, 2], .3, .3, torch.nn.ReLU(True), 3).to(self.device)
            self._states_close(oracle.state_dict(), candidate.state_dict(), 'initialization-phase-' + str(phase), atol=0)
            adapter = self.UAGA(6, 4, 2, num_heads=2, num_out_heads=2, dropout=.3, epoch=1,
                                separation_epochs=1, top_k=3, device=self.device, verbose=0)
            adapter.model = candidate
            features = graph.x.to(self.device)[inputs]
            coeff = torch.ones(len(blocks[-1].srcdata['_ID']), 1, device=self.device)
            coeff[::2] = -1
            sb = SimpleNamespace(blocks=blocks, x=features.clone().requires_grad_(), y=outputs % 2,
                                 domain=torch.zeros(len(coeff), device=self.device))
            tb = SimpleNamespace(blocks=blocks, x=(features+.01).clone().requires_grad_(),
                                 pseudo_label=torch.tensor([-1, 2, 0, 1], device=self.device), grl=coeff,
                                 domain=torch.ones(len(coeff), device=self.device))
            sx = sb.x.detach().clone().requires_grad_()
            tx = tb.x.detach().clone().requires_grad_()
            torch.manual_seed(37)
            s = oracle(blocks, sx, 1, False, phase)
            t = oracle(blocks, tx, 1 if phase == 0 else coeff, True, phase)
            loss = torch.nn.functional.cross_entropy(s[1], sb.y)
            if phase == 0:
                loss = loss + torch.nn.functional.binary_cross_entropy(t[1].softmax(1)[:, -1], torch.full_like(t[1][:, -1], .5))
            else:
                mask = tb.pseudo_label != -1
                loss = loss + .1 * torch.nn.functional.cross_entropy(t[1][mask], tb.pseudo_label[mask])
                loss = loss + torch.nn.functional.binary_cross_entropy(torch.cat((s[2], t[2])), torch.cat((sb.domain, tb.domain)))
            torch.manual_seed(37)
            adapted_loss = adapter.forward_model(sb, tb, phase)
            self.assertTrue(torch.allclose(loss, adapted_loss, atol=2e-6, rtol=1e-5))
            loss.backward()
            adapted_loss.backward()
            grads_o = {n: p.grad for n, p in oracle.named_parameters() if p.grad is not None}
            grads_c = {n: p.grad for n, p in candidate.named_parameters() if p.grad is not None}
            self._states_close(grads_o, grads_c, 'gradients-phase-' + str(phase))
            self._states_close(dict(source=sx.grad, target=tx.grad), dict(source=sb.x.grad, target=tb.x.grad),
                               'input-gradients-phase-' + str(phase))
            for model in (oracle, candidate):
                params = ([{'params': model.generator.parameters()}, {'params': model.classifier.parameters()}]
                          if phase == 0 else model.parameters())
                torch.optim.Adam(params, lr=.001, weight_decay=.0001).step()
            self._states_close(oracle.state_dict(), candidate.state_dict(), 'adam-phase-' + str(phase))
            oracle.eval()
            candidate.eval()
            with torch.no_grad():
                a = oracle(g, graph.x.to(self.device), 1, True, phase)
                b = candidate(g, graph.x.to(self.device), 1, True, phase)
            self._states_close({str(i): v for i,v in enumerate(a)}, {str(i): v for i,v in enumerate(b)},
                               'full-graph-outputs-phase-' + str(phase))

    def test_reversal_coefficients_are_per_call(self):
        from pygda.nn.uaga_base import _Reverse
        x = torch.ones(2, device=self.device, requires_grad=True)
        (_Reverse.apply(x, 1).sum() + 2 * _Reverse.apply(x, -1).sum()).backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_target_label_independence_and_new_graph(self):
        source = _cycle(12)
        source.y = torch.arange(12) % 3
        target = _cycle(9)
        class HiddenLabels:
            x, edge_index = target.x, target.edge_index
            @property
            def y(self):
                raise AssertionError('fit accessed target ground-truth labels')
        opts = dict(in_dim=6, hid_dim=4, num_classes=2, num_heads=2, num_out_heads=2,
                    epoch=1, separation_epochs=1, top_k=3, seed=11, device=self.device, verbose=0)
        first = self.UAGA(**opts)
        self.assertIs(first.fit(source, HiddenLabels()), first)
        target.y = torch.arange(9) % 5
        second = self.UAGA(**opts).fit(source, target)
        self._states_close(first.model.state_dict(), second.model.state_dict(), 'target-label-independence')
        new_graph = _cycle(7)
        logits, labels = first.predict(new_graph)
        self.assertEqual(tuple(logits.shape), (7, 3))
        self.assertIsNone(labels)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(source.x.shape[0], 12)
        self.assertEqual(source.y.tolist(), (torch.arange(12) % 3).tolist())

    def test_real_data_two_phase_reference_parity(self):
        root = Path(os.environ.get('UAGA_DATA_ROOT', '/missing-uaga-data'))
        if not (root / 'citation-v1.mat').is_file() or not (self.author / 'train.py').is_file():
            self.skipTest('Set UAGA_DATA_ROOT and UAGA_AUTHOR_DIR for real-data parity.')
        from scipy.io import loadmat
        official = _load('_uaga_real_oracle_models', self.author / 'models.py')
        utils = _load('_uaga_real_oracle_utils', self.author / 'utils.py')
        raw = (self.author / 'train.py').read_text()
        tree = ast.parse(raw)
        snapshots = {}
        scope = dict(__name__='__uaga_reference_parity__')
        def capture(phase, epoch):
            snapshots[phase] = {k: v.detach().clone() for k, v in scope['network'].state_dict().items()}
        scope['_capture'] = capture
        output = Path(os.environ['UAGA_TEST_OUTPUT']) / 'official-parity.txt'
        class Limit(ast.NodeTransformer):
            def visit_Assign(self, node):
                if len(node.targets) == 1 and getattr(node.targets[0], 'id', '') in ('epochs0', 'epochs1'):
                    node.value = ast.Constant(value=1)
                return self.generic_visit(node)
            def visit_For(self, node):
                if (isinstance(node.target, ast.Tuple) and
                        [getattr(n, 'id', None) for n in node.target.elts] == ['source', 'target']):
                    node.iter = ast.parse("[('citation-v1', 'dblp-v4')]", mode='eval').body
                if getattr(node.target, 'id', '') == 'seed':
                    node.iter = ast.parse('range(1)', mode='eval').body
                node = self.generic_visit(node)
                if getattr(node.target, 'id', '') == 'epoch':
                    phase = 0 if 'epochs0' in ast.unparse(node.iter) else 1
                    node.body.append(ast.parse('_capture('+str(phase)+', epoch)').body[0])
                return node
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant(value=str(output)), node) if node.value == 'UAGA.txt' else node
        tree = ast.fix_missing_locations(Limit().visit(tree))
        funcs = lambda t: {n.name: ast.dump(n) for n in t.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(funcs(tree), funcs(ast.parse(raw)))
        old_modules = {k: sys.modules.get(k) for k in ('models', 'utils')}
        old_dir = Path.cwd()
        sys.modules['models'], sys.modules['utils'] = official, utils
        try:
            os.chdir(root.parent)
            exec(compile(tree, str(self.author / 'train.py'), 'exec'), scope)
        finally:
            os.chdir(old_dir)
            for key, module in old_modules.items():
                if module is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = module
        def graph(name):
            data = loadmat(root / (name + '.mat'))
            adj = data['adjacency_matrix'].tocoo()
            return SimpleNamespace(x=torch.tensor(data['features'].toarray(), dtype=torch.float32),
                edge_index=torch.stack((torch.from_numpy(adj.row.astype('int64')), torch.from_numpy(adj.col.astype('int64')))),
                y=torch.tensor(data['labels'][0], dtype=torch.long))
        source, target = graph('citation-v1'), graph('dblp-v4')
        adapter = self.UAGA(7786, 32, 5, epoch=1, separation_epochs=1, seed=0, device=self.device, verbose=0)
        prepared = adapter.process_graph(source, include_labels=True)
        self.assertTrue(torch.equal(prepared.x, scope['X_s']))
        self.assertTrue(torch.equal(prepared.y, scope['y_s']))
        for a, b in zip(adapter._graph(prepared).edges(), scope['G_s'].edges()):
            self.assertTrue(torch.equal(a, b), 'Induced source edge order differs from official SciPy conversion.')
        class NoLabels:
            x, edge_index = target.x, target.edge_index
            @property
            def y(self):
                raise AssertionError('Adapter read real target labels')
        def check(model, phase, epoch):
            self._states_close(snapshots[phase], model.model.state_dict(), 'real-data-phase-' + str(phase))
        adapter.fit(source, NoLabels(), epoch_callback=check)
        logits, labels = adapter.predict(target)
        scope['network'].eval()
        with torch.no_grad():
            expected = scope['network'](scope['G_t'], scope['X_t'], scope['grl_lambda_t'], True, 1)[1]
        self._states_close(dict(logits=expected), dict(logits=logits), 'real-data-prediction', atol=1e-5)
        self.assertEqual(tuple(logits.shape), (8653, 6))
        self.assertTrue(torch.equal(labels, target.y.to(self.device).clamp(max=5)))
        RESULTS['real-data-scope'] = dict(task='citation-v1 -> dblp-v4', seed=0, separation_epochs=1,
                                          adaptation_epochs=1, official_function_asts_unchanged=True)


if __name__ == '__main__':
    unittest.main()
