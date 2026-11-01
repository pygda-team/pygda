# ADAlign

ADAlign implements *Learning Adaptive Distribution Alignment with Neural
Characteristic Function for Graph Domain Adaptation* (ICLR 2026).

- [Paper](https://arxiv.org/abs/2602.10489)
- [Author implementation](https://github.com/gxingyu/ADAlign), commit
  `d6d7a700049782f9402b2bfa891c60f1cba552c7`.

## Usage

Supply source and target PyG `Data` graphs with the same feature dimension.
Features must be finite floating-point tensors; `edge_index` must contain
valid integer node indices. Source labels must be integers in
`[0, num_classes)`. All source labels are used; `train_mask` is not applied.
Target labels are optional and excluded from the internal training graph.

```python
from pygda.models import ADAlign

model = ADAlign(in_dim=source_data.x.size(1), hid_dim=128,
                num_classes=5, device='cuda:0')
model.fit(source_data, target_data)
logits, labels = model.predict(target_data)
```

`fit` returns the model. `predict(data, source=False)` returns logits and
the supplied graph's labels, or `None` when labels are absent. Use
`source=True` for source propagation. Input graphs are cloned, and no
feature normalization or topology conversion is performed.

Only full-batch, single-source node adaptation is supported:
`batch_size=0`, `num_neigh=-1`, `mode='node'`, `adv=False`.

## Algorithm and defaults

The backbone is PyGDA's `A2GNNBase`. Each epoch first maximizes the alignment
loss through the frequency sampler, then minimizes source classification
loss plus `weight * alignment_loss` through the backbone. Both optimizers
are Adam with the same learning rate and weight decay.

`alpha` is the amplitude coefficient kappa, and `weight` is lambda.
`t_batchsize` is the nominal frequency count M: M/16 vectors are generated
and repeated 16 times. It must be a multiple of 16 and at least 32.
Defaults follow the author CLI configuration, rather than task-specific
selected parameters: 150 epochs, hidden size 128, two layers, dropout 0.25,
lr/weight decay 0.001, source/target propagation 0/10, lambda 0.4,
M=2048 and kappa=0.5.

The implementation follows the author code where it differs from the paper:

- Sampler ascent precedes backbone descent; paper Algorithm 1 lists the
  reverse order.
- The sampler repeats learned frequencies rather than implementing the
  paper's explicit Gaussian scale mixture.
- Classification and alignment use separate forwards and new frequencies
  in each phase. The sampler remains in training mode during backbone
  descent, including BatchNorm buffer updates, but its optimizer is not stepped.
- The phase component is clamped to 1e-12 before the weighted square root
  and mean. Identical distributions may therefore have a nonzero loss.

Two engineering changes support the public API: the sampler dimension
follows `hid_dim`, and prediction operates on the supplied graph. Training
does not evaluate target labels or select an epoch using target performance.
See the [Citation benchmark](../benchmark/ADAlign.md) for evaluation details.

## Tests and numerical boundaries

Run the self-contained tests from the repository root:

```bash
python -B -m unittest discover -s tests -p 'test_adalign*.py' -v
```

Two optional author-comparison tests require a clean checkout at the commit
listed above. Without the environment variable they are explicitly skipped;
an invalid configured checkout fails the tests.

```bash
ADALIGN_OFFICIAL_DIR=/path/to/ADAlign python -B -m unittest discover -s tests -p 'test_adalign*.py' -v
```

Tests cover the public interface, invalid numeric parameters, finite regular-point
gradients, source/target symmetry, target-label independence, prediction on a
new graph, and training equivalence with diagnostic monitoring enabled.
Author comparisons additionally check frequencies, loss, feature gradients,
sampler parameter gradients and model forwards. CPU and CUDA are checked
when available.

The author loss has singular gradients at zero complex modulus and at
`alpha=1` with zero discrepancy. Those boundaries are preserved. Successful
Citation experiments do not establish stability for all inputs.
Use the benchmark's `--audit-training` option for per-phase finite-value checks.

## Attribution and permission

The CF and sampler implementation is derived from the author repository at
the commit above. The contributor reports that the author has agreed to
the integration and public distribution. Retain the authorization record
and any attribution or licensing conditions agreed with the author.
This statement does not assert that the author repository has added a license
or specify terms beyond that authorization.

## API reference

::: pygda.models.adalign
    options:
      docstring_style: numpy
      show_source: false
      merge_init_into_class: true
      ignore_init_summary: true
      heading_level: 6
