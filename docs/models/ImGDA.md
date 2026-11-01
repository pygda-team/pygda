# ImGDA

`ImGDA` implements the released code for *Dual Prototype-Enhanced Contrastive Framework for Class-Imbalanced Graph Domain Adaptation* (NeurIPS 2025).

```python
from pygda.datasets import load_imgda_pair
from pygda.models import ImGDA

source, target = load_imgda_pair('/path/to/released/files', device='cuda:0')
model = ImGDA(source.x.size(1), 512, source.imgda_class_counts.numel(),
              epoch=200, seed=0, device='cuda:0')
model.fit(source, target)
logits, labels = model.predict(target)  # labels is None for the unlabeled loader output
```

The released implementation uses a fixed two-layer encoder, per-source-node compensation and shared local/global convolution weights. Classification uses global source embeddings, while cross-domain alignment uses summed target embeddings. Full-source class counts drive the released adaptive temperature. These implementation details are retained even where the paper description differs.

Training requires prepared source labels, `imgda_class_counts` from the original full source graph, and explicit `global_adj` / `global_norm` caches for both domains. Each source class must contain at least two selected nodes. Only node-level, full-graph training with all neighbors is supported. Missing global caches are rejected; weights, including negative values in the released caches, are not regenerated, clamped or normalized again.

The author stores Dropout layers in an ordinary Python list. This implementation preserves that behavior: Dropout remains active during evaluation and `predict` is stochastic. Each training epoch performs one unlabeled global-target forward to match the original evaluation's RNG consumption. `fit` retains final weights, never reads target labels and performs no target-score checkpoint selection. A callback can evaluate each epoch externally:

```python
def evaluate(epoch, losses, target_logits):
    # External evaluation may use separately held target labels.
    pass

model.fit(source, target, callback=evaluate)
```

The benchmark implements the author protocol: maximum all-target Micro-F1 over 200 epochs and Macro-F1 from the same epoch, averaged across seeds 0–4 with population standard deviation (`ddof=0`). This uses target labels to select the reported epoch. It differs from reporting the final prediction or selecting a checkpoint without target labels.

An unmodified-author A→C, IF20 five-seed baseline obtained Micro-F1 72.77±3.95% and Macro-F1 67.54±4.67%, below the paper's 77.30±1.05% and 73.92±0.83%. This is a documented released-code baseline, not a claim that the paper scores or all six transfer tasks were reproduced. The released archives inspected contain ACMv9 and Citationv1, but no DBLPv7.

Source: [maxin88scu/ImGDA](https://github.com/maxin88scu/ImGDA/tree/d4237cdb142a4618057090b6d23af0ff85aa6ae9), MIT license. Author requirements matched in the reference experiment: Torch 2.4.0+cu124, PyG 2.6.1, torch_scatter 2.1.2+pt24cu124, NumPy 1.26.4, scikit-learn 1.5.2 and tqdm 4.66.4. Python 3.12/CUDA 12.4 and SciPy 1.14.1 were our explicit environment choices; the author did not specify a complete original environment.

::: pygda.models.ImGDA

## Full-training interface validation

Both implementations completed A→C, IF20, 200 epochs for seeds 0–4 in the
same Python 3.12 / RTX 4090 environment. Population standard deviations are
reported, following the author benchmark.

| Implementation (same Torch 2.5.1+cu124 environment) | Micro-F1 | Macro-F1 |
| --- | ---: | ---: |
| Unmodified author code | 72.60 ± 4.42% | 67.36 ± 5.32% |
| Public PyGDA interface | 72.96 ± 4.22% | 68.37 ± 5.51% |

The public-minus-author mean differences are +0.36 percentage points for
Micro-F1 and +1.01 for Macro-F1 in this run. Together with real-data one-step
loss/gradient/Adam checks and CPU/CUDA API tests, this supports the released-code
integration on the tested task. It does not establish bitwise or statistical
equivalence, reproduce the paper scores, or validate other transfer directions
and imbalance factors. GPU runs can diverge despite equal seeds; no deterministic
algorithm settings were added and no seeds or hyperparameters were selected
after observing results. A full documentation-site build has not been verified.
