# UAGA

Unknown-Excluded Adversarial Graph Domain Alignment for open-set node adaptation
(AAAI 2025), adapted from [the official implementation](https://github.com/3480430977/UAGA)
at commit `4ef2f8fa6c2a18760ef9d9f8ef90d4ee4f92c227`.

```python
from pygda.models import UAGA
from pygda.datasets.uaga import load_uaga_graph

source = load_uaga_graph('/root/uaga-reproduce/data', 'citation-v1')
target = load_uaga_graph('/root/uaga-reproduce/data', 'dblp-v4')
model = UAGA(in_dim=7786, hid_dim=32, num_classes=5, seed=0)
model.fit(source, target)
logits, labels = model.predict(target)
```

`num_classes` counts the **known** classes K. Logits contain K+1 columns, and
column K is the unknown class. With the author data, source classes 5..8 are
removed together with their incident edges; all target nodes remain. `fit` does
not access target labels or masks. Labels returned by `predict` are optional,
with target labels K and above mapped to K for external evaluation.

The implementation retains DGL GATConv, full-neighbor sampling, classifier-first
initialization, phase-specific gradient reversal, repeated smaller-domain node
permutations, a fresh Adam optimizer for adaptation, float64 KMeans inputs,
top-2000 unknown initialization, and domain losses over classifier-block source
nodes. `edge_weight` is ignored, as in the official conversion from SciPy.
Self-loops are replaced by one per node. Prediction uses the graph supplied by
the caller; it does not reuse a cached training graph.

DGL is an optional dependency, loaded on model construction/training rather than
on importing PyGDA. The verified author baseline uses Python 3.9.13, PyTorch
1.7.1+cu110, DGL 0.7.2, NumPy 1.21.5, SciPy 1.9.1 and scikit-learn 1.0.2 on an
RTX 4090. This is a reference environment, not a claim that the complete current
PyGDA dependency set runs under PyTorch 1.7.1. The public PyGDA import and graph
preparation are checked separately in the existing PyGDA environment. GPU
operator probes and real-data reference training pass; CPU reference workers
currently segfault after completing their synthetic checks. The unified GPU
integration profile described below has passed public-entry-point acceptance;
other DGL/Torch combinations and CPU execution remain unverified.

| Task | dropout / attention dropout | weight_decay |
|---|---:|---:|
| citation-v1 → dblp-v4 | 0.4 | 0.0001 |
| citation-v1 → acm-v8 | 0.5 | 0.0001 |
| dblp-v4 → citation-v1 | 0.5 | 0.0001 |
| dblp-v4 → acm-v8 | 0.5 | 0.00001 |
| acm-v8 → citation-v1 | 0.6 | 0.0001 |
| acm-v8 → dblp-v4 | 0.5 | 0.00001 |

Other author defaults are 30 separation epochs, 200 adaptation epochs, lr=0.001,
batch_size=2048, one encoder layer, eight encoder heads, two classifier heads,
threshold=0.5 and target_weight=0.1. `epoch` counts adaptation epochs;
`separation_epochs` counts separation epochs. `seed=None` preserves caller RNG
state; set a seed for a controlled run. `epoch_callback(model, phase, epoch)` may
inspect state without providing target supervision.

Report final adaptation-epoch metrics over seeds 0..4, using population standard
deviation (`ddof=0`), not the best target epoch. OS/OS* are macro class recalls.
HS uses micro accuracy among known target nodes and unknown recall; it is not
the harmonic mean of OS* and unknown recall. UNK_AUC uses the unknown softmax
column. Official citation-v1 → dblp-v4 five-seed reference results are
OS 69.2146±0.2145, OS* 73.9999±0.2997, UNK_AUC 73.9153±1.0747 and
HS 56.4283±0.4844 percent. These are official-code results; adapter parity is a
separate validation step. Fourteen recorded adapter/reference comparisons have
zero maximum absolute difference in the tested short runs. Five other tasks have
not yet been validated.

The upstream repository has no discovered license file. The contributor reports
author permission to port and publish; the authorization record is still to be
attached before public distribution. This adaptation does not assign an upstream
license that the author has not granted.

## Benchmark and integration environment

See [the benchmark instructions](../benchmark/UAGA.md) for all six fixed task
configurations and the scope of the available reproduction evidence. The
integration requirements are in `benchmark/node/uaga-requirements.txt` and target
Python 3.9/Linux x86_64. They include DGL as an optional model dependency rather
than adding it to every PyGDA installation. CUDA 11.0 shared libraries are also
required by that DGL build; installing the Python wheel alone is insufficient.

The integration profile is separate from the author reference environment.
On RTX 4090, Python 3.9.13, Torch 1.13.1+cu117, PyG 2.3.1 and DGL 0.7.2,
public import, real PyG Data training, optional dependency behavior, six tests,
the benchmark entry point and documentation build passed in one environment.
Real-data acceptance uses one epoch per phase and is not a full-schedule accuracy
claim. Other Torch/DGL combinations are not claimed compatible without testing.

::: pygda.models.UAGA
