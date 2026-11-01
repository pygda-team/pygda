# SAGA

SAGA is a multi-view graph domain adaptation model. This integration targets
the public implementation at author commit
`55ff00a4e426a2d0fbeb913f11330ab69b68a506`.

```python
from pygda.datasets import SAGADataset
from pygda.models import SAGA

source = SAGADataset('/path/to/saga-data', 'ACM1')[0]
target = SAGADataset('/path/to/saga-data', 'ACM2', include_labels=False)[0]
model = SAGA(in_dim=512, hid_dim=256, num_classes=3,
             embed_dim=128, batch_size=1000, epoch=50, seed=0)
model.fit(source, target)
logits, labels = model.predict(target)
```

The example configuration follows the public ACM1-ACM2 CLI/config defaults.
It does not assert the configuration matches the paper's parameter table.

## Input and evaluation

Each domain is one PyG Data object with x, ordered view_names,
edge_index_dict, edge_weight_dict and optional integer y. Source y is required.
Domains must have the same feature dimension, class semantics and ordered
views; node counts may differ. Edge weights and view order are preserved.
All source labels are used. No train/validation/test masks are generated.

fit excludes target y before device transfer and preprocessing. It returns
self and stores scalar epoch losses and node-batch sizes, without target
metrics, checkpoints or output files. predict processes the supplied graph,
not a stored training target, and returns logits plus optional input labels.
Features are converted to float32, matching the author's data loader.

The author's training entry evaluates target labels after every epoch and
prints the maximum target accuracy. This library fit uses fixed epochs;
an external benchmark must distinguish final-epoch evaluation from the
author's maximum-over-epoch reporting.

## Preserved public-code behavior

- Fixed k_hop propagation and row L2 normalization precede node batching.
- The shared encoder/decoder has two Linear layers with ELU and Dropout.
- Views are concatenated for the classifier, in their declared order.
- Per-batch dominant views are selected against each domain's raw feature Gram.
- SAD uses uncentered feature second moments. IA aligns to view zero,
  retaining the public implementation's unused dominant-index arguments.
- CA retains the author's row-wise log/mean order and epsilon.
- One Adam optimizer updates all trainable parameters. Each epoch uses
  separate RandomSampler permutations. When one domain is exhausted,
  remaining batches use source-only or target-only losses.

No dynamic search over propagation depths has been added. These choices
reproduce the public code even where its formulas differ from the paper.

## Explicit interface changes and boundaries

The core class is renamed SAGABase; an unused classifier class and unused
wildcard import are omitted. Core methods and Entropy are otherwise retained.
The wrapper uses explicit device selection, validates inputs, avoids reading
target labels in fit, and predicts on new graphs. The author's unused initial
dominant-view precomputation is omitted; that field is initialized to zero and
does not control the core forward path.

Paired batches containing only one node in either domain are rejected before
initializing the network because the projection head uses BatchNorm. No
samples are silently dropped or repeated. Non-finite losses stop fit; the
underlying loss is not clamped or replaced. Very small tau, large magnitudes,
or exponentiated SAD weights can still encounter the author's numerical
boundaries. Passing regular-point tests is not a stability guarantee.

## Tests

Run from the repository root:

```bash
python -B -m unittest discover -s tests -p 'test_saga*.py' -v
```

Author comparisons are optional when SAGA_OFFICIAL_DIR is absent; an invalid
configured checkout fails. Set it to a clean checkout at the pinned commit:

```bash
SAGA_OFFICIAL_DIR=/path/to/SAGA python -B -m unittest discover -s tests -p 'test_saga*.py' -v
```

Tests compare individual losses, feature/parameter gradients, two epochs of
Adam updates with unequal-domain batches, and predictions with the author
core. They also check target-label independence, new-graph prediction and
input preservation, using CPU and CUDA when available. Real-data performance
experiments cover five seeds on the ACM pair; see [the benchmark record](../benchmark/SAGA.md)
for results, evaluation protocol and scope. MAG real-data validation is pending.

## Attribution

[Author repository](https://github.com/f1shungry/SAGA) and
[MIT license](https://github.com/f1shungry/SAGA/blob/55ff00a4e426a2d0fbeb913f11330ab69b68a506/LICENSE).
Copyright (c) 2026 Jingyu Zhao. The full notice is retained in saga_base.py
and the dataset adapter's SAGA_LICENSE.txt.

::: pygda.models.saga
    options:
      docstring_style: numpy
      show_source: false
      merge_init_into_class: true
      ignore_init_summary: true
