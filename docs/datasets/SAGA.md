# SAGA datasets

`SAGADataset` reads the author's multi-view graph tensors without downloading,
creating caches or changing the input files. Obtain the data from the
[author repository](https://github.com/f1shungry/SAGA) and use only trusted
tensor files: the original archive uses Python pickle serialization.

```python
from pygda.datasets import SAGADataset

source = SAGADataset('/path/to/data', 'ACM1')[0]
target = SAGADataset('/path/to/data', 'ACM2', include_labels=False)[0]
```

Each domain directory contains `feat.pt`, `label.pt`, and two sparse COO
adjacencies: `pap.pt` and `psp.pt` for ACM; `pap.pt` and `pp.pt` for MAG.
The adapter exposes ordered views, PyG edge indices and original edge weights.
It converts one-hot labels to integer class indices. With `include_labels=False`,
it never opens `label.pt`. Accessing a dataset item returns a cloned graph.

ACM1 and ACM2 have real-data validation. The adapter also defines the author's
CN, US, JP, FR, DE and RU MAG domains; these domains have not been validated
with real data in this integration. Separate source and target graphs must
share feature dimensions, class semantics and ordered views.

::: pygda.datasets.saga.SAGADataset
    options:
      show_source: false
