# PALA author material

Use verified, trusted files from the pinned author repository. The helpers
load pickle-backed `processed/data.pt` and `.pkl` caches; no automatic download,
new split generation or random-walk cache fallback is performed.

```python
from pygda.datasets import (load_pala_graph, load_pala_cache,
                            prepare_pala_source, prepare_pala_target)

source = prepare_pala_source(
    load_pala_graph(data_root, 'ACMv9'),
    load_pala_cache(cache_dir / 'ACMv920.pkl'),
    imbalance_factor=20, device='cuda:0')
target = prepare_pala_target(
    load_pala_graph(data_root, 'Citationv1', include_labels=False),
    load_pala_cache(cache_dir / 'Citationv1.pkl'), device='cuda:0')
```

Prepare the source before setting a training seed, matching the author entrypoint.
The original processed masks are kept; only selected training nodes remain in
the induced/reindexed source graph. `pala_original_node_ids` records their order.

Source caches use `<domain><IF>.pkl`, target caches `<domain>.pkl`. Negative
finite normalization weights in the released caches are preserved. Endpoint
and shape validation alone cannot prove semantic node alignment; retain the
verified node order and material provenance.

The verified release includes ACMv9 and Citationv1, with IF 10/20/50 caches.
DBLPv7 material was not present in that release and has not been validated here.
