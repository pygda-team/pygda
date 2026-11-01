# ImGDA released inputs

`load_imgda_pair(root, source='ACMv9', target='Citationv1', imbalance_factor=20, device='cuda:0')` reads the author's processed data and global caches without downloads, split regeneration or cache writes. Only load trusted files: `.pt` and `.pkl` use pickle serialization.

Expected layout:

```text
root/
  data/ACMv9/processed/data.pt
  data/Citationv1/processed/data.pt
  tmp/ACMv920.pkl
  tmp/Citationv1.pkl
```

The processed graph retains the released feature order and source `train_mask`. Source selection uses the author degree-based long-tail procedure, then removes and reindexes nodes in sorted original-node order. `imgda_class_counts` holds counts from the full source labels, not the selected training subset. `imgda_original_node_ids` records the selected IDs. The returned target has no labels.

Select on the same device used for the associated cache: CPU/CUDA degree ties can produce different selected nodes. Bounds and population counts do not independently prove the node-semantic provenance of precomputed caches. The loader retains this known limitation and does not silently substitute caches or data from another reproduction.

For custom data use `prepare_imgda_source` with a source cache that already matches its selected node population. Prepared graphs require finite float32 `x`, valid long `edge_index` and `global_adj`, and finite float32 `global_norm`. Cache weights may be negative because some released values are negative; they must already have the intended normalization.

::: pygda.datasets.load_imgda_pair

::: pygda.datasets.prepare_imgda_source
