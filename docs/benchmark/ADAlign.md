# ADAlign Citation benchmark

The entry point `benchmark/node/adalign.py` runs full-batch node adaptation
on existing ACMv9, Citationv1 and DBLPv7 caches. Other dataset families are
not covered by this benchmark. It reads trusted local `(Data, slices)` PyG
caches; it does not download datasets or rebuild caches. Citation graph
sizes, 6,775 feature columns and label IDs 0 through 4 are checked.

Run from the repository root, supplying existing data and a results directory:

```bash
python -B benchmark/node/adalign.py \
  --data-root /path/to/citation-data \
  --output-dir /path/to/results \
  --all-transfers --seeds 42 43 44 45 46 --epochs 150 --device cuda:0
```

For one transfer, omit `--all-transfers` and set `--source ACMv9 --target
Citationv1`. `--help` lists the exposed parameters. An optional
`--audit-report /path/to/data-audit.json` verifies previously recorded raw/cache
comparisons and all recorded file fingerprints. Without it, input hashes are
recorded, but raw/cache equivalence is not certified. Feature normalization,
topology conversion and mask-based splitting are not performed by the entry.

## Evaluation and persistence

Each seed starts an independent model and reseeds Python, NumPy and PyTorch.
All source labels are used. Actual target labels are held outside `fit`;
the supplied target graph has synthetic invalid placeholder labels, and each
training forward checks that the model removed them. No split masks or
TDSS-specific cached attributes enter the model. F1 is computed externally
once, at the end of the configured epochs, on all target nodes.

Every run saves training losses, environment versions, GPU allocator
statistics and final F1 to JSON. Add `--audit-training` to check phase order,
finite losses/gradients/parameters/buffers and parameter updates during training.
This optional diagnostic mode saves scalar phase records and flushes a JSONL
journal after each optimizer pair; it adds synchronization and parameter copies.
Without this option the detailed phase lists and journal remain empty, and no
optimizer method is patched. Both modes check final prediction finiteness.
`--verbose 0` suppresses training progress; final status and result paths are
still printed.

`batch.json` and `summary.csv` aggregate successful runs. Incomplete groups
are explicitly marked. The standard deviation is the population standard
deviation across seeds (`ddof=0`). JSON scores are in [0,1]; CSV scores are
percentages. Seeded runs do not enable deterministic PyTorch algorithms and
are not promised to be bit-for-bit reproducible across devices or versions.

## Observed Citation results

The following results were collected with the stage7 integration, before
the maintenance changes to tests, logging and optional benchmark diagnostics.
They are retained as historical evidence, not a validation of the revised package.
The loss, sampler and training update sequence are unchanged in this revision.

All 30 runs completed on an RTX 4090 with Python 3.12.3, PyTorch 2.5.1+cu124
and PyG 2.8.0.post1. Each used 150 epochs and seeds 42 through 46. Defaults
were hidden size 128, two layers, dropout 0.25, lr/weight decay 0.001,
source/target propagation 0/10, lambda 0.4, M=2048, kappa 0.5.
These are audited author CLI defaults, not demonstrated per-task paper-best
settings. Results below use final-epoch evaluation, whereas the author
reports separate maxima over target Micro/Macro-F1 during training.

| Source → target | Micro-F1 (%) | Macro-F1 (%) |
| --- | ---: | ---: |
| ACMv9 → Citationv1 | 77.86 ± 0.49 | 73.73 ± 0.88 |
| ACMv9 → DBLPv7 | 70.26 ± 2.36 | 65.84 ± 2.65 |
| Citationv1 → ACMv9 | 72.91 ± 0.39 | 73.35 ± 0.62 |
| Citationv1 → DBLPv7 | 75.81 ± 1.41 | 71.92 ± 1.69 |
| DBLPv7 → ACMv9 | 65.51 ± 0.94 | 58.63 ± 1.61 |
| DBLPv7 → Citationv1 | 76.65 ± 0.39 | 69.89 ± 1.47 |

Total batch time was 260.11 seconds; peak allocated GPU memory ranged from
1.18 to 1.58 GiB. Recorded file hashes and workspace states were unchanged.
The exact per-seed scores and provenance are in
[adalign_citation_results.json](adalign_citation_results.json).
This is a Citation validation result, not a claim of full paper reproduction
or performance relative to other methods under a matched evaluation protocol.

## Known boundaries and provenance

The model and neural CF implementation are based on author commit
`d6d7a700049782f9402b2bfa891c60f1cba552c7` and PyGDA base commit
`8cef1de93ce67f0e6feb9bbbc61ac15851cd67a6`. Algorithm-order and frequency
sampler differences between the paper and author code are documented in
[ADAlign](../models/ADAlign.md). Singular gradient behavior at zero complex
modulus and at kappa=1 with zero discrepancy is retained from the author code;
successful Citation runs do not establish stability at those boundaries.

The contributor reports author permission for integration and public
distribution. See [ADAlign](../models/ADAlign.md) for attribution and permission
notes, the model API, and instructions for optional author-comparison tests.
