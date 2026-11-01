# PALA results and validation scope

Source: [maxin88scu/PALA](https://github.com/maxin88scu/PALA), commit
`612b35d059c23b96b4878a6c5e4420bfc6845d11`. The reference task is ACMv9 →
Citationv1, IF20, 200 epochs, seeds 0–4. The unmodified author run overflowed in
seed 1 at epoch 0; both completed versions below use the documented log-domain
contrastive reduction.

| Metric (%) | Stabilized author reference | Public PyGDA benchmark | Paper table 1 |
| --- | --- | --- | --- |
| Micro-F1 | 72.21 ± 7.06 | 72.94 ± 6.22 | 76.18 ± 1.28 |
| Macro-F1 | 59.50 ± 7.12 | 61.59 ± 10.04 | 66.91 ± 2.90 |

For each seed, select the epoch with maximum target Micro-F1 and take Macro-F1
from that same epoch. Ties retain the first epoch. Standard deviation uses
ddof=0. This is target-label oracle selection, not unsupervised model selection.
All five seeds, including the weaker ones, are retained. Per-seed best and
final-epoch metrics, environment versions and report hashes are recorded in
[the result JSON](pala_acm_citation_if20.json).

The stabilized author reference used torch 2.4.0, torch-geometric 2.6.1,
torch-scatter 2.1.2, NumPy 1.26.4, scikit-learn 1.5.2 and tqdm 4.66.4.
The public PyGDA benchmark ran through `benchmark/node/pala.py` with full
package imports on torch 2.5.1+cu124, torch-geometric 2.8.0.post1, NumPy 2.1.3
and scikit-learn 1.9.1. Both five-seed runs used default nondeterministic CUDA.
The measured mean scores remain below the paper and have greater variability.
These version/mode comparisons do not identify the cause of the paper gap.

## Interface and trajectory checks

- Actual public PyGDA imports, CPU/CUDA fit and predict, input immutability,
  target-label independence and four package unit tests passed.
- Author-dependency checks used isolated PALA submodules with the real BaseGDA
  and logger, without installing unrelated legacy-model dependencies.
- Under deterministic CUDA in author dependencies, the public wrapper logic
  and stabilized official reference matched all 200 epoch losses and target
  logits for seed 0. Final parameters, memory, queue, Adam state and RNG matched
  exactly. This is a one-seed controlled trajectory check.
- The public benchmark completed 200 epochs for each of seeds 0–4 with finite
  losses and gradients. It is a separate default-CUDA performance run.

CPU/GPU synthetic core checks also cover the ELR activation boundary. Default
CUDA produced small differences even in author self-repeats; bitwise default-
CUDA long-trajectory reproducibility is not claimed.

Only A→C IF20 has a completed public five-seed benchmark. DBLPv7 was absent
from the verified material; the full paper table is not validated. No claim
is made that the paper scores have been reproduced.

## Running the benchmark

Use a checkout importable by the selected Python environment:

```bash
PYTHONPATH="$PWD" python benchmark/node/pala.py --data-root /path/to/materials/data \
  --cache-dir /path/to/materials/cache --source ACMv9 --target Citationv1 \
  --imbalance-factor 20 --epochs 200 --runs 5 --output pala-results.json
```

`--deterministic` enables a separate controlled-validation mode. Report this
flag and the actual environment when comparing results. The original author
hardware, Python version and complete runtime were not specified sufficiently
to claim an identical original environment.
