# SAGA ACM benchmark

This integration follows the public implementation at author commit
`55ff00a4e426a2d0fbeb913f11330ab69b68a506` and PyGDA base commit
`8cef1de93ce67f0e6feb9bbbc61ac15851cd67a6`.

Run from the repository root with the trusted author ACM tensor directories:

```bash
python -B benchmark/node/saga.py --data-root /path/to/data --device cuda:0 --output /path/to/new-result.json
```

The default runs both directions with seeds 0, 1, 2, 3, 4. It uses 50 epochs
and dropout 0 for ACM1→ACM2, and 100 epochs and dropout 0.05 for ACM2→ACM1.
The batch size is 1000. Public configuration values are k_hop=2, alpha=0.95,
beta=1 and delta=0.01/0 respectively. These follow the public CLI/configuration;
they are not a claim that the paper's parameter table was used.

For a short interface check:

```bash
python -B benchmark/node/saga.py --data-root /path/to/data --direction ACM1-ACM2 --seeds 0 --epochs 1 --output /path/to/new-smoke.json
```

Only an explicitly requested new JSON output is written. Existing output files
are rejected. Input file hashes are recorded and checked after the run.
There are no downloads, checkpoints or cache writes. Target labels are loaded
for evaluation after fixed-budget training and prediction. No target labels
select seeds, hyperparameters or epochs.

## Observed results

Final-epoch metrics, mean ± sample standard deviation over seeds 0–4,
expressed as percentages (standard deviations in percentage points):

| Direction | Accuracy / Micro-F1 | Macro-F1 |
| --- | ---: | ---: |
| ACM1→ACM2 | 51.12 ± 0.94 | 28.59 ± 3.74 |
| ACM2→ACM1 | 37.47 ± 2.95 | 36.65 ± 2.77 |

Environment: Python 3.12.3, PyTorch 2.5.1+cu124, PyG 2.8.0.post1,
RTX 4090. The ten monitored runs completed in about 100 seconds. This is an
observed run time, not a benchmark speed guarantee. The compact records and
data hashes are in [saga_acm_results.json](saga_acm_results.json).

The results are from the fixed-configuration validation runner using the
same library implementation, rather than a completed run of this new CLI.
The new CLI has a separate smoke check in the integration validation script.

These scores do not reproduce the paper's reported performance. ACM1→ACM2
has very low recall for class 1 across the observed seeds. Input verification,
code parity and finite gradients do not establish paper-level accuracy.

## Author comparison and limits

At seed 0 and identical propagated input tensors, both full training budgets
gave identical batch losses, all epoch metrics and all 28 final model states
between the wrapper and author model. Independent CUDA preprocessing differs
by about 10⁻⁷; independent bitwise parameter equality is not guaranteed.

A separate audited invocation of the author's main() with independent data
loading and preprocessing reproduced the seed-0 scores to its four-decimal
stdout precision. That invocation used an exact full-node identity-slice
shortcut to avoid per-edge GPU scalar transfers; uint8 and float32 sparse
fixtures were checked against the original routine. It was not an entirely
unmodified runtime. Original author source files were unchanged.

The author's main() reports the maximum target accuracy over epochs. Its
seed-0 maxima were 52.97% and 40.13%. Those are target-label oracle diagnostics;
they must be separated from the final-epoch, five-seed results above. The
author also saves the final state under `best_0.pth`, not the best epoch state.

Real-data validation covers the ACM pair only. No MAG performance claim or
full-paper reproduction claim is made. Core public-code behavior is preserved,
including fixed-hop preprocessing and view-zero IA. The paper's dynamic
neighborhood-order search is not added by this integration.
