# UAGA benchmark

Run from the repository root in an environment containing Torch, PyG and DGL:

```bash
python benchmark/node/uaga.py --data-root /path/to/uaga/data \
  --source citation-v1 --target dblp-v4 --seeds 0 1 2 3 4 \
  --output uaga-citation-dblp.json
```

Use `--epochs 1 --separation-epochs 1 --seeds 0` for an entry-point smoke test.
All six directed domain pairs are supported with fixed author dropout and weight
decay settings. The default schedule is 30 separation and 200 adaptation epochs.
The output records each seed, configuration, data hashes, elapsed time, final
metrics and population standard deviations (`ddof=0`). Existing output files are
not overwritten. Intermediate completed-seed results are saved as the run proceeds.

The benchmark trains without target labels and evaluates once after the final
adaptation epoch. It does not select an epoch using target scores. Metric values
in JSON are fractions; multiply means and standard deviations by 100 for the
paper's percentage format. HS combines micro known-node accuracy with unknown
recall. An all-wrong known/unknown prediction receives HS=0 instead of NaN.

## Reference evidence

For citation-v1 → dblp-v4, official source at commit
`4ef2f8fa6c2a18760ef9d9f8ef90d4ee4f92c227`, original nine-class author data,
seeds 0..4, final adaptation epoch and RTX 4090:

| Metric (%) | Official-code reference | Paper |
|---|---:|---:|
| OS | 69.2146 ± 0.2145 | 69.21 ± 0.17 |
| OS* | 73.9999 ± 0.2997 | 74.00 ± 0.15 |
| UNK_AUC | 73.9153 ± 1.0747 | 73.72 ± 1.03 |
| HS | 56.4283 ± 0.4844 | 56.41 ± 0.49 |

The reference environment is Python 3.9.13, Torch 1.7.1+cu110, DGL 0.7.2,
NumPy 1.21.5, SciPy 1.9.1 and scikit-learn 1.0.2. These are official-code
reference results, not a five-seed run of the new benchmark.

Adapter checks in that environment passed four tests without skips. Fourteen
recorded comparisons had maximum absolute difference zero, including parameter
initialization, both phases' gradients and Adam updates, full-graph outputs,
real-data parameters after one epoch per phase, final predictions, and target
label independence. Public PyGDA import/graph preparation were separately checked
using Torch 2.5.1 and PyG 2.8.0.post1. Full public-entry-point training in the
integration profile has also passed separately. Five other paper tasks have not
yet been validated. CPU execution in the legacy reference environment has an
unresolved exit SIGSEGV and is not part of the supported reference claim.

The integration profile in `benchmark/node/uaga-requirements.txt` targets Python
3.9 on Linux x86_64 with Torch 1.13.1+cu117, PyG 2.3.1 and the official DGL 0.7.2
CUDA 11.0 build. In this unified environment on RTX 4090, editable installation,
public import, six tests without skips, fourteen zero-difference comparisons,
real PyG Data training through the benchmark (one epoch per phase, seed 0),
optional-dependency import behavior and documentation build passed. These checks
use the real package initialization and no private module bridge. They do not
establish full-schedule paper accuracy in the newer integration environment.

DGL 0.7.2 logs that `libtensoradapter_pytorch_1.13.1.so` is unavailable in this
profile. GPU training, tests and benchmark completed with clean process exits
despite that diagnostic; availability of TensorAdapter acceleration is not claimed.
