# PALA

Class-imbalanced Graph Domain Adaptation via Prototype-anchored Learning and
Alignment (IJCAI 2025). Source: [maxin88scu/PALA](https://github.com/maxin88scu/PALA),
commit `612b35d059c23b96b4878a6c5e4420bfc6845d11`, MIT license.

```python
from pygda.models import PALA

model = PALA(in_dim=source.x.shape[1], hid_dim=512, num_classes=5,
             epoch=200, seed=0, device='cuda:0')
model.fit(source, target)
logits, _ = model.predict(target, return_labels=False)
```

Inputs are prepared graphs with `x`, `edge_index`, `ppmi_edge_index` and
`ppmi_edge_weight`. The source additionally has contiguous integer labels `y`.
Prepare the long-tail source with the [dataset helpers](../datasets/PALA.md).
The explicit caches must use each input graph's exact node order.

Only full-batch, two-layer ReLU training is supported. Each source class needs
at least `2*gamma` nodes. Target labels are not retrieved by `fit`; an optional
`epoch_callback(epoch, logits, record)` can evaluate the existing post-update
target logits outside training, without introducing extra stochastic forwards.

The implementation follows the public code, including its three-dimensional
normalization axis, prototype broadcasting, unregistered Dropout list and
ELR activation at zero-based epoch 51. Dropout remains active during prediction,
as in the author implementation. An extra prediction consumes random numbers.

The prototype contrastive reduction uses logsumexp/logaddexp to avoid observed
float32 overflow in the original code. The formula is algebraically equivalent,
but floating point trajectories need not be bitwise identical. Target memory
class count comes from source labels, caches are explicit tensors, and the
library does not generate or write PPMI files during training.

See the [benchmark record](../benchmark/PALA.md) for measured results and limits.

::: pygda.models.PALA
