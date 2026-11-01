# ImGDAEncoder

The released two-layer shared-weight local/global encoder with per-source-node compensation. This adaptation accepts explicitly supplied global cache tensors and follows the input tensor device. It performs no cache file I/O or random-walk regeneration. Ordinary-list Dropout preserves the released stochastic evaluation behavior.

Source and target global edges and normalized weights are injected through each convolution's `set_global_cache` method. Cache weights are used unchanged. Compensation is enabled only for training source embeddings.

::: pygda.nn.ImGDAEncoder
