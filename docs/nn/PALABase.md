# PALABase

The two-encoder PALA core consumes prepared full-batch graphs and explicit cache
tensors. It preserves the author encoder state layout and source compensation,
and derives target memory width from the source class count. The prototype
contrastive reduction is evaluated in the log domain for numerical stability.

::: pygda.nn.PALABase
