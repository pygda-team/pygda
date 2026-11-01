# UAGABase

The UAGA backbone uses DGL GATConv for the encoder and graph classifier.
Importing this symbol does not load DGL; constructing a backbone requires DGL.
The parameter names and classifier-first initialization follow the official
implementation. Reversal coefficients are stored per autograd invocation.

::: pygda.nn.UAGABase
