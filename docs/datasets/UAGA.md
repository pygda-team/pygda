# UAGA citation data

`load_uaga_graph(root, name)` reads the original nine-class MAT files:
`citation-v1.mat`, `dblp-v4.mat` and `acm-v8.mat`. The files must contain sparse
`features`, sparse `adjacency_matrix` and a row vector `labels`.

The loader returns PyG Data with dense float32 `x`, long `edge_index` and long
`y`. It preserves the COO edge order produced by SciPy and retains all raw
classes. Filtering source-private classes is the model's responsibility. For
the paper setting use `num_classes=5`: classes 0..4 are known and 5..8 are private.

These files differ from PyGDA's five-class ACMv9/Citationv1/DBLPv7 citation data.
No data are downloaded automatically. Use the data supplied by the
[official UAGA repository](https://github.com/3480430977/UAGA).

::: pygda.datasets.uaga.load_uaga_graph
