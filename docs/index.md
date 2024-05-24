# Welcome to PyGDA

![](pygda_logo.png)
-----

PyGDA is a **Python library** for **Graph Domain Adaptation** built upon [PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) to easily train graph domain adaptation models in a [sklearn](https://scikit-learn.org/stable/) style. PyGDA includes **15+** graph domain adaptation models. See examples with PyGDA below!

**Graph Domain Adaptation Using PyGDA with 5 Lines of Code**
```
from pygda.models import A2GNN

# choose a graph domain adaptation model
model = A2GNN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)

# train the model
model.fit(source_data, target_data)

# evaluate the performance
logits, labels = model.predict(target_data)
```
**PyGDA is featured for:**

* **Consistent APIs and comprehensive documentation.**
* **Cover 15+ graph domain adaptation models.**
* **Scalable architecture that efficiently handles large graph datasets through mini-batching and sampling techniques.**
* **Seamlessly integrated data processing with PyG, ensuring full compatibility with PyG data structures.**

## Installation
Note: PyGDA depends on [PyTorch](https://pytorch.org/), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/), [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) and [Pytorch Scatter](https://github.com/rusty1s/pytorch_scatter). PyGDA does not automatically install these libraries for you. Please install them separately in order to run PyGDA successfully.

**Required Dependencies:**
* torch>=1.13.1
* torch_geometric>=2.4.0
* torch_sparse>=0.6.15
* torch_scatter>=2.1.0
* python3
* scipy
* sklearn
* numpy
* cvxpy
* tqdm

**Installing with pip**
```
pip install pygda
```

**Installation for local development:**
```
git clone https://github.com/pygda-team/pygda
cd pygda
pip install -e .
```

## Quick Start

### Step 1: Load Data
```
from pygda.datasets import CitationDataset

source_dataset = CitationDataset(path, args.source)
target_dataset = CitationDataset(path, args.target)
```

### Step 2: Build Model
```
from pygda.models import A2GNN

model = A2GNN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, device=args.device)
```

### Step 3: Fit Model
```
model.fit(source_data, target_data)
```

### Step 4: Evaluation
```
from pygda.metrics import eval_micro_f1, eval_macro_f1

logits, labels = model.predict(target_data)
preds = logits.argmax(dim=1)
mi_f1 = eval_micro_f1(labels, preds)
ma_f1 = eval_macro_f1(labels, preds)
```

## Create your own GDA model
In addition to the easy application of existing GDA models, PyGDA makes it simple to implement custom models.

* the customed model should inherit ``BaseGDA`` class.
* implement your ``fit()``, ``forward_model()``, and ``predict()`` functions.
