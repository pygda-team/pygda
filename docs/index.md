# Welcome to PyGDA

![](pygda_logo.png)
-----
[![PyPI - Version](https://img.shields.io/pypi/v/pygda?style=flat)](https://pypi.org/project/pygda/)
[![Documentation Status](https://readthedocs.org/projects/pygda/badge/?version=stable)](https://pygda.readthedocs.io/en/stable/?badge=stable)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/pygda-team/pygda/issues)

PyGDA is a **Python library** for **Graph Domain Adaptation** built upon [PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) to easily train graph domain adaptation models in a [sklearn](https://scikit-learn.org/stable/) style. PyGDA includes **20+** graph domain adaptation models. See examples with PyGDA below!

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
* **Cover 20+ graph domain adaptation models.**
* **Scalable architecture that efficiently handles large graph datasets through mini-batching and sampling techniques.**
* **Seamlessly integrated data processing with PyG, ensuring full compatibility with PyG data structures.**

## What's New?
**[12/2024]**. We now support source-free setting of graph domain adaptation.

- 3 recent models including `GTrans`, `SOGA` and `GraphCTA` are supported.

**[08/2024]**. We support graph-level domain adaptation task.

- 7 models including `A2GNN`, `AdaGCN`, `CWGCN`, `DANE`, `GRADE`, `SAGDA`, `UDAGCN` are supported.
- Various TUDatasets are supported including `FRANKENSTEIN`, `Mutagenicity` and `PROTEINS`.
- To perform a graph-level domain adaptation task, only one parameter is added to the model as follows:
```
model = A2GNN(in_dim=num_features, hid_dim=args.nhid, num_classes=num_classes, mode='graph', device=args.device)
```

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

**Installing with pip:**
```
pip install pygda
```

or 

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

## Reference

| **ID** | **Paper** | **Method** | **Venue** |
|--------|---------|:----------:|:--------------:|
| 01      | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)      |    Vanilla GCN     |   ICLR 2017    |
| 02      | [DANE: Domain Adaptive Network Embedding](https://www.ijcai.org/proceedings/2019/606)      |    DANE     |   IJCAI 2019    |
| 03      | [Adversarial Deep Network Embedding for Cross-network Node Classification](https://arxiv.org/abs/2002.07366) |    ACDNE     |   AAAI 2020    |
| 04      | [Unsupervised Domain Adaptive Graph Convolutional Networks](https://dl.acm.org/doi/10.1145/3366423.3380219)  |   UDAGCN   |    WWW 2020    |
| 05      | [Adversarial Separation Network for Cross-Network Node Classification](https://dl.acm.org/doi/abs/10.1145/3459637.3482228)  |    ASN    |  CIKM 2021  |
| 06      | [Graph Transfer Learning via Adversarial Domain Adaptation with Graph Convolution](https://arxiv.org/abs/1909.01541)  |    AdaGCN    | TKDE 2022 |
| 07      | [Non-IID Transfer Learning on Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/26231)  |  GRADE   |   AAAI 2023    |
| 08      | [Graph Domain Adaptation via Theory-Grounded Spectral Regularization](https://openreview.net/forum?id=OysfLgrk8mk)  |   SpecReg    |   ICLR 2023    |
| 09      | [Structural Re-weighting Improves Graph Domain Adaptation](https://arxiv.org/abs/2306.03221) |   StruRW    |    ICML 2023    |
| 10      | [Improving Graph Domain Adaptation with Network Hierarchy](https://dl.acm.org/doi/10.1145/3583780.3614928)  | JHGDA |  CIKM 2023  |
| 11     | [Bridged-GNN: Knowledge Bridge Learning for Effective Knowledge Transfer](https://dl.acm.org/doi/10.1145/3583780.3614796)  |    KBL     |    CIKM 2023    |
| 12     | [Domain-adaptive Message Passing Graph Neural Network](https://www.sciencedirect.com/science/article/abs/pii/S0893608023002253)  |   DMGNN    |    NN 2023    |
| 13     | [Correntropy-Induced Wasserstein GCN: Learning Graph Embedding via Domain Adaptation](https://ieeexplore.ieee.org/document/10179964)  |   CWGCN    |    TIP 2023    |
| 14     | [SA-GDA: Spectral Augmentation for Graph Domain Adaptation](https://dl.acm.org/doi/10.1145/3581783.3612264)  |  SAGDA   |    MM 2023    |
| 15     | [Graph Domain Adaptation: A Generative View](https://dl.acm.org/doi/10.1145/3631712)  |   DGDA   |    TKDD 2024    |
| 16     | [Rethinking Propagation for Unsupervised Graph Domain Adaptation](https://arxiv.org/abs/2402.05660)      |    A2GNN    |   AAAI 2024    |   
| 17     | [Pairwise Alignment Improves Graph Domain Adaptation](https://arxiv.org/abs/2403.01092)      |   PairAlign   |   ICML 2024    |
| 18     | [Empowering Graph Representation Learning with Test-Time Graph Transformation](https://arxiv.org/abs/2210.03561)      |   GTrans   |   ICLR 2023    |
| 19     | [Source Free Unsupervised Graph Domain Adaptation](https://arxiv.org/abs/2112.00955)      |   SOGA   |   WSDM 2024    |
| 20     | [Collaborate to Adapt: Source-Free Graph Domain Adaptation via Bi-directional Adaptation](https://dl.acm.org/doi/10.1145/3589334.3645507)      |   GraphCTA   |   WWW 2024    |


## Cite

If you compare with, build on, or use aspects of PyGDA, please consider citing "[Revisiting, Benchmarking and Understanding Unsupervised Graph Domain Adaptation](https://arxiv.org/abs/2407.11052)":

```
@inproceedings{liu2024revisiting,
title={Revisiting, Benchmarking and Understanding Unsupervised Graph Domain Adaptation},
author={Meihan Liu and Zhen Zhang and Jiachen Tang and Jiajun Bu and Bingsheng He and Sheng Zhou},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://openreview.net/forum?id=ZsyFwzuDzD}
}
```
