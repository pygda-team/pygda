# Benchmarks Overview

PyGDA provides extensive benchmarking capabilities across different types of graph domain adaptation tasks. This document outlines our three main benchmark suites.

## Node Classification Benchmark

### Overview
- Evaluates 16 different methods
- Tests on 5 distinct datasets
- Each experiment repeated 3 times for statistical significance

### Running the Benchmark
```
cd benchmark/node
./run.sh
```

## Graph Classification Benchmark

### Overview
- Evaluates 7 different methods
- Tests on 3 graph classification datasets:
    
    * PROTEINS
    * FRANKENSTEIN
    * Mutagenicity

- Each experiment repeated 3 times for statistical significance

### Running the Benchmark
```
cd benchmark/graph
# Run benchmarks for each dataset
./run_all_F.sh  # FRANKENSTEIN
./run_all_M.sh  # Mutagenicity
./run_all_P.sh  # PROTEINS
```

## LLM-Enhanced Benchmark

### Overview
- Evaluates 5 different methods
- Focuses on ogbn-arxiv dataset with LLM predictions and explanations
- Each experiment repeated 3 times for statistical significance
- Tests different feature encoding approaches

### Dataset Preprocessing Options

#### **Original Features**
```
python origin_preprocess.py
```

#### **LLM with Word2Vec**
```
python llm_w2v_preprocess.py
```
- Combines title, abstract, and LLM outputs
- Processes using word2vec embeddings

#### **LLM with BERT**
```
python llm_bert_preprocess.py
```
- Combines title, abstract, and LLM outputs
- Uses DeBERTa for sentence embeddings
- Unsupervised approach (no fine-tuning)

### Data Requirements
- **ogbn-arxiv**: Download title and abstract data from [OGB](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz)
- **LLM Responses**: Download from [TAPE paper data](https://drive.google.com/file/d/1A6mZSFzDIhJU795497R6mAAM2Y9qutI5/view?usp=sharing)

### Chronological Split
Dataset is divided into 3 groups based on publication years:

- Group A: Papers before 2016
- Group B: Papers from 2016-2018
- Group C: Papers from 2018-2020

### Running the Benchmark
```
cd benchmark/llm
./run1.sh
./run2.sh
./run3.sh
```

## General Guidelines

### Running Benchmarks
- Ensure all required datasets are downloaded
- Install all dependencies
- Run benchmarks from their respective directories
- Results will be saved in the corresponding output directories

### Reproducibility
- Fixed random seeds are used
- Multiple runs (3x) for statistical significance
- Standardized evaluation metrics across all experiments

### Resource Requirements
- Node classification: Moderate GPU memory
- Graph classification: Lower GPU memory
- LLM benchmark: Higher GPU memory (for BERT embeddings)


This overview:

1. Provides a clear structure for each benchmark suite
2. Includes detailed setup and running instructions
3. Specifies data requirements and preprocessing steps
4. Offers guidelines for reproducibility
5. Maintains consistent formatting throughout
6. Includes resource requirements