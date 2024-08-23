# Benchmark
Evaluation scripts for 5 methods on the ogbn-arxiv datasets with LLM predictions and explanations. Each experiment is repeated 3 times.

## LLM as Feature Encoder
To investigate whether the distribution gap narrows after utilizing the LLM as the feature encoder, we utilize the prompts from TAPE (ICLR 2024, Explanations as Features: LLM-Based Features for Text-Attributed Graphs), which allows us to assess the impact of LLM-based features on the model's performance.

The dataset is chronologically divided into 3 groups according to the publication years of the papers. We construct 3 graphs encompassing papers published before 2016 (Group A), 2016-2018 (Group B), and 2018-2020 (Group C). 

### Datasets Preprocess
- Original node attributes, which are obtained by averaging the embeddings of words in its title and abstract via word2vec.
```
python origin_preprocess.py
```
- LLM enhanced text with word2vec embedding, which combines the title, abstract, and LLM-generated predictions and explanations into a single input. This composite text is then fed into word2vec. Then, the node features are obtained by averaging the embeddings of its combined input.
```
python llm_w2v_preprocess.py
```
- LLM enhanced text with BERT embedding, which combines the title, abstract, and LLM-generated predictions and explanations into a single input. This composite text is then fed into a pretrained DeBERTa. Then, the node features are obtained by sentence embedding. **Note that, we did not finetune the DeBERTa like TAPE paper, since we study unsupervised graph domain adaptation**.
```
python llm_bert_preprocess.py
```

### Data Download
- ogbn-arixv. The [OGB](https://ogb.stanford.edu/docs/nodeprop/) provides the mapping from MAG paper IDs into the raw texts of titles and abstracts. Download the title and abstract data [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz).
- LLM-responses. Download the LLM responses data [here](https://drive.google.com/file/d/1A6mZSFzDIhJU795497R6mAAM2Y9qutI5/view?usp=sharing) from TAPE paper.

## Run

Run via
```
./run1.sh
./run2.sh
./run3.sh
```