# Dataset Overview

This section provides detailed documentation for all supported datasets in PyGDA, including their domains and sources.

### Citation Networks

#### [Arxiv](Arxiv.md)
- **Domains**: 3 domains based on publication years
- **Source**: [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
- **Processing**: See [ArxivDataset](https://github.com/pygda-team/pygda/blob/main/pygda/datasets/arxiv.py)
- **Features**: Generated from paper abstracts
- **Note**: Can be preprocessed with scripts in benchmark folder

#### [Citation](Citation.md)
- **Domains**: ACMv9, Citationv1, DBLPv7
- **Source**: Adopted from [ASN](https://arxiv.org/abs/2103.13355)
- **Processing**: See [CitationDataset](https://github.com/pygda-team/pygda/blob/main/pygda/datasets/citation.py)
- **Download**: [Download Link](https://drive.google.com/drive/folders/1ntNt3qHE4p9Us8Re9tZDaB-tdtqwV8AX?usp=share_link)

#### [MAG](MAG.md)
- **Domains**: CN, DE, FR, JP, RU, US
- **Source**: Originally from [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag)
- **Processing**: See [MAGDataset](https://github.com/pygda-team/pygda/blob/main/pygda/datasets/mag.py)
- **Download**: [Download Link](https://drive.google.com/drive/folders/1HinhjpNPPivyqoubiYOr8X2jq-rjw3e9?usp=share_link)
- **Note**: Separated into 6 countries by [PairAlign](https://arxiv.org/abs/2403.01092)

### Social Networks

#### [Blog](Blog.md)
- **Domains**: Blog1, Blog2
- **Source**: Adopted from [ACDNE](https://arxiv.org/abs/2002.07366)
- **Processing**: See [BlogDataset](https://github.com/pygda-team/pygda/blob/main/pygda/datasets/blog.py)
- **Download**: [Download Link](https://drive.google.com/drive/folders/1jKKG0o7rEY-BaVEjBhuGijzwwhU0M-pQ?usp=share_link)

#### [Twitch](Twitch.md)
- **Domains**: DE, EN, ES, FR, PT, RU
- **Source**: [Twitch Social Networks](https://github.com/benedekrozemberczki/datasets#twitch-social-networks)
- **Processing**: See [TwitchDataset](https://github.com/pygda-team/pygda/blob/main/pygda/datasets/twitch.py)
- **Download**: [Download Link](https://drive.google.com/drive/folders/1GWMyyJOZ4CeeqP_H5dCA5voSQHT0WlXG?usp=share_link)

### Infrastructure Networks

#### [Airport](Airport.md)
- **Domains**: Brazil, Europe, USA
- **Source**: Adopted from [struc2vec](https://arxiv.org/abs/1704.03165)
- **Processing**: See [AirportDataset](https://github.com/pygda-team/pygda/blob/main/pygda/datasets/airport.py)
- **Features**: Constructed using OneHotDegree for each node
- **Download**: [Download Link](https://drive.google.com/drive/folders/1zlluWoeukD33ZxwaTRQi3jCdD0qC-I2j?usp=share_link)

### Graph Classification Benchmarks

#### [TUGraph](TUGraph.md)
- **Datasets**: 
    
    * PROTEINS
    * FRANKENSTEIN
    * Mutagenicity

- **Domains**: 2 domains based on density for each dataset
- **Source**: Adopted from [TUDataset](https://chrsmrrs.github.io/datasets/docs/datasets/)
- **Processing**: See [GraphTUDataset](https://github.com/pygda-team/pygda/blob/main/pygda/datasets/tugraph.py)
- **Download**: [Download Link](https://drive.google.com/drive/folders/1NbPK71Dy0ulH3CdNyfvMwQECj_Oh867I?usp=sharing)

### Usage Example

```python
from pygda.datasets import CitationDataset

# Load the Citation dataset
dataset = CitationDataset(root='data/citation', name='ACMv9')
data = dataset[0]

# Access the data
x = data.x  # Node features
edge_index = data.edge_index  # Graph connectivity
y = data.y  # Labels
```

Each dataset documentation includes:

- Detailed domain descriptions
- Data sources and references
- Processing instructions
- Download information
- Usage examples
- Implementation details

For specific details about each dataset, please visit their respective documentation pages linked above.