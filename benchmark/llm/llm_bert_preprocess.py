import numpy as np
import torch
import random
import json
import pandas as pd
import argparse
from transformers import AutoTokenizer, DebertaModel


def take_second(element):
    return element[1]

def load_data_raw_text():
    nodeidx2paperid = pd.read_csv('./data/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv('./data/ogbn_arxiv_orig/titleabs.tsv.gz', compression='gzip', sep='\t', header=None, names=['paper id', 'title', 'abs'])
    raw_text = raw_text[raw_text['paper id'].notna()]
    mask = raw_text['paper id'] == 'titleabs.tsv'
    raw_text = raw_text[~mask]
    raw_text = raw_text.astype({"paper id": int})
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    
    return text

def load_data_gpt_text(num_nodes):
    folder_path = './data/ogbn_arxiv_gpt/'

    text = []

    for i in range(num_nodes):
        filename = str(i) + '.json'
        file_path = folder_path + filename
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            content = json_data['choices'][0]['message']['content']
            text.append(content)
    
    return text

def load_ogb_arxiv(data_dir, year_bound = [2018, 2020], proportion = 1.0):
    import ogb.nodeproppred

    dataset = ogb.nodeproppred.NodePropPredDataset(name='ogbn-arxiv', root=data_dir)
    graph = dataset.graph

    node_years = graph['node_year']
    # print(node_years) year for each node
    n = node_years.shape[0]
    # print(n) number of nodes
    node_years = node_years.reshape(n)

    gpt_text = load_data_gpt_text(n)
    raw_text = load_data_raw_text()

    d = np.zeros(len(node_years))
    print(d.shape)

    edges = graph['edge_index']
    for i in range(edges.shape[1]):
        if node_years[edges[0][i]] <= year_bound[1] and node_years[edges[1][i]] <= year_bound[1] and node_years[edges[0][i]] > year_bound[0] and node_years[edges[1][i]] > year_bound[0]:
            d[edges[0][i]] += 1
            d[edges[1][i]] += 1

    nodes = []
    for i, year in enumerate(node_years):
        if year <= year_bound[1] and year > year_bound[0]:
            nodes.append([i, d[i]])

    nodes.sort(key = take_second, reverse = True)

    nodes = nodes[: int(proportion * len(nodes))]

    random.shuffle(nodes)

    result_edges = []
    result_features = []
    result_labels = []
    result_text = []

    for node in nodes:
        result_features.append(graph['node_feat'][node[0]])
        content = raw_text[node[0]] + '\n' + 'Prediction and Explanation: ' + gpt_text[node[0]]
        result_text.append(content)
    result_features = np.array(result_features)

    # construct nodes
    device = 'cuda:3'
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    model = DebertaModel.from_pretrained("microsoft/deberta-base").to(device)

    emb = None

    cnt = 0

    for text in result_text:
        print(cnt)
        cnt += 1
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state.detach().cpu()
        # Use CLS Emb as sentence emb.
        last_hidden_states = last_hidden_states.permute(1, 0, 2)[0]

        if emb is None:
            emb = last_hidden_states
        else:
            emb = torch.cat([emb, last_hidden_states], dim=0)

    ids = {}
    for i, node in enumerate(nodes):
        ids[node[0]] = i

    for i in range(edges.shape[1]):
        if edges[0][i] in ids and edges[1][i] in ids:
            result_edges.append([ids[edges[0][i]], ids[edges[1][i]]])
    result_edges = np.array(result_edges).transpose(1, 0)

    result_labels = dataset.labels[[node[0] for node in nodes]]

    edge_index = torch.tensor(result_edges, dtype=torch.long)
    
    # emb: llm enhanced bert features
    node_feat = torch.tensor(emb, dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': node_feat.size(0)}
    dataset.label = torch.tensor(result_labels)

    return dataset

def main(args):
    data_dir = './data'

    # 3 domains: [1950, 2016], [2016, 2018], [2018, 2020]

    start_year = 1950
    end_year = 2016

    dataset = load_ogb_arxiv(data_dir, year_bound=[start_year, end_year])

    dataset.n = dataset.graph['num_nodes']
    dataset.c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    dataset.d = dataset.graph['node_feat'].shape[1]

    print(torch.min(dataset.graph['edge_index']))
    print(torch.max(dataset.graph['edge_index']))
    print(len(torch.unique(dataset.graph['edge_index'])))
    print(len(dataset.graph['edge_index'][1]))
    print(dataset.graph['node_feat'].size())
    print(len(dataset.label))

    print(f"num nodes {dataset.n}| num classes {dataset.c} | num node feats {dataset.d}")

    import pickle

    filename = 'llm-bert-arxiv-' + str(start_year) + '-' + str(end_year) + '.pkl'

    fw = open(filename, 'wb')
    pickle.dump(dataset, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    main(args)
