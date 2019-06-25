#coding:utf-8
import sys
import numpy as np
import networkx as nx

from netorch.lookup import GraphLookup
from netorch.dataset import load_edgelist, load_labels
from netorch.coarsening.aco import ACOCoarsening
from netorch.models.walkbased import DeepWalk
from netorch.models.hierarchical import ConcatPCAModel
from netorch.evaluate import evaluate

DATASET = 'blogcatalog'
DATASET_DIR = 'datasets'
EDGES_TXT = '{}/{}_edgelist.txt'.format(DATASET_DIR, DATASET)
LABELS_TXT = '{}/{}_labels.txt'.format(DATASET_DIR, DATASET)

g, labels = load_edgelist(EDGES_TXT), load_labels(LABELS_TXT)
lookup = GraphLookup(g)
g = nx.convert_node_labels_to_integers(g)
model = ConcatPCAModel(
    graph = g,
    dimension = 128, 
    Model = lambda graph, dimension: DeepWalk(graph, dimension=dimension, batch_size=5000, iterations=3),
    Coarsening = lambda graph: ACOCoarsening(graph, phe_power=.5, iterations=2),
)
embedding = model.train().get_embeddings()
result = evaluate({lookup.index_to_label(index):embedding[index] for index in range(g.number_of_nodes())}, labels, clf_ratio=0.5)
print(result)
