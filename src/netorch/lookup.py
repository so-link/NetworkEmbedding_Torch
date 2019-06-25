#coding:utf-8

import networkx as nx

class GraphLookup(object):

    def __init__(self, graph):
        self._index2label = list(graph.nodes)
        self._label2index = {node:i for i, node in enumerate(self._index2label)}
    
    def label_to_index(self, label):
        return self._label2index[label]

    def index_to_label(self, index):
        return self._index2label[index]
