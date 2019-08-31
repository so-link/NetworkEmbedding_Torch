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

class HeteroGraphLookup(object):

    def __init__(self, graph):
        self._gindex2label = list(graph.nodes)
        self._label2gindex = {node:i for i, node in enumerate(self._gindex2label)}

        self._gindex2tag = [graph.nodes[u]['tag'] for u in graph.nodes]
        self._tags = list(set(self._gindex2tag))
        
        self._tag2gindices = {t:[] for t in self._tags}
        self._gindex2tindex = []
        for u, tag in enumerate(self._gindex2tag):
            self._gindex2tindex.append(len(self._tag2gindices[tag]))
            self._tag2gindices[tag].append(u)

    def g_index_to_label(self, g_index):
        return self._gindex2label[g_index]

    def g_index_to_tag(self, g_index):
        return self._gindex2tag[g_index]

    def g_index_to_t_index(self, g_index):
        return self._gindex2tindex[g_index]

    def label_to_g_index(self, label):
        return self._label2gindex[label]

    def label_to_t_index(self, label):
        g_index = self.label_to_g_index(label)
        return self.g_index_to_t_index(g_index)

    def label_to_tag(self, label):
        g_index = self.label_to_g_index(label)
        return self.g_index_to_tag(g_index)
    
    def t_index_to_g_index(self, tag, t_index):
        return self._tag2gindices[tag][t_index]

    def t_index_to_label(self, tag, t_index):
        g_index = self.t_index_to_g_index(tag, t_index)
        return self.g_index_to_label(g_index)

    def num_tag_nodes(self, tag):
        return len(self._tag2gindices[tag])