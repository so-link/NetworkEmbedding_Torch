#coding:utf-8
import numpy as np
import networkx as nx
import pprint
from scipy.sparse import coo_matrix


class BaseCoarsening(object):

    def __init__(self, graph, threshold=0.2, weighted=True):
        nx.set_node_attributes(graph, 1, 'size')
        self.original_graph = graph
        self.graphs = [graph]
        self.mappings = [{node:[node] for node in graph.nodes}]
        self.threshold = threshold
        self.weighted = weighted

    def merge(self, graph):
        '''
        RETURN merge result list of list
        example [
            [node1, node2],
            [node3, node4, node5],
            ...
        ]
        '''
        raise NotImplementedError

    def make_mappings_to_original_graph(self):
        mappings_orig = [self.mappings[0]]
        for mapping in self.mappings[1:]:
            mappings_orig.append(self.extend_mapping(mappings_orig[-1], mapping))
        return mappings_orig

    def reverse_mapping(self, mapping):
        return {node:super_node for super_node, nodes in mapping.items() for node in nodes}

    def merge_result_to_mapping(self, merge_result):
        return dict(enumerate(merge_result))

    def extend_mapping(self, mapping_last, mapping_current):
        extended_mapping = {super_node:[] for super_node in mapping_current}
        for super_node, nodes in mapping_current.items():
            for node in nodes:
                extended_mapping[super_node].extend(mapping_last[node])
        return extended_mapping


    def gen_merged_graph(self, graph, merge_result):
        rev_mapping = {node:super_node for super_node, nodes in enumerate(merge_result) for node in nodes}
        num_new_nodes = len(merge_result)
        rows, cols, vals = [], [], []
        for u, v in graph.edges:
            su = rev_mapping[u]
            sv = rev_mapping[v]
            if su==sv:
                continue
            w = graph[u][v]['weight']
            rows.append(su)
            cols.append(sv)
            rows.append(sv)
            cols.append(su)
            vals.append(w)
            vals.append(w)
        new_graph = nx.from_scipy_sparse_matrix(
            coo_matrix(
                (vals, (rows, cols)),
                shape=(num_new_nodes, num_new_nodes),
            ).tocsr()
        )
        if not self.weighted:
            nx.set_edge_attributes(new_graph, 1.0, 'weight')

        for super_node, nodes in enumerate(merge_result):
            sum_size = np.sum([graph.nodes[node]['size'] for node in nodes])
            new_graph.nodes[super_node]['size'] = sum_size
        
        return new_graph

    def recursive_merge(self):
        edge_threshold = self.original_graph.number_of_edges()*self.threshold
        node_threshold = self.original_graph.number_of_nodes()*self.threshold
        while True:
            cur_graph = self.graphs[-1]
            merge_result = self.merge(cur_graph)
            if len(merge_result)==cur_graph.number_of_nodes(): # no edge in the graph
                break
            new_graph = self.gen_merged_graph(cur_graph, merge_result)
            self.graphs.append(new_graph)
            self.mappings.append(self.merge_result_to_mapping(merge_result))
            if new_graph.number_of_nodes()<node_threshold or new_graph.number_of_edges()<edge_threshold:
                break
