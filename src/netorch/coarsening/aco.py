#coding:utf-8

import numpy as np
import networkx as nx

from .base import BaseCoarsening

from util.data_structure import DisjoinSet
from util.stackoverflow import find_best_trade_off

from necython import Graph as CGraph, aco_walk

class ACOCoarsening(BaseCoarsening):

    def __init__(self, graph, threshold=0.2, window_size=10, num_walks=10, walk_length=80, phe_power=1., evapo_rate=0., iterations=1):
        super().__init__(graph, threshold)
        self.window_size = window_size
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.phe_power = phe_power
        self.evapo_rate = evapo_rate
        self.iterations = iterations

    def merge(self, graph):
        m_graph = CGraph.from_nx_graph(graph)

        edge_and_weights = aco_walk(m_graph, self.num_walks, self.window_size, self.iterations, self.phe_power, self.evapo_rate, 20)
        edge_and_weights.sort(key=lambda item: item[2], reverse=True)

        weights = np.array([item[2] for item in edge_and_weights])
        trade_off_index = find_best_trade_off(weights)

        ds = DisjoinSet(graph.number_of_nodes())
        for u, v, weight in edge_and_weights[:trade_off_index+1]:
            ds.union(u, v)

        return ds.make_mapping()