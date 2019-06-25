#coding:utf-8
import random

import networkx as nx

from .base import BaseCoarsening

class RandomCoarsening(BaseCoarsening):

    def __init__(self, graph, threshold=0.2, weighted=False):
        super().__init__(graph, threshold, weighted)

    def star_collapsing(self, graph, merge_result, merged_nodes):
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        for node in nodes:
            star_neighbors = [neighbor for neighbor in graph.neighbors(node) if neighbor not in merged_nodes]
            random.shuffle(star_neighbors)
            cnt = len(star_neighbors)//2
            merge = list(zip(star_neighbors[:cnt], star_neighbors[cnt:]))
            for u, v in merge:
                merged_nodes.add(u)
                merged_nodes.add(v)
            merge_result.extend(merge)

    def edge_collapsing(self, graph, merge_result, merged_nodes):
        edges = list(graph.edges)
        random.shuffle(edges)
        for u, v in edges:
            if u in merged_nodes or v in merged_nodes:
                continue
            merged_nodes.add(u)
            merged_nodes.add(v)
            merge_result.append((u, v))

    def merge(self, graph):
        merged_nodes = set()
        merge_result = []
        self.edge_collapsing(graph, merge_result, merged_nodes)
        self.star_collapsing(graph, merge_result, merged_nodes)
        for node in graph.nodes:
            if node not in merged_nodes:
                merge_result.append((node,))
        return merge_result
