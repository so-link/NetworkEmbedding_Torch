#coding:utf-8
import multiprocessing as mp
import numpy as np
import networkx as nx

from necython import Graph as CGraph, Walker as CWalker, BiasedWalker as CBiasedWalker

CPU_COUNT = mp.cpu_count()

class Walker(object):

    def __init__(self, num_walks=10, walk_length=80, weighted=False, multi_process=CPU_COUNT):
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.weighted = weighted
        self.multi_process = multi_process

    def walk(self, graph):
        g = CGraph.from_nx_graph(graph)
        w = CWalker()
        w.init_distributions_from_graph(g, self.weighted)
        sequences = w.walk(self.num_walks, self.walk_length, self.multi_process)
        return sequences

class BiasedWalker(object):

    def __init__(self, num_walks=10, walk_length=80, p=1., q=1., multi_process=CPU_COUNT):
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.multi_process = multi_process

    def walk(self, graph):
        g = CGraph.from_nx_graph(graph)
        w = CBiasedWalker()
        w.init_distributions_from_graph(g, self.p, self.q)
        sequences = w.walk(self.num_walks, self.walk_length, self.multi_process)
        return sequences
