#coding:utf-8

import numpy as np
import torch
import networkx as nx
from sklearn.preprocessing import normalize

from netorch.lookup import GraphLookup
from netorch.models.common import NodeEmbedding
from .sampling import NegativeSampling
from .walker import Walker, BiasedWalker

class WalkBasedEmbedding(object):

    def __init__(self, graph, dimension, iterations, walker, sampler, model):
        self.graph = graph
        self.dimension = dimension
        self.iterations = iterations

        self.walker = walker
        self.sampler = sampler
        self.model = model

    def train(self):
        sequences = self.walker.walk(self.graph)

        for it in range(self.iterations):
            for samples in self.sampler.sample(sequences):
                self.model.feed(*samples)
            self.model.lr_decay()
        
        return self
        
    def set_embeddings(self, embeddings):
        self.model.set_embeddings(embeddings)

    def get_embeddings(self):
        return self.model.get_embeddings()

    def set_contexts(self, contexts):
        self.model.set_contexts(contexts)

    def get_contexts(self):
        return self.model.get_contexts()

def DeepWalk(graph, *,
        dimension = 128,
        num_walks = 10,
        walk_length = 80,
        window_size = 10,
        iterations = 3,
        neg_ratio = 5,
        learning_rate = 0.001,
        batch_size = 10000,
        down_sample_threshold = 1e-3,
        weighted_walk=False):
    return WalkBasedEmbedding(graph,
        dimension,
        iterations,
        walker = Walker(num_walks, walk_length, weighted=weighted_walk),
        sampler = NegativeSampling(window_size, batch_size, neg_ratio=neg_ratio, down_sampling=down_sample_threshold),
        model = NodeEmbedding(graph.number_of_nodes(), dimension, learning_rate),
    )

def Node2Vec(graph, *, p, q,
        dimension = 128,
        num_walks = 10,
        walk_length = 80,
        window_size = 10,
        iterations = 3,
        neg_ratio = 5,
        learning_rate = 0.001,
        batch_size = 10000,
        down_sample_threshold = 1e-3):
    return WalkBasedEmbedding(graph,
        dimension,
        iterations,
        walker = BiasedWalker(num_walks, walk_length, p=p, q=q),
        sampler = NegativeSampling(window_size, batch_size, neg_ratio=neg_ratio, down_sampling=down_sample_threshold),
        model = NodeEmbedding(graph.number_of_nodes(), dimension, learning_rate),
    )

def Triplet(graph,*,
        dimension = 128,
        num_walks = 10,
        walk_length = 80,
        window_size = 10,
        iterations = 3,
        learning_rate = 0.001,
        batch_size = 10000,
        down_sample_threshold = 1e-3):
    return WalkBasedEmbedding(graph,
        dimension,
        iterations,
        walker = Walker(num_walks, walk_length),
        sampler = TripletSampling(window_size, batch_size, down_sampling=down_sample_threshold),
        model = TripletNodeEmbedding(graph.number_of_nodes(), dimension, learning_rate),
    )

