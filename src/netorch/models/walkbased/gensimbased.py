#coding:utf-8
import multiprocessing as mp
from gensim.models import Word2Vec
from .walker import Walker
import numpy as np
class DeepWalk(object):
    
    def __init__(self, graph, *,
            dimension = 128,
            num_walks = 10,
            walk_length = 80,
            window_size = 10,
            iterations = 3):
        self.graph = graph
        self.dimension = dimension
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.window_size = window_size
        self.iterations = iterations

    def train(self):
        sequences = Walker(self.num_walks, self.walk_length).walk(self.graph)
        sequences = list(map(
            lambda seq: list(map(lambda node: str(node), seq)),
            sequences,
        ))
        model = Word2Vec(sequences, size=self.dimension, workers=mp.cpu_count(), hs=1, iter=self.iterations, window=self.window_size)
        embeddings = np.ndarray((self.graph.number_of_nodes(), self.dimension))
        for i in range(embeddings.shape[0]):
            embeddings[i] = model[str(i)]
        self.embeddings = embeddings
        return self

    def get_embeddings(self):
        return self.embeddings

