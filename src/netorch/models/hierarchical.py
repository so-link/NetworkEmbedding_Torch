#coding:utf-8

import numpy as np
import networkx as nx
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

class ConcatPCAModel(object):

    def __init__(self, graph, dimension, Model, Coarsening, num_scales=4):
        self.original_graph = graph
        self.dimension = dimension
        self.Model = Model
        self.Coarsening = Coarsening
        self.num_scales = num_scales
        self.embeddings = np.zeros((graph.number_of_nodes(), 0))

    def dimension_reduction(self, embeddings, dimension):
        return PCA(n_components=dimension).fit_transform(embeddings)

    def train(self):
        dimension = self.dimension

        coarsening = self.Coarsening(self.original_graph)
        if len(coarsening.graphs)==1:
            coarsening.recursive_merge()

        graphs = coarsening.graphs
        mappings = coarsening.make_mappings_to_original_graph()

        prev_nodes = None
        indices = []
        for i, g in enumerate(graphs):
            if prev_nodes is None or g.number_of_nodes()<prev_nodes*0.97:
                indices.append(i)
                prev_nodes = g.number_of_nodes()

        num_scales = min(len(indices), self.num_scales)
        step = len(indices)/num_scales
        selected_indices = [int(np.ceil(i*step)) for i in range(num_scales)]
        selected_indices = [indices[i] for i in selected_indices]

        train_graphs = [graphs[index] for index in selected_indices]
        train_mappings = [mappings[index] for index in selected_indices]

        dimensions = [self.dimension for g in train_graphs]
        
        for i, (graph, mapping, dimension) in enumerate(zip(train_graphs, train_mappings, dimensions)):
            print('Training graph#{} #nodes={} #edges={}'.format(i, graph.number_of_nodes(), graph.number_of_edges()))
            rev_mapping = coarsening.reverse_mapping(mapping)
            model = self.Model(graph, dimension)
            results = model.train().get_embeddings()
            embeddings = np.ndarray(shape=(self.original_graph.number_of_nodes(), dimension))
            for node, super_node in rev_mapping.items():
                embeddings[node,:] = results[super_node,:]
            self.embeddings = np.concatenate([self.embeddings, embeddings], axis=1)

        self.embeddings = self.dimension_reduction(self.embeddings, self.dimension)
        
        return self
        
    def get_embeddings(self):
        return self.embeddings


def model_prolong(prev_model, new_model, mapping):
    prev_emb = prev_model.get_embeddings()
    prev_ctx = prev_model.get_contexts()

    new_emb = np.ndarray((new_model.graph.number_of_nodes(), new_model.dimension), dtype=np.float32)
    new_ctx = np.ndarray((new_model.graph.number_of_nodes(), new_model.dimension), dtype=np.float32)

    for super_node, nodes in mapping.items():
        for node in nodes:
            new_emb[node] = prev_emb[super_node]
            new_ctx[node] = prev_ctx[super_node]

    new_model.set_embeddings(new_emb)
    new_model.set_contexts(new_ctx)


class RecursiveProlong(object):

    def __init__(self, graph, dimension, Model, Coarsening):
        self.original_graph = graph
        self.dimension = dimension
        self.Model = Model
        self.Coarsening = Coarsening

    def train(self):
        coarsening = self.Coarsening(self.original_graph)
        if len(coarsening.graphs)==1:
            coarsening.recursive_merge()

        graphs = coarsening.graphs[::-1]
        mappings = coarsening.mappings
        mappings.append(None)
        mappings = mappings[::-1]

        prev_model = None
        for i, (graph, mapping) in enumerate(zip(graphs, mappings)):
            print('Training graph#{} #nodes={} #edges={}'.format(i, graph.number_of_nodes(), graph.number_of_edges()))
            model = self.Model(graph, self.dimension)
            if prev_model is not None:
                model_prolong(prev_model, model, mapping)
            if prev_model is None or prev_model.graph.number_of_nodes()<graph.number_of_nodes()*0.95 or graph.number_of_nodes()==self.original_graph.number_of_nodes():
                model.train()
            prev_model = model


        self.embeddings = prev_model.get_embeddings()
        return self

    def get_embeddings(self):
        return self.embeddings
            
