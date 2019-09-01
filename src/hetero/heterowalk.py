#coding:utf-8
import multiprocessing as mp
import networkx as nx
import numpy as np

from necython import Walker as CWalker

def trans_prob_normalization(u, neighbors, tags, weights):
    if len(neighbors)==0:
        return [u], [1.]
    tag_set = set(tags)
    neibor_weights = {t:{'v':[],'weight':[]} for t in tags}
    for v, t, w in zip(neighbors, tags, weights):
        neibor_weights[t]['v'].append(v)
        neibor_weights[t]['weight'].append(w)

    for t in tag_set:
        weights = np.array(neibor_weights[t]['weight'])
        neibor_weights[t]['weight'] = weights/np.sum(weights)
    
    ret_neibors = []
    ret_weights = []
    for t in tag_set:
        ret_neibors.extend(neibor_weights[t]['v'])
        ret_weights.extend(neibor_weights[t]['weight'])
    return ret_neibors, ret_weights

def init_walker(graph):
    walker = CWalker()
    walker.set_node_list(list(graph.nodes))
    trans_weights = {}
    for u in graph.nodes:
        neibors = list(graph.neighbors(u))
        tags = [graph.nodes[v]['tag'] for v in neibors]
        weights = [graph[u][v]['weight'] for v in neibors]
        neibors, weights = trans_prob_normalization(u, neibors, tags, weights)

        walker.set_transition_weights(u, neibors, weights)
        trans_weights[u] = {neibor:w for neibor, w in zip(neibors, weights)}
    return walker, trans_weights

def hetero_walk(graph, num_walks=10, walk_length=80):
    walker, _ = init_walker(graph)
    sequences = walker.walk(num_walks, walk_length, mp.cpu_count())
    return sequences

def hetero_walker(graph, num_walks=10, walk_length=80):
    walker, weights = init_walker(graph)
    def walk(start_node):
        return [walker.simulate_walk(start_node, walk_length) for i in range(num_walks)]
    def path_score(path):
        score = 1.0
        for i in range(1, len(path)):
            score *= weights[path[i-1]][path[i]]
        return score
    return walk, path_score


