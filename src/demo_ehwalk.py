#coding:utf-8
import random
from collections import Counter
from pprint import pprint
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from netorch.dataset import load_hetero_graph
from netorch.models.walkbased import Triplet, DeepWalk
from netorch.lookup import HeteroGraphLookup, GraphLookup
from netorch.models.walkbased.sampling import TripletSampling
from netorch.models.common import TripletNodeEmbedding
from hetero.heterowalk import hetero_walk, hetero_walker

DATASET_DOWNLOAD_LINK = 'https://so-link.org/xxxxxx/'

DATASET_DIR = 'datasets/scholar-graph'
NODES_TXT = '{}/nodes.txt'.format(DATASET_DIR)
EDGES_TXT = '{}/edges.txt'.format(DATASET_DIR)
GROUND_TRUTH_TXT = '{}/author_paper_label.txt'.format(DATASET_DIR)
RESULT_TXT = 'results.txt'

EMBEDDING_DIM = 128

static_subnetworks = ['conf','word']
dynamic_subnetworks = ['homepage','paper']

def check_dataset():
    try:
        open(GROUND_TRUTH_TXT)
    except:
        print('Scholar graph dataset not found.')
        print('Please download the dataset at {}'.format(DATASET_DOWNLOAD_LINK))
        return False
    return True

def sigmoid(mat):
    return 1. / (1. + np.exp(-mat))

def load_category_node(file_name):
    category_node = {}
    fp = open(file_name)
    for line in fp:
        node, category = line.strip().split()
        if category not in category_node:
            category_node[category] = []
        category_node[category].append(node)
    fp.close()
    return category_node

def load_author_paper_label(file_name):
    fp = open(file_name)
    lst = []
    for line in fp:
        author, paper, label = line.strip().split()
        lst.append((author, paper, label))
    return lst

def make_subnetwork(tags, graph):
    if isinstance(tags, list):
        def cond(node):
            return graph.nodes[node]['tag'] in tags
    else:
        def cond(node):
            return graph.nodes[node]['tag'] == tags

    return graph.subgraph(filter(lambda node: cond(node), graph.nodes))

def network_embedding(num_nodes, sequences, iterations=2, batch_size=10000, lr=0.0025):
    model = TripletNodeEmbedding(num_nodes, EMBEDDING_DIM, lr)
    sampler = TripletSampling(window_size=5, batch_size=batch_size, down_sampling=-1)
    for it in range(iterations):
        for samples in sampler.sample(sequences):
            model.feed(*samples)
        model.lr_decay()
    return model.get_embeddings()

def filter_subnetwork_sequences(tag, lookup, sequences):
    seqs = []
    for seq in sequences:
        seqs.append([lookup.g_index_to_t_index(node) for node in seq if lookup.g_index_to_tag(node)==tag])
    return seqs

def subnetwork_embedding(tag, lookup, sequences, iterations=5, batch_size=10000, lr=0.01):
    num_nodes = lookup.num_tag_nodes(tag)
    subnetwork_seqs = filter_subnetwork_sequences(tag, lookup, sequences)
    embeddings = network_embedding(num_nodes, subnetwork_seqs, iterations, batch_size, lr)
    embeddings_dict = {}
    for i in range(embeddings.shape[0]):
        embeddings_dict[lookup.t_index_to_g_index(tag, i)] = embeddings[i]
    return embeddings_dict

def visualize_embedding(tag, lookup, embedding, category_node):
    colors = ['red', 'pink', 'gold', 'green', 'blue', 'cyan', 'purple', 'black', 'gray']
    embedding_arr = np.ndarray((len(embedding), EMBEDDING_DIM))
    for node, emb in embedding.items():
        embedding_arr[lookup.g_index_to_t_index(node), :] = embedding[node]
    arr = TSNE().fit_transform(embedding_arr)
    for i, (category, nodes) in enumerate(category_node.items()):
        selected_tsne_vis = arr[[lookup.label_to_t_index(node) for node in nodes], :]
        plt.scatter(selected_tsne_vis[:,0], selected_tsne_vis[:,1], c=colors[i])
    save_file = '{}_tsne.png'.format(tag)
    print("TSNE saved file {}".format(save_file))
    plt.savefig(save_file)

def make_embedding_edges(tag, embedding, lookup, graph, min_degree):
    subnetwork = make_subnetwork(tag, graph)
    emb = np.ndarray((len(embedding), EMBEDDING_DIM))
    for i in range(emb.shape[0]):
        emb[i,:] = embedding[lookup.t_index_to_g_index(tag, i)]
    dot_prod = np.matmul(emb, np.transpose(emb))
    dot_prod = sigmoid(dot_prod)
    for node in subnetwork.nodes:
        if subnetwork.degree(node) > min_degree:
            for neibor in subnetwork.neighbors(node):
                graph[node][neibor]['weight'] = dot_prod[lookup.label_to_t_index(node)][lookup.label_to_t_index(node)]
        else:
            t_index = lookup.label_to_t_index(node)
            k_th_max = np.partition(dot_prod[t_index], min_degree)[min_degree]
            for i, val in enumerate(dot_prod[t_index]):
                if val > k_th_max:
                    neibor = lookup.t_index_to_label(tag, i)
                    if graph.has_edge(node, neibor):
                        graph[node][neibor]['weight'] = val
                    else:
                        graph.add_edge(node, neibor, weight=val)





def main():
    if not check_dataset():
        exit()
    print('loading')
    graph = load_hetero_graph(NODES_TXT, EDGES_TXT)
    ground_truth = load_author_paper_label(GROUND_TRUTH_TXT)

    print('build subgraph')
    g_static = make_subnetwork(static_subnetworks, graph)
    lookup_g_static = HeteroGraphLookup(g_static)
    g_n_static = nx.convert_node_labels_to_integers(g_static)

    print('heterogeneous random walk')
    walk_sequences = hetero_walk(g_n_static, num_walks=5, walk_length=80)

    for tag in static_subnetworks:
        print("train subnetwork {}".format(tag))
        embedding = subnetwork_embedding(tag, lookup_g_static, walk_sequences)
        tag_category = load_category_node('{}/{}_category.txt'.format(DATASET_DIR, tag))
        print("visualize embedding {}".format(tag))
        # visualize_embedding(tag, lookup_g_static, embedding, tag_category) # Very time-consuming. 
        #                                                                    # Comment this step to make the whole process faster.
        print("construct embedding network of {}".format(tag))
        make_embedding_edges(tag, embedding, lookup_g_static, graph, min_degree=50)

    lookup_g = HeteroGraphLookup(graph)
    graph_n = nx.convert_node_labels_to_integers(graph)
    walker = hetero_walker(graph_n, num_walks=1000, walk_length=10)
    output = open(RESULT_TXT, 'w')
    for author, paper, label in ground_truth:
        cnt = 0
        seqs = walker(lookup_g.label_to_g_index('#'+paper))
        for seq in seqs:
            cnt += seq.count(lookup_g.label_to_g_index('@'+author))
        output.write('{} {} {} {}\n'.format(author, paper, label, cnt/1000))
    output.close()



if __name__ == '__main__':
    main()
