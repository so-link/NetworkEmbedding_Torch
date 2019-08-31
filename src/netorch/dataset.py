#coding:utf-8

#coding:utf-8

import networkx as nx

def load_edgelist(file_name):
    graph = nx.Graph()
    edges = []
    with open(file_name) as file:
        for line in file:
            if line.startswith('#'):
                continue
            l = line.strip().split()
            if len(l)==2:
                u, v = l
                edges.append((u, v, 1))
            else:
                u, v, weight = l
                edges.append((u, v, float(weight)))

    for u, v, weight in edges:
        if not graph.has_edge(u, v):
            graph.add_edge(u, v, weight=weight)
        else:
            graph[u][v]['weight'] += weight

    return graph

def save_embedding(embeddings, file_name):
    with open(file_name, 'w') as file:
        for key, val in embeddings.items():
            file.write('{} {}\n'.format(
                key,
                ' '.join(list(map(str, val))),
            ))

def load_labels(file_name):
    labels = {}
    with open(file_name) as file:
        for line in file:
            l = line.strip().split()
            node = l[0]
            labels[node] = [int(label) for label in l[1:]]
    return labels

def load_dataset(dataset, dataset_dir='datasets'):
    graph = load_edgelist('{}/{}_edgelist.txt'.format(dataset_dir, dataset))
    labels = load_labels('{}/{}_labels.txt'.format(dataset_dir, dataset))
    return graph, labels

def load_hetero_nodes(file_name):
    nodes = []
    with open(file_name) as file:
        for line in file:
            node, tag = line.strip().split()
            nodes.append((node, tag))
    return nodes

def load_hetero_edges(file_name):
    edges = []
    with open(file_name) as file:
        for line in file:
            l = line.strip().split()
            if len(l)==2:
                edges.append((l[0], l[1], 1.0))
            else:
                edges.append((l[0], l[1], float(l[2])))
    return edges

def load_hetero_graph(nodes_file_name, edges_file_name):
    nodes = load_hetero_nodes(nodes_file_name)
    edges = load_hetero_edges(edges_file_name)

    graph = nx.Graph()

    for node, tag in nodes:
        graph.add_node(node, tag=tag)
    
    for u, v, weight in edges:
        graph.add_edge(u, v, weight=weight)
    
    return graph
