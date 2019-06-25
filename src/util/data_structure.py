#coding:utf-8
from collections import defaultdict
import heapq
from pprint import pprint

class DisjoinSet(object):

    def __init__(self, size):
        self.size = size
        self.num_components = size
        self.arr = list(range(size))
    
    def union(self, u, v):
        u_root = self.find(u)
        v_root = self.find(v)
        if u_root!=v_root:
            self.arr[u_root] = v_root
            self.num_components -= 1

    def find(self, u):
        if self.arr[u]==u:
            return u
        u_root = self.find(self.arr[u])
        self.arr[u] = u_root
        return u_root

    # def make_clusters(self):
    #     arr = list(map(self.find, self.arr))
    #     clusters = []
    #     added = set()
    #     for index in arr:
    #         if index in added:
    #             continue
    #         added.add(index)
    #         clusters.append(index)
    #     cluster_index = {cluster:index for index, cluster in enumerate(clusters)}
    #     self.node_to_cluster = {node:cluster_index[cluster] for node, cluster in enumerate(arr)}
    #     self.cluster_to_nodes = {cluster_index[cluster]:[] for cluster in clusters}
    #     for node, cluster in enumerate(arr):
    #         self.cluster_to_nodes[cluster_index[cluster]].append(node)

    def make_mapping(self):
        arr = list(map(self.find, self.arr))
        mapping = defaultdict(list)
        for node, cluster in enumerate(arr):
            mapping[cluster].append(node)
        return list(mapping.values())

class HuffmanTreeNode(object):

    def __init__(self, weight, data=None, lchild=None, rchild=None):
        self.weight = weight
        self.data = data
        self.lchild=lchild
        self.rchild=rchild
    
    def is_leaf(self):
        return self.lchild is None and self.rchild is None

    def __lt__(self, node2):
        return self.weight<node2.weight

class HuffmanTree(object):

    def __init__(self, items, weights):
        pq = []
        for item, weight in zip(items, weights):
            pq.append(HuffmanTreeNode(weight=weight, data=item))
        heapq.heapify(pq)

        num = 0

        while len(pq)>1:
            node1 = heapq.heappop(pq)
            node2 = heapq.heappop(pq)
            new_node = HuffmanTreeNode(weight=node1.weight+node2.weight, data=num, lchild=node1, rchild=node2)
            heapq.heappush(pq, new_node)
            num += 1
        
        self.root = heapq.heappop(pq)

    def traverse_encode(self, cur_node, parent_list, sign_list, result):
        if cur_node.is_leaf():
            result[cur_node.data] = (
                tuple(parent_list),
                tuple(sign_list),
            )
            return

        self.traverse_encode(cur_node.lchild, parent_list+[cur_node.data], sign_list+[1], result)
        self.traverse_encode(cur_node.rchild, parent_list+[cur_node.data], sign_list+[-1], result)

    def huffman_encode(self):
        result = {}
        self.traverse_encode(self.root, [], [], result)
        return result

if __name__=='__main__':
    tree = HuffmanTree([0,1,2,3,4,5,6,7], [6,2,7,3,7,1,8,4])
    pprint(tree.huffman_encode())