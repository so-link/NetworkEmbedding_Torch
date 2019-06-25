#include <unordered_set>
#include <unordered_map>
#include <random>
#include <functional>

#include "graph.hpp"

#ifndef NETWORK_EMBEDDING_WALKER_H
#define NETWORK_EMBEDDING_WALKER_H

namespace network_embedding {

class ParallelWalker {
    public:

    virtual ~ParallelWalker() {}
    virtual const NodeList & get_node_list() const = 0;
    virtual NodeList SimulateWalk(const Node & start_node, std::size_t walk_length) = 0;
    virtual std::vector<NodeList> Walk(std::size_t num_walks, std::size_t walk_length, std::size_t num_threads);
};

class Walker : public ParallelWalker {
    public:

    Walker() {}

    virtual inline const NodeList & get_node_list() const {
        return node_list_;
    }

    void set_node_list(const NodeList & nodes);

    void InitDistributionsFromGraph(const Graph &graph, bool weighted);
    void SetTransitionWeights(const Node & node, const NodeList & neighbors, const std::vector<double> & weights);
    virtual NodeList SimulateWalk(const Node & start_node, std::size_t walk_length);

    private:

    NodeList node_list_;
    std::unordered_map<Node, std::pair<NodeList, std::discrete_distribution<int>>> distributions_;
};

class BiasedWalker : public ParallelWalker {

    public:

    BiasedWalker() {}

    virtual inline const NodeList & get_node_list() const {
        return node_list_;
    }

    void InitDistributionsFromGraph(const Graph &graph, double p, double q);

    virtual NodeList SimulateWalk(const Node & start_node, std::size_t walk_length);

    private:

    NodeList node_list_;
    std::unordered_map<Node, std::pair<NodeList, std::uniform_int_distribution<int>>> node_distributions_;
    std::unordered_map<Edge, std::pair<NodeList, std::discrete_distribution<int>>, edge_hash> edge_distributions_;
};

};

#endif
