#include "walker.hpp"

#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <thread>
#include <functional>
#include <random>

#include "graph.hpp"

namespace network_embedding {

using std::size_t;
using std::vector;
using std::sqrt;
using std::default_random_engine;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::discrete_distribution;
using std::round;
using std::unordered_map;
using std::unordered_set;
using std::get;
using std::move;
using std::hash;
using std::pair;
using std::for_each;
using std::thread;
using std::min;
using std::max;
using std::random_shuffle;
using std::function;

extern thread_local default_random_engine random_number_generator;

vector<NodeList> ParallelWalker::Walk(size_t num_walks, size_t walk_length, size_t num_threads) {
    const NodeList & node_list = get_node_list();
    size_t num_nodes = node_list.size();

    vector<NodeList> sequences(num_nodes*num_walks);

    if (num_nodes <= num_threads)
        num_threads = 1;

    vector<thread> threads;
    for (size_t i = 0; i < num_threads; i++) {
        size_t start_idx = static_cast<size_t>(round(static_cast<double>(num_nodes)/num_threads*i));
        size_t end_idx = static_cast<size_t>(round(static_cast<double>(num_nodes)/num_threads*(i+1)));

        threads.push_back(thread([this, start_idx, end_idx, num_walks, walk_length, &node_list, &sequences]{
            for (size_t u = start_idx; u < end_idx; u++) {
                for (size_t w = 0; w < num_walks; w++) {
                    sequences[u*num_walks+w] = move(SimulateWalk(node_list[u], walk_length));
                }
            }
        }));
    }

    for (auto &t : threads) {
        t.join();
    }

    return sequences;
}

void Walker::set_node_list(const NodeList & nodes) {
    node_list_.clear();
    for (auto && node : nodes)
        node_list_.push_back(node);
}

void Walker::SetTransitionWeights(const Node & node, const NodeList & neighbors, const vector<double> & weights) {
    distributions_[node] = {neighbors, discrete_distribution<int>(weights.begin(), weights.end())};
}

void Walker::InitDistributionsFromGraph(const Graph &graph, bool weighted) {
    auto make_node_distribution = [weighted, &graph] (const Node & u) -> pair<NodeList, discrete_distribution<int>> {
        const NodeSet & n_u = graph.neighbors(u);
        NodeList neighbors(n_u.begin(), n_u.end());
        vector<double> weights;

        if (weighted) {
            for (const auto & v : neighbors) {
                weights.push_back(graph.weight(u, v));
            }
        }
        else {
            for (const auto & v : neighbors) {
                weights.push_back(1.);
            }
        }
        return {neighbors, discrete_distribution<int>(weights.begin(), weights.end())};
    };

    node_list_.clear();
    distributions_.clear();
    for (auto && node : graph.nodes()) {
        node_list_.push_back(node);
        distributions_[node] = move(make_node_distribution(node));
    }
}

NodeList Walker::SimulateWalk(const Node & start_node, size_t walk_length) {
    NodeList seq;
    int curr_node = start_node;
    seq.push_back(curr_node);
    for (size_t i = 0; i < walk_length; i++) {
        auto &val = distributions_.at(curr_node);
        auto &neibors = val.first;
        auto &dist = val.second;

        int next_node = neibors[dist(random_number_generator)];
        seq.push_back(next_node);
        curr_node = next_node;
    }
    return seq;
}

void BiasedWalker::InitDistributionsFromGraph(const Graph &graph, double p, double q) {
    auto make_node_distribution = [&graph] (const Node & u) -> pair<NodeList, uniform_int_distribution<int>> {
        const NodeSet & n_u = graph.neighbors(u);
        NodeList neighbors(n_u.begin(), n_u.end());
        return {neighbors, uniform_int_distribution<int>(0, neighbors.size()-1)};
    };

    auto make_edge_distribution = [p, q, &graph] (const Edge & edge) -> pair<NodeList, discrete_distribution<int>> {
        const Node & u = edge.first;
        const Node & v = edge.second;

        const NodeSet & n_u = graph.neighbors(u);
        const NodeSet & n_v = graph.neighbors(v);

        NodeList nodes;
        vector<double> weights;

        nodes.push_back(u);
        weights.push_back(1.0/p);

        for (const auto &node : n_v) {
            double weight;
            if (n_u.find(node) != n_u.end())
                weight = 1.0;
            else
                weight = 1.0/q;
            nodes.push_back(node);
            weights.push_back(weight);
        }

        return {nodes, discrete_distribution<int>(begin(weights), end(weights))};
    };

    node_list_.clear();
    node_distributions_.clear();
    edge_distributions_.clear();

    auto nodes = graph.nodes();
    auto edges = graph.directed_edges();

    for (const auto & node : nodes) {
        node_list_.push_back(node);
        node_distributions_[node] = move(make_node_distribution(node));
    }

    for (const auto & edge : edges) {
        edge_distributions_[edge] = move(make_edge_distribution(edge));
    }
}

NodeList BiasedWalker::SimulateWalk(const Node & start_node, size_t walk_length) {
    NodeList seq;

    auto &v = node_distributions_.at(start_node);
    auto &neibors = v.first;
    auto &dist = v.second;

    int prev_node = start_node, curr_node = neibors[dist(random_number_generator)];
    seq.push_back(prev_node);
    seq.push_back(curr_node);
    for (size_t i = 1; i < walk_length; i++) {
        auto &v = edge_distributions_.at({prev_node, curr_node});
        auto &neibors = v.first;
        auto &dist = v.second;

        int next_node = neibors[dist(random_number_generator)];
        seq.push_back(next_node);
        prev_node = curr_node;
        curr_node = next_node;
    }

    return seq;
}

};  // namespace network_embedding
