#include "aco.hpp"

#include <vector>
#include <unordered_map>

namespace network_embedding {

using std::size_t;
using std::vector;
using std::unordered_map;

static void CalculateLoopPheromone(
        const vector<NodeList> & sequences,
        size_t max_step,
        size_t num_threads,
        Graph & pheromone) {
    for (const auto & sequence : sequences) {
        for (size_t i = 0; i < sequence.size(); i++) {
            for (size_t len = 1; len < max_step && i+len < sequence.size(); len++) {
                if (sequence[i] == sequence[i+len]) {
                    for (size_t k = i; k < i+len; k++) {
                        const Node & u = sequence[k];
                        const Node & v = sequence[k+1];
                        pheromone.SetEdgeWeight(u, v, pheromone.weight(u, v) + 1.0/len);
                    }
                    break;
                }
            }
        }
    }
}

static void CalculatePheromoneWithLabel(
        const vector<NodeList> & sequences,
        const unordered_map<Node, vector<int>> labels,
        size_t max_step,
        size_t num_threads,
        Graph & pheromone) {
    for (const auto & sequence : sequences) {
        for (size_t i = 0; i < sequence.size(); i++) {
            for (size_t len = 1; len < max_step && i+len < sequence.size(); len++) {
                if (labels.find(sequence[i+len]) != labels.end()) {
                    for (size_t k = i; k < i+len; k++) {
                        const Node & u = sequence[k];
                        const Node & v = sequence[k+1];
                        pheromone.SetEdgeWeight(u, v, pheromone.weight(u, v) + 1.0/len);
                    }
                    break;
                }
            }
        }
    }
}

Graph ACOWalk(
        const Graph & graph,
        size_t num_walks,
        size_t max_step,
        size_t num_iterations,
        double alpha,
        double evaporate,
        size_t num_threads) {
    Graph g(graph);
    Graph pheromone(graph);
    Graph total_pheromone(graph);
    Graph mixed(graph);

    for (const auto & edge : g.edges()) {
        total_pheromone.SetEdgeWeight(edge, 0.0);
    }

    for (size_t i = 0; i < num_iterations; i++) {
        Walker walker;
        walker.InitDistributionsFromGraph(mixed, true);
        auto sequences = walker.Walk(num_walks, 80, num_threads);

        for (const auto & edge : g.edges()) {
            pheromone.SetEdgeWeight(edge, 0.0);
        }

        CalculateLoopPheromone(sequences, max_step, num_threads, pheromone);

        for (const auto & edge : g.edges()) {
            total_pheromone.SetEdgeWeight(edge, total_pheromone.weight(edge) * (1.0-evaporate) + pheromone.weight(edge));
        }

        for (const auto & edge : g.edges()) {
            mixed.SetEdgeWeight(edge, g.weight(edge) * pow(total_pheromone.weight(edge), alpha));
        }
    }

    for (const auto & edge : g.edges()) {
        total_pheromone.SetEdgeWeight(edge, pow(total_pheromone.weight(edge), alpha));
    }

    return total_pheromone;
}

Graph ACOWalkWithLabel(
        const Graph & graph,
        const unordered_map<Node, vector<int>> labels,
        size_t num_walks,
        size_t max_step,
        size_t num_iterations,
        double alpha,
        double evaporate,
        size_t num_threads) {

    Graph g(graph);
    Graph pheromone(graph);
    Graph total_pheromone(graph);
    Graph mixed(graph);

    for (const auto & edge : g.edges()) {
        total_pheromone.SetEdgeWeight(edge, 0.0);
    }

    for (size_t i = 0; i < num_iterations; i++) {
        Walker walker;
        walker.InitDistributionsFromGraph(mixed, true);
        auto sequences = walker.Walk(num_walks, 80, num_threads);

        for (const auto & edge : g.edges()) {
            pheromone.SetEdgeWeight(edge, 0.0);
        }

        CalculateLoopPheromone(sequences, max_step, num_threads, pheromone);
        CalculatePheromoneWithLabel(sequences, labels, max_step, num_threads, pheromone);

        for (const auto & edge : g.edges()) {
            total_pheromone.SetEdgeWeight(edge, total_pheromone.weight(edge) * (1.0-evaporate) + pheromone.weight(edge));
        }

        for (const auto & edge : g.edges()) {
            mixed.SetEdgeWeight(edge, g.weight(edge) * pow(total_pheromone.weight(edge), alpha));
        }
    }

    return total_pheromone;
}

};
