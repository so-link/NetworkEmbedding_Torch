#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cmath>

#include "graph.hpp"
#include "walker.hpp"

#ifndef NETWORK_EMBEDDING_ACO_H
#define NETWORK_EMBEDDING_ACO_H

namespace network_embedding {

Graph ACOWalk(
        const Graph & graph,
        std::size_t num_walks,
        std::size_t max_step,
        std::size_t num_iterations,
        double alpha,
        double evaporate,
        std::size_t num_threads);

Graph ACOWalkWithLabel(
        const Graph & graph,
        const std::unordered_map<Node, std::vector<int>> labels,
        std::size_t num_walks,
        std::size_t max_step,
        std::size_t num_iterations,
        double alpha,
        double evaporate,
        std::size_t num_threads);

};

#endif // NETWORK_EMBEDDING_ACO_H