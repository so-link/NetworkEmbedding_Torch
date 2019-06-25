#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <functional>

#include "graph.hpp"

#ifndef NETWORK_EMBEDDING_SAMPLING_H
#define NETWORK_EMBEDDING_SAMPLING_H

namespace network_embedding {

std::vector<NodeList> DownSampling(const std::vector<NodeList> & sequences, double threshold);
std::vector<NodeList> SlidingWindow(const std::vector<NodeList> & sequences, std::size_t window_size);
std::vector<NodeList> Skipping(const std::vector<NodeList> & sequences, std::size_t distance);
std::vector<NodeList> SequencesProcessingPipeline(const std::vector<NodeList> & sequences, const std::vector<std::function<void(std::vector<NodeList>&)>> & pipeline);
std::vector<NodeList> WindowSampling(const std::vector<NodeList> & sequences, std::size_t window_size, double down_sampling, bool shuffle);
std::vector<NodeList> SkipSampling(const std::vector<NodeList> & sequences, std::size_t distance, double down_sampling, bool shuffle);

}; // namespace network_embedding

#endif // NETWORK_EMBEDDING_SAMPLING_h