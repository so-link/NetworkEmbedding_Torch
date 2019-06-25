#include "sampling.hpp"

#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <functional>

namespace network_embedding {

using std::vector;
using std::unordered_map;
using std::default_random_engine;
using std::uniform_real_distribution;
using std::min;
using std::max;
using std::function;
using std::cout;
using std::endl;

extern thread_local default_random_engine random_number_generator;

vector<NodeList> DownSampling(const vector<NodeList> & sequences, double threshold) {
    vector<NodeList> result;
    unordered_map<Node, double> probs;
    size_t sequences_size = 0;
    for (const auto & sequence : sequences) {
        for (const auto & node : sequence) {
            if (probs.find(node) == probs.end())
                probs[node] = 0.0;
            probs[node] += 1.0;
            sequences_size += 1;
        }
    }

    for (auto & node_prob : probs) {
        double & prob = node_prob.second;
        prob = prob / sequences_size;
        prob = sqrt(threshold/prob) + threshold/prob;
    }

    uniform_real_distribution<> dist(0,1);
    for (const auto & sequence : sequences) {
        NodeList filtered;
        for (const auto & node : sequence) {
            const double & prob = probs[node];
            if (dist(random_number_generator) - prob < 0) {
                filtered.push_back(node);
            }
        }
        if(filtered.size() > 2)
            result.push_back(filtered);
    }

    return result;
}

vector<NodeList> SlidingWindow(const vector<NodeList> & sequences, size_t window_size) {
    vector<NodeList> samples;
    int distance = (window_size+1)/2;
    for (const auto & seq : sequences) {
        for (int i = 0; i < seq.size(); i++) {
            int lb = max(0, i-distance), ub = min(int(seq.size()), i+distance+1);
            for(int j = lb; j < i; j++) {
                samples.push_back({seq[i], seq[j]});
            }
            for(int j = i+1; j < ub; j++) {
                samples.push_back({seq[i], seq[j]});
            }
        }
    }
    return samples;
}

vector<NodeList> Skipping(const vector<NodeList> & sequences, size_t distance) {
    vector<NodeList> samples;
    for (const auto & seq : sequences) {
        size_t ub = seq.size()-distance;
        for (size_t i = 0; i < ub; i++) {
            samples.push_back({seq[i], seq[i+distance]});
        }
    }
    return samples;
}

vector<NodeList> SequencesProcessingPipeline(const vector<NodeList> & sequences, const vector<function<void(vector<NodeList>&)>> & pipeline) {
    vector<NodeList> sequences_buffer = sequences;
    for (auto & func : pipeline) {
        func(sequences_buffer);
    }
    return sequences_buffer;
}

vector<NodeList> WindowSampling(const vector<NodeList> & sequences, size_t window_size, double down_sampling, bool shuffle) {
    vector<function<void(vector<NodeList>&)>> pipeline;

    if (down_sampling > 0) {
        pipeline.push_back(
            [&](vector<NodeList> & seq) -> void {
                auto buffer = DownSampling(seq, down_sampling);
                seq = move(buffer);
            }
        );
    }

    pipeline.push_back(
        [&](vector<NodeList> & seq) -> void {
            auto buffer = SlidingWindow(seq, window_size);
            seq = move(buffer);
        }
    );

    if (shuffle) {
        pipeline.push_back(
            [&](vector<NodeList> & seq) ->void {
                random_shuffle(seq.begin(), seq.end());
            }
        );
    }

    return SequencesProcessingPipeline(sequences, pipeline);
}

// vector<NodeList> WindowSampling(const vector<NodeList> & sequences, size_t window_size, double down_sampling, bool shuffle) {
//     return SlidingWindow(sequences, window_size);
// }

vector<NodeList> SkipSampling(const vector<NodeList> & sequences, size_t distance, double down_sampling, bool shuffle) {
    vector<function<void(vector<NodeList>&)>> pipeline;

    if (down_sampling > 0) {
        pipeline.push_back(
            [&](vector<NodeList> & seq) -> void {
                seq = move(DownSampling(seq, down_sampling));
            }
        );
    }

    pipeline.push_back(
        [&](vector<NodeList> & seq) -> void {
            seq = move(Skipping(seq, distance));
        }
    );

    if (shuffle) {
        pipeline.push_back(
            [&](vector<NodeList> & seq) ->void {
                random_shuffle(seq.begin(), seq.end());
            }
        );
    }

    return SequencesProcessingPipeline(sequences, pipeline);
}

};