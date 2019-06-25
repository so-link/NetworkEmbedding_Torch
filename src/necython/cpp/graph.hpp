#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <algorithm>

#ifndef NETWORK_EMBEDDING_GRAPH_H
#define NETWORK_EMBEDDING_GRAPH_H

namespace network_embedding {

using Node = int;
using Edge = std::pair<Node, Node>;

struct edge_hash {
    std::size_t operator() (const Edge &p) const {
        auto h1 = std::hash<int>{}(p.first);
        auto h2 = std::hash<int>{}(p.second);
        return h1^(h2 + 0x9e3779b9 + (h1<<6) + (h1>>2));
    }
};

using NodeList = std::vector<Node>;
using NodeSet = std::unordered_set<Node>;
using AdjacencyList = std::unordered_map<Node, NodeSet>;
using EdgeWeight = std::unordered_map<Edge, double, edge_hash>;

class Graph {
    public:

    class NodeView {
        public:

        NodeView() : graph_(nullptr) {}

        NodeView(const Graph * graph) : graph_(graph) {}

        class Iterator {
            public:

            Iterator() : node_view_(nullptr) {}

            Iterator(const Iterator & it) : node_view_(it.node_view_), adjacency_list_iterator_(it.adjacency_list_iterator_) {}

            Iterator(const NodeView * node_view, const AdjacencyList::const_iterator & it) : node_view_(node_view), adjacency_list_iterator_(it) {}

            inline Iterator & operator=(const Iterator &rhs) {
                node_view_ = rhs.node_view_;
                adjacency_list_iterator_ = rhs.adjacency_list_iterator_;
                return *this;
            }

            inline bool operator==(const Iterator &rhs) const {
                return adjacency_list_iterator_ == rhs.adjacency_list_iterator_;
            }

            inline bool operator!=(const Iterator &rhs) const {
                return adjacency_list_iterator_ != rhs.adjacency_list_iterator_;
            }

            inline Iterator & operator++ () {
                ++adjacency_list_iterator_;
                return *this;
            }

            inline Iterator operator++(int) {
                Iterator tmp(*this);
                ++adjacency_list_iterator_;
                return tmp;
            }

            inline const Node & operator*() const {
                return adjacency_list_iterator_->first;
            }
        
            private:
            
            const NodeView * node_view_;
            AdjacencyList::const_iterator adjacency_list_iterator_;
        };

        inline Iterator begin() const {
            return Iterator(this, graph_->adjacency_list_.cbegin());
        }

        inline Iterator end() const {
            return Iterator(this, graph_->adjacency_list_.cend());
        }

        private:

        const Graph * graph_;
    }; // class Graph::NodeView

    class EdgeView {
        public:

        EdgeView() : graph_(nullptr) {}

        EdgeView(const Graph * graph) : graph_(graph) {}

        class Iterator {
            public:

            Iterator() : edge_view_(nullptr) {}

            Iterator(const Iterator & it) : edge_view_(it.edge_view_), edge_weight_iterator_(it.edge_weight_iterator_) {}

            Iterator(const EdgeView * edge_view, const EdgeWeight::const_iterator & it) : edge_view_(edge_view), edge_weight_iterator_(it) {}

            inline Iterator & operator=(const Iterator &rhs) {
                edge_weight_iterator_ = rhs.edge_weight_iterator_;
                return *this;
            }

            inline bool operator==(const Iterator &rhs) const {
                return edge_weight_iterator_ == rhs.edge_weight_iterator_;
            }

            inline bool operator!=(const Iterator &rhs) const {
                return edge_weight_iterator_ != rhs.edge_weight_iterator_;
            }

            inline Iterator & operator++ () {
                while (true) {
                    ++edge_weight_iterator_;
                    if (edge_weight_iterator_ == edge_view_->graph_->edge_weight_.cend())
                        break;
                    const auto & edge = edge_weight_iterator_->first;
                    const Node & u = edge.first;
                    const Node & v = edge.second;
                    if (u < v)
                        break;
                }
                return *this;
            }

            inline Iterator operator++(int) {
                Iterator tmp(*this);
                while (true) {
                    ++edge_weight_iterator_;
                    if (edge_weight_iterator_ == edge_view_->graph_->edge_weight_.cend())
                        break;
                    const auto & edge = edge_weight_iterator_->first;
                    const Node & u = edge.first;
                    const Node & v = edge.second;
                    if (u < v)
                        break;
                }
                return tmp;
            }

            inline const Edge & operator*() const {
                return edge_weight_iterator_->first;
            }
        
            private:
            
            const EdgeView * edge_view_;
            EdgeWeight::const_iterator edge_weight_iterator_;
        };

        inline Iterator begin() {
            return Iterator(this, graph_->edge_weight_.cbegin());
        }

        inline Iterator end() {
            return Iterator(this, graph_->edge_weight_.cend());
        }

        private:

        const Graph * graph_;
    }; // class Graph::EdgeView

    class DirectedEdgeView {
        public:

        DirectedEdgeView() : graph_(nullptr) {}

        DirectedEdgeView(const Graph * graph) : graph_(graph) {}

        class Iterator {
            public:

            Iterator() : directed_edge_view_(nullptr) {}

            Iterator(const Iterator & it) :
                directed_edge_view_(it.directed_edge_view_),
                edge_weight_iterator_(it.edge_weight_iterator_) {}

            Iterator(const DirectedEdgeView * directed_edge_view, const EdgeWeight::const_iterator & it) :
                directed_edge_view_(directed_edge_view),
                edge_weight_iterator_(it) {}

            inline Iterator & operator=(const Iterator &rhs) {
                edge_weight_iterator_ = rhs.edge_weight_iterator_;
                return *this;
            }

            inline bool operator==(const Iterator &rhs) const {
                return edge_weight_iterator_ == rhs.edge_weight_iterator_;
            }

            inline bool operator!=(const Iterator &rhs) const {
                return edge_weight_iterator_ != rhs.edge_weight_iterator_;
            }

            inline Iterator & operator++ () {
                ++edge_weight_iterator_;
                return *this;
            }

            inline Iterator operator++(int) {
                Iterator tmp(*this);
                ++edge_weight_iterator_;
                return tmp;
            }

            inline const Edge & operator*() const {
                return edge_weight_iterator_->first;
            }
        
            private:
            
            const DirectedEdgeView * directed_edge_view_;
            EdgeWeight::const_iterator edge_weight_iterator_;
        };

        inline Iterator begin() {
            return Iterator(this, graph_->edge_weight_.cbegin());
        }

        inline Iterator end() {
            return Iterator(this, graph_->edge_weight_.cend());
        }

        private:

        const Graph * graph_;
    }; // class Graph::DirectedEdgeView
    
    Graph() {}

    Graph(const Graph & g) : adjacency_list_(g.adjacency_list_), edge_weight_(g.edge_weight_) {}

    Graph & operator=(const Graph & g) {
        adjacency_list_ = g.adjacency_list_;
        edge_weight_ = g.edge_weight_;
        return *this;
    }

    inline void AddEdge(const Node & u, const Node & v, double weight) {
        adjacency_list_[u].insert(v);
        adjacency_list_[v].insert(u);
        edge_weight_[{u,v}] = weight;
        edge_weight_[{v,u}] = weight;
    }

    inline void RemoveEdge(const Node & u, const Node & v) {
        adjacency_list_[u].erase(v);
        adjacency_list_[v].erase(u);
        edge_weight_.erase({u,v});
        edge_weight_.erase({v,u});
    }

    inline void SetEdgeWeight(const Node & u, const Node & v, double weight) {
        edge_weight_.at({u,v}) = weight;
        edge_weight_.at({v,u}) = weight;
    }

    inline void SetEdgeWeight(const Edge & edge, double weight) {
        edge_weight_.at(edge) = weight;
        edge_weight_.at({edge.second, edge.first}) = weight;
    }

    inline NodeView nodes() const {
        return NodeView(this);
    }

    inline EdgeView edges() const {
        return EdgeView(this);
    }

    inline DirectedEdgeView directed_edges() const {
        return DirectedEdgeView(this);
    }

    inline double weight(const Node & u, const Node & v) const {
        return edge_weight_.at({u,v});
    }
    inline double weight(const Edge & edge) const {
        return edge_weight_.at(edge);
    }

    inline const NodeSet & neighbors(const Node & u) const {
        return adjacency_list_.at(u);
    }

    inline std::size_t number_of_nodes() const {
        return adjacency_list_.size();
    }

    inline std::size_t number_of_edges() const {
        return edge_weight_.size()/2;
    }

    inline std::size_t degree(const Node & u) const {
        return adjacency_list_.at(u).size();
    }

    private:
    
    AdjacencyList adjacency_list_;
    EdgeWeight edge_weight_;
}; // class Graph

}; // namespace network_embedding

#endif // NETWORK_EMBEDDING_GRAPH_H