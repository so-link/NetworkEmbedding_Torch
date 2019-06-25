from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map

cdef extern from "cpp/graph.hpp" namespace "network_embedding" nogil:
    ctypedef int Node
    ctypedef pair[Node, Node] Edge
    ctypedef vector[Node] NodeList
    ctypedef unordered_set[Node] NodeSet

cdef extern from "cpp/graph.hpp" namespace "network_embedding" nogil:
    cdef cppclass Graph:
        cppclass NodeView:
            cppclass Iterator:
                Iterator() except +
                Iterator(const Iterator & it) except +
                Iterator & operator=(const Iterator & it)
                bint operator==(const Iterator& it)
                bint operator!=(const Iterator& it)
                Iterator& operator++()
                Iterator operator++(int)
                const Node& operator*()

            NodeView() except +
            NodeView(const Graph * graph) except +
            Iterator begin()
            Iterator end()

        cppclass EdgeView:
            cppclass Iterator:
                Iterator() except +
                Iterator(const Iterator & it) except +
                Iterator & operator=(const Iterator & it)
                bint operator==(const Iterator& it)
                bint operator!=(const Iterator& it)
                Iterator& operator++()
                Iterator operator++(int)
                const Edge& operator*()

            EdgeView() except +
            EdgeView(const Graph * graph) except +
            Iterator begin()
            Iterator end()
        
        cppclass DirectedEdgeView:
            cppclass Iterator:
                Iterator() except +
                Iterator(const Iterator & it) except +
                Iterator & operator=(const Iterator & it)
                bint operator==(const Iterator& it)
                bint operator!=(const Iterator& it)
                Iterator& operator++()
                Iterator operator++(int)
                const Edge& operator*()

            DirectedEdgeView() except +
            DirectedEdgeView(const Graph * graph) except +
            Iterator begin()
            Iterator end()
        
        Graph() except +
        Graph & operator=(const Graph & g)
        void AddEdge(const Node& u, const Node& v, double weight)
        void RemoveEdge(const Node& u, const Node& v)
        NodeView nodes()
        EdgeView edges()
        DirectedEdgeView directed_edges()
        double weight(const Node& u, const Node& v)
        double weight(const Edge& edge)
        const NodeSet& neighhors(const Node& u)
        size_t number_of_nodes()
        size_t number_of_edges()
        size_t degree(const Node& u)

cdef extern from "cpp/walker.hpp" namespace "network_embedding" nogil:
    cdef cppclass Walker:
        Walker() except +
        void set_node_list(const NodeList& nodes)
        void SetTransitionWeights(const Node& node, const NodeList& neighbors, const vector[double]& weights)
        void InitDistributionsFromGraph(const Graph& graph, bint weighted)
        NodeList SimulateWalk(const Node& start_node, size_t walk_length)
        vector[NodeList] Walk(size_t num_walks, size_t walk_length, size_t num_threads)
    
    cdef cppclass BiasedWalker:
        BiasedWalker() except +
        void InitDistributionsFromGraph(const Graph& graph, double p, double q)
        NodeList SimulateWalk(const Node& start_node, size_t walk_length)
        vector[NodeList] Walk(size_t num_walks, size_t walk_length, size_t num_threads)

cdef extern from "cpp/sampling.hpp" namespace "network_embedding" nogil:
    vector[NodeList] WindowSampling(const vector[NodeList] & sequences, size_t window_size, double down_sampling, bool shuffle);
    vector[NodeList] SkipSampling(const vector[NodeList] & sequences, size_t distance, double down_sampling, bool shuffle);

cdef extern from "cpp/aco.hpp" namespace "network_embedding" nogil:
    Graph ACOWalk(const Graph & graph, size_t num_walks, size_t max_step, size_t num_iterations, double alpha, double evaporate, size_t num_threads)