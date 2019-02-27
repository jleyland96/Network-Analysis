#na3pa.py
#matthew johnson 19 january 2017

#####################################################

"""the following code creates a PA graph and is based on code from
http://www.codeskulptor.org/#alg_dpa_trial.py; """

import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import approximation
from collections import defaultdict
from networkx.generators.classic import empty_graph, complete_graph
import sys


def _random_subset(seq,m):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets=set()
    while len(targets)<m:
        x=random.choice(seq)
        targets.add(x)
    return targets

def barabasi_albert_graph(n, m, seed=None):
    """Return random graph using Barabási-Albert preferential attachment model.

    A graph of n nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Notes
    -----
    The initialization is a graph with with m nodes and no edges.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    if m < 1 or  m >= n:
        raise nx.NetworkXError(\
              "Barabási-Albert network must have m>=1 and m<n, m=%d,n=%d" % (m, n))
    if seed is not None:
        random.seed(seed)

    # Add m initial nodes (m0 in barabasi-speak)
    G = complete_graph(m)
    # # # # G = empty_graph(m) # I REMOVED THIS
    G.name = "barabasi_albert_graph(%s,%s)" % (n, m)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    for i in range(0, m):
        repeated_nodes += [i]*(m-1)
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source]*m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source]*m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        targets = _random_subset(repeated_nodes, m)
        source += 1
    return G

#first we need
class PATrial:
    """
    Used when each new node is added in creation of a PA graph.
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are in proportion to the
    probability that it is linked to.
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a PATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities
        
        Returns:
        Set of nodes
        """       
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        # update the list of node numbers so that each node number 
        # appears in the correct ratio

        # add num_nodes (= new node number) to list
        self._node_numbers.append(self._num_nodes)

        # add new neighbours to the list
        self._node_numbers.extend(list(new_node_neighbors))

        # update the number of nodes
        self._num_nodes += 1

        return new_node_neighbors
    
def make_complete_graph(num_nodes):
    """Takes the number of nodes num_nodes and returns a dictionary
    corresponding to a complete directed graph with the specified number of
    nodes. A complete graph contains all possible edges subject to the
    restriction that self-loops are not allowed. The nodes of the graph should
    be numbered 0 to num_nodes - 1 when num_nodes is positive. Otherwise, the
    function returns a dictionary corresponding to the empty graph."""
    # initialize empty graph
    complete_graph = {}
    # consider each vertex
    for vertex in range(num_nodes):
        # add vertex with list of neighbours
        complete_graph[vertex] = set([j for j in range(num_nodes) if j != vertex])
    return complete_graph
    
def make_PA_Graph(total_nodes, out_degree):
    """creates a PA_Graph on total_nodes where each vertex is iteratively
    connected to a number of existing nodes equal to out_degree"""
    # initialize graph by creating complete graph and trial object
    pa_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)

    # total_nodes - out_degree new nodes
    for vertex in range(out_degree, total_nodes):
        # add edges from vertex to neighbours
        pa_graph[vertex] = trial.run_trial(out_degree)
    return pa_graph

def compute_in_degrees(digraph):
    """Takes a directed graph and computes the in-degrees for the nodes in the
    graph. Returns a dictionary with the same set of keys (nodes) and the
    values are the in-degrees."""
    # initialize in-degrees dictionary with zero values for all vertices
    in_degree = {}
    for vertex in digraph:
        in_degree[vertex] = 0
    # consider each vertex
    for vertex in digraph:
        # amend in_degree[w] for each outgoing edge from v to w
        for neighbour in digraph[vertex]:
            in_degree[neighbour] += 1
    return in_degree

def in_degree_distribution(digraph):
    """Takes a directed graph and computes the unnormalized distribution of the
    in-degrees of the graph.  Returns a dictionary whose keys correspond to
    in-degrees of nodes in the graph and values are the number of nodes with
    that in-degree. In-degrees with no corresponding nodes in the graph are not
    included in the dictionary."""
    # find in_degrees
    in_degree = compute_in_degrees(digraph)
    # initialize dictionary for degree distribution
    degree_distribution = {}
    # consider each vertex
    for vertex in in_degree:
        # update degree_distribution
        if in_degree[vertex] in degree_distribution:
            degree_distribution[in_degree[vertex]] += 1
        else:
            degree_distribution[in_degree[vertex]] = 1
    return degree_distribution

def vertex_brilliance(graph, centre):
    # returns the largest k such that vertex is the centre of a k-star

    # create subgraph consisting of the neighbors of vertex
    neighbours = []
    edges = []

    # add each neighbour of the centre vertex to a subset of vertices
    for n in graph[centre]:
        neighbours += [n]

    # add all the edges to this graph
    for n1 in neighbours:
        for n2 in neighbours:
            if n2 in graph[n1]:
                if (n1, n2) not in edges:
                    edges += [(n1, n2)]
                if (n2, n1) not in edges:
                    edges += [(n2, n1)]

    # create the networkx equivalent so we can use the independent set function
    G = nx.Graph()
    G.add_nodes_from(neighbours)
    G.add_edges_from(edges)

    # we want to find the largest independent set of centre's neighbors
    largest_independent_set = nx.algorithms.approximation.maximum_independent_set(G)
    return len(largest_independent_set)


def investigate_brilliance():
    MAX_LOOPS = 1
    brilliance_distribution = defaultdict(int)

    for i in range(MAX_LOOPS):
        print("making graph...")
        my_graph = make_PA_Graph(1559, 30)
        print("made graph...")

        # check the brilliance of each node
        brilliance = defaultdict(int)

        for vertex in my_graph:
            brilliance[vertex] = vertex_brilliance(my_graph, vertex)
            print("brilliance of", vertex, "is", brilliance[vertex])
            # print("vertex", vertex, "has brilliance", brilliance[vertex])
            if vertex % 100 == 0:
                print(vertex)

        for v in brilliance:
            brilliance_distribution[brilliance[v]] += 1/(1559*MAX_LOOPS)  # Or is it 1/MAX_LOOPS

        print(brilliance)
        print(brilliance_distribution)
        print("loop", i+1, "complete")
        print()

    x_data = []
    y_data = []
    for b in brilliance_distribution:
        x_data += [b]
        y_data += [brilliance_distribution[b]]

    plt.xlabel('brilliance')
    plt.ylabel('frequency')
    plt.title('distribution of vertex brilliance in PA graphs')
    plt.plot(x_data, y_data, marker='.', linestyle='None', color='b')
    plt.savefig('brilliance graph - PA - ' + str(MAX_LOOPS) + ' .png')


def new_brilliance_methods(nx_graph, centre):
    temp_G = nx.Graph()

    for n1 in nx_graph[centre]:
        temp_G.add_node(n1)
        # print("NODE", n1)
        for n2 in nx_graph[n1]:
            if n2 in nx_graph[centre]:
                # print("EDGE", (n1, n2))
                temp_G.add_edge(n1, n2)

    largest_set = nx.algorithms.approximation.maximum_independent_set(temp_G)
    return len(largest_set)


MAX_LOOPS = 10
brilliance_dict = defaultdict(int)

for i in range(MAX_LOOPS):
    my_G = barabasi_albert_graph(1559, 30)  # 1559, 30
    print("\nLOOP", i+1)

    for vertex in my_G:
        b = new_brilliance_methods(my_G, vertex)
        print("vertex", vertex, "has brilliance", b)
        brilliance_dict[b] += 1/MAX_LOOPS               # or 1/MAX_LOOPS*1559 when normalizing

x_data = []
y_data = []
for brill in brilliance_dict:
    x_data += [brill]
    y_data += [brilliance_dict[brill]]

# plot degree distribution
plt.xlabel('Brilliance')
plt.ylabel('Frequency')
plt.title('Brilliance distribution in PA graph (n=1559, m=30)')
plt.plot(x_data, y_data, marker='.', linestyle='None', color='b')
plt.savefig('pa_brilliance_new_normalised_PLOT.png')
plt.show()


