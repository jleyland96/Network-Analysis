import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from networkx.algorithms import approximation


def load_graph(graph_txt):
    file = open(graph_txt)
    graph = {}
    for vertex in range(1, 1560):
        graph[vertex] = set()

    line_count = 0
    for line in file:
        # edges from line 1569 to the end of the file
        if line_count > 1568:
            info = line.split(' ')
            u, v = int(info[2]), int(info[3])
            if u != v:
                graph[u].add(v)
                graph[v].add(u)
        line_count += 1

    return graph


def check_edge_existence(graph, u, v):
    return (v in graph[u]) or (u in graph[v])


def vertex_brilliance(graph, centre):
    # returns the largest k such that vertex is the centre of a k-star

    # create subgraph consisting of the neighbors of vertex
    neighbours = []
    edges = []

    for n in graph[centre]:
        neighbours += [n]

    for n1 in neighbours:
        for n2 in neighbours:
            if n2 in graph[n1]:
                if (n1, n2) not in edges:
                    edges += [(n1, n2)]
                if (n2, n1) not in edges:
                    edges += [(n2, n1)]

    # want to find the largest independent set amongst neighbours
    G = nx.Graph()
    G.add_nodes_from(neighbours)
    G.add_edges_from(edges)

    largest_independent_set = nx.algorithms.approximation.maximum_independent_set(G)
    return len(largest_independent_set)


my_graph = load_graph('coauthorship.txt')

# check the brilliance of each node
brilliance_distribution = defaultdict(int)
brilliance = defaultdict(int)
for vertex in my_graph:
    b = vertex_brilliance(my_graph, vertex)
    brilliance[vertex] = b
    brilliance_distribution[b] += 1
    print("vertex", vertex, "has brilliance", b)

print(brilliance)
print(brilliance_distribution)

x_data = []
y_data = []
for b in brilliance_distribution:
    x_data += [b]
    y_data += [brilliance_distribution[b]]

plt.xlabel('brilliance')
plt.ylabel('frequency')
plt.title('distribution of vertex brilliance')
plt.plot(x_data, y_data, marker='.', linestyle='None', color='b')
plt.savefig('brilliance graph - COAUTHORSHIP')




