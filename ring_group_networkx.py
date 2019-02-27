import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx
from networkx.algorithms import approximation


def make_ring_group_graph(m, k, p, q):
    ring_group_graph = {}

    # ring_group_graph[vertex] = [group, [neighbours]]
    edges = 0

    # add vertices to graph
    for i in range(0, m*k):
        ring_group_graph[i] = []

    # create M groups of size K, sequentially
    group = 0
    for i in range(0, m):
        for j in range(0, k):
            ring_group_graph[i*k + j].append(i)

    # give each vertex a
    for i in range(0, m*k):
        ring_group_graph[i].append(set())

    # for each pair of distinct vertices, sort out edges
    for u in range(0, m*k):
        for v in range(0, m*k):
            if u < v:
                u_group = ring_group_graph[u][0]
                v_group = ring_group_graph[v][0]
                random_number = random.random()

                if (u_group == v_group) or (abs(u_group - v_group % m) == 1) or (v_group - u_group % m == m-1) or (u_group - v_group % m == m-1):

                    if random_number < p:
                        # print("adding P edge between", u, "and", v)
                        ring_group_graph[u][1].add(v)
                        ring_group_graph[v][1].add(u)
                        edges += 1
                else:
                    if random_number < q:
                        # print("adding Q edge between", u, "and", v)
                        ring_group_graph[u][1].add(v)
                        ring_group_graph[v][1].add(u)
                        edges += 1
    print("edges:", edges)
    return ring_group_graph


def vertex_brilliance(graph, centre):
    # returns the largest k such that vertex is the centre of a k-star

    # create subgraph consisting of the neighbors of vertex
    neighbours = []
    edges = []

    # add each neighbour of the centre vertex to a subset of vertices
    for n in graph[centre][1]:
        neighbours += [n]

    # add all the edges to this graph
    for n1 in neighbours:
        for n2 in neighbours:
            if n2 in graph[n1][1]:
                if (n1, n2) not in edges:
                    edges += [(n1, n2)]
                if (n2, n1) not in edges:
                    edges += [(n2, n1)]

    print(neighbours)
    print(edges)

    # create the networkx equivalent so we can use the independent set function
    G = nx.Graph()
    G.add_nodes_from(neighbours)
    G.add_edges_from(edges)

    # we want to find the largest independent set of centre's neighbors
    largest_independent_set = nx.algorithms.approximation.maximum_independent_set(G)
    return len(largest_independent_set)


# --- MAIN ---
def investigate_diameter():
    p = 0.05
    q = 0.05
    loops = 10

    x_data = []  # p
    y_data = []  # diameter

    while p < 1.0:
        print("p =", p)
        diameter_sum = 0
        for i in range(0, loops):
            my_graph = make_ring_group_graph(15, 15, p, q)
            G = nx.Graph()
            vertices = []
            edges = []

            for vertex in my_graph:
                vertices.append(vertex)
                for neighbour in my_graph[vertex][1]:
                    edges.append((vertex, neighbour))

            G.add_nodes_from(vertices)
            G.add_edges_from(edges)

            diameter_sum += nx.diameter(G)

        print("avg_diameter = ", diameter_sum / loops, "\n")
        x_data += [p]
        y_data += [diameter_sum / loops]
        p += 0.01

    plt.xlabel('p')
    plt.ylabel('Diameter')
    plt.title('How diameter changes with p in ring_group_graph')
    plt.axis([0, 1, 0, 5])
    plt.plot(x_data, y_data, marker='.', linestyle='None', color='b')
    plt.plot(x_data, np.poly1d(np.polyfit(x_data, y_data, 1))(x_data))
    plt.savefig('diameter graph 4')

def investigate_brilliance():
    print("making graph...")
    MAX_LOOPS = 10
    brilliance_distribution = defaultdict(int)

    for i in range(MAX_LOOPS):
        my_graph = make_ring_group_graph(40, 40, 0.23, 0.02)

        # check the brilliance of each node
        brilliance = defaultdict(int)

        for vertex in my_graph:
            brilliance[vertex] = vertex_brilliance(my_graph, vertex)
            # print("vertex", vertex, "has brilliance", brilliance[vertex])
            if vertex % 100 == 0:
                print(vertex)

        for v in brilliance:
            brilliance_distribution[brilliance[v]] += 1/(1600*MAX_LOOPS)  # Or is it 1/MAX_LOOPS

        print(brilliance)
        print(brilliance_distribution)
        print()

    x_data = []
    y_data = []
    for b in brilliance_distribution:
        x_data += [b]
        y_data += [brilliance_distribution[b]]

    plt.xlabel('brilliance')
    plt.ylabel('frequency')
    plt.title('distribution of vertex brilliance in Ring Group graphs')
    plt.plot(x_data, y_data, marker='.', linestyle='None', color='b')
    plt.savefig('brilliance graph - RING GROUP -' + str(MAX_LOOPS) + ' .png')


investigate_diameter()


