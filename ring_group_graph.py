import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import defaultdict


def make_ring_group_graph(m, k, p, q):
    ring_group_graph = {}
    print("making graph on", m*k, "nodes...")

    # ring_group_graph[vertex] = [group, [neighbours]]

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
                    # print(u, "and", v)

                    if random_number < p:
                        # print("adding P edge between", u, "and", v)
                        ring_group_graph[u][1].add(v)
                        ring_group_graph[v][1].add(u)
                else:
                    if random_number < q:
                        # print("adding Q edge between", u, "and", v)
                        ring_group_graph[u][1].add(v)
                        ring_group_graph[v][1].add(u)
    return ring_group_graph


def compute_degrees(digraph):
    # initialize in-degrees dictionary with zero values for all vertices
    degree = {}
    for vertex in digraph:
        degree[vertex] = 0

    # consider each vertex
    for vertex in digraph:
        # amend in_degree[w] for each outgoing edge from v to w
        for neighbour in digraph[vertex][1]:
            degree[neighbour] += 1

    return degree


def degree_distribution(digraph):
    # find in_degrees
    degree = compute_degrees(digraph)

    # initialize dictionary for degree distribution
    degree_distribution = {}

    # consider each vertex
    for vertex in degree:
        # update degree_distribution
        if degree[vertex] in degree_distribution:
            degree_distribution[degree[vertex]] += 1
        else:
            degree_distribution[degree[vertex]] = 1

    return degree_distribution


def bfs_from_source(graph, source):
    # # Keeping track of execution progress
    # if source % 100 == 0:
    #     print(source, "being explored")

    # storing if vertex has been visited from source vertex
    visited = defaultdict(bool)
    for vertex in graph:
        visited[vertex] = False

    # storing shortest distance from source to vertex
    distance = defaultdict(int)
    for vertex in graph:
        distance[vertex] = -1

    queue = [source]
    visited[source] = True
    distance[source] = 0

    while len(queue) > 0:
        current_vertex = queue.pop()

        for neighbour in graph[current_vertex][1]:
            if not visited[neighbour]:
                queue.append(neighbour)
                visited[neighbour] = True
                distance[neighbour] = distance[current_vertex] + 1

    longest_distance = max(distance.values())
    return longest_distance


def get_diameter(graph):
    maximum_distance = 0
    for vertex in graph:
        distance = bfs_from_source(graph, vertex)
        if distance > maximum_distance:
            maximum_distance = distance
            print("distance", distance, "is the new biggest!")
    return maximum_distance


def investigate_degree_distribution_repeats():
    # p > q
    # p + q = 0.5
    m = 50
    k = 50
    p = 0.26
    q = 0.24
    MAX_LOOPS = 5

    index = 0
    while p < 0.51:
        normalized_dist = defaultdict(float)
        for i in range(MAX_LOOPS):

            print("p =", p, "q =", q)
            my_graph = make_ring_group_graph(m, k, p, q)
            degrees_distribution = degree_distribution(my_graph)

            for degree in degrees_distribution:
                normalized_dist[degree] += degrees_distribution[degree]/MAX_LOOPS

        colours = ['r', 'g', 'b', 'y', 'k', 'm', 'c']

        # create arrays for plotting
        x_data = []
        y_data = []
        for degree in normalized_dist:
            x_data += [degree]
            y_data += [normalized_dist[degree]]

        # plot degree distribution
        plt.xlabel('Degree')
        plt.ylabel('Frequency Rate')
        plt.title('Degree Distribution of ring_group_graph')
        plt.plot(x_data, y_data, marker='.', linestyle='None', color=colours[index])

        p += 0.01
        q -= 0.01
        index = (index + 1) % 7
        print()

    plt.savefig('REPEATS ' + str(m) + ' - ' + str(k) + ' - p_range_0.26_0.50.png')
    plt.show()

def investigate_degree_distribution():
    # p > q
    # p + q = 0.5
    m = 5
    k = 500
    p = 0.26
    q = 0.24

    index = 0
    while p < 0.51:
        print("p =", p, "q =", q)
        my_graph = make_ring_group_graph(m, k, p, q)
        degrees_distribution = degree_distribution(my_graph)

        colours = ['r', 'g', 'b', 'y', 'k', 'm', 'c']

        # create arrays for plotting
        x_data = []
        y_data = []
        for degree in degrees_distribution:
            x_data += [degree]
            y_data += [degrees_distribution[degree]]

        # plot degree distribution
        plt.xlabel('Degree')
        plt.ylabel('Frequency Rate')
        plt.title('Degree Distribution of ring_group_graph')
        plt.plot(x_data, y_data, marker='.', linestyle='None', color=colours[index])

        p += 0.24
        q -= 0.24
        index = (index + 1) % 7
        print()

    # plt.savefig('Normalized' + str(m) + ' - ' + str(k) + ' - p_range_0.26_0.50.png')
    plt.show()


def investigate_diameter():
    # fixed q
    # p > q
    m = 10
    k = 10
    p = 0.31
    q = 0.30

    x_data = []
    y_data = []

    while p < 0.51:
        print("p =", p, "q =", q)
        my_graph = make_ring_group_graph(m, k, p, q)
        print("Getting diameter...")
        longest_u_v_path = get_diameter(my_graph)
        print("longest path from source to another node is", longest_u_v_path, "\n")

        x_data.append(p)
        y_data.append(longest_u_v_path)

        p += 0.01

    # PLOT
    plt.xlabel('p')
    plt.ylabel('Diameter')
    plt.title('How diameter changes with p in ring_group_graph ')
    plt.plot(x_data, y_data, marker='.', linestyle='None')


investigate_degree_distribution_repeats()

# m = 10
# k = 100
# p = 0.4
# q = 0.1
# MAX_LOOPS = 100
# normalized_degree_dist = defaultdict(float)
#
# for it in range(MAX_LOOPS):
#     GRAPH = make_ring_group_graph(m, k, p, q)
#     degree_dist = degree_distribution(GRAPH)
#
#     print("normalizing degrees...")
#     for i in degree_dist:
#         normalized_degree_dist[i] += degree_dist[i] / (m*k*MAX_LOOPS)  # NORMALIZED
#         # normalized_degree_dist[i] += degree_dist[i] / MAX_LOOPS
#     print("iteration", it, "done\n")
#
# x_data = []
# y_data = []
#
# for degree in normalized_degree_dist:
#     x_data += [degree]
#     y_data += [normalized_degree_dist[degree]]
#
# plt.xlabel('degree')
# plt.ylabel('Normalized Rate')
# plt.title('In-Degree Distribution of Ring group graph')
# plt.plot(x_data, y_data, marker='.', linestyle='None', color='b')
# plt.savefig('Ring Group Normalized Distribution - ' + str(MAX_LOOPS) + ' .png')








