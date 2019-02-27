import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import defaultdict


def make_random_graph(num_nodes, prob):
    # initialize empty graph
    random_graph = defaultdict(set)
    # consider each vertex
    for vertex in range(1, num_nodes+1):
        for neighbour in range(1, num_nodes+1):
            if vertex < neighbour:
                random_number = random.random()
                if random_number < prob:
                    random_graph[vertex].add(neighbour)
                    random_graph[neighbour].add(vertex)
    return random_graph


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


def get_neighbours(graph, vertex):
    neighbours = []
    for n in graph[vertex]:
        neighbours += [n]
    random.shuffle(neighbours)
    return neighbours


def search_random(graph, source, target):
    # COMPLETELY RANDOM SEARCH
    current_vertex = source
    queries = 0

    while current_vertex != target:
        N = get_neighbours(graph, current_vertex)
        d = len(N)
        current_vertex = N[0]
        queries += 1

    return queries


def search_better(graph, source, target):
    # IDEA - if number of neighbours is a bit larger, try checking the ids of all neighbouring vertices
    current_vertex = source
    queries = 0

    while current_vertex != target:
        N = get_neighbours(graph, current_vertex)
        d = len(N)
        # print("\ncurrent vertex", current_vertex, "has degree", d, ". queries = ", queries)

        # if we reach a vertex with a 'high' degree, then we check all of its neighbours
        if d > 55:
            for i in range(d):
                # print("checking", N[i])
                queries += 1
                if N[i] == target:
                    current_vertex = N[i]
                    break
            # print("now we'll just move randomly")
            current_vertex = N[0]
        else:
            # print("simply moving to", N[0])
            current_vertex = N[0]
            queries += 1

    return queries


def investigate_random_search():
    LOOPS = 10
    SUB_LOOPS = 5
    search_time_total_sum = 0
    time_dict = defaultdict(float)
    graph_search_time = 0

    for i in range(LOOPS):
        print("LOOP", i + 1)
        my_graph = make_random_graph(100, 0.5)
        # print(my_graph)
        searches = 0
        search_time_total_sum = 0

        for u in my_graph:
            print("from node", u)
            for v in my_graph:
                if u != v:
                    search_time = 0
                    for j in range(SUB_LOOPS):
                        # search_time += search_ring_random(my_graph, u, v) / LOOPS
                        search_time += search_random(my_graph, u, v) / SUB_LOOPS
                    # print(u, "to", v, "averaged", "{0:.2f}".format(search_time), "queries")
                    search_time_total_sum += search_time
                    time_dict[int(search_time)] += 1 / LOOPS
                    searches += 1

        this_graph_search_time = search_time_total_sum / searches
        print("Graph search time = ", this_graph_search_time)
        print()
        graph_search_time += this_graph_search_time / LOOPS

    x_data = []
    y_data = []
    for time in time_dict:
        x_data += [time]
        y_data += [time_dict[time]]

    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title("Search time distribution in random graphs. (n=100, p=0.5).\n Mean time = " + "{0:.2f}".format(graph_search_time))
    # plt.axis([0, 200, 0, 1000])
    plt.plot(x_data, y_data, marker='.', linestyle='None', color='b')
    plt.savefig('Searching random different instances.png')
    plt.show()


def investigate_better_search():
    my_graph = make_random_graph(100, 0.5)

    searches = 0
    search_time_total_sum = 0
    LOOPS = 3

    for u in my_graph:
        for v in my_graph:
            if u != v:
                search_time = 0
                for i in range(LOOPS):
                    search_time += search_better(my_graph, u, v) / LOOPS
                print(u, "to", v, "averaged", "{0:.2f}".format(search_time), "queries")
                search_time_total_sum += search_time
                searches += 1
    print("Graph search time = ", search_time_total_sum / searches)


# my_graph = make_random_graph(100, 0.5)
# print(my_graph)
# investigate_random_search()
investigate_random_search()