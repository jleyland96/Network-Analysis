import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from collections import defaultdict


def make_ring_group_graph(m, k, p, q):
    ring_group_graph = {}
    print("making graph on", m*k, "nodes...")

    # p_edges = 0
    # q_edges = 0

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
                        # p_edges += 1
                else:
                    if random_number < q:
                        # print("adding Q edge between", u, "and", v)
                        ring_group_graph[u][1].add(v)
                        ring_group_graph[v][1].add(u)
                        # q_edges += 1

    # print(p_edges, "p_edges")
    # print(q_edges, "q_edges")
    return ring_group_graph


def get_neighbours(graph, vertex):
    neighbours = []
    for n in graph[vertex][1]:
        neighbours += [n]
    random.shuffle(neighbours)
    return neighbours


def search_ring_random(graph, source, target):
    # COMPLETELY RANDOM SEARCH
    current_vertex = source
    queries = 0

    while current_vertex != target:
        N = get_neighbours(graph, current_vertex)
        d = len(N)
        group = graph[current_vertex][0]
        current_vertex = N[0]
        queries += 1

    return queries


def search_ring_use_p(graph, source, target, m, k):
    # COMPLETELY RANDOM SEARCH
    current_vertex = source
    queries = 0
    target_group = target // k

    closest_neighbour_dist = 10
    within_adjacent_or_same_group = False

    # while the target vertex hasn't been found
    while current_vertex != target:
        N = get_neighbours(graph, current_vertex)
        d = len(N)
        current_group = graph[current_vertex][0]
        backup_next_move = N[d-1]

        # print("Searching from", current_vertex, ", in group", current_group, ". Queries = ", queries)

        i = 0
        next_found = False

        # check through the neighbours
        while i < d:
            queries += 1
            neighbour_group = graph[N[i]][0]
            # print("checking", N[i], "in group", neighbour_group)

            neighbour_to_target_dist = min((neighbour_group - target_group) % m, (target_group - neighbour_group) % m)

            # if we have found the target vertex
            if N[i] == target:
                current_vertex = target
                next_found = True
                break

            # if we can home in on a neighbouring group
            if neighbour_to_target_dist <= 1:
                # print("NEIGHBOURING GROUP FOUND")
                if closest_neighbour_dist > 0:
                    closest_neighbour_dist = 1

                if not within_adjacent_or_same_group:
                    # print("NEIGHBOURING GROUP, move to", N[i])
                    within_adjacent_or_same_group = True
                    current_vertex = N[i]
                    next_found = True
                    break
                else:
                    backup_next_move = N[i]

            # if we can home in on the correct group
            if neighbour_group == target_group:
                # print("SAME GROUP FOUND")
                closest_neighbour_dist = 0

                if not within_adjacent_or_same_group:
                    # print("SAME GROUP, move to", N[i])
                    within_adjacent_or_same_group = True
                    current_vertex = N[i]
                    next_found = True
                    break
                else:
                    backup_next_move = N[i]

            # if this neighbour is our closest yet to the target, we move here as backup
            if neighbour_to_target_dist < closest_neighbour_dist:
                # print("- closer node found")
                closest_neighbour_dist = neighbour_to_target_dist

                current_vertex = N[i]
                next_found = True
                break

            i += 1

        if not next_found:
            # if no vertex in the right group found, just move to best found in this loop
            # print("just move to", backup_next_move, "anyway")
            current_vertex = backup_next_move

    return queries


def investigate_ring_search():
    LOOPS = 50
    SUB_LOOPS = 100
    search_time_total_sum = 0
    time_dict = defaultdict(float)
    graph_search_time = 0

    for i in range(LOOPS):
        print("LOOP", i+1)
        my_graph = make_ring_group_graph(10, 10, 0.3, 0.05)
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
                        search_time += search_ring_use_p(my_graph, u, v, 10, 10) / SUB_LOOPS
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
    plt.title("Search time distribution in ring group graphs (m=10, k=10, p=0.3, q=0.05).\n Mean time = " + "{0:.2f}".format(graph_search_time))
    # plt.axis([0, 200, 0, 1000])
    plt.plot(x_data, y_data, marker='.', linestyle='None', color='b')
    plt.savefig('Searching ring group different instances.png')
    plt.show()


# my_graph = make_ring_group_graph(50, 50, 0.3, 0.05)
# print(my_graph)
# print(get_neighbours(my_graph, 0))
investigate_ring_search()
