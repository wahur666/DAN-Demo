from itertools import cycle, combinations
from copy import deepcopy
from typing import Tuple

from huffman_tree import calculate_all_push_up_trees, calculate_all_bfs_trees
from ego_trees import calculate_all_locally_balanced_egotrees, calculate_all_ego_balanced_egotrees

import numpy as np
from src.adt import *
import re

PREFIX = "T"


class Network:

    def __init__(self, demand_matrix: List[List[float]]):
        self.vertices = []
        self.edges = []
        self.avg_deg = 0
        self.demand_matrix = demand_matrix
        self.trees: List[Tree] = []

        self.new_demand_matrix = deepcopy(demand_matrix)
        self.routing_scheme = []  # a.k.a new edges

        self.build_graph()

    def add_vertex(self, vertex: Vertex):
        self.vertices.append(vertex)

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def __repr__(self):
        return str(self.edges)

    def __str__(self):
        return str(self.edges)

    def build_graph(self):

        for i in range(len(self.demand_matrix)):
            self.vertices.append(Vertex(PREFIX, i))

        for i in range(len(self.demand_matrix)):
            for j in range(i + 1, len(self.demand_matrix)):
                if self.demand_matrix[i][j] > 0:
                    e = Edge(self.vertices[i], self.vertices[j], self.demand_matrix[i][j])
                    self.add_edge(e)

    def get_vertex(self, index: int):
        return list(filter(lambda x: x.index == index, self.vertices))

    def create_dan(self, delta):
        self.select_points(delta)
        self.add_helpers()
        self.calculate_trees()
        self.union_trees()
        self.calculate_congestion_and_avglen()
        self.print_summary()

    def select_points(self, delta=None):
        self.delta = delta
        # Classifying points
        degs = []
        for vert in self.vertices:
            deg = sum(e.v1 is vert or e.v2 is vert for e in self.edges)
            degs.append((vert, deg))
        # print(degs)
        self.avg_deg = sum(x[1] for x in degs) / len(degs)

        degs.sort(key=lambda x: x[1], reverse=True)

        H = degs[0:round(len(degs) / 2)]
        L = degs[round(len(degs) / 2):]

        while len(L) > 0 and L[0][1] > self.avg_deg:
            H.append(L.pop(0))
        L.sort(key=lambda x: x[1])

        print(H)
        print(L)

        self.H = H
        self.L = L

    def add_helpers(self):
        c_L = cycle(self.L)
        self.helper_struct: List[Tuple[Vertex, Vertex, Vertex]] = []
        already_assinged = {}
        for edge in self.edges:
            H_v = [x[0] for x in self.H]
            if edge.v1 in H_v and edge.v2 in H_v:
                weight = edge.probability
                u_index = edge.v1.index
                v_index = edge.v2.index

                a1 = [x.index for x in already_assinged[edge.v1]] if edge.v1 in already_assinged else []
                a2 = [x.index for x in already_assinged[edge.v2]] if edge.v2 in already_assinged else []

                a1.extend(a2)
                s = set(a1)
                l = next(c_L)
                counter = len(self.L)
                while l[0].index in s and counter:
                    l = next(c_L)
                    counter -= 1

                l_index = l[0].index
                self.new_demand_matrix[u_index][v_index] = 0
                self.new_demand_matrix[v_index][u_index] = 0

                self.new_demand_matrix[u_index][l_index] += weight
                self.new_demand_matrix[l_index][u_index] += weight

                self.new_demand_matrix[v_index][l_index] += weight
                self.new_demand_matrix[l_index][v_index] += weight

                # print(u_index, weight, v_index, l_index)
                print(edge.v1, edge.v2, "Helper:", l[0])
                if edge.v1 not in already_assinged:
                    already_assinged[edge.v1] = [l[0]]
                else:
                    already_assinged[edge.v1].append(l[0])
                if edge.v2 not in already_assinged:
                    already_assinged[edge.v2] = [l[0]]
                else:
                    already_assinged[edge.v2].append(l[0])
                self.helper_struct.append((edge.v1, edge.v2, l[0]))

    def calculate_trees(self):
        for i in self.demand_matrix:
            print(i)

        self.H_i = [x[0].index for x in self.H]
        self.L_i = [x[0].index for x in self.L]

        if self.delta is None:
            self.delta = 12 * int(round(self.avg_deg))
        elif re.match('\d+$', str(self.delta)):
            self.delta = int(self.delta)
        elif re.match("\d+d$", str(self.delta)):
            self.delta = int(self.delta[:-1]) * int(round(self.avg_deg))
        else:
            raise Exception("Invalid delta format, accepted format \d+$ or \d+d$")
        self.calculate()
        for i in self.trees:
            print(i)

    @abstractmethod
    def calculate(self):
        raise Exception("Calculate function has to be implemented!")

    def union_trees(self):
        for item in self.new_demand_matrix:
            print(item)
        print("------")

        queue = []
        for tree in self.trees:
            print(tree)
            queue.append(tree)
            while queue:
                tree_to_process = queue.pop(0)
                for subtree in tree_to_process.leaves:
                    queue.append(subtree)
                    edge = Edge(tree_to_process.root, subtree.root, subtree.weight())
                    if edge not in self.routing_scheme:
                        if edge.probability > 0:
                            self.routing_scheme.append(edge)
                    else:
                        print("Noveljuk az elet")
                        ind = self.routing_scheme.index(edge)
                        self.routing_scheme[ind].probability += edge.probability

        L_v = [x[0] for x in self.L]

        L_perms = combinations(L_v, 2)
        print(str(self.routing_scheme))

        for u, v in L_perms:
            u_index = u.index
            v_index = v.index
            edge = Edge(u, v, self.new_demand_matrix[u_index][v_index])
            if edge.probability > 0:
                if edge not in self.routing_scheme:
                    self.routing_scheme.append(edge)
                else:
                    ind = self.routing_scheme.index(edge)
                    self.routing_scheme[ind].probability += edge.probability

        print(str(self.routing_scheme))


    def calculate_congestion_and_avglen(self):
        all_path = {}
        for tree in self.trees:
            tree.build_routes()
            tree_paths = []
            helpers_added = []
            for struct in self.helper_struct:
                if struct[0].index == tree.root.index or struct[1].index == tree.root.index:
                    r = tree.get_path(struct[2])
                    r.reverse()
                    tree_paths.append(r)
                    helpers_added.append(struct[2])
            leaves = [x.root for x in tree.leaves]
            leaves_indices = [x.index for x in leaves]
            helper_indices = [x.index for x in helpers_added]
            for index, leaf in enumerate(leaves_indices):
                if leaf not in helper_indices:
                    tree_paths.append([tree.root, leaves[index]])

            all_path[tree_paths[0][0].index] = tree_paths

        # max az utak trolodasa, tordoldas sum az u-v ut osszes elet
        congestion = 0
        most_congested_route = None

        # sum az ut hossza megszorozva ut valoszinusege
        avg_route_len = 0

        full_weight = sum_aa(self.demand_matrix)

        for i in range(len(self.new_demand_matrix) - 1):
            for j in range(i + 1, len(self.new_demand_matrix)):
                if self.new_demand_matrix[i][j]:
                    print(i, j)
                    if i in self.H_i and j in self.H_i:
                        #Mikor H x H van
                        #Ket kulonbozo pont osszeforgatasa
                        route = []
                        for struct in self.helper_struct:
                            if struct[0].index == i and struct[1].index == j:
                                route = self.find_route(all_path, i, struct[2].index)
                                route2 = self.find_route(all_path, j, struct[2].index)[:-1]
                                route2.reverse()
                                route.extend(route2)
                                break
                    elif i in self.H_i and j in self.L_i:
                        #Megkeresessük az utat
                        route = self.find_route(all_path, i, j)
                    elif i in self.L_i and j in self.H_i:
                        #Megkeresessük az utat
                        route = self.find_route(all_path, j, i)
                    else:
                        # Mikor mindketto L
                        # Direkt kapcsolat, hossz 1
                        for edge in self.routing_scheme:
                            if edge.v1.index == i and edge.v2.index == j or edge.v1.index == j and edge.v2.index == i:
                                con = edge.probability
                                route = [edge.v1, edge.v2]
                                break
                    if route:
                        con = self.calculate_congestion(route)
                        congestion, most_congested_route = self.update_congestion(con, congestion, most_congested_route,
                                                                                  route)
                        route_len = (len(route) - 1) * self.new_demand_matrix[i][j] * 2
                        avg_route_len += route_len
                        print("Congestion:", con, route)
                        print("Route Length:", route_len)
                        print("Route LEN:", len(route) - 1)

        #full_weight = sum_aa(self.demand_matrix)

        control_demand_matrix = np.zeros((len(self.demand_matrix), len(self.demand_matrix)))

        for edge in self.routing_scheme:
            control_demand_matrix[edge.v1.index][edge.v2.index] += edge.probability
            control_demand_matrix[edge.v2.index][edge.v1.index] += edge.probability

        max_delta = 0
        for row in control_demand_matrix:
            delta = sum(1 if x > 0 else 0 for x in row)
            max_delta = max(max_delta, delta)
            
        # if avg_route_len / full_weight < 1:
        #     breakpoint()

        self.summary = {}
        self.summary['congestion'] = congestion
        self.summary['real_congestion'] = congestion / full_weight
        self.summary['most_congested_route'] = most_congested_route
        self.summary['avg_route_len'] = avg_route_len / full_weight
        self.summary['delta'] = self.delta
        self.summary['max_delta'] = max_delta


        #print(all_path)

    def print_summary(self):
        print("------- Summary -------")
        print("Most congested route:", self.summary['congestion'], self.summary['real_congestion'],
              self.summary['most_congested_route'])
        print("Average weighted route length:", self.summary['avg_route_len'])
        print("Delta:", self.delta)
        print("max delta:", self.summary['max_delta'])
        print("-----------------------")

    def update_congestion(self, con, congestion, most_congested_route, route):
        if con > congestion:
            congestion = con
            most_congested_route = route
        return congestion, most_congested_route

    def calculate_congestion(self, route):
        con = 0
        for k in range(len(route) - 1):
            start = route[k]
            end = route[k + 1]
            self.routing_scheme: List[Edge]
            for edge in self.routing_scheme:
                if edge.v1.index == start.index and edge.v2.index == end.index or \
                        edge.v1.index == end.index and edge.v2.index == start.index:
                    con += edge.probability
                    break
        return con

    def find_route(self, all_path, i, j) -> List[Node]:
        if not i in all_path:
            return None
        for route in all_path[i]:
            assert route, List[Node]
            if route[0].index == i and route[-1].index == j:
                return route.copy()

    def get_summary(self):
        return self.summary


class OriginalDanNetwork(Network):

    def calculate(self):
        self.trees, self.new_demand_matrix = calculate_all_locally_balanced_egotrees(self.demand_matrix, self.delta, self.H_i, PREFIX,
                                                                                     self.helper_struct)

class EgoBalanceDanNetwork(Network):

    def calculate(self):
        self.trees, self.new_demand_matrix = calculate_all_ego_balanced_egotrees(self.demand_matrix, self.delta, self.H_i, PREFIX,
                                                                                     self.helper_struct)


class HuffmanDanNetwork(Network):

    def calculate(self):
        self.trees = calculate_all_push_up_trees(self.new_demand_matrix, self.delta, self.H_i, PREFIX)


class BfsDanNetwork(Network):

    def calculate(self):
        self.trees = calculate_all_bfs_trees(self.new_demand_matrix, self.delta, self.H_i, PREFIX)


def sum_aa(aa):
    return sum([sum(x) for x in aa])



def normalize100(demand_distribution):
    sum_of_items = sum_aa(demand_distribution)
    multiplier = 100 / sum_of_items
    normalized = [list(map(lambda z: z * multiplier, x)) for x in demand_distribution]
    return normalized


