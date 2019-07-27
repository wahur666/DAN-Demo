import re
from copy import deepcopy
from itertools import cycle, combinations
import numpy as np

from adt import *

PREFIX = "T"


class HuffmanDanNetwork:

    def __init__(self, demand_matrix: List):
        self.vertices = []
        self.edges = []
        self.avg_deg = 0
        self.demand_matrix = demand_matrix

        self.new_demand_matrix = deepcopy(demand_matrix)
        self.new_new_demand_matrix = deepcopy(demand_matrix)
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

    def create_dan(self, delta=None):
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

        self.add_helpers()

    def add_helpers(self):
        c_L = cycle(self.L)
        self.helper_struct = []
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
        self.calculate_trees()

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
        self.egotrees = calculate_all_bfs_trees(self.new_demand_matrix, self.delta, self.H_i)
        for i in self.egotrees:
            print(i)
        self.union_egotrees()

    def union_egotrees(self):
        for item in self.new_demand_matrix:
            print(item)
        print("------")

        queue = []
        for tree in self.egotrees:
            print(tree)
            queue.append(tree)
            while queue:
                tree_to_process = queue.pop(0)
                for subtree in tree_to_process.leaves:
                    queue.append(subtree)
                    edge = Edge(tree_to_process.root, subtree.root, subtree.weight())
                    if edge not in self.routing_scheme:
                        # u_index = edge.v1.index
                        # v_index = edge.v2.index
                        # edge.probability = self.new_new_demand_matrix[u_index][v_index]
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

        self.calculate_congestion_and_avglen()

    def calculate_congestion_and_avglen(self):
        all_path = {}
        for tree in self.egotrees:
            tree.build_routes()
            tree_paths = []
            for struct in self.helper_struct:
                if struct[0].index == tree.root.index or struct[1].index == tree.root.index:
                    r = tree.get_path(struct[2])
                    r.reverse()
                    tree_paths.append(r)
            if tree_paths:
                all_path[tree_paths[0][0].index] = tree_paths

        self.H_i = [x[0].index for x in self.H]
        self.L_i = [x[0].index for x in self.L]

        # max az utak trolodasa, tordoldas sum az u-v ut osszes elet
        congestion = 0
        most_congested_route = None

        # sum az ut hossza megszorozva ut valoszinusege
        avg_route_len = 0

        for i in range(len(self.demand_matrix)-1):
            for j in range(i + 1, len(self.demand_matrix)):
                if self.demand_matrix[i][j]:
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

                        if route:
                            con = self.calculate_congestion(route)
                            congestion, most_congested_route = self.update_congestion(con, congestion,
                                                                                      most_congested_route,
                                                                                      route)
                            avg_route_len += self.demand_matrix[i][j] * len(route)
                            print("Congestion:", con, route)
                            print("Route Length:", self.demand_matrix[i][j] * len(route))


                        #print("Full combined:", route1)
                    elif i in self.H_i and j in self.L_i:
                        #Megkeresessük az utat
                        route = self.find_route(all_path, i, j)
                        if route:
                            con = self.calculate_congestion(route)
                            congestion, most_congested_route = self.update_congestion(con, congestion, most_congested_route,
                                                                                      route)
                            avg_route_len += self.demand_matrix[i][j] * len(route)
                            print("Congestion:", con, route)
                            print("Route Length:", self.demand_matrix[i][j] * len(route))
                            #print(route)

                    elif i in self.L_i and j in self.H_i:
                        route = self.find_route(all_path, j, i)
                        if route:
                            con = self.calculate_congestion(route)
                            congestion, most_congested_route = self.update_congestion(con, congestion, most_congested_route,
                                                                                      route)
                            avg_route_len += self.demand_matrix[i][j] * len(route)
                            print("Congestion:", con, route)
                            print("Route Length:", self.demand_matrix[i][j] * len(route))

                        #print(route)
                    else:
                        # Mikor mindketto L
                        # Direkt kapcsolat, hossz 1
                        con, route = 0, []
                        for edge in self.routing_scheme:
                            if edge.v1.index == i and edge.v2.index == j or edge.v1.index == j and edge.v2.index == i:
                                con = edge.probability
                                route = [edge.v1, edge.v2]
                                break
                        congestion, most_congested_route = self.update_congestion(con, congestion, most_congested_route,
                                                                                  route)
                        avg_route_len += self.demand_matrix[i][j] * len(route)
                        print("Congestion:", con, route)
                        print("Route Length:", self.demand_matrix[i][j] * len(route))

        full_weight = sum_aa(self.demand_matrix)

        control_demand_matrix = np.zeros((len(self.demand_matrix), len(self.demand_matrix)))

        for edge in self.routing_scheme:
            control_demand_matrix[edge.v1.index][edge.v2.index] += edge.probability
            control_demand_matrix[edge.v2.index][edge.v1.index] += edge.probability

        mismatch_counter = 0
        for i in range(len(self.demand_matrix)):
            for j in range(len(self.demand_matrix)):
                if control_demand_matrix[i][j] != self.new_new_demand_matrix[i][j]:
                    mismatch_counter += 1
        print("Mismatching items count:",mismatch_counter)

        max_delta = 0
        for row in control_demand_matrix:
            delta = sum(1 if x > 0 else 0 for x in row)
            max_delta = max(max_delta, delta)

        self.summary = {}
        self.summary['congestion'] = congestion
        self.summary['real_congestion'] = congestion / full_weight
        self.summary['most_congested_route'] = most_congested_route
        self.summary['avg_route_len'] = avg_route_len/full_weight
        self.summary['delta'] = self.delta
        self.summary['max_delta'] = max_delta


        self.print_summary()
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

    def find_route(self, all_path, i, j) -> List[HuffmanDanNode]:
        if not i in all_path:
            return None
        for route in all_path[i]:
            assert route, List[HuffmanDanNode]
            if route[0].index == i and route[-1].index == j:
                return route.copy()

    def get_summary(self):
        return self.summary

# --------------------------------------------------------------------------------------------------------------------
def calculate_all_bfs_trees(demand_distribution, delta, indices: List = None):
    bfs_trees = []

    dd = demand_distribution

    for i in range(len(dd)):

        if indices and i not in indices:
            continue

        nodes = []
        source: HuffmanDanNode = None

        for j in range(len(dd)):
            if i == j:
                source = HuffmanDanNode(PREFIX, i, 0)
            elif dd[i][j] + dd[j][i] > 0:
                nodes.append(HuffmanDanNode(PREFIX, j, dd[i][j] + dd[j][i]))

        tree = create_bfs_tree(delta, nodes)
        tree.root = source

        bfs_trees.append(tree)
    return bfs_trees


def get_nodes(tree: HuffmanDanTree) -> List[HuffmanDanNode]:
    nodes = []
    for index, leaf in enumerate(tree.leaves):
        if isinstance(leaf, HuffmanDanNode):
            nodes.append(leaf)
        elif isinstance(leaf, HuffmanDanTree):
            nodes.extend(get_nodes(leaf))
    return nodes


# Naive Solution


def detach_leaf(tree: HuffmanDanTree, path: List[int]):
    temp_tree = tree
    for i in path[:-1]:
        temp_tree = temp_tree.leaves[i]
    temp_tree.leaves[path[-1]] = None


def calculate_node_paths(tree: HuffmanDanTree):
    for index, leaf in enumerate(tree.leaves):
        for i in tree.path:
            leaf.path.append(i)
        leaf.path.append(index)
        if isinstance(leaf, HuffmanDanTree):
            calculate_node_paths(leaf)


def remove_dead_branches(root: HuffmanDanTree):
    leaves = []
    for leaf in root.leaves:
        leaves.append(leaf)

    while leaves:
        leaf = leaves.pop(0)
        if isinstance(leaf, HuffmanDanNode):
            continue
        leaf.leaves = [x for x in leaf.leaves if x is not None]
        leaves.extend(leaf.leaves)


def naive_push_up(delta, nodes):
    root = HuffmanDanTree(None, delta, nodes)
    calculate_node_paths(root)
    leaves = root.leaves.copy()
    while leaves:
        leaf = leaves.pop(0)
        if leaf is not None and isinstance(leaf, HuffmanDanTree):
            nodes = get_nodes(leaf)
            print(nodes)
            max_weight_leaf = max(nodes, key=lambda x: x.weight())
            leaf.root = max_weight_leaf
            detach_leaf(root, max_weight_leaf.path)
            leaves.extend(leaf.leaves)
    remove_dead_branches(root)
    return root


# Breadth-first solution


def find_next_free_position(tree: HuffmanDanTree, node: HuffmanDanNode):
    if len(tree.leaves) < tree.delta:
        new_leaf = HuffmanDanTree(node, 2, [])
        tree.leaves.append(new_leaf)
    else:
        leaves = tree.leaves.copy()
        while leaves:
            leaf = leaves.pop(0)
            if len(leaf.leaves) < leaf.delta:
                new_leaf = HuffmanDanTree(node, 2, [])
                leaf.leaves.append(new_leaf)
                break
            else:
                leaves.extend(leaf.leaves)


def bfs_push_up(delta, nodes):
    root = HuffmanDanTree(None, delta, nodes)
    new_root = HuffmanDanTree(None, delta, [])
    leaves = root.leaves.copy()
    while leaves:
        leaf = leaves.pop(0)
        if leaf is not None:
            if isinstance(leaf, HuffmanDanTree):
                nodes = get_nodes(leaf)
                nodes.sort(key=lambda x: x.weight(), reverse=True)
                print(nodes)
                if len(nodes) > 1:
                    n = nodes.pop(0)
                    new_tree = HuffmanDanTree(n, 2, [])
                    for node in nodes:
                        find_next_free_position(new_tree, node)
                    new_root.leaves.append(new_tree)
            else:
                tree = HuffmanDanTree(leaf, 2, [])
                new_root.leaves.append(tree)
    return new_root


def create_push_up_tree(delta, nodes: List[AbstractHuffman]):
    while len(nodes) > delta:
        node1 = min(nodes, key=lambda x: x.weight())
        nodes.pop(nodes.index(node1))
        node2 = min(nodes, key=lambda x: x.weight())
        nodes.pop(nodes.index(node2))
        tree = HuffmanDanTree(None, 2, [node1, node2])
        nodes.append(tree)
    # root = naive_push_up(delta, nodes)
    root = bfs_push_up(delta, nodes)

    return root


def create_bfs_tree(delta, nodes: List[HuffmanDanNode]):
    nodes.sort(key=lambda x: x.weight(), reverse=True)
    root = HuffmanDanTree(None, delta, [])
    for node in nodes:
        find_next_free_position(root, node)
    return root

def sum_aa(aa):
    return sum([sum(x) for x in aa])