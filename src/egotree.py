from typing import List, Dict
from itertools import cycle, combinations
from copy import deepcopy

class Vertex:

    def __init__(self, label: str):
        self.label = label

    def __repr__(self):
        return "Vertex: " + self.label

    def __str__(self):
        return "Vertex: " + self.label


class Edge:

    def __init__(self, v1: Vertex, v2: Vertex, probability):
        self.v1 = v1
        self.v2 = v2
        self.probability = probability

    def __repr__(self):
        return self.v1.label + "-" + str(self.probability) + "-" + self.v2.label

    def __str__(self):
        return self.v1.label + "-" + str(self.probability) + "-" + self.v2.label

    def __eq__(self, other):
        return {self.v1, self.v2} == {other.v1, other.v2}


class Graph:

    def __init__(self, demand_matix : List):
        self.vertices = []
        self.edges = []
        self.avg_deg = 0
        self.demand_matrix = demand_matix

        self.new_demand_matrix = deepcopy(demand_matix)
        self.routing_scheme = [] # a.k.a new edges

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
            self.vertices.append(Vertex("T" + str(i)))

        for i in range(len(self.demand_matrix)):
            for j in range(i + 1, len(self.demand_matrix)):
                if self.demand_matrix[i][j] > 0:
                    e = Edge(self.vertices[i], self.vertices[j], self.demand_matrix[i][j])
                    self.add_edge(e)

    def get_vertex(self, label):
        return list(filter(lambda x : x.label == label, self.vertices))

class Node:

    def __init__(self, label: str, probability: float):
        self.label = label
        self.probability = probability

    def __str__(self):
        return self.label + " W: " + str(self.probability)

    def __repr__(self):
        return self.label + " W: " + str(self.probability)


class Tree:

    def __init__(self, root: Node):
        self.root = root
        self.leaves: List = []

    def dept(self):
        return 1 + max(self.leaves, key=lambda x: x.dept()).dept() if self.leaves else 0

    def __str__(self):
        return self.root.label + " L: " + str(self.leaves)

    def __repr__(self):
        return self.root.label + " L: " + str(self.leaves)

    def weight(self):
        return sum(n.weight() for n in self.leaves) + self.root.probability

    def get_edges(self) -> List[Edge]:
        edges = []
        for leave in self.leaves:
            v1 = g.get_vertex(self.root.label)[0]
            v2 = g.get_vertex(leave.root.label)[0]
            weight = int(leave.weight() / 2)
            edges.append(Edge(v1, v2, weight))
            edges.extend(leave.get_edges())
        return edges

    def get_dependent_nodes(self) -> List[Node]:
        nodes = []
        for leave in self.leaves:
            nodes.append(leave.root)
            nodes.extend(leave.get_dependent_nodes())
        return sorted(nodes, key=lambda x: x.probability, reverse=True)

    def get_trees(self) -> List:
        trees = []
        for leave in self.leaves:
            trees.append(leave)
            trees.extend(leave.get_trees())
        return trees



class BinTree(Tree):

    def __init__(self, root: Node):
        super().__init__(root)
        self.leaves: List[BinTree] = []

    def push(self, bintree):
        if len(self.leaves) != 2:
            self.leaves.append(bintree)
        else:
            lightest_leave = min(self.leaves, key=lambda x: x.weight())
            lightest_leave.push(bintree)


class EgoTree(Tree):

    def __init__(self, root: Node, delta: int):
        super().__init__(root)
        self.delta = delta
        self.leaves: List[BinTree] = []

    def push(self, bintree: BinTree):
        if len(self.leaves) == self.delta:
            lightest_leave = min(self.leaves, key=lambda x: x.weight())
            lightest_leave.push(bintree)
        else:
            self.leaves.append(bintree)


def map_probabilities(p: List[Node]) -> Dict:
    return {n: n.probability for n in p}


def create_egotree(source: Node, p: List[Node], delta: int) -> EgoTree:
    p1 = map_probabilities(p)
    p1 = sorted(p1.items(), key=lambda kv: kv[1], reverse=True)
    egotree = EgoTree(source, delta)
    for key, value in p1:
        egotree.push(BinTree(key))
    return egotree


source = Node("T0", 0)
p = [Node("T1", 24),
     Node("T2", 20),
     Node("T3", 10),
     Node("T4", 10),
     Node("T5", 10),
     Node("T6", 10),
     Node("T7", 7),
     Node("T8", 5),
     Node("T9", 2),
     Node("T10", 1),
     Node("T11", 1)]
n = 4

# egotree = create_egotree(source, p, n)
#
# print("Root: " + str(egotree.root))
# for item in egotree.leaves:
#     print(item)

# print(egotree.dept())
# print(egotree.weight())


def sum_aa(aa):
    return sum([sum(x) for x in aa])


def normalize100(demand_distribution):
    sum_of_items = sum_aa(demand_distribution)
    multiplier = 100 / sum_of_items
    normalized = [list(map(lambda z: z * multiplier, x)) for x in demand_distribution]
    return normalized


demand_distribution = [[0, 3, 4, 1, 1, 1, 1],
                       [3, 0, 2, 0, 1, 0, 4],
                       [4, 2, 0, 2, 0, 0, 4],
                       [1, 0, 2, 0, 3, 0, 0],
                       [1, 1, 0, 3, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 3],
                       [1, 4, 4, 0, 0, 3, 0]]

# print( normalize100(demand_distribution) )
#
# print(sum_aa(normalize100(demand_distribution)))


def calculate_all_egotrees(demand_distribution, delta, indexes : List = None):
    egotrees = []

    dd = demand_distribution

    for i in range(len(demand_distribution)):

        if indexes and i not in indexes:
            continue
        nodes = []
        source: Node

        for j in range(len(demand_distribution)):
            if i == j:
                source = Node("T"+str(i), 0)
            elif dd[i][j] + dd[j][i] > 0:
                nodes.append(Node("T"+str(j), dd[i][j] + dd[j][i]))

        egotrees.append(create_egotree(source, nodes, delta))

    return egotrees

v = calculate_all_egotrees(demand_distribution, 3)

# for tree in v:
#     print(tree.weight())


g = Graph(demand_distribution)

print(g)


def create_dan(g: Graph):
    # Classifying points
    degs = []
    for vert in g.vertices:
        deg = sum( e.v1 is vert or e.v2 is vert for e in g.edges )
        degs.append((vert, deg))
    # print(degs)
    g.avg_deg = sum(x[1] for x in degs) / len(degs)

    degs.sort(key=lambda x: x[1], reverse=True)

    H = degs[0:round(len(degs)/2)]
    L = degs[round(len(degs)/2):]

    while len(L) > 0 and L[0][1] > g.avg_deg:
        H.append(L.pop(0))
    L.sort(key=lambda x: x[1])

    print(H)
    print(L)

    add_helpers(g, H, L)


def add_helpers(g, H, L):
    c_L = cycle(L)
    helper_struct = []
    already_assinged = {}
    for edge in g.edges:
        H_v = [ x[0] for x in H ]
        if edge.v1 in H_v and edge.v2 in H_v:
            weight = edge.probability
            u_index = int(edge.v1.label[1:])
            v_index = int(edge.v2.label[1:])

            a1 = [ x.label for x in already_assinged[edge.v1]] if edge.v1 in already_assinged else []
            a2 = [ x.label for x in already_assinged[edge.v2]] if edge.v2 in already_assinged else []

            a1.extend(a2)
            s = set(a1)
            l = next(c_L)
            while l[0].label in s:
                l = next(c_L)

            l_index = int(l[0].label[1:])
            g.new_demand_matrix[u_index][v_index] = 0
            g.new_demand_matrix[v_index][u_index] = 0

            g.new_demand_matrix[u_index][l_index] += weight
            g.new_demand_matrix[l_index][u_index] += weight

            g.new_demand_matrix[v_index][l_index] += weight
            g.new_demand_matrix[l_index][v_index] += weight


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
            helper_struct.append((edge.v1, edge.v2, l[0]))
    calculate_egotrees(g, H, L, helper_struct, 3)

def calculate_egotrees(g: Graph, H: List, L: List, helper_struct: List, delta = None):
    for i in g.demand_matrix:
        print(i)

    H_i= [ int(x[0].label[1:]) for x in H ]

    deltaN = delta or 12 * int(round(g.avg_deg))

    v = calculate_all_egotrees(g.demand_matrix, deltaN, H_i)
    for i in v:
        print(i)

    change_nodes_in_egotrees(g, v, H, L, helper_struct)

def change_nodes_in_egotrees(g: Graph, egotrees: List[EgoTree], H: List, L: List, helper_struct: List):

    print("---- EGO TREES BEFORE ----")
    for tree in egotrees:
        print(tree)
    print("--------------------------")
    for tree in egotrees:
        print("Tree before", tree)
        for struct in helper_struct:
            leave_labels = [x.label for x in tree.get_dependent_nodes()]
            if tree.root.label == struct[0].label and struct[1].label in leave_labels:
                v_tree = [ x for x in tree.get_trees() if x.root.label == struct[1].label ][0]
            elif tree.root.label == struct[1].label and struct[0].label in leave_labels:
                v_tree = [ x for x in tree.get_trees() if x.root.label == struct[0].label ][0]
            else:
                continue

            if v_tree in tree.leaves:
                v_parent = tree
            else:
                v_parent = [x for x in tree.get_trees() if v_tree in x.leaves][0]

            if not struct[2].label in leave_labels:
                # Mikor L nincs benne a faban
                print("L atveszi V helyet, L=0")
                # print(tree)
                # print("V", v_tree)
                l_tree = BinTree(Node(struct[2].label, 0))
                l_tree.leaves = v_tree.leaves

                ind = v_parent.leaves.index(v_tree)
                v_parent.leaves[ind] = l_tree
                # print("L", l_tree)
                # print(tree)

            else:
                # Mikor L benne van a faban
                l_tree = [x for x in tree.get_trees() if x.root.label == struct[2].label][0]
                if l_tree in tree.leaves:
                    l_parent = tree
                else:
                    l_parent = [x for x in tree.get_trees() if l_tree in x.leaves][0]

                u_index = int(tree.root.label[1:])
                v_index = int(v_tree.root.label[1:])
                l_index = int(l_tree.root.label[1:])

                if g.demand_matrix[u_index][l_index] > g.demand_matrix[u_index][v_index]:
                    # Toroljuk V-t a fabol
                    nodes_to_redistribute = v_tree.get_dependent_nodes()
                    ind = v_parent.leaves.index(v_tree)
                    v_parent.leaves.pop(ind)
                    print("V-t toroljuk a fabol")
                else:
                    # L atveszi V helyet
                    nodes_to_redistribute = l_tree.get_dependent_nodes()
                    if v_tree.root in nodes_to_redistribute:
                        nodes_to_redistribute.remove(v_tree.root)
                    ind = l_parent.leaves.index(l_tree)
                    l_parent.leaves.pop(ind)
                    l_tree.leaves = v_tree.leaves

                    if len(v_parent.leaves) > 0:
                        ind = v_parent.leaves.index(v_tree)
                        v_parent.leaves[ind] = l_tree
                    else:
                        if v_parent != l_tree:
                            v_parent.push(l_tree)
                    print("L atveszi V helyet, L>0")

                for item in nodes_to_redistribute:
                    tree.push(BinTree(item))
        print("Tree after", tree)

    print("---- EGO TREES AFTER ----")

    for tree in egotrees:
        print(tree)

    print("-------------------------")

    union_egotrees(g, egotrees, L)

def union_egotrees(g: Graph, egotrees: List[EgoTree], L: List):
    for item in g.new_demand_matrix:
        print(item)
    for tree in egotrees:
        print(tree)
        for edge in tree.get_edges():
            if edge not in g.routing_scheme:
                u_index = int(edge.v1.label[1:])
                v_index = int(edge.v2.label[1:])
                #TODO Újra kell számolni a trolódást
                edge.probability = g.new_demand_matrix[u_index][v_index]
                g.routing_scheme.append(edge)

    L_v = [x[0] for x in L]

    L_perms = combinations(L_v, 2)
    print(str(g.routing_scheme))

    for u, v in L_perms:
        u_index = int(u.label[1:])
        v_index = int(v.label[1:])
        edge = Edge(u, v, g.new_demand_matrix[u_index][v_index])
        if edge not in g.routing_scheme:
            g.routing_scheme.append(edge)

    print(str(g.routing_scheme))





create_dan(g)