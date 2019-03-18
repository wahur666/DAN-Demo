from typing import List, Dict
from itertools import cycle, permutations

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

        self.new_demand_matrix = demand_matix.copy()
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
    # L.sort(key=lambda x: x[1])

    print(H)
    print(L)

    add_helpers(g, H, L)


def add_helpers(g, H, L):
    c_L = cycle(L)
    for edge in g.edges:
        H_v = [ x[0] for x in H ]
        if edge.v1 in H_v and edge.v2 in H_v:
            weight = edge.probability
            u_index = int(edge.v1.label[1:])
            v_index = int(edge.v2.label[1:])
            l = next(c_L)
            l_index = int(l[0].label[1:])
            g.new_demand_matrix[u_index][v_index] = 0
            g.new_demand_matrix[v_index][u_index] = 0

            g.new_demand_matrix[u_index][l_index] += weight
            g.new_demand_matrix[l_index][u_index] += weight

            g.new_demand_matrix[v_index][l_index] += weight
            g.new_demand_matrix[l_index][v_index] += weight


            # print(u_index, weight, v_index, l_index)
            print(edge.v1, edge.v2, "Helper:", l)
    calculate_egotrees_with_new_demand(g, H, L)

def calculate_egotrees_with_new_demand(g: Graph, H: List, L: List):
    for i in g.new_demand_matrix:
        print(i)

    H_i= [ int(x[0].label[1:]) for x in H ]

    v = calculate_all_egotrees(g.new_demand_matrix, 12 * int(round(g.avg_deg)), H_i)
    for i in v:
        print(i)

    union_egotrees(g, v, L)

def union_egotrees(g: Graph, egotrees: List[EgoTree], L: List):
    for tree in egotrees:
        for leave in tree.leaves:
            v1 = g.get_vertex(tree.root.label)[0]
            v2 = g.get_vertex(leave.root.label)[0]
            weight = int(leave.weight()/2)
            g.routing_scheme.append(Edge(v1, v2, weight))

    L_v = [x[0] for x in L]

    L_perms = permutations(L_v, 2)
    print(str(g.routing_scheme))

    for u, v in L_perms:
        u_index = int(u.label[1:])
        v_index = int(v.label[1:])
        edge = Edge(u, v, g.new_demand_matrix[u_index][v_index])
        if edge not in g.routing_scheme: g.routing_scheme.append(edge)

    print(str(g.routing_scheme))





create_dan(g)