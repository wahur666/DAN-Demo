from abc import abstractmethod
from typing import List


class Vertex:

    def __init__(self, prefix: str, index: int):
        self.label = prefix + str(index)
        self.index = index

    def __repr__(self):
        return "Vertex: " + self.label

    def __str__(self):
        return "Vertex: " + self.label


class Node(Vertex):

    def __init__(self, prefix: str, index: int, probability: float):
        super(Node, self).__init__(prefix, index)
        self.probability = probability
        self.parent = None

    def __str__(self):
        return self.label + " W: " + str(self.probability)

    def __repr__(self):
        return self.label + " W: " + str(self.probability)

    def set_parent(self, parent):
        self.parent = parent


class Edge:

    def __init__(self, v1: Vertex, v2: Vertex, probability: float):
        self.v1 = v1
        self.v2 = v2
        self.probability = probability

    def __repr__(self):
        return self.v1.label + "-" + str(self.probability) + "-" + self.v2.label

    def __str__(self):
        return self.v1.label + "-" + str(self.probability) + "-" + self.v2.label

    def __eq__(self, other):
        return {self.v1.index, self.v2.index} == {other.v1.index, other.v2.index}


class Tree:

    def __init__(self, root: Node):
        self.root = root
        self.leaves: List[Tree] = []
        self.routes_built = False


    def __str__(self):
        return self.root.label + " L: " + str(self.leaves)

    def __repr__(self):
        return self.root.label + " L: " + str(self.leaves)

    def weight(self):
        return sum(n.weight() for n in self.leaves) + self.root.probability

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

    def get_node_dept(self, node: Node):
        dept = 1
        queue: List[Tree] = self.leaves
        while queue:
            nq = []
            for item in queue:
                if item.root.index == node.index:
                    return dept
                else:
                    nq.extend(item.leaves)
            dept += 1
            queue = nq
        return -1

    def build_routes(self):
        for leave in self.leaves:
            leave : Tree
            leave.root.set_parent(self.root)
            leave.build_routes()
        self.routes_built = True

    def get_path(self, vertex: Vertex) -> List:
        route = []
        if not self.routes_built:
            self.build_routes()
        for node in self.get_dependent_nodes():
            if node.index == vertex.index:
                nx = node
                while nx:
                    route.append(nx)
                    if nx.parent is not None:
                        nx = nx.parent
                    else:
                        return route
        return route


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


class AbstractHuffman:

    def __init__(self):
        self.path = []

    @abstractmethod
    def weight(self):
        pass


class HuffmanDanNode(AbstractHuffman, Node):

    def __init__(self, prefix, index, probability):
        AbstractHuffman.__init__(self)
        Node.__init__(self, prefix, index, probability)

    def weight(self):
        return self.probability

    def __str__(self):
        return f"N:{self.label},{self.probability},{self.path}"

    def __repr__(self):
        return f"N:{self.label},{self.probability},{self.path}"


class HuffmanDanTree(AbstractHuffman, Tree):

    def __init__(self, root: HuffmanDanNode, delta: int, nodes: List[AbstractHuffman]):
        AbstractHuffman.__init__(self)
        Tree.__init__(self, root)
        self.delta = delta
        self.leaves: List = nodes
        self.root: HuffmanDanNode = root

    def weight(self):
        a = sum(x.weight() for x in self.leaves)
        b = self.root.weight() if self.root else 0
        return a + b

