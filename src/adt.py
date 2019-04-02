from typing import List, Dict

class Node:

    def __init__(self, label: str, probability: float):
        self.label = label
        self.probability = probability

    def __str__(self):
        return self.label + " W: " + str(self.probability)

    def __repr__(self):
        return self.label + " W: " + str(self.probability)

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
