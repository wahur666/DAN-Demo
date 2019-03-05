from typing import List, Dict


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


def create_egotree(source: Node, p: List[Node], delta: int):
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

egotree = create_egotree(source, p, n)

print("Root: " + str(egotree.root))
for item in egotree.leaves:
    print(item)

print(egotree.dept())
print(egotree.weight())
