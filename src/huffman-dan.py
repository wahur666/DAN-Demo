from abc import abstractmethod
from typing import List, Dict


class AbstractHuffman:

    def __init__(self):
        self.path = []

    @abstractmethod
    def weight(self):
        pass


class HuffmanDanNode(AbstractHuffman):

    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = value

    def weight(self):
        return self.value

    def __str__(self):
        return f"N:{self.name},{self.value},{self.path}"

    def __repr__(self):
        return f"N:{self.name},{self.value},{self.path}"

class HuffmanDanTree(AbstractHuffman):

    def __init__(self, delta, nodes: List[AbstractHuffman]):
        super().__init__()
        self.delta = delta
        self.leaves = nodes
        self.base = None

    def weight(self):
        return sum(x.weight() for x in self.leaves)


def get_nodes(tree: HuffmanDanTree) -> List[HuffmanDanNode]:
    nodes = []
    for index, leaf in enumerate(tree.leaves):
        for i in tree.path:
            leaf.path.append(i)
        leaf.path.append(index)
        if isinstance(leaf, HuffmanDanNode):
            nodes.append(leaf)
        elif isinstance(leaf, HuffmanDanTree):
            nodes.extend(get_nodes(leaf))
    return nodes


def create_push_up_tree(delta, nodes: List[AbstractHuffman]):
    while len(nodes) != delta:
        node1 = min(nodes, key=lambda x: x.weight())
        nodes.pop(nodes.index(node1))
        node2 = min(nodes, key=lambda x: x.weight())
        nodes.pop(nodes.index(node2))
        tree = HuffmanDanTree(2, [node1, node2])
        nodes.append(tree)
    root = HuffmanDanTree(delta, nodes)
    for index, leaf in enumerate(root.leaves):
        leaf.path.append(index)
        if isinstance(leaf, HuffmanDanTree):
            nodes = get_nodes(leaf)
            print(nodes)


    return root



if __name__ == '__main__':
    codes = [
        HuffmanDanNode('t0', 1),
        HuffmanDanNode('t1', 1),
        HuffmanDanNode('t2', 1),
        HuffmanDanNode('t3', 1),
        HuffmanDanNode('t4', 1),
        HuffmanDanNode('t5', 1),
        HuffmanDanNode('t6', 2),
        HuffmanDanNode('t7', 2),
        HuffmanDanNode('t8', 2),
        HuffmanDanNode('t9', 2),
        HuffmanDanNode('t10', 2),
        HuffmanDanNode('t11', 2),
        HuffmanDanNode('t12', 3),
        HuffmanDanNode('t13', 4),
        HuffmanDanNode('t14', 4),
        HuffmanDanNode('t15', 7),
    ]

    a = create_push_up_tree(4, codes)
    print(a)

