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
        self.leaves: List = nodes
        self.base: HuffmanDanNode = None

    def weight(self):
        a = sum(x.weight() for x in self.leaves)
        b = self.base.weight() if self.base else 0
        return a + b


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

def get_nodes(tree: HuffmanDanTree) -> List[HuffmanDanNode]:
    nodes = []
    for index, leaf in enumerate(tree.leaves):
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
    root = naive_push_up(delta, nodes)

    return root


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
    root = HuffmanDanTree(delta, nodes)
    calculate_node_paths(root)
    leaves = root.leaves.copy()
    while leaves:
        leaf = leaves.pop(0)
        if leaf is not None and isinstance(leaf, HuffmanDanTree):
            nodes = get_nodes(leaf)
            print(nodes)
            max_weight_leaf = max(nodes, key=lambda x: x.weight())
            leaf.base = max_weight_leaf
            detach_leaf(root, max_weight_leaf.path)
            leaves.extend(leaf.leaves)
    remove_dead_branches(root)
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

