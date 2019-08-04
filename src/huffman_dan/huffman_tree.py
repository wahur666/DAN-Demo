from typing import List

from adt import HuffmanDanNode, AbstractHuffman, HuffmanDanTree, Node, Tree


def get_nodes(tree: HuffmanDanTree) -> List[HuffmanDanNode]:
    nodes = []
    for index, leaf in enumerate(tree.leaves):
        if isinstance(leaf, HuffmanDanNode):
            nodes.append(leaf)
        elif isinstance(leaf, HuffmanDanTree):
            nodes.extend(get_nodes(leaf))
    return nodes

# Simple push up on each branch


def calculate_all_push_up_trees(demand_distribution: List[List[float]], delta: int, indices: List[Node] = None, prefix: str = "T") -> List[Tree]:
    pushup_trees = []

    dd = demand_distribution

    for i in range(len(dd)):

        if indices and i not in indices:
            continue

        nodes = []
        source: HuffmanDanNode = None

        for j in range(len(dd)):
            if i == j:
                source = HuffmanDanNode(prefix, i, 0)
            elif dd[i][j] + dd[j][i] > 0:
                nodes.append(HuffmanDanNode(prefix, j, dd[i][j] + dd[j][i]))

        tree = create_push_up_tree(delta, nodes)
        tree.root = source

        pushup_trees.append(tree)
    return pushup_trees


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

# Breadth-first solution

def calculate_all_bfs_trees(demand_distribution: List[List[float]], delta: int, indices: List[Node] = None, prefix: str = "T") -> List[Tree]:
    bfs_trees = []

    dd = demand_distribution

    for i in range(len(dd)):

        if indices and i not in indices:
            continue

        nodes = []
        source: HuffmanDanNode = None

        for j in range(len(dd)):
            if i == j:
                source = HuffmanDanNode(prefix, i, 0)
            elif dd[i][j] + dd[j][i] > 0:
                nodes.append(HuffmanDanNode(prefix, j, dd[i][j] + dd[j][i]))

        tree = create_bfs_tree(delta, nodes)
        tree.root = source

        bfs_trees.append(tree)
    return bfs_trees

def create_bfs_tree(delta, nodes: List[HuffmanDanNode]):
    nodes.sort(key=lambda x: x.weight(), reverse=True)
    root = HuffmanDanTree(None, delta, [])
    for node in nodes:
        find_next_free_position(root, node)
    return root


# Naive Solution

#
# def detach_leaf(tree: HuffmanDanTree, path: List[int]):
#     temp_tree = tree
#     for i in path[:-1]:
#         temp_tree = temp_tree.leaves[i]
#     temp_tree.leaves[path[-1]] = None
#
#
# def calculate_node_paths(tree: HuffmanDanTree):
#     for index, leaf in enumerate(tree.leaves):
#         for i in tree.path:
#             leaf.path.append(i)
#         leaf.path.append(index)
#         if isinstance(leaf, HuffmanDanTree):
#             calculate_node_paths(leaf)
#
#
# def remove_dead_branches(root: HuffmanDanTree):
#     leaves = []
#     for leaf in root.leaves:
#         leaves.append(leaf)
#
#     while leaves:
#         leaf = leaves.pop(0)
#         if isinstance(leaf, HuffmanDanNode):
#             continue
#         leaf.leaves = [x for x in leaf.leaves if x is not None]
#         leaves.extend(leaf.leaves)
#
#
# def naive_push_up(delta, nodes):
#     root = HuffmanDanTree(None, delta, nodes)
#     calculate_node_paths(root)
#     leaves = root.leaves.copy()
#     while leaves:
#         leaf = leaves.pop(0)
#         if leaf is not None and isinstance(leaf, HuffmanDanTree):
#             nodes = get_nodes(leaf)
#             print(nodes)
#             max_weight_leaf = max(nodes, key=lambda x: x.weight())
#             leaf.root = max_weight_leaf
#             detach_leaf(root, max_weight_leaf.path)
#             leaves.extend(leaf.leaves)
#     remove_dead_branches(root)
#     return root
#
