from copy import deepcopy
from typing import List, Dict

from src.adt import Node, EgoTree, BinTree


def calculate_all_egotrees(demand_distribution, delta, indexes: List, prefix, helper_struct):
    egotrees = []

    dd = demand_distribution

    for i in range(len(demand_distribution)):

        if indexes and i not in indexes:
            continue
        nodes = []
        from src.adt import Node
        source: Node

        for j in range(len(demand_distribution)):
            if i == j:
                source = Node(prefix, i, 0)
            elif dd[i][j] + dd[j][i] > 0:
                nodes.append(Node(prefix, j, dd[i][j] + dd[j][i]))

        egotrees.append(create_egotree(source, nodes, delta))


    egotrees, new_demand_matrix = change_nodes_in_egotrees(demand_distribution, delta, egotrees, prefix, helper_struct)
    return egotrees, new_demand_matrix


def change_nodes_in_egotrees(demand_matrix, delta, egotrees, prefix, helper_struct):
    new_demand_matrix = deepcopy(demand_matrix)
    print("---- EGO TREES BEFORE ----")
    for tree in egotrees:
        print(tree)
    print("--------------------------")
    for tree in egotrees:
        print("Tree before", tree)
        for struct in helper_struct:
            leaf_indices = [x.index for x in tree.get_dependent_nodes()]
            if tree.root.index == struct[0].index and struct[1].index in leaf_indices:
                v_tree = [x for x in tree.get_trees() if x.root.index == struct[1].index][0]
            elif tree.root.index == struct[1].index and struct[0].index in leaf_indices:
                v_tree = [x for x in tree.get_trees() if x.root.index == struct[0].index][0]
            else:
                continue

            if v_tree in tree.leaves:
                v_parent = tree
            else:
                v_parent = [x for x in tree.get_trees() if v_tree in x.leaves][0]

            if not struct[2].index in leaf_indices:
                # Mikor L nincs benne a faban
                print("L atveszi V helyet, L=0")
                # print(tree)
                # print("V", v_tree)
                l_tree = BinTree(Node(prefix, struct[2].index, 0))
                l_tree.leaves = v_tree.leaves

                ind = v_parent.leaves.index(v_tree)
                v_parent.leaves[ind] = l_tree
                # print("L", l_tree)
                # print(tree)
                u_index = tree.root.index
                v_index = v_tree.root.index
                l_index = l_tree.root.index

            else:
                # Mikor L benne van a faban
                l_tree = [x for x in tree.get_trees() if x.root.index == struct[2].index][0]
                if l_tree in tree.leaves:
                    l_parent = tree
                else:
                    l_parent = [x for x in tree.get_trees() if l_tree in x.leaves][0]

                u_index = tree.root.index
                v_index = v_tree.root.index
                l_index = l_tree.root.index

                if new_demand_matrix[u_index][l_index] > new_demand_matrix[u_index][v_index]\
                        or tree.get_node_dept(v_tree.root) > tree.get_node_dept(l_tree.root):
                    # Toroljuk V-t a fabol

                    leaves: List[BinTree] = v_tree.leaves
                    ind = v_parent.leaves.index(v_tree)
                    v_parent.leaves.pop(ind)

                    rebalance_tree(v_parent, leaves)

                    print("V-t toroljuk a fabol")
                else:
                    # L atveszi V helyet
                    leaves: List[BinTree] = l_tree.leaves
                    if v_tree in leaves:
                        leaves.remove(v_tree)

                    ind = l_parent.leaves.index(l_tree)
                    l_parent.leaves.pop(ind)
                    l_tree.leaves = v_tree.leaves
                    if len(v_parent.leaves) > 0:
                        ind = v_parent.leaves.index(v_tree)
                        v_parent.leaves[ind] = l_tree
                    else:
                        if v_parent != l_tree:
                            v_parent.push(l_tree)

                    rebalance_tree(l_parent, leaves)

                    print("L atveszi V helyet, L>0")

            weight = new_demand_matrix[u_index][v_index]

            new_demand_matrix[u_index][v_index] = 0
            new_demand_matrix[v_index][u_index] = 0

            new_demand_matrix[u_index][l_index] += weight
            new_demand_matrix[l_index][u_index] += weight

            new_demand_matrix[v_index][l_index] += weight
            new_demand_matrix[l_index][v_index] += weight
            l_tree.root.probability = new_demand_matrix[u_index][l_index]


        if len(tree.leaves) < delta:
            for leave in tree.leaves:
                if len(leave.leaves) > 0:
                    tree_to_move = leave.leaves.pop(0)
                    tree.leaves.append(tree_to_move)
                    break

        print("Tree after", tree)

    return egotrees, new_demand_matrix


def rebalance_tree(l_parent, leaves):
    while leaves:
        if len(leaves) == 2:
            if leaves[0].weight() > leaves[1].weight():
                new_leaves = leaves[0].leaves
                leaves[0].leaves = [leaves[1]]
                l_parent.leaves.append(leaves[0])
                l_parent = leaves[0]
                leaves = new_leaves
            else:
                new_leaves = leaves[1].leaves
                leaves[1].leaves = [leaves[0]]
                l_parent.leaves.append(leaves[1])
                l_parent = leaves[1]
                leaves = new_leaves
        elif len(leaves) == 1:
            l_parent.leaves.extend(leaves)
            break

def create_egotree(source: Node, p: List[Node], delta: int) -> EgoTree:
    p1 = map_probabilities(p)
    p1 = sorted(p1.items(), key=lambda kv: kv[1], reverse=True)
    egotree = EgoTree(source, delta)
    for key, value in p1:
        egotree.push(BinTree(key))
    return egotree

def map_probabilities(p: List[Node]) -> Dict[Node, float]:
    return {n: n.probability for n in p}





def change_nodes_in_egotrees2(trees, helper_struct):
    print("---- EGO TREES BEFORE ----")
    for tree in trees:
        print(tree)
    print("--------------------------")
    for tree in trees:
        print("Tree before", tree)
        for struct in helper_struct:
            leaf_indices = [x.index for x in tree.get_dependent_nodes()]
            if tree.root.index == struct[0].index and struct[1].index in leaf_indices:
                v_tree = [x for x in tree.get_trees() if x.root.index == struct[1].index][0]
            elif tree.root.index == struct[1].index and struct[0].index in leaf_indices:
                v_tree = [x for x in tree.get_trees() if x.root.index == struct[0].index][0]
            else:
                continue

            if v_tree in tree.leaves:
                v_parent = tree
            else:
                v_parent = [x for x in tree.get_trees() if v_tree in x.leaves][0]

            if not struct[2].index in leaf_indices:
                # Mikor L nincs benne a faban
                print("L atveszi V helyet, L=0")
                # print(tree)
                # print("V", v_tree)
                l_tree = BinTree(Node(PREFIX, struct[2].index, 0))
                l_tree.leaves = v_tree.leaves

                ind = v_parent.leaves.index(v_tree)
                v_parent.leaves[ind] = l_tree
                # print("L", l_tree)
                # print(tree)
                u_index = tree.root.index
                v_index = v_tree.root.index
                l_index = l_tree.root.index

            else:
                # Mikor L benne van a faban
                l_tree = [x for x in tree.get_trees() if x.root.index == struct[2].index][0]
                if l_tree in tree.leaves:
                    l_parent = tree
                else:
                    l_parent = [x for x in tree.get_trees() if l_tree in x.leaves][0]

                u_index = tree.root.index
                v_index = v_tree.root.index
                l_index = l_tree.root.index

                if self.new_new_demand_matrix[u_index][l_index] > self.new_new_demand_matrix[u_index][v_index] \
                        or tree.get_node_dept(v_tree.root) > tree.get_node_dept(l_tree.root):
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
                    print("REEEEEEEEEE")
                    tree.push(BinTree(item))

            weight = self.new_new_demand_matrix[u_index][v_index]

            self.new_new_demand_matrix[u_index][v_index] = 0
            self.new_new_demand_matrix[v_index][u_index] = 0

            self.new_new_demand_matrix[u_index][l_index] += weight
            self.new_new_demand_matrix[l_index][u_index] += weight

            self.new_new_demand_matrix[v_index][l_index] += weight
            self.new_new_demand_matrix[l_index][v_index] += weight
            l_tree.root.probability = self.new_new_demand_matrix[u_index][l_index]


        if len(tree.leaves) < self.delta:
            for leave in tree.leaves:
                if len(leave.leaves) > 0:
                    tree_to_move = leave.leaves.pop(0)
                    tree.leaves.append(tree_to_move)
                    break

        print("Tree after", tree)
