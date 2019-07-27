from typing import List


class AdaptiveNode:

    def __init__(self, code='', internal_node=False, nyt=False):
        self.parent: AdaptiveNode = None
        self.left: AdaptiveNode = None
        self.right: AdaptiveNode = None
        self.code = code
        self.frequency = 0
        self.internal_node = internal_node
        self.nyt = nyt

class AdaptiveTree:

    def __init__(self):
        self.root: AdaptiveNode = None
        self.leaf_to_increment: AdaptiveNode = None
        self.codes_in_the_tree: List[AdaptiveNode] = []

    def update(self, code):

        q = AdaptiveNode(code)

        if self.root is None:
            self.root = AdaptiveNode(internal_node=True)
            self.root.left = AdaptiveNode(nyt=True)
            self.root.left.parent = self.root
            self.codes_in_the_tree.append(self.root.left)

            node = q

            self.root.right = node
            self.root.right.parent = self.root

            q = self.root
            self.leaf_to_increment = q.right
            self.codes_in_the_tree.append(node)
        else:
            # q
            item = next((x for x in self.codes_in_the_tree if x.code == code), None)
            if item:
                # Benne van mar a faban a karakter
                leader_of_block = next((x for x in self.codes_in_the_tree if x.frequency == item.frequency), None)
                index_of_leader = self.codes_in_the_tree.index(leader_of_block)
                index_of_item = self.codes_in_the_tree.index(item)
                self.codes_in_the_tree[index_of_leader], self.codes_in_the_tree[index_of_item] = self.codes_in_the_tree[index_of_item], self.codes_in_the_tree[index_of_leader]

                tmpNode = AdaptiveNode()
                tmpNode.parent = leader_of_block.parent
                tmpNode.left = leader_of_block.left
                tmpNode.right = leader_of_block.right

                leader_of_block.parent = item.parent
                leader_of_block.right = item.right
                leader_of_block.left = item.left

                item.parent = tmpNode.parent
                item.right = tmpNode.right
                item.left = tmpNode.left

                q = item

                if q.parent.left.nyt:
                    self.leaf_to_increment = q
                    q = item.parent

            else:
                item = next((x for x in self.codes_in_the_tree if x.nyt == True), None)
                index_of_nyt = self.codes_in_the_tree.index(item)
                parent_of_nyt = item.parent

                new_node = AdaptiveNode(internal_node=True)
                new_node.left = AdaptiveNode(nyt=True)
                new_node.left.parent = new_node

                node = q

                new_node.right = node
                new_node.right.parent = new_node

                q = new_node
                self.leaf_to_increment = new_node.right

                parent_of_nyt.left = new_node

                self.codes_in_the_tree[index_of_nyt] = new_node.left

        while q is not self.root:
            self.slide_and_increment(q)

        if self.leaf_to_increment is not None:
            self.slide_and_increment(self.leaf_to_increment)

    def slide_and_increment(self, leaf: AdaptiveNode):
        wt = leaf.frequency
        pass


if __name__ == '__main__':
    codes = {
        'e': 4,
        'n': 2,
        'o': 1,
        'u': 1,
        'a': 4,
        't': 2,
        'm': 2,
        'i': 2,
        'x': 1,
        'p': 1,
        'h': 2,
        's': 2,
        'r': 1,
        'l': 1,
        'f': 3,
        ' ': 7
    }
