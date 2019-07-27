codes = {
    'e':4,
    'n':2,
    'o':1,
    'u':1,
    'a':4,
    't':2,
    'm':2,
    'i':2,
    'x':1,
    'p':1,
    'h':2,
    's':2,
    'r':1,
    'l':1,
    'f':3,
    ' ':7
}

class HuffmanNode:

    def __init__(self, name, value):
        self.name = name
        self.prob = value

    def value(self):
        return self.prob

    def __repr__(self):
        return f"N:{self.name} V:{self.value()}"

    def __str__(self):
        return f"N:{self.name} V:{self.value()}"


class HuffmanDeltaTree:

    def __init__(self, delta, *nodes):
        if delta != len(nodes):
            raise Exception(f"Argument number does not equal! {delta} != {len(nodes)}")
        self.leaves = nodes

    def value(self):
        return sum(x[1].value() for x in self.leaves)

    def __repr__(self):
        return f"HDT{len(self.leaves), [{self.leaves}]}"

class HuffmanBinTree(HuffmanDeltaTree):

    def __init__(self, *nodes):
        super().__init__(2, *nodes)


class HuffmanNetwork:

    def __init__(self, codes):
        self.codes = codes
        self.huffman_delta_tree = None

    def create_delta_tree(self, delta):
        while len(self.codes) != delta:
            node1 = min(self.codes.items(), key=lambda x: x[1].value())
            self.codes.pop(node1[0])
            node2 = min(self.codes.items(), key=lambda x: x[1].value())
            self.codes.pop(node2[0])
            tree = HuffmanBinTree(node1, node2)
            self.codes[node1[0] + node2[0]] = tree
            # print(tree)
        self.huffman_delta_tree = HuffmanDeltaTree(delta, *self.codes)
        return self.huffman_delta_tree


if __name__ == '__main__':

    codes = { k: HuffmanNode(k, v) for k, v in codes.items() }

    hnet = HuffmanNetwork(codes)
    hdt = hnet.create_delta_tree(4)

    print(hdt)


