import networkx as nx
import matplotlib.pyplot as plt
from network import Network, List
from adt import EgoTree


def render_egotrees(egotrees: List[EgoTree]):
    ind = 100
    for tree in egotrees:
        ind += 1
        Ge = nx.Graph()
        queue = [tree]
        while queue:
            ctree = queue.pop(0)
            root = ctree.root.label
            Ge.add_node(root, label=root)
            for leave in ctree.leaves:
                queue.append(leave)
                Ge.add_node(leave.root.label)
                Ge.add_edge(root, leave.root.label, w=leave.weight())
        colors = []
        for node in Ge:
            if node == tree.root.label:
                # sarga a gyoker szine, mivel nincs tree layout
                colors.append('yellow')
            else:
                colors.append('red')
        pos = nx.kamada_kawai_layout(Ge)
        plt.figure(ind)
        nx.draw(Ge, pos, node_color=colors)
        labels = nx.get_node_attributes(Ge, 'label')
        nx.draw_networkx_labels(Ge, pos, labels)
        weights = nx.get_edge_attributes(Ge, 'w')
        nx.draw_networkx_edge_labels(Ge, pos, weights)


demand_distribution = [[0, 3, 4, 1, 1, 1, 1],
                       [3, 0, 2, 0, 1, 0, 4],
                       [4, 2, 0, 2, 0, 0, 4],
                       [1, 0, 2, 0, 3, 0, 0],
                       [1, 1, 0, 3, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 3],
                       [1, 4, 4, 0, 0, 3, 0]]

g = Network(demand_distribution)
print(g)

# Itt adjuk meg, hogy mennyi legyen a Delta az egoree generalaskor
# None -> 12 * atlag fokszam
# x > 1 -> annyi fokszam
g.create_dan(3)

G = nx.Graph()
for i in range(len(demand_distribution)):
    G.add_node(i, label=str(i))

for i in range(len(demand_distribution)-1):
    for j in range(i + 1, len(demand_distribution)):
        if demand_distribution[i][j] > 0:
            G.add_edge(i, j, w=demand_distribution[i][j])

pos = nx.circular_layout(G)
plt.figure(200)
nx.draw(G, pos)
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels)
weights = nx.get_edge_attributes(G, 'w')
nx.draw_networkx_edge_labels(G, pos, weights)


Gn = nx.Graph()
for i in range(len(demand_distribution)):
    Gn.add_node(i, label=str(i))

for edge in g.routing_scheme:
    Gn.add_edge(int(edge.v1.label[1:]), int(edge.v2.label[1:]), w=edge.probability)

pos = nx.circular_layout(Gn)
plt.figure(300)

nx.draw(Gn, pos, with_labels=False)
labels = nx.get_node_attributes(Gn, 'label')
nx.draw_networkx_labels(Gn, pos, labels)
weights = nx.get_edge_attributes(Gn, 'w')
nx.draw_networkx_edge_labels(Gn, pos, weights)

render_egotrees(g.egotrees)

plt.show()
