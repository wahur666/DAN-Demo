import networkx as nx
import matplotlib.pyplot as plt
from network import Network

demand_distribution = [[0, 3, 4, 1, 1, 1, 1],
                       [3, 0, 2, 0, 1, 0, 4],
                       [4, 2, 0, 2, 0, 0, 4],
                       [1, 0, 2, 0, 3, 0, 0],
                       [1, 1, 0, 3, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 3],
                       [1, 4, 4, 0, 0, 3, 0]]

g = Network(demand_distribution)
print(g)
g.create_dan()

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

plt.show()
