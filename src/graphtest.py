import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

demand_distribution = [[0, 3, 4, 1, 1, 1, 1],
                       [3, 0, 2, 0, 1, 0, 4],
                       [4, 2, 0, 2, 0, 0, 4],
                       [1, 0, 2, 0, 3, 0, 0],
                       [1, 1, 0, 3, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 3],
                       [1, 4, 4, 0, 0, 3, 0]]

G.add_nodes_from(list(range(7)))

for i in range(len(demand_distribution)-1):
    for j in range(i + 1, len(demand_distribution)):
        if demand_distribution[i][j] > 0:
            G.add_edge(i, j, weight=demand_distribution[i][j])

pos = nx.circular_layout(G)
plt.figure(200)
nx.draw(G, pos)


Gn = nx.Graph()
new_demand = [[0, 0, 0, 5, 2, 4, 0],
              [0, 0, 0, 4, 3, 3, 0],
              [0, 0, 0, 6, 2, 4, 0],
              [5, 4, 6, 0, 3, 0, 4],
              [2, 3, 2, 3, 0, 0, 1],
              [4, 3, 4, 0, 0, 0, 7],
              [0, 0, 0, 4, 1, 7, 0]]

Gn.add_nodes_from(list(range(7)))

for i in range(len(new_demand)-1):
    for j in range(i + 1, len(new_demand)):
        if new_demand[i][j] > 0:
            Gn.add_edge(i, j, w=new_demand[i][j])

pos = nx.circular_layout(Gn)
plt.figure(300)

nx.draw(Gn, pos, with_labels=False)
labes = nx.get_edge_attributes(Gn, 'w')
nx.draw_networkx_edge_labels(Gn, pos, labes)

plt.show()

#
# # Step 1: Build up a graph
# G = nx.Graph()
# G.add_node('n1', alias='source')
# G.add_node('n2', alias='sink')
# G.add_edge('n1', 'n2', route='route 1')
#
# # Step 2: Draw a graph and suppress node labels (node id)
# pos = nx.spring_layout(G)
# nx.draw_networkx(G, pos, with_labels=False)  # OR: nx.draw(G, pos)
#
# # Step 3: Draw the graph with the specific node labels and edge labels
# # node labels
# node_labels = nx.get_node_attributes(G, 'alias')
# nx.draw_networkx_labels(G, pos, node_labels)
#
# # edge labels
# edge_labels = nx.get_edge_attributes(G, 'route')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#
# # plt.axis('off')
# plt.show()