from itertools import cycle, combinations
from copy import deepcopy
from adt import *

PREFIX = "T"


class Network:

    def __init__(self, demand_matix : List):
        self.vertices = []
        self.edges = []
        self.avg_deg = 0
        self.demand_matrix = demand_matix

        self.new_demand_matrix = deepcopy(demand_matix)
        self.new_new_demand_matrix = deepcopy(demand_matix)
        self.routing_scheme = []  # a.k.a new edges

        self.build_graph()

    def add_vertex(self, vertex: Vertex):
        self.vertices.append(vertex)

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def __repr__(self):
        return str(self.edges)

    def __str__(self):
        return str(self.edges)

    def build_graph(self):

        for i in range(len(self.demand_matrix)):
            self.vertices.append(Vertex(PREFIX, i))

        for i in range(len(self.demand_matrix)):
            for j in range(i + 1, len(self.demand_matrix)):
                if self.demand_matrix[i][j] > 0:
                    e = Edge(self.vertices[i], self.vertices[j], self.demand_matrix[i][j])
                    self.add_edge(e)

    def get_vertex(self, index: int):
        return list(filter(lambda x : x.index == index, self.vertices))

    def create_dan(self, delta = None):
        self.delta = delta
        # Classifying points
        degs = []
        for vert in self.vertices:
            deg = sum(e.v1 is vert or e.v2 is vert for e in self.edges)
            degs.append((vert, deg))
        # print(degs)
        self.avg_deg = sum(x[1] for x in degs) / len(degs)

        degs.sort(key=lambda x: x[1], reverse=True)

        H = degs[0:round(len(degs) / 2)]
        L = degs[round(len(degs) / 2):]

        while len(L) > 0 and L[0][1] > self.avg_deg:
            H.append(L.pop(0))
        L.sort(key=lambda x: x[1])

        print(H)
        print(L)

        self.H = H
        self.L = L

        self.add_helpers()

    def add_helpers(self):
        c_L = cycle(self.L)
        self.helper_struct = []
        already_assinged = {}
        for edge in self.edges:
            H_v = [x[0] for x in self.H]
            if edge.v1 in H_v and edge.v2 in H_v:
                weight = edge.probability
                u_index = edge.v1.index
                v_index = edge.v2.index

                a1 = [x.index for x in already_assinged[edge.v1]] if edge.v1 in already_assinged else []
                a2 = [x.index for x in already_assinged[edge.v2]] if edge.v2 in already_assinged else []

                a1.extend(a2)
                s = set(a1)
                l = next(c_L)
                while l[0].index in s:
                    l = next(c_L)

                l_index = l[0].index
                self.new_demand_matrix[u_index][v_index] = 0
                self.new_demand_matrix[v_index][u_index] = 0

                self.new_demand_matrix[u_index][l_index] += weight
                self.new_demand_matrix[l_index][u_index] += weight

                self.new_demand_matrix[v_index][l_index] += weight
                self.new_demand_matrix[l_index][v_index] += weight

                # print(u_index, weight, v_index, l_index)
                print(edge.v1, edge.v2, "Helper:", l[0])
                if edge.v1 not in already_assinged:
                    already_assinged[edge.v1] = [l[0]]
                else:
                    already_assinged[edge.v1].append(l[0])
                if edge.v2 not in already_assinged:
                    already_assinged[edge.v2] = [l[0]]
                else:
                    already_assinged[edge.v2].append(l[0])
                self.helper_struct.append((edge.v1, edge.v2, l[0]))
        self.calculate_egotrees(self.delta)

    def calculate_egotrees(self, delta=None):
        for i in self.demand_matrix:
            print(i)

        H_i = [x[0].index for x in self.H]

        deltaN = delta or 12 * int(round(self.avg_deg))
        if not self.delta:
            self.delta = deltaN
        self.egotrees = calculate_all_egotrees(self.demand_matrix, deltaN, H_i)
        for i in self.egotrees:
            print(i)

        self.change_nodes_in_egotrees()

    def change_nodes_in_egotrees(self):

        print("---- EGO TREES BEFORE ----")
        for tree in self.egotrees:
            print(tree)
        print("--------------------------")
        for tree in self.egotrees:
            print("Tree before", tree)
            for struct in self.helper_struct:
                leave_indices = [x.index for x in tree.get_dependent_nodes()]
                if tree.root.index == struct[0].index and struct[1].index in leave_indices:
                    v_tree = [x for x in tree.get_trees() if x.root.index == struct[1].index][0]
                elif tree.root.index == struct[1].index and struct[0].index in leave_indices:
                    v_tree = [x for x in tree.get_trees() if x.root.index == struct[0].index][0]
                else:
                    continue

                if v_tree in tree.leaves:
                    v_parent = tree
                else:
                    v_parent = [x for x in tree.get_trees() if v_tree in x.leaves][0]

                if not struct[2].index in leave_indices:
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

                    weight = self.new_new_demand_matrix[u_index][v_index]

                    self.new_new_demand_matrix[u_index][v_index] = 0
                    self.new_new_demand_matrix[v_index][u_index] = 0

                    self.new_new_demand_matrix[u_index][l_index] += weight
                    self.new_new_demand_matrix[l_index][u_index] += weight

                    self.new_new_demand_matrix[v_index][l_index] += weight
                    self.new_new_demand_matrix[l_index][v_index] += weight

                    l_tree.root.probability = self.new_new_demand_matrix[u_index][l_index]
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

                    if self.new_new_demand_matrix[u_index][l_index] > self.new_new_demand_matrix[u_index][v_index]\
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

                    weight = self.new_new_demand_matrix[u_index][v_index]

                    self.new_new_demand_matrix[u_index][v_index] = 0
                    self.new_new_demand_matrix[v_index][u_index] = 0

                    self.new_new_demand_matrix[u_index][l_index] += weight
                    self.new_new_demand_matrix[l_index][u_index] += weight

                    self.new_new_demand_matrix[v_index][l_index] += weight
                    self.new_new_demand_matrix[l_index][v_index] += weight
                    l_tree.root.probability = self.new_new_demand_matrix[u_index][l_index]

                    for item in nodes_to_redistribute:
                        print("REEEEEEEEEE")
                        tree.push(BinTree(item))

            if len(tree.leaves) < self.delta:
                for leave in tree.leaves:
                    if len(leave.leaves) > 0:
                        tree_to_move = leave.leaves.pop(0)
                        tree.leaves.append(tree_to_move)
                        break

            print("Tree after", tree)


        self.union_egotrees()


    def union_egotrees(self):
        for item in self.new_demand_matrix:
            print(item)
        print("------")
        for item in self.new_new_demand_matrix:
            print(item)

        queue = []
        for tree in self.egotrees:
            print(tree)
            queue.append(tree)
            while queue:
                tree_to_process = queue.pop(0)
                for subtree in tree_to_process.leaves:
                    queue.append(subtree)
                    edge = Edge(tree_to_process.root, subtree.root, subtree.weight())
                    if edge not in self.routing_scheme:
                        # u_index = edge.v1.index
                        # v_index = edge.v2.index
                        # edge.probability = self.new_new_demand_matrix[u_index][v_index]
                        if edge.probability > 0:
                            self.routing_scheme.append(edge)
                    else:
                        print("Noveljuk az elet")
                        ind = self.routing_scheme.index(edge)
                        self.routing_scheme[ind].probability += edge.probability

        L_v = [x[0] for x in self.L]

        L_perms = combinations(L_v, 2)
        print(str(self.routing_scheme))

        for u, v in L_perms:
            u_index = u.index
            v_index = v.index
            edge = Edge(u, v, self.new_new_demand_matrix[u_index][v_index])
            if edge.probability > 0:
                if edge not in self.routing_scheme:
                    self.routing_scheme.append(edge)
                else:
                    ind = self.routing_scheme.index(edge)
                    self.routing_scheme[ind].probability += edge.probability

        print(str(self.routing_scheme))

        self.calculate_congestion_and_avglen()

    def calculate_congestion_and_avglen(self):
        all_path = {}
        for tree in self.egotrees:
            tree.build_routes()
            tree_paths = []
            for struct in self.helper_struct:
                if struct[0].index == tree.root.index or struct[1].index == tree.root.index:
                    r = tree.get_path(struct[2])
                    r.reverse()
                    tree_paths.append(r)
            all_path[tree_paths[0][0].index] = tree_paths

        self.H_i = [x[0].index for x in self.H]
        self.L_i = [x[0].index for x in self.L]

        # max az utak trolodasa, tordoldas sum az u-v ut osszes elet
        congestion = 0
        most_congested_route = None

        # sum az ut hossza megszorozva ut valoszinusege
        avg_route_len = 0

        for i in range(len(self.demand_matrix)-1):
            for j in range(i + 1, len(self.demand_matrix)):
                if self.demand_matrix[i][j]:
                    print(i, j)
                    if i in self.H_i and j in self.H_i:
                        #Mikor H x H van
                        #Ket kulonbozo pont osszeforgatasa
                        route = []
                        for struct in self.helper_struct:
                            if struct[0].index == i and struct[1].index == j:
                                route = self.find_route(all_path, i, struct[2].index)
                                route2 = self.find_route(all_path, j, struct[2].index)[:-1]
                                route2.reverse()
                                route.extend(route2)
                                break

                        if route:
                            con = self.calculate_congestion(route)
                            congestion, most_congested_route = self.update_congestion(con, congestion,
                                                                                      most_congested_route,
                                                                                      route)
                            avg_route_len += self.demand_matrix[i][j] * len(route)
                            print("Congestion:", con, route)
                            print("Route Length:", self.demand_matrix[i][j] * len(route))


                        #print("Full combined:", route1)
                    elif i in self.H_i and j in self.L_i:
                        #Megkeresessük az utat
                        route = self.find_route(all_path, i, j)
                        if route:
                            con = self.calculate_congestion(route)
                            congestion, most_congested_route = self.update_congestion(con, congestion, most_congested_route,
                                                                                      route)
                            avg_route_len += self.demand_matrix[i][j] * len(route)
                            print("Congestion:", con, route)
                            print("Route Length:", self.demand_matrix[i][j] * len(route))
                            #print(route)

                    elif i in self.L_i and j in self.H_i:
                        route = self.find_route(all_path, j, i)
                        if route:
                            con = self.calculate_congestion(route)
                            congestion, most_congested_route = self.update_congestion(con, congestion, most_congested_route,
                                                                                      route)
                            avg_route_len += self.demand_matrix[i][j] * len(route)
                            print("Congestion:", con, route)
                            print("Route Length:", self.demand_matrix[i][j] * len(route))

                        #print(route)
                    else:
                        # Mikor mindketto L
                        # Direkt kapcsolat, hossz 1
                        con, route = 0, []
                        for edge in self.routing_scheme:
                            if edge.v1.index == i and edge.v2.index == j or edge.v1.index == j and edge.v2.index == i:
                                con = edge.probability
                                route = [edge.v1, edge.v2]
                                break
                        congestion, most_congested_route = self.update_congestion(con, congestion, most_congested_route,
                                                                                  route)
                        avg_route_len += self.demand_matrix[i][j] * len(route)
                        print("Congestion:", con, route)
                        print("Route Length:", self.demand_matrix[i][j] * len(route))

        full_weight = sum_aa(self.demand_matrix)
        print("------- Summary -------")

        print("Most congested route:", congestion, congestion/full_weight, most_congested_route)
        print("Average weighted routlength:", avg_route_len/full_weight)

        print("-----------------------")
        #print(all_path)

    def update_congestion(self, con, congestion, most_congested_route, route):
        if con > congestion:
            congestion = con
            most_congested_route = route
        return congestion, most_congested_route

    def calculate_congestion(self, route):
        con = 0
        for k in range(len(route) - 1):
            start = route[k]
            end = route[k + 1]
            self.routing_scheme: List[Edge]
            for edge in self.routing_scheme:
                if edge.v1.index == start.index and edge.v2.index == end.index or \
                        edge.v1.index == end.index and edge.v2.index == start.index:
                    con += edge.probability
                    break
        return con

    def find_route(self, all_path, i, j) -> List[Node]:
        for route in all_path[i]:
            assert route, List[Node]
            if route[0].index == i and route[-1].index == j:
                return route.copy()


def map_probabilities(p: List[Node]) -> Dict:
    return {n: n.probability for n in p}


def create_egotree(source: Node, p: List[Node], delta: int) -> EgoTree:
    p1 = map_probabilities(p)
    p1 = sorted(p1.items(), key=lambda kv: kv[1], reverse=True)
    egotree = EgoTree(source, delta)
    for key, value in p1:
        egotree.push(BinTree(key))
    return egotree


def sum_aa(aa):
    return sum([sum(x) for x in aa])


def normalize100(demand_distribution):
    sum_of_items = sum_aa(demand_distribution)
    multiplier = 100 / sum_of_items
    normalized = [list(map(lambda z: z * multiplier, x)) for x in demand_distribution]
    return normalized


def calculate_all_egotrees(demand_distribution, delta, indexes : List = None):
    egotrees = []

    dd = demand_distribution

    for i in range(len(demand_distribution)):

        if indexes and i not in indexes:
            continue
        nodes = []
        source: Node

        for j in range(len(demand_distribution)):
            if i == j:
                source = Node(PREFIX, i, 0)
            elif dd[i][j] + dd[j][i] > 0:
                nodes.append(Node(PREFIX, j, dd[i][j] + dd[j][i]))

        egotrees.append(create_egotree(source, nodes, delta))

    return egotrees

