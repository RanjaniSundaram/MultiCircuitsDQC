import networkx as nx
import random as rn
from MinQAP import*
import copy

class NetworkGraph:
    def __init__(self, numMems, num):
        self.numMems = numMems
        self.num = num
        self.pieces = self.create_pieces()
        self.qG= None
        self.node_map = {}  
        self.network_graph = self.create_network_graph()
        self.main_network_graph= self.create_main_network_graph()
    
    def create_pieces(self):
        pieces = []
        a = self.numMems
        if a == self.num:
            pieces = [1] * self.num
        else:
            for idx in range(self.num - 1):
                #pieces.append(randint(1, a - sum(pieces) - self.num + idx))
                pieces.append(self.numMems//self.num)
            pieces.append(a - sum(pieces))
        print(pieces)
        return pieces

    def create_network_graph(self):
        self.qG = Create_qG(self.num)
        return qGraph_to_Networkx(self.qG)

    def create_main_network_graph(self):
        main_network_graph = nx.Graph()
        print(sorted(self.network_graph.nodes()))
        for node in sorted(self.network_graph.nodes()):
            print(node)
            print(self.pieces[node])
            temp_graph = gnp_random_connected_graph(self.pieces[node], 0.9, str(node))
            main_network_graph = nx.disjoint_union(main_network_graph, temp_graph)

        for subnode, att in main_network_graph.nodes(data=True):
            self.node_map[subnode] = int(att['Label'])
        for edge in self.network_graph.edges():
            left_nodes = [x for x, y in main_network_graph.nodes(data=True) if y['Label'] == str(edge[0])]
            right_nodes = [x for x, y in main_network_graph.nodes(data=True) if y['Label'] == str(edge[1])]
            for left in left_nodes:
                for right in right_nodes:
                    main_network_graph.add_edge(left, right, weight=10)

        return main_network_graph
    
    def AddDummyNodes(self,n):
        G = nx.Graph()
        for i in range(n):
            G.add_node(self.numMems+i, Label=str(self.numMems+i))
            if i>=1:
                G.add_edge(self.numMems+i-1, self.numMems+i, weight=10000000)
        NewNetworkGraph=nx.disjoint_union(self.main_network_graph, G)
        NewNetworkGraph.add_edge(0,self.numMems, weight=10)
        #fig, axes = plt.subplots(5,5,dpi=100)

        elarge = [(u, v) for (u, v, d) in NewNetworkGraph.edges(data=True)]
        esmall = [(u, v) for (u, v, d) in NewNetworkGraph.edges(data=True)]
        pos = nx.spring_layout(NewNetworkGraph, seed=7)
        nx.draw_networkx_nodes(NewNetworkGraph, pos, node_size=10)

        # edges
        nx.draw_networkx_edges(NewNetworkGraph, pos, edgelist=elarge, width=1)
        #nx.draw_networkx_edges(NewNetworkGraph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
        edge_labels = nx.get_edge_attributes(NewNetworkGraph, "weight")
        nx.draw_networkx_edge_labels(NewNetworkGraph, pos, edge_labels)
        plt.savefig("biggraph.png") 

        return NewNetworkGraph

    def __repr__(self):
        return f"NetworkGraph(numqubits={self.numqubits}, num={self.num}, pieces={self.pieces})"