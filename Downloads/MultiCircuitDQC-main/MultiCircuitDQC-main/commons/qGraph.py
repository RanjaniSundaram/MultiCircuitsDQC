from typing import List
from .qChannel import qChannel
from .qNode import qNode
import networkx as nx
import matplotlib.pyplot as plt
from .tools import *


class qGraph:
    """qGraph, G=(V, E), is a graph that representing a quantum network.
    qGraph is assumed to be undirected graph; two edges for an adjacent pair of qNodes.
    Nodes and edges (qChannels) are assumed to be distinct and no parallel edges between a pair of nodes is
    assumed to be exist."""

    def __init__(self, V: List[qNode] = None, E: List[qChannel] = None, time_matrix=None, avg_classical_time=0.0):
        """
        :param V: List of qNodes. Duplicates are removed
        :param E: List of qChannels that connect qNodes together. Parallel edges are removed.
        """
        self._v_map, self._e_map = {}, {}  # variables to avoid duplicates
        self._V, self._E = [], []  # immutable
        if time_matrix:
            #self.distance_mat=distance_mat
            self.time_matrix=time_matrix
            self.avg_classical_time=avg_classical_time
        else:
            self.time_matrix=None
            self.avg_classical_time=self.ComputeDistanceMatrix(V)
        
        for v in V:
            self.add_node(v)
        for e in E:
            self.add_edge(e)

    def add_node(self, node: qNode):
        """add an immutable qNode. Avoiding duplicates"""
        if node.id not in self._v_map:
            self._v_map[node.id] = node.clone()
            self._V.append(self._v_map.get(node.id))

    def ComputeDistanceMatrix(self,V:List[qNode]):
        n=len(V)
        tot_time=0.0
        #distance_mat=[[0]*n for i in range(n)]
        self.time_matrix=[[0]*n for i in range(n)]
        for node1 in V:
            for node2 in V:
                d=node1.loc.distance(node2.loc)
                #distance_mat[node1.id][node2.id]=d
                tot_time+=d/LIGHT_SPEED
                self.time_matrix[node1.id][node2.id]=d/LIGHT_SPEED
        #self.avg_classical_time=tot_time/(n*n)
        print("Distance Matrix computed ")
        return tot_time/(n*n)
    def add_edge(self, e: qChannel):
        """add an immutable qChannel. Avoiding duplicates"""
        e1_id = "{q1}-{q2}".format(q1=e.this.id, q2=e.other.id)
        e2_id = "{q1}-{q2}".format(q1=e.other.id, q2=e.this.id)
        if e1_id not in self._e_map and e2_id not in self._e_map:
            self._E.append(qChannel(self._v_map.get(e.this.id), self._v_map.get(e.other.id),
                                    channels_num=e.channels_num, optical_bsm_rate=e.optical_bsm_rate))
            self._e_map[e1_id] = self._E[-1]
            # attach correct qNodes objects

    def get_edge(self, node1_id: int, node2_id: int) -> qChannel:
        e1_id = "{q1}-{q2}".format(q1=node1_id, q2=node2_id)
        e2_id = "{q1}-{q2}".format(q1=node2_id, q2=node1_id)
        if e1_id in self._e_map:
            return self._e_map[e1_id]
        elif e2_id in self._e_map:
            return self._e_map[e2_id]
        return None
    def del_edge(self, node1_id: int, node2_id: int):
        newE=[]
        newEmap={}
        for e in self._E:
            if e.this.id!=node1_id or e.other.id!=node2_id:
                newE.append(e)
                e1_id = "{q1}-{q2}".format(q1=e.this.id, q2=e.other.id)
                newEmap[e1_id]=e
        self._e_map=newEmap
        self._E=newE
    @property
    def V(self) -> List[qNode]:
        return self._V

    @property
    def E(self) -> List[qChannel]:
        return self._E

    def get_node(self, node_id: int) -> qNode:
        return self._v_map[node_id] if node_id in self._v_map else None

    def clone(self) -> "qGraph":
        return qGraph(self.V, self.E, self.time_matrix)

    def visualize(self, filename: str = ''):
        G = nx.Graph()
        nodes = {}
        edges = []
        edges2width = {}
        widths = []
        for node in self.V:
            nodes[node.id] = node.loc.x, node.loc.y
        for qchannel in self.E:
            u, v = qchannel.this.id, qchannel.other.id
            edges.append(get_min_max(u, v))
            edges2width[get_min_max(u, v)] = qchannel.channels_num
        G.add_edges_from(edges)
        for u, v in G.edges:
            print(get_min_max(u, v))
            widths.append(edges2width[get_min_max(u, v)])
        if len(self.V) < 20:
            fig, ax = plt.subplots()
        if len(self.V) >= 20:
            a, b = 20, 20
            fig, ax = plt.subplots(figsize=(20, 20))
        if len(self.V) >= 50:
            fig, ax = plt.subplots(figsize=(40, 40))
        nx.draw_networkx(G, nodes, width=widths)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
