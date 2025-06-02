from routing_algorithms.dp_shortest_path import dp_shortest_path as dp
from routing_algorithms.optimal_linear_programming import optimal_linear_programming as lp
from commons.qGraph import qGraph
from commons.qNode import qNode
from commons.tree_node import tree_node
from commons.qChannel import qChannel
from collections import namedtuple
from typing import List
import heapq as hq
from commons.ghz_fusion_nodes import fusion_retain_node
import math
from commons.tools import get_classical_time, DisJointSets

Edge = namedtuple('Edge', ['v', 'w'])  # an abstract for edges between end-nodes


class ghz_fusion_retain:
    """This class is designed to do multipartite routing using only Fusion Retain version of GHZ generation where
    only end nodes do the fusion process to merge two GHZ states into a bigger one. Two states has only one node
    in common whose two qubits will be consumed and resulted in only one remaining.

    The algorithm is summarized:
        1- All bipartite shortest paths, optimal GHZ-2 or bell pairs (DP)
        2- Create a new COMPLETE graph including only FINAL end nodes and edges with weight of EP rates calculated in 1
        3- Find the BEST spanning tree T, in the new graph, where the MINIMUM edge rate is MAXIMIZED
            * Doing this, we implicitly assume the final Fusion Tree is balanced in terms of number of GHZ-2
        4- Do LP over the edges of the tree T to do GHZ-2 (using multiple paths)
        5- Do Fusion retain in a balanced manner

    """

    def __init__(self, G: qGraph, end_nodes: set[qNode], decoherence_time: float):
        self._G = G
        self._end_nodes = list(end_nodes)
        n = len(end_nodes)  # number of end nodes for the GHZ state
        # map end nodes id to new ids 0..|end_nodes|-1
        self._en_id_map = {node.id: idx for idx, node in enumerate(self._end_nodes)}

        # 1
        self._bi_dp = dp(G, decoherence_time)

        # 2. COMPLETE graph
        complete_graph = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                w = 1.0 / self._bi_dp.get_shortest_path_without_update(
                    self._end_nodes[i], self._end_nodes[j]).avr_ent_time
                complete_graph[i].append(Edge(j, w))
                complete_graph[j].append(Edge(i, w))

        # 3. BEST spanning tree
        best_spt = spt_prim(complete_graph)

        # 4. Extract the new graph, using edges in the above tree and do LP
        spt_nodes, spt_edges = set(), set()
        spt_srcs, spt_dsts = [], []

        for u, v, _ in best_spt:
            uv_tree = self._bi_dp.get_shortest_path_without_update(self._end_nodes[u], self._end_nodes[v])
            tree_nodes, tree_edges = tree_node.used_nodes_edges(uv_tree)
            spt_nodes, spt_edges = spt_nodes.union(tree_nodes), spt_edges.union(tree_edges)
            spt_srcs.append(self._end_nodes[u])
            spt_dsts.append(self._end_nodes[v])
        spt_edges_seen = set()
        spt_channels = []
        for spt_edge in spt_edges:
            u, v = spt_edge.split("-")
            if spt_edge not in spt_edges_seen and f"{v}-{u}" not in spt_edges_seen:
                spt_edges_seen.add(spt_edge)
                spt_channels.append(G.get_edge(int(u), int(v)))
        spt_graph = qGraph([self._G.V[idx] for idx in spt_nodes], spt_channels)

        # 4. LP, multi-paths for GHZ-2
        spt_lp = lp(spt_graph, [spt_graph.get_node(spt_src.id) for spt_src in spt_srcs],
                    [spt_graph.get_node(spt_dst.id) for spt_dst in spt_dsts], max_min=True)
        self._bi_trees = spt_lp.trees
        # updating GHZ-2 rates based on LP output
        updated_best_spt = []
        for ghz2_idx in range(len(best_spt)):
            total_rate = 0.0
            for tree in self._bi_trees[ghz2_idx]:
                total_rate += 1.0 / tree.avr_ent_time
            if total_rate == 0.0:
                raise ValueError("No solution was found!")
            updated_best_spt.append((best_spt[ghz2_idx][0], best_spt[ghz2_idx][1], total_rate))

        # 5. Constructing ghz fusion tree
        self._ghz_fusion_tree = ghz_fusion_retain.optimum_fusion_tree(spt_tree=updated_best_spt,
                                                                      end_nodes=self._end_nodes)
        self._bi_trees = {f"{src.id}_{dst.id}": trees for src, dst, trees in zip(spt_srcs, spt_dsts, self._bi_trees)}

    @property
    def bi_trees(self):
        return self._bi_trees

    @property
    def ghz_fusion_tree(self):
        return self._ghz_fusion_tree

    @staticmethod
    def optimum_fusion_tree(spt_tree: List, end_nodes: List[qNode]) -> fusion_retain_node:
        """DP-based fusion tree construction"""
        n = len(end_nodes)  # number of end nodes
        ghz_states = [{} for _ in range(n - 1)]  # optimal ghz2..ghzn states
        # GHZ-2
        for u, v, rate in spt_tree:
            ghz_states[0]["-".join([str(num) for num in sorted([u, v])])] = \
                fusion_retain_node(end_nodes={end_nodes[u].id, end_nodes[v].id}, avr_ent_time=1.0 / rate)

        for ghz_num in range(3, n + 1):
            for ghz_sub1_num in range(2, math.ceil(ghz_num / 2) + 1):
                for ghz_sub1_end_nodes, ghz_sub1 in ghz_states[ghz_sub1_num - 2].items():
                    for ghz_sub2_end_nodes, ghz_sub2 in ghz_states[ghz_num - ghz_sub1_num + 1 - 2].items():
                        ghz1_nodes = set([int(num) for num in ghz_sub1_end_nodes.split("-")])
                        ghz2_nodes = set([int(num) for num in ghz_sub2_end_nodes.split("-")])
                        common_node = ghz1_nodes.intersection(ghz2_nodes)
                        if len(common_node) == 1:
                            node = end_nodes[common_node.pop()]
                            union_nodes = "-".join([str(num) for num in sorted(list(ghz1_nodes.union(ghz2_nodes)))])
                            ghz_latency = (1.5 * max(ghz_sub1.avr_ent_time, ghz_sub2.avr_ent_time) +
                                           max([get_classical_time(node.loc.distance(end_nodes[e_node].loc)) for e_node
                                                in ghz1_nodes.union(ghz2_nodes)]) + node.fusion_time) /\
                                          node.fusion_success_rate
                            if union_nodes not in ghz_states[ghz_num - 2] or \
                                    ghz_latency < ghz_states[ghz_num - 2][union_nodes].avr_ent_time:
                                ghz_states[ghz_num - 2][union_nodes] = \
                                    fusion_retain_node(node=node,
                                                       sub1=ghz_sub1, sub2=ghz_sub2,
                                                       end_nodes=ghz_sub1.end_nodes.union(ghz_sub2.end_nodes),
                                                       avr_ent_time=ghz_latency)

        return ghz_states[-1]["-".join([str(num) for num in range(len(end_nodes))])]


def spt_prim(graph: List[List[Edge]]) -> List:
    """Spanning Tree which maximize the minimum edge selected, using Prim's algorithm."""
    seen = set()
    tree = []  # list of final edges (u, v, w)
    # using heapq to find max at each step
    maxheap = [(0, 0, 0)]  # start randomly at node 0 (w, u, v)
    while len(seen) != len(graph):
        w, u, v = hq.heappop(maxheap)
        if u not in seen:
            tree.append((u, v, -w))
            seen.add(u)
            for n_edge in graph[u]:
                if n_edge.v not in seen:
                    hq.heappush(maxheap, (-n_edge.w, n_edge.v, u))
    return tree[1:]


def spt_kruskal(graph: List[List[Edge]]) -> List:
    """Spanning Tree which maximize the minimum edge selected, using Kruskal's algorithm."""
    n = len(graph)
    graph_edges = []  # (u, v, rate) with no duplicates
    seen = set()  # to remove duplicate edges
    for u in range(n):
        for e in graph[u]:
            if f"{u}-{e.v}" not in seen and f"{e.v}-{u}" not in seen:
                seen.add(f"{u}-{e.v}")
                graph_edges.append((u, e.v, e.w))
    spt_tree = []  # maximum spanning tree, stops when have n - 1 edges
    graph_edges.sort(key=lambda x: x[2], reverse=True)
    union_find = DisJointSets(n)
    for u, v, w in graph_edges:
        if not union_find.connected(u, v):
            spt_tree.append((u, v, w))
            union_find.union(u, v)
        if len(spt_tree) == n - 1:
            break
    return spt_tree
