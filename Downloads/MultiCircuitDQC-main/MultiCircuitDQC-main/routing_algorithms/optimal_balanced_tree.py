import sys

sys.path.append("..")

from commons.qChannel import qChannel
from commons.qGraph import qGraph
from commons.qNode import qNode
from commons.tree_node import tree_node
from typing import List
from commons.tools import get_classical_time


class optimal_balanced_tree:
    """
    This class is the implementation of Calefi paper of finding optimal balanced tree between two nodes of a graph.
    """

    def __init__(self, G: qGraph, decoherence_time: float):
        """
        :param G: qGraph representation of the quantum network. The graph is assumed to be undirected
        :param decoherence_time: decoherence time when an entanglement disappears (# time slots)
        """
        n = len(G.V)  # number of nodes
        self._decoherence_time = decoherence_time
        self._node_map = {v: i for i, v in enumerate(G.V)}
        self._adj_list = [[] for _ in range(n)]
        self._edge_map = {f"{self._node_map[e.this]}_{self._node_map[e.other]}": idx for idx, e in enumerate(G.E)}
        self._G = G
        for e in G.E:
            self._adj_list[self._node_map[e.this]].append(self._node_map[e.other])
            self._adj_list[self._node_map[e.other]].append(self._node_map[e.this])

    def get_best_tree(self, src: qNode, dst: qNode) -> tree_node:
        all_paths = optimal_balanced_tree.all_paths(self._adj_list, self._node_map[src], self._node_map[dst])
        best_latency, best_tree = float('inf'), None
        for path in all_paths:
            tree = optimal_balanced_tree.calculate_latency(path, self._edge_map, self._decoherence_time, self._G)
            if tree.avr_ent_time < best_latency:
                best_latency = tree.avr_ent_time
                best_tree = tree
        # convert the best path into a balanced tree
        return best_tree if best_tree.avr_ent_time < float('inf') else None

    @staticmethod
    def all_paths(adj_list: List[List[int]], src_idx: int, dst_idx: int) -> List[List[int]]:
        def _helper(u: int, dst: int, current_path: List):
            if u == dst and len(current_path) > 0:
                all_paths.append(current_path + [u])
                return
            seen[u] = True
            for v in adj_list[u]:
                if not seen[v]:
                    _helper(v, dst, current_path + [u])
            seen[u] = False

        all_paths = []
        seen = [False] * len(adj_list)
        _helper(src_idx, dst_idx, [])
        return all_paths

    @staticmethod
    def calculate_latency(path: List[int], edge_map: dict, decoherence_time: float, G: qGraph) -> tree_node:
        """path is a list of consecutive links"""
        def _helper(lidx, ridx) -> [float, tree_node]:
            if ridx == lidx:
                return 0, 0
            if ridx == lidx + 1:  # base case of reaching a link between node path[lidx] -> path[ridx]
                if f"{path[lidx]}_{path[ridx]}" in edge_map:
                    e_idx = edge_map[f"{path[lidx]}_{path[ridx]}"]
                else:
                    e_idx = edge_map[f"{path[ridx]}_{path[lidx]}"]
                e = G.E[e_idx]
                if e.avr_success_time > decoherence_time:  # link generation greater than decoherence time
                    return float('inf'), tree_node()
                return e.avr_ent_time - e.avr_success_time, \
                       tree_node(data=qChannel(G.V[path[lidx]], G.V[path[ridx]], e.channels_num),
                                 avr_ent_time=e.avr_ent_time, one_round_ent_time=e.avr_success_time,
                                 classical_time=get_classical_time(e.length / 2))
            mid = (lidx + ridx) // 2
            first_left_qubit_created, left_tree = _helper(lidx, mid)
            first_right_qubit_created, right_tree = _helper(mid, ridx)
            mid_node = G.V[path[mid]]
            classical_time = max(get_classical_time(mid_node.loc.distance(G.V[path[lidx]].loc)),
                                 get_classical_time(mid_node.loc.distance(G.V[path[ridx]].loc)))
            success_time = max(left_tree.one_round_ent_time, right_tree.one_round_ent_time) + \
                           classical_time + mid_node.bsm_time
            avr_ent_time = (max(left_tree.avr_ent_time, right_tree.avr_ent_time) + classical_time + mid_node.bsm_time) /\
                           mid_node.bsm_success_rate
            if success_time - min(first_left_qubit_created, first_right_qubit_created) > decoherence_time:
                return float('inf'), tree_node
            return min(first_left_qubit_created, first_right_qubit_created), \
                   tree_node(data=mid_node, left=left_tree, right=right_tree, avr_ent_time=avr_ent_time,
                             one_round_ent_time=success_time, left_avr_ent_time=left_tree.avr_ent_time,
                             right_avr_ent_time=right_tree.avr_ent_time, classical_time=classical_time)
        return _helper(0, len(path) - 1)[1]
