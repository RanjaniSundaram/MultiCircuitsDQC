from networkx.readwrite.json_graph import tree
from commons.qChannel import qChannel
import sys

sys.path.append("..")

from commons.qGraph import qGraph
from commons.qNode import qNode
# from quantum_routing.commons.qChannel import qChannel
from commons.tools import get_classical_time
from typing import List
from commons.tree_node import tree_node


class dp_shortest_path:
    """
    This class finds all shortest paths (between any pair) via a dynamic programming technique in q quantum network
    (qGraph).
    Due to physical nature of quantum communication and need for BSM operations on middle nodes, a shortest path is
    in binary tree structure. i.e. a binary tree that connects two nodes in the shortest time will be find.
    """

    def __init__(self, G: qGraph, decoherence_time: float, USAGE: bool = False):
        """
        :param G: qGraph representation of the quantum network. The graph is assumed to be undirected
        :param decoherence_time: decoherence time when an entanglement disappears (# time slots)
        :param USAGE: Other nodes' generation capacities will be used if True, otherwise always 50-50
        """
        n = len(G.V)  # number of nodes
        self._decoherence_time = decoherence_time
        self._node_map = {v: i for i, v in enumerate(G.V)}  # a dictionary that maps nodes to the indices of G.V
        self._USAGE = USAGE
        self._UPDATED = [[False] * n for _ in range(n)]
        self._G = G
        if not USAGE:
            self._dp = [[tree_node()] * i + [tree_node(data=G.V[i], avr_ent_time=0, one_round_ent_time=0)] +
                        [tree_node()] * (n - 1 - i) for i in range(n)]
            self._nonusage_dp(G)
        else:
            self._interval = 20
            self._dpu = [[[[tree_node()] * i + [tree_node(data=G.V[i], avr_ent_time=0, one_round_ent_time=0)] +
                           [tree_node()] * (n - 1 - i) for i in range(n)] for _ in range(100 // self._interval + 1)]
                         for _ in range(100 // self._interval + 1)]
            self._usage_dp(G)

    def clone(self, shortest_path: "dp_shortest_path"):
        # TODO not completed. need tree_node 1st to be completed
        cloned_shortest = dp_shortest_path(qGraph([], []), self._decoherence_time, self._USAGE)
        n = len(self._dp)
        cloned_shortest._UPDATED = [[False] * n for _ in range(n)]
        if not self._USAGE:
            cloned_shortest._dp = [[tree_node()] * n for _ in range(n)]

        else:
            shortest_path._interval = self._interval
            self._dpu = [[[[tree_node()] * n for i in range(_)] for _ in range(100 // self._interval + 1)]
                         for _ in range(100 // self._interval + 1)]

    def _nonusage_dp(self, G: qGraph):
        n = len(G.V)
        # node(t=0, ts=0) are shortest paths from nodes to themselves (0 edges)
        for e in G.E:
            u, v = self._node_map.get(e.this), self._node_map.get(e.other)  # e = (u -> v)
            self._dp[u][v] = tree_node(e,
                                       avr_ent_time=e.avr_ent_time
                                       if e.avr_success_time < self._decoherence_time else float('inf'),
                                       one_round_ent_time=e.avr_success_time
                                       if e.avr_success_time < self._decoherence_time else float('inf'))
            # self._dp[v][u] = self._dp[u][v]
            self._dp[v][u] = dp_shortest_path.inverse_tree(self._dp[u][v])

        # O(n^4)
        for _ in range(n - 2):  # shortest paths between any pair should be achieved in n - 1 runs,above for loop is one
            new_dp = [[tree_node()] * i + [tree_node(data=G.V[i], avr_ent_time=0, one_round_ent_time=0)] +
                      [tree_node()] * (n - i - 1) for i in range(n)]
            CHANGED = False
            for src in range(n):  # src id
                src_node = G.V[src]
                for dst in range(src + 1, n):  # dst id
                    dst_node = G.V[dst]
                    if src == dst:  # dp(src, src) always zero
                        continue
                    shortest_src_dst = self._dp[src][dst]
                    for k in range(n):
                        if k == src or k == dst:  # one side is zero
                            continue
                        k_node = G.V[k]  # the node bsm happens
                        # average total ent time
                        src_to_k_classical = get_classical_time(k_node.loc.distance(src_node.loc))
                        k_to_dst_classical = get_classical_time(k_node.loc.distance(dst_node.loc))
                        # t = None
                        # if self._dp[src][k].avr_ent_time == float('inf') or \
                        #         self._dp[k][dst].avr_ent_time == float('inf'):
                        #     t = float('inf')
                        # else:
                        #     t = ceil((self._dp[src][k].avr_ent_time + self._dp[k][dst].avr_ent_time -
                        #          1.0 / (1.0 / self._dp[src][k].avr_ent_time + 1.0 / self._dp[k][dst].avr_ent_time) +
                        #               max(src_to_k_classical, k_to_dst_classical) +
                        #               k_node.success_bsm_time()) / k_node.bsm_success_rate)
                        t = (1.5 * max(self._dp[src][k].avr_ent_time, self._dp[k][dst].avr_ent_time) +
                             # time to generate both left and right subtrees
                             max(src_to_k_classical, k_to_dst_classical) +  # time to send BSM res back
                             k_node.bsm_time) / k_node.bsm_success_rate
                        if t>=0.5: 
                            t=0.1
                        elif t>=0.1:
                            t=0.05
                        elif t>=0.01:
                            t=0.01
                        # average one round of entanglement (waiting included)
                        # t_s = max(src_to_k_classical, k_to_dst_classical) + k_node.bsm_time
                        # if self._dp[src][k].avr_ent_time < self._dp[k][dst].avr_ent_time:
                        #     t_s += max(self._dp[src][k].one_round_ent_time + self._dp[k][dst].avr_ent_time -
                        #                self._dp[src][k].avr_ent_time, self._dp[k][dst].one_round_ent_time)
                        # else:
                        #     t_s += max(self._dp[src][k].one_round_ent_time, self._dp[k][dst].one_round_ent_time
                        #                + self._dp[src][k].avr_ent_time - self._dp[k][dst].avr_ent_time)
                        # t_s = max(self._dp[src][k].one_round_ent_time, self._dp[k][dst].one_round_ent_time) + \
                        #       max(src_to_k_classical, k_to_dst_classical) + \
                        #       k_node.success_bsm_time()

                        if t < shortest_src_dst.avr_ent_time:
                            # and t_s < decoherence_time:  # found a shorter path
                            # left = self._dp[src][src] if self._dp[src][k].left is None else self._dp[src][k]
                            # right = self._dp[dst][dst] if self._dp[k][dst].left is None else self._dp[k][dst]
                            left = self._dp[src][k]
                            right = self._dp[k][dst]
                            shortest_tmp = tree_node(data=k_node, left=left.clone(), right=right.clone(),
                                                     avr_ent_time=t, one_round_ent_time=0,
                                                     classical_time=max(src_to_k_classical, k_to_dst_classical),
                                                     left_avr_ent_time=self._dp[src][k].avr_ent_time,
                                                     right_avr_ent_time=self._dp[k][dst].avr_ent_time)
                            if tree_node.max_node_occurrences(shortest_tmp) <= 2:
                                shortest_src_dst = shortest_tmp
                                CHANGED = True
                    new_dp[src][dst] = shortest_src_dst
                    if shortest_src_dst.data is not None:
                        new_dp[dst][src] = dp_shortest_path.inverse_tree(shortest_src_dst)  # same time but inverted
            self._dp = new_dp
            if not CHANGED:
                break

    def _usage_dp(self, G: qGraph):
        # TODO: add supporting Superlinks
        n = len(G.V)
        # node(t=0, ts=0) are shortest paths from nodes to themselves (0 edges)
        for e in G.E:
            u, v = self._node_map.get(e.this), self._node_map.get(e.other)  # e = (u -> v)
            for usage_u in range(100 // self._interval + 1):
                for usage_v in range(100 // self._interval + 1):
                    self._dpu[usage_u][usage_v][u][v] = tree_node(e,
                                                                  avr_ent_time=e.get_avr_ent_time(
                                                                      usage_u * self._interval,
                                                                      usage_v * self._interval)
                                                                  if e.avr_success_time < self._decoherence_time else
                                                                  float('inf'),
                                                                  one_round_ent_time=e.avr_success_time
                                                                  if e.avr_success_time < self._decoherence_time else
                                                                  float('inf'))
                    # self._dp[v][u] = self._dp[u][v]
                    self._dpu[usage_v][usage_u][v][u] = dp_shortest_path.inverse_tree(self._dpu[usage_u][usage_v][u][v])

        # O(n^4)
        for _ in range(n - 2):  # shortest paths between any pair should be achieved in n - 1 runs,above for loop is one
            CHANGED = False
            for src in range(n):  # src id
                src_node = G.V[src]
                for dst in range(n):  # dst id
                    if src == dst:  # dp(src, src) always zero
                        continue
                    dst_node = G.V[dst]
                    possible_ks = [k for k in range(n) if (k != src and k != dst and
                                                           self._dpu[-1][-1][src][k].avr_ent_time != float('inf')
                                                           and self._dpu[-1][-1][k][dst].avr_ent_time != float('inf'))]
                    for k in possible_ks:
                        # print(f"src: {src}, k:{k}, dst: {dst}")
                        k_node = G.V[k]  # the node bsm happens
                        # average total ent time
                        src_to_k_classical = get_classical_time(k_node.loc.distance(src_node.loc))
                        k_to_dst_classical = get_classical_time(k_node.loc.distance(dst_node.loc))
                        for src_usage in range(1, 100 // self._interval + 1):
                            for dst_usage in range(1, 100 // self._interval + 1):
                                shortest_src_dst = self._dpu[src_usage][dst_usage][src][dst]
                                for kleft_usage in range(1, 100 // self._interval):
                                    left = self._dpu[src_usage][kleft_usage][src][k]
                                    right = self._dpu[100 // self._interval - kleft_usage][dst_usage][k][dst]
                                    t = (1.5 * max(left.avr_ent_time,
                                                   right.avr_ent_time) +
                                         max(src_to_k_classical, k_to_dst_classical) +  # time to send BSM res back
                                         k_node.bsm_time) / k_node.bsm_success_rate
                                    if t>=0.5: 
                                        t=0.1
                                    elif t>=0.1:
                                        t=0.05
                                    elif t>=0.01:
                                        t=0.01
                                    if t < shortest_src_dst.avr_ent_time:
                                        shortest_src_dst = tree_node(data=k_node, left=left, right=right,
                                                                     avr_ent_time=t,
                                                                     one_round_ent_time=0,
                                                                     classical_time=max(src_to_k_classical,
                                                                                        k_to_dst_classical),
                                                                     left_avr_ent_time=left.avr_ent_time,
                                                                     right_avr_ent_time=right.avr_ent_time)
                                        CHANGED = True
                                self._dpu[src_usage][dst_usage][src][dst] = shortest_src_dst
            # self._dp = new_dp
            if not CHANGED:
                break

    @property
    def shortest_paths(self) -> List[List[tree_node]]:
        """
        :return: Shortest path matrix with indices based on appearances on G.V
        """
        return self._dp

    def get_shortest_path_without_update_by_id(self, src_id: int, dst_id: int):
        return self.get_shortest_path_without_update(self._G.get_node(src_id), self._G.get_node(dst_id))

    def get_shortest_path_without_update(self, src: qNode, dst: qNode):
        if src not in self._node_map or dst not in self._node_map:
            return None
        src_idx, dst_idx = self._node_map[src], self._node_map[dst]
        if not self._USAGE:
            shortest_path = self._dp[src_idx][dst_idx]
        else:
            shortest_path = self._dpu[-1][-1][self._node_map[src]][self._node_map[dst]]
        # if shortest_path.data is None:
        #     return None
        return shortest_path

    def get_shortest_path_by_id(self, src_id: int, dst_id: int, NON_THROTTLING: bool = False):
        return self.get_shortest_path(self._G.get_node(src_id), self._G.get_node(dst_id), NON_THROTTLING)

    def get_shortest_path(self, src: qNode, dst: qNode, NON_THROTTLING: bool = False):
        """
        :param src: source qNode
        :param dst: destination qNode
        :return: shortest path(tree) between src and dst
        """
        if src not in self._node_map or dst not in self._node_map:
            return None
        src_idx, dst_idx = self._node_map[src], self._node_map[dst]
        if not self._USAGE:
            shortest_path = self._dp[src_idx][dst_idx]
        else:
            shortest_path = self._dpu[-1][-1][self._node_map[src]][self._node_map[dst]]
        if src_idx == dst_idx:
            return shortest_path
        if shortest_path.data is None:
            return None
        if not self._UPDATED[src_idx][dst_idx]:
            _, _, shortest_path = dp_shortest_path.update_ent_time_tree(shortest_path, shortest_path.avr_ent_time,
                                                                        NON_THROTTLING)
            self._UPDATED[src_idx][dst_idx] = True
            if not self._USAGE:
                self._dp[src_idx][dst_idx] = shortest_path
            else:
                self._dpu[-1][-1][src_idx][dst_idx] = shortest_path
        return shortest_path if shortest_path.one_round_ent_time < self._decoherence_time else None

    @staticmethod
    def inverse_tree(node: tree_node):
        if isinstance(node.data, qChannel):
            # invert the edge
            return tree_node(data=qChannel(node.data.other, node.data.this, node.data.channels_num),
                             avr_ent_time=node.avr_ent_time, one_round_ent_time=node.one_round_ent_time)
            # ,IS_SL=node.IS_SL, SL_RATE=node.SL_RATE)
        new_right_subtree = dp_shortest_path.inverse_tree(node.left)
        new_left_subtree = dp_shortest_path.inverse_tree(node.right)
        return tree_node(data=node.data, left=new_left_subtree, right=new_right_subtree,
                         avr_ent_time=node.avr_ent_time, one_round_ent_time=node.one_round_ent_time,
                         left_avr_ent_time=node.right_avr_ent_time, right_avr_ent_time=node.left_avr_ent_time,
                         classical_time=node.classical_time)
        # , IS_SL=node.IS_SL, SL_RATE=node.SL_RATE)

    @staticmethod
    def update_ent_time_tree(node: tree_node, avr_ent_time: float, NON_THROTTLING: bool = False):
        if node is None or node.data is None:
            return
        if isinstance(node.data, qChannel):
            new_node = tree_node(data=node.data, avr_ent_time=node.avr_ent_time, classical_time=node.classical_time)
            if not NON_THROTTLING:
                new_node.avr_ent_time = avr_ent_time
                # node.avr_ent_time = avr_ent_time
            # node.one_round_ent_time = node.data.avr_success_time
            new_node.one_round_ent_time = node.data.avr_success_time
            return node.one_round_ent_time, node.one_round_ent_time, new_node
        # if not node.IS_SL:
        node.avr_ent_time = avr_ent_time

        # child_ent_time = ((1.0/node.SL_RATE if node.IS_SL else node.avr_ent_time) * node.data.bsm_success_rate -
        #                   node.data.bsm_time - node.classical_time) / 1.5
        child_ent_time = (node.avr_ent_time * node.data.bsm_success_rate -
                          node.data.bsm_time - node.classical_time) / 1.5
        left_left_age, left_root_age, left_root = dp_shortest_path.update_ent_time_tree(node.left, child_ent_time,
                                                                                        NON_THROTTLING)
        right_root_age, right_right_age, right_root = dp_shortest_path.update_ent_time_tree(node.right, child_ent_time,
                                                                                            NON_THROTTLING)
        root_new_node = tree_node(node.data, left=left_root, right=right_root, avr_ent_time=avr_ent_time,
                                  classical_time=node.classical_time, left_avr_ent_time=left_root.avr_ent_time,
                                  right_avr_ent_time=right_root.avr_ent_time)
        # node.left_avr_ent_time = node.right_avr_ent_time = child_ent_time
        waiting_time = child_ent_time / 2
        # node.one_round_ent_time = max([left_left_age, left_root_age, right_root_age, right_right_age]) + waiting_time
        root_new_node.one_round_ent_time = max([left_left_age, left_root_age, right_root_age, right_right_age]) + \
                                               waiting_time
        return left_left_age + waiting_time, right_right_age + waiting_time, root_new_node
