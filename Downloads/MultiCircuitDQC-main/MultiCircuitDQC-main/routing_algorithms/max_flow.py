import sys
sys.path.append("..")
#P_HT, V_H, V_T = 1.0, 0.99, 0.99 
#LIGHT_SPEED = 4e8 
#TAU_P, TAU_H, TAU_T, TAU_D, TAU_O = 5.9e-7, 20e-7, 10e-7, 100e-7, 10e-7
from routing_algorithms.dp_alternate import DP_Alt
from routing_algorithms.dp_shortest_path import dp_shortest_path, tree_node
from commons.qGraph import qGraph, qNode
from commons.qChannel import qChannel
from commons.tools import time_slot_num
from collections import defaultdict, namedtuple
from typing import List
import math
from routing_algorithms.naive_shortest_path import hop_shortest_path

flow_edge_cancel = namedtuple("flow_edge_cancel", ("this", "other", "inverse_flow"))  # edges in augmenting path and
flow_edge = namedtuple("flow_edge", ("edge", "inverse_flow"))


class max_flow:
    """This is class is designed to calculate the maximum number (and all paths and trees) of qubits we can
    send from a node to another one. Capacities are assumed to be be the inverse of time.

    Beside the way of updating the edges (in a tree-like manner) after finding augmenting paths, it behaves like a
    classical max-flow algorithm. We use our dp algorithm to find the augmenting paths.
    """

    def __init__(self, G: qGraph, srcs: List[qNode], dsts: List[qNode], decoherence_time: float,
                 multiple_pairs_search: str, alternate: bool = True, NON_THROTTLE: bool = False,
                 PATH_ALL_PAIR: bool = False, IS_NAIVE: bool = False):
        """
        :param G: given graph of the network
        :param srcs: list of source nodes where we want to send qubits from
        :param dsts: list of destination nodes where we want to send qubits to
        :param decoherence_time: decoherence time (s)
        :param multiple_pairs_search: multiple source-destination pair support {"best_paths", "round_robin"}
        :param alternate: using heuristic instead of the optimal dp
        :param IS_NAIVE: use naive shortest path if set to True
        """
        self._worst_to_best_ratio = 0.02  # as 2percent
        # check if src and dst are two nodes of G.V
        if srcs is None or dsts is None or len(srcs) != len(dsts):
            raise ValueError("source/destination list is not correct.")

        self._NON_THROTTLE = NON_THROTTLE
        self._PATH_ALL_PAIR = PATH_ALL_PAIR
        self._max_flow = 0  # maximum flow that can be send from src to dst
        self._flows = [[] for _ in range(len(dsts))]
        self._flow_trees = [[] for _ in range(len(dsts))]  # list of the trees that we can send the flows
        self._flow_edges = [[] for _ in range(len(dsts))]
        self._path_all_pair_shortest_path = [[] for _ in range(len(dsts))]  # store all sps for a path, used in predistributed
        for src, dst in zip(srcs, dsts):
            if not any([src is v for v in G.V]) and not any([dst is v for v in G.V]):
                raise ValueError("src or dst is not a node in the graph")
            if src is dst:
                self._max_flow = float('inf')  # infinity flow if src and dst are the same node
                return
                
        self.G_copy=G.clone()
        self._residual_graph = G
        if not alternate:
            # augmenting path using dp
            if multiple_pairs_search.lower() == "best_paths" or multiple_pairs_search.lower() == "best":
                self._best_paths(srcs, dsts, decoherence_time, IS_NAIVE)
            elif multiple_pairs_search.lower() == "fair" or multiple_pairs_search.lower() == "round_robin":
                self._round_robin(srcs, dsts, decoherence_time)
        else:
            # augmenting path using heuristic and dijkstra
            if multiple_pairs_search.lower() == "best_paths" or multiple_pairs_search.lower() == "best":
                self._best_paths_alternate(srcs, dsts, decoherence_time)
            elif multiple_pairs_search.lower() == "fair" or multiple_pairs_search.lower() == "round_robin":
                self._round_robin_alternate(srcs, dsts, decoherence_time)

    def _best_paths(self, srcs, dsts, decoherence_time, IS_NAIVE:bool = False):
        rnds = 0
        best_best_rate = -float('inf')
        while True:
            if not IS_NAIVE:
                all_pairs_shortest = dp_shortest_path(self._residual_graph, decoherence_time)
            else:
                all_pairs_shortest = hop_shortest_path(self._residual_graph, decoherence_time)
            pairs_ent_time = []
            for idx, (src, dst) in enumerate(zip(srcs, dsts)):
                idx_shortest_path = all_pairs_shortest.get_shortest_path(src, dst, self._NON_THROTTLE)
                if idx_shortest_path is not None:
                    pairs_ent_time.append((idx, idx_shortest_path))
            # pairs_ent_time = [(idx, all_pairs_shortest.get_shortest_path(src, dst)) for idx, (src, dst)
            #                   in enumerate(zip(srcs, dsts))]  # sorting to find the best pair
            if len(pairs_ent_time) == 0:
                return
            pairs_ent_time.sort(key=lambda x: x[1].avr_ent_time)
            best_pair_idx = pairs_ent_time[0][0]
            best_pair_tree = pairs_ent_time[0][1]
            if best_pair_tree.avr_ent_time == float('inf') or (best_best_rate != -float('inf')
                                                               and 1.0 / best_pair_tree.avr_ent_time <
                                                               self._worst_to_best_ratio * best_best_rate):  # no more paths
                return
            # print(f"round: {rnds}, src: {srcs[best_pair_idx].id} (loc: {srcs[best_pair_idx].loc}), "
            #       f"dst: {dsts[best_pair_idx].id} (loc: {dsts[best_pair_idx].loc}), "
            #       f"rate: {1.0/best_pair_tree.avr_ent_time:.3f}, "
            #       f"decoherence_time: {best_pair_tree.one_round_ent_time:.3f} s")
            aug_path_tree = best_pair_tree
            if 1.0 / aug_path_tree.avr_ent_time > best_best_rate:
                best_best_rate = 1.0 / aug_path_tree.avr_ent_time
            self._flows[best_pair_idx].append(1.0 / aug_path_tree.avr_ent_time)
            self._max_flow += self._flows[best_pair_idx][-1]
            self._flow_trees[best_pair_idx].append(aug_path_tree)
            if self._PATH_ALL_PAIR:
                used_nodes, _ = tree_node.used_nodes_edges(aug_path_tree)
                sp_from_src, sp_to_dst = {}, {}
                for node_id in used_nodes:
                    if node_id != srcs[best_pair_idx].id and node_id != dsts[best_pair_idx].id:
                        sp_from_src[node_id] = all_pairs_shortest.get_shortest_path_without_update(
                            src=srcs[best_pair_idx], dst=self._residual_graph.get_node(node_id))
                        sp_to_dst[node_id] = all_pairs_shortest.get_shortest_path_without_update(
                            src=self._residual_graph.get_node(node_id), dst=dsts[best_pair_idx])
                self._path_all_pair_shortest_path[best_pair_idx].append([sp_from_src, sp_to_dst])
            aug_path_edges = []
            self._calculate_flow_edges(aug_path_tree,  # aug_path_tree.avr_ent_time,
                                       aug_path_edges,
                                       srcs[best_pair_idx], dsts[best_pair_idx])
            self._flow_edges[best_pair_idx].append(aug_path_edges)
            self._residual_nodes()
            rnds += 1
            del all_pairs_shortest
    
    def _best_paths_alternate(self, srcs, dsts, decoherence_time):
        rnds = 0
        best_best_rate = float('-inf')
        while True:
            dp_alt = DP_Alt(self._residual_graph, decoherence_time=decoherence_time)
            min_time = float('inf')
            best_pair_idx, best_tree_root = None, None
            for idx, src, dst in zip(range(len(srcs)), srcs, dsts):
                path = dp_alt.dijsktra(src, dst)
                if path.fail:
                    return
                tree_root = dp_alt.path2tree(path, decoherence_time=decoherence_time)
                if tree_root and tree_root.data and tree_root.avr_ent_time < min_time:
                    min_time = tree_root.avr_ent_time
                    best_tree_root = tree_root
                    best_pair_idx = idx
            if best_pair_idx is None or \
                    (best_best_rate != float('-inf') and
                     1./best_tree_root.avr_ent_time < best_best_rate * self._worst_to_best_ratio):
                return

            print(f"round: {rnds}, src: {srcs[best_pair_idx].id} (loc: {srcs[best_pair_idx].loc}), "
                  f"dst: {dsts[best_pair_idx].id} (loc: {dsts[best_pair_idx].loc}), "
                  f"rate: {1.0/best_tree_root.avr_ent_time:.3f}, "
                  f"decoherence_time: {best_tree_root.one_round_ent_time:.3f} s")

            aug_path_tree = best_tree_root
            # below is the same comparing to self._best_paths
            if 1.0 / aug_path_tree.avr_ent_time > best_best_rate:
                best_best_rate = 1.0 / aug_path_tree.avr_ent_time
            self._flows[best_pair_idx].append(1.0 / aug_path_tree.avr_ent_time)
            self._max_flow += self._flows[best_pair_idx][-1]
            self._flow_trees[best_pair_idx].append(aug_path_tree)
            aug_path_edges = []
            self._calculate_flow_edges(aug_path_tree, aug_path_edges, srcs[best_pair_idx], dsts[best_pair_idx])
            self._flow_edges[best_pair_idx].append(aug_path_edges)
            self._residual_nodes()
            rnds += 1

    def _round_robin(self, srcs, dsts, decoherence_time, is_alternate: bool = False):
        pair_idx_set = set([i for i in range(len(dsts))])
        pair_idx_set_remained = set()
        count=0
        while len(pair_idx_set) > 0:
            if len(pair_idx_set_remained) == 0:
                pair_idx_set_remained = set([i for i in pair_idx_set])
            if not is_alternate:
                all_pairs_shortest = dp_shortest_path(self._residual_graph, decoherence_time)
            else:
                all_pairs_shortest = DP_Alt(G=self._residual_graph, decoherence_time=decoherence_time)
            """
            if all_pairs_shortest.get_shortest_path(srcs[next(iter(pair_idx_set_remained))],
                                                          dsts[next(iter(pair_idx_set_remained))]) is None and count<=len(srcs)/5:
                self.residual_graph=self.G_copy.clone()
                all_pairs_shortest = DP_Alt(G=self.residual_graph, decoherence_time=decoherence_time, avg_classical_time=self.residual_graph.avg_classical_time)
                count+=1
                #print("updated ", count)
            """                  
            while len(pair_idx_set_remained) > 0 and \
                    (all_pairs_shortest.get_shortest_path(srcs[next(iter(pair_idx_set_remained))],
                                                          dsts[next(iter(pair_idx_set_remained))]) is None or
                     all_pairs_shortest.get_shortest_path(srcs[next(iter(pair_idx_set_remained))],
                                                          dsts[next(iter(pair_idx_set_remained))]).avr_ent_time ==
                     float('inf')or
                     1.0 / all_pairs_shortest.get_shortest_path(srcs[next(iter(pair_idx_set_remained))],
                                                                dsts[next(iter(pair_idx_set_remained))]).avr_ent_time <
                     0.05 * sum(self._flows[next(iter(pair_idx_set_remained))])):
                
                pair=pair_idx_set_remained.pop()
                #print("throwing out ", srcs[pair], " ", dsts[pair])
                pair_idx_set.remove(pair)  # this pair will no longer has augmenting path
            if len(pair_idx_set_remained) == 0:
                continue
            #print("doing things")
            pair_target_idx = pair_idx_set_remained.pop()
            aug_path_tree = all_pairs_shortest.get_shortest_path(srcs[pair_target_idx], dsts[pair_target_idx])
            self._flows[pair_target_idx].append(1.0 / aug_path_tree.avr_ent_time)
            self._max_flow += self._flows[pair_target_idx][-1]
            self._flow_trees[pair_target_idx].append(aug_path_tree)
            aug_path_edges = []
            self._calculate_flow_edges(aug_path_tree, aug_path_edges,
                                       srcs[pair_target_idx], dsts[pair_target_idx])
            self._flow_edges[pair_target_idx].append(aug_path_edges)
            self._residual_nodes()

    def _round_robin_alternate(self, srcs, dsts, decoherence_time):
        pair_idx_list = [i for i in range(len(dsts))]
        while len(pair_idx_list) > 0:
            no_longer_augment_path = self._round_robin_alternate_one_round(srcs, dsts, pair_idx_list, decoherence_time)
            for idx in no_longer_augment_path:
                pair_idx_list.remove(idx)

    def _round_robin_alternate_one_round(self, srcs, dsts, pair_idx_list, decoherence_time):
        '''a helper function for self._round_robin_alternate
        '''
        no_longer_augment_path = []
        for pair_target_idx in pair_idx_list:
            dp_alt = DP_Alt(self._residual_graph, decoherence_time=decoherence_time)
            src, dst = srcs[pair_target_idx], dsts[pair_target_idx]
            path = dp_alt.dijsktra(src, dst)
            if path.fail:
                no_longer_augment_path.append(pair_target_idx)
                continue
            tree_layers, tree_root = dp_alt.path2tree_balanced(path, decoherence_time=decoherence_time)
            if tree_root is None or tree_root.avr_ent_time == float('inf'):
                no_longer_augment_path.append(pair_target_idx)    # this pair will no longer has augmenting path
                continue
            aug_path_tree = tree_root
            # below is the same as self._round_robin
            self._flows[pair_target_idx].append(1.0 / aug_path_tree.avr_ent_time)
            self._max_flow += self._flows[pair_target_idx][-1]
            self._flow_trees[pair_target_idx].append(aug_path_tree)
            aug_path_edges = []
            self._calculate_flow_edges(aug_path_tree, aug_path_edges,
                                       srcs[pair_target_idx], dsts[pair_target_idx])
            self._flow_edges[pair_target_idx].append(aug_path_edges)
            self._residual_nodes()
        return no_longer_augment_path

    def _residual_nodes(self):
        """This method update residual capacity of nodes.
        Two memories will be reduced from each node (except src and dst) in the path. One for src and dst."""
        adj_list = defaultdict(list)
        for e in self._residual_graph.E:
            adj_list[e.this].append(e)
            adj_list[e.other].append(e)
        for v, edges in adj_list.items():
            v.used_capacity = max_flow._calculate_node_used_capacity(v, edges)

    @staticmethod
    def _residual_nodes_cancel(old_nodes: List[qNode], aug_path_edges: List[flow_edge_cancel], src: qNode, dst: qNode):
        """After finding edges of the augmenting path, memories of nodes should be updated.
        Two memories will be reduced from each node (except src and dst) in the path. One for src and dst.
        A new set of nodes (including new src and dst) will be created
        :param old_nodes: old nodes
        :param aug_path_edges: edges in the augmenting path
        :param src: source node
        :param dst: destination node
        :returns: tuple(new_src, new_dst, new_nodes, nodes_map)
                   WHERE new_src and new_dst are the new qNode objects for old src and dst
                   and
                   new_nodes return a list set of new nodes including new_src and new_dst.
                   The list would be empty if either new_src or new_dst has no enough resources.
                   and
                   nodes_map which is mapping between old and new nodes' object"""
        node_path_set = set()
        for e in aug_path_edges:
            node_path_set.add(e.this)
            node_path_set.add(e.other)
        new_src, new_dst = src.clone(src.memory - 1), dst.clone(dst.memory - 1)
        if new_src.memory <= 0 or new_dst.memory <= 0:
            return new_src, new_dst, None, None  # no more augmenting path can't be find
        new_nodes = [new_src, new_dst]
        nodes_map = {src: new_src, dst: new_dst}
        for ov in old_nodes:
            if ov in node_path_set and ov is not src and ov is not dst:
                if ov.memory >= 4:  # inter nodes should have at least 2 memories to be included in the next res graph
                    new_nodes.append(ov.clone(ov.memory - 2))
                    nodes_map[ov] = new_nodes[-1]
            elif ov not in node_path_set:
                new_nodes.append(ov.clone())  # nodes not in aug path would not change
                nodes_map[ov] = new_nodes[-1]
        return new_src, new_dst, new_nodes, nodes_map

    @staticmethod
    def _residual_edges(old_edges: List[qChannel], aug_path_edges: List[flow_edge_cancel], nodes_map: dict):
        """Not being used"""
        if nodes_map is None:
            return None
        new_edges = []
        adj_lst = defaultdict(list)
        for e in old_edges:
            adj_lst[e.this].append(e)
        updated_edges = {}
        # updating edges in the path
        for fe in aug_path_edges:
            for e in adj_lst[fe.this]:
                if e.other == fe.other:  # reduce the residual capacity
                    reducing_flow = 1.0 / max(fe.inverse_flow, e.avr_ent_time)
                    try:
                        updated_edges[e] = math.ceil(1.0 / (1.0 / e.avr_ent_time - reducing_flow))
                    except ZeroDivisionError:
                        updated_edges[e] = float('inf')
                    for eo in adj_lst[fe.other]:  # increase the residual capacity of the inverse edge
                        if eo.other == fe.this:
                            updated_edges[eo] = math.ceil(1.0 / (1.0 / eo.avr_ent_time + reducing_flow))
                            break  # found the inverse edge
                    break  # found the edge
        # creating new edges
        for oe in old_edges:
            if oe.this in nodes_map and oe.other in nodes_map:  # edges with one deleted end will be discarded
                new_this, new_other = nodes_map[oe.this], nodes_map[oe.other]
                if oe in updated_edges:  # edge in augmenting path with new avr_ent_time
                    new_edges.append(qChannel(new_this, new_other, updated_edges[oe]))
                else:  # edge not in augmenting path and no avr_ent_time changes
                    new_edges.append(qChannel(new_this, new_other, oe.avr_ent_time))
        return new_edges

    def _calculate_flow_edges(self, node: tree_node, flow_edges, src: qNode, dst: qNode):
        """This is wrote for undirected-edges output of dp"""
        # recursively find all edges in an augmenting path and their appropriate new flow
        if node is None:
            return flow_edges
        if isinstance(node.data, qChannel):
            flow_edges.append(flow_edge(node.data, node.avr_ent_time))
            # channels might be recreated in dp, get from residual graph
            channel = self._residual_graph.get_edge(node.data.this.id, node.data.other.id)
            # if self._NON_THROTTLE:
            #     channel.current_flow += channel.residual_capacity
            # else:
            channel.current_flow += 1 / node.avr_ent_time  # update the flow of the edge
            # update memory
            node.data.this.memory -= 1
            if node.data.this.memory <= 1 and node.data.this is not src and node.data.this is not dst:
                node.data.this.memory = 0  # intermediate nodes should have at least two memories
            node.data.other.memory -= 1
            # if node.data.other.memory < 1 and node.data.other is not src and node.data.other is not dst:
            #     node.data.other.memory = 0  # intermediate nodes should have at least two memories
            return
        # child_ent_time = (node.avr_ent_time * node.data.bsm_success_rate -
        #                   node.data.bsm_time - node.classical_time) / 1.5
        # go left
        self._calculate_flow_edges(node.left, flow_edges, src, dst)
        # go right
        self._calculate_flow_edges(node.right, flow_edges, src, dst)

    def get_path_all_pair_shortest_path(self, sd_id: int):
        if sd_id < len(self._path_all_pair_shortest_path):
            return self._path_all_pair_shortest_path[sd_id]
        return None

    @staticmethod
    def _calculate_flow_edges_cancelling(node: tree_node, time: int, flow_edges: List[flow_edge_cancel], parent=None,
                                         is_left: bool = False):
        """This was wrote for having cancellation"""
        # recursively find all edges in a path and their appropriate flow
        if node is None:
            return flow_edges
        if node.left is None or node.right is None:  # we reach to an edge
            if parent is None:
                flow_edges.append(flow_edge_cancel(node.data.this, node.data.other, time))
            else:
                if is_left:
                    flow_edges.append(flow_edge_cancel(node.data, parent.data, time))  # edge from parent -> node
                else:
                    flow_edges.append(flow_edge_cancel(parent.data, node.data, time))  # edge from node -> parent
            return
        child_ent_time = node.avr_ent_time * node.data.bsm_success_rate - node.data.bsm_time - node.classical_time
        # go left
        max_flow._calculate_flow_edges_cancelling(node.left, child_ent_time, flow_edges, node, True)
        # go right
        max_flow._calculate_flow_edges_cancelling(node.right, child_ent_time, flow_edges, node, False)

    @staticmethod
    def _calculate_node_used_capacity(node: qNode, edges: List[qChannel]):
        used_capacity = 0
        for e in edges:
            if node is e.this or node is e.other:
                used_capacity += e.current_flow / (e.channel_success_rate ** 2 * e.optical_bsm_rate *
                                                   e.this.gen_success_rate * e.other.gen_success_rate)
        return used_capacity

    @property
    def get_flow_total(self):
        return self._max_flow

    def get_flows(self, idx):
        if idx >= len(self._flow_trees):
            return None
        return [(flow, tree, edges) for tree, edges, flow in zip(self._flow_trees[idx],
                                                                 self._flow_edges[idx],
                                                                 self._flows[idx])]

    @property
    def trees(self):
        return self._flow_trees

    def count_non_empty_flow(self):
        '''used in computing fairness, return the non empty flow count
        '''
        counter = 0
        for flow in self._flows:
            if flow:
                counter += 1
        return counter


    def print_tmp(self):
        fl = 0
        for i_pair, pair in enumerate(self._flows):
            for i_flow, flow in enumerate(pair):
                edges = ""
                for flow_edge in self._flow_edges[i_pair][i_flow]:
                    edges += str(flow_edge.edge.this.id) + "-" + str(flow_edge.edge.other.id) + "--"
                edges = edges[0:-2]
                print(fl, ", route =", edges, ", rate =", flow)
                fl += 1
