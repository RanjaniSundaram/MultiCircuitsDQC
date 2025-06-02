import math

from commons.qNode import qNode
from commons.qGraph import qGraph
from commons.qChannel import qChannel
from commons.tree_node import tree_node
from commons.ghz_fusion_nodes import general_fusion_node
from routing_algorithms.dp_shortest_path import dp_shortest_path as dp
from routing_algorithms.ghz_fusion_retain import Edge, spt_kruskal
from typing import List
from collections import deque, defaultdict
from itertools import product, islice

import pickle

WHITE, GRAY, BLACK = 0, 1, 2  # colors for doing dfs and find out a graph is connected over some nodes
EQUAL_EDGES_PARTITIONING, LATENCY_PARTITIONING = 0, 1  # ways of partitioning on a node


class ghz_steiner_tree:
    """This class is designed to do multipartite routing using general fusion of GHZ generation where any node,
    if included in final fusion tree, can perform fusion operation. Whether a node is an end or intermediate node,
    it may or may not remain after fusing two GHZ states into a bigger one.

    The algorithm is summarized as:
        1- Sort the edges in a decreasing way based on EP generation rate (or increasing based on EP latency)
        2. For each edge e, delete all other edges with lower rate and find the Steiner Tree which gives us an
            approximated cost assuming the final fusion tree would be balanced.
        3. Given the best Steiner Tree from step  (2), construct the final fusion tree in a balanced way recursively,
            either in terms of number of edges or the cost of sub-trees.

    """

    def __init__(self, G: qGraph, end_nodes: set[qNode], decoherence_time: float):
        self._G = G
        self._end_nodes = list(end_nodes)
        self._decoherence_time = decoherence_time

        self._best_st = self._best_steiner_tree()
        # self._best_st = pickle.load(open("best_st_25nodes_3endnodes", "rb"))
        # self._best_st = pickle.load(open("best_st_50nodes_6endnodes2.p", "rb"))

    @property
    def best_steiner_tree(self):
        return self._best_st

    def fusion_tree(self, partitioning_method: EQUAL_EDGES_PARTITIONING) -> general_fusion_node:
        nodes_degree = defaultdict(int)
        for edge in self._best_st:
            nodes_degree[edge.this.id] += 1
            nodes_degree[edge.other.id] += 1
        return ghz_steiner_tree.balanced_fusion_tree(self.best_steiner_tree, nodes_degree,
                                                     {node.id: 1 for node in self._end_nodes},
                                                     method=partitioning_method)

    def _best_steiner_tree(self):
        sorted_edges = sorted(self._G.E, key=lambda e: e.avr_ent_time)

        # To have a steiner tree, end-nodes should be connected. Find the first time they're connected.
        # At least, n - 1 edges is required where n is number of end nodes
        num_edges = len(self._end_nodes) - 1
        while not ghz_steiner_tree.end_nodes_connected(total_number_of_nodes=len(self._G.V),
                                                       edges=sorted_edges[:num_edges],
                                                       end_nodes=self._end_nodes):
            num_edges += 1

        best_steiner_latency = float('inf')
        best_steiner_tree = None
        while num_edges < len(self._G.E):
            steiner_latency, steiner_tree = ghz_steiner_tree.approx_steiner_tree(sorted_edges[:num_edges],
                                                                                 self._end_nodes,
                                                                                 self._decoherence_time)
            if steiner_latency < best_steiner_latency:
                best_steiner_latency = steiner_latency
                best_steiner_tree = steiner_tree
            num_edges += 1

        return best_steiner_tree

    @staticmethod
    def balanced_fusion_tree(steiner_tree: List[qChannel], nodes_degree: dict[int],
                             end_nodes: dict,
                             method: int = EQUAL_EDGES_PARTITIONING,
                             ) -> general_fusion_node:
        """Given a steiner tree, this method tries to construct a balanced fusion tree in terms of the number of edges.
        The following idea is performed recursively to get the final structure."""

        def dfs_helper(target_node):
            """Given a tree, this functions finds all edges connected to any branch rooted from a given node."""
            if len(adj_list[target_node]) == 1:
                # leaf node, one branch include all edges
                return [list(steiner_tree)]
            branches = [[] for _ in range(len(adj_list[target_node]))]
            queue = deque()
            seen = set()
            branch_id = 0
            for neighbor, e in adj_list[target_node].items():
                branches[branch_id].append(e)
                queue.append((neighbor, branch_id))
                seen.add((target_node, neighbor))
                branch_id += 1
            while queue:
                node, branch_id = queue.pop()
                for neighbor, e in adj_list[node].items():
                    if (node, neighbor) not in seen and (neighbor, node) not in seen:
                        branches[branch_id].append(e)
                        queue.append((neighbor, branch_id))
                        seen.add((node, neighbor))
            return branches

        # base cases
        if len(steiner_tree) == 0:
            return None
        if len(steiner_tree) == 1:
            # equal to EP over a channel
            return general_fusion_node(all_nodes={steiner_tree[0].this.id, steiner_tree[0].other.id},
                                       end_nodes={steiner_tree[0].this.id, steiner_tree[0].other.id},
                                       avr_ent_time=steiner_tree[0].get_avr_ent_time(
                                           this_usage=100/nodes_degree[steiner_tree[0].this.id],
                                           other_usage=100/nodes_degree[steiner_tree[0].other.id]))

        nodes, adj_list = {}, defaultdict(dict)
        for edge in steiner_tree:
            nodes[edge.this.id] = edge.this
            nodes[edge.other.id] = edge.other
            adj_list[edge.this.id][edge.other.id] = edge
            adj_list[edge.other.id][edge.this.id] = edge
        # find number of total edges for all branches rooted from all the node, then for non-leaf nodes, find the best
        # two ways partition where the number of edges is most equally divided into two parts
        max_partition_metric = float('inf')
        best_partitions, best_root = None, -1
        for node_id in nodes.keys():
            nodes_branches = dfs_helper(target_node=node_id)
            if method == EQUAL_EDGES_PARTITIONING:
                part1_metric, part1, part2_metric, part2 = equal_edges_partitioning([len(branch) for branch in
                                                                                     nodes_branches])
            elif method == LATENCY_PARTITIONING:
                part1_metric, part1, part2_metric, part2 = latency_balanced_partitioning(
                    branches=[(len(branch), max([e.avr_ent_time for e in branch])) for branch in nodes_branches],
                    fusion_success_rate=nodes_branches[0][0].this.fusion_success_rate,
                    fusion_time=nodes_branches[0][0].this.fusion_time)
            else:
                raise ValueError("No known partitioning method found")
            if max(part1_metric, part2_metric) < max_partition_metric:
                max_partition_metric = max(part1_metric, part2_metric)
                best_root = node_id
                partition1, partition2 = [], []
                for part1_idx in part1:
                    partition1 += nodes_branches[part1_idx]
                for part2_idx in part2:
                    partition2 += nodes_branches[part2_idx]
                best_partitions = (partition1, partition2)

        # recursively create the partitions
        if best_root not in end_nodes:
            end_nodes[best_root] = 0
        end_nodes[best_root] += 1  # the root at this point should be among end nodes in partitions to keep the qubit
        ghz_partition1 = ghz_steiner_tree.balanced_fusion_tree(best_partitions[0], nodes_degree, end_nodes,
                                                               method=method)
        ghz_partition2 = ghz_steiner_tree.balanced_fusion_tree(best_partitions[1], nodes_degree,
                                                               end_nodes, method=method)
        # remove the root again. If root is not one of the final end nodes, it shouldn't be retained anymore
        end_nodes[best_root] -= 1
        if end_nodes[best_root] == 0:
            end_nodes.pop(best_root)

        root = nodes[best_root]
        root_end_nodes = ghz_partition1.end_nodes.union(ghz_partition2.end_nodes)
        if best_root not in end_nodes and best_root in root_end_nodes:
            root_end_nodes.remove(best_root)
        # TODO ignoring classical time for now
        return general_fusion_node(node=root,
                                   all_nodes=ghz_partition1.all_nodes.union(ghz_partition2.all_nodes),
                                   end_nodes=root_end_nodes,
                                   avr_ent_time=(1.5 * max(ghz_partition1.avr_ent_time, ghz_partition2.avr_ent_time)
                                                 + root.fusion_time) / root.fusion_success_rate,
                                   sub1=ghz_partition1, sub2=ghz_partition2,
                                   is_retain=True if best_root in end_nodes else False)

    @staticmethod
    def approx_steiner_tree(edges: List[qChannel], end_nodes: List[qNode], decoherence_time: float):
        """This function calculates an approximated steiner tree in a following way.
            * First, shortest paths between any pair of end nodes is calculated.
            * Given the shortest paths, a complete graph is constructed among end nodes.
            * Minimum (maximum) spanning tree will gives us the approximated steiner tree."""
        # construct the quantum graph
        nodes = []
        seen = set()
        n = len(end_nodes)
        for edge in edges:
            if edge.this.id not in seen:
                seen.add(edge.this.id)
                nodes.append(edge.this)
            if edge.other.id not in seen:
                seen.add(edge.other.id)
                nodes.append(edge.other)

        g = qGraph(V=nodes, E=edges)
        g_dp = dp(G=g, decoherence_time=decoherence_time)

        adj_list = [[] for _ in range(n)]
        for u in range(n - 1):
            for v in range(u + 1, n):
                latency = g_dp.get_shortest_path_without_update_by_id(end_nodes[u].id, end_nodes[v].id).avr_ent_time
                adj_list[u].append(Edge(v, 1.0 / latency))
                adj_list[v].append(Edge(u, 1.0 / latency))

        steiner_tree = spt_kruskal(adj_list)
        steiner_tree_edges = []
        seen = set()
        for edge in steiner_tree:
            tree = g_dp.get_shortest_path_without_update_by_id(end_nodes[edge[0]].id, end_nodes[edge[1]].id)
            _, tree_edges = tree_node.used_nodes_edges(tree)
            seen = seen.union(tree_edges)
        min_edge_rate = float('inf')  # defines the bottleneck
        for edge in edges:
            if (f"{edge.this.id}-{edge.other.id}" in seen or f"{edge.other.id}-{edge.this.id}" in seen) and\
                    not graph_has_cycle(steiner_tree_edges + [edge]):
                min_edge_rate = min(min_edge_rate, 1.0 / edge.avr_ent_time)
                steiner_tree_edges.append(edge)

        # TODO ignoring classical time
        return ghz_steiner_tree.balanced_fusion_tree_latency(num_edges=len(steiner_tree_edges) // 2,
                                                             max_edge_latency=1.0 / min_edge_rate,
                                                             fusion_success_rate=end_nodes[0].fusion_success_rate,
                                                             fusion_time=end_nodes[0].fusion_time), \
               steiner_tree_edges

    @staticmethod
    def end_nodes_connected(total_number_of_nodes: int, edges: List[qChannel], end_nodes: List[qNode]) -> bool:
        def dfs_helper(visiting_node: int, comp_id: int):
            nodes_color[visiting_node] = GRAY
            for neighbor in adj_list[visiting_node]:
                if nodes_color[neighbor] == WHITE:
                    dfs_helper(neighbor, comp_id)
            nodes_color[visiting_node] = BLACK
            nodes_compartment[visiting_node] = comp_id

        nodes_color = [WHITE] * total_number_of_nodes
        nodes_compartment = [-1] * total_number_of_nodes

        adj_list = [[] for _ in range(total_number_of_nodes)]
        for e in edges:
            adj_list[e.this.id].append(e.other.id)
            adj_list[e.other.id].append(e.this.id)

        compartment_id = -1
        for node_id in range(total_number_of_nodes):
            if nodes_color[node_id] == WHITE:
                compartment_id += 1
                dfs_helper(node_id, compartment_id)

        end_nodes_compartment_id = set([nodes_compartment[node.id] for node in end_nodes])
        return len(end_nodes_compartment_id) == 1

    @staticmethod
    def balanced_fusion_tree_latency(num_edges: int, max_edge_latency: float, fusion_success_rate: float,
                                     fusion_time: float, classical_time: float = 0) -> float:
        """Given number of edges, the bottleneck, fusion success probability, fusion time, and maximum classical time,
        this function return an estimated latency assuming the final fusion tree be balanced."""
        if num_edges == 0:
            return 0
        height = math.ceil(math.log2(num_edges))
        const = 1.5 / fusion_success_rate
        return const ** height * max_edge_latency + \
               ((classical_time + fusion_time) / fusion_success_rate) * ((const ** height - 1) / (const - 1))


def equal_edges_partitioning(numbers: List[int]):
    """Given a set of numbers, this function, using dynamic programming, return the best two partitions such that
    the difference between them is minimized."""
    if len(numbers) <= 1:
        return sum(numbers), [0], 0, []
    if len(numbers) <= 2:
        return numbers[0], [0], numbers[1], [1]
    sum_ = sum(numbers)
    n = len(numbers)
    first_partition = [[None] * (n + 1) for _ in range(sum_ // 2 + 1)]
    first_partition[0] = [[]] * (n + 1)
    for part1_sum in range(1, sum_ // 2 + 1):
        for num_idx in range(1, n + 1):
            number = numbers[num_idx - 1]
            if part1_sum - number >= 0:
                if first_partition[part1_sum - number][num_idx - 1] is not None:
                    first_partition[part1_sum][num_idx] = first_partition[part1_sum - number][num_idx - 1] + [num_idx
                                                                                                              - 1]
                elif first_partition[part1_sum][num_idx - 1] is not None:
                    first_partition[part1_sum][num_idx] = list(first_partition[part1_sum][num_idx - 1])
            else:
                if first_partition[part1_sum][num_idx - 1] is not None:
                    first_partition[part1_sum][num_idx] = list(first_partition[part1_sum][num_idx - 1])
    closest_partition = sum_ // 2
    while first_partition[closest_partition][-1] is None:
        closest_partition -= 1
    second_partition, sum_second_partition = [], 0
    first_partition_indices = set(first_partition[closest_partition][-1])
    for idx in range(len(numbers)):
        if idx not in first_partition_indices:
            second_partition.append(idx)
            sum_second_partition += numbers[idx]
    return sum_ - sum_second_partition, sorted(first_partition[closest_partition][-1]), \
           sum_second_partition, second_partition


def latency_balanced_partitioning(branches: List[tuple[int, float]], fusion_success_rate: float, fusion_time: float,
                                  classical_time: float = 0):
    """Given a set of branches, number of edges and the bottleneck(maximum latency) for each branch, this function
    tries to partition them into two parts such that the latency between them is minimized."""
    n = len(branches)
    if n == 0:
        return 0, [], 0, []
    max_latency, best_partition = float('inf'), None
    best_latency1, best_latency2 = 0, 0
    best_part1_num_edges, best_part2_num_edges = 0, 0
    for partition in islice(product([False, True], repeat=n), 2 ** n // 2):
        part1_num_edges, part1_bottleneck = 0, 0
        part2_num_edges, part2_bottleneck = 0, 0
        for branch_idx in range(n):
            if partition[branch_idx]:
                # included in the first partition
                part1_num_edges += branches[branch_idx][0]
                part1_bottleneck = max(part1_bottleneck, branches[branch_idx][1])
            else:  # included in the second partition
                part2_num_edges += branches[branch_idx][0]
                part2_bottleneck = max(part2_bottleneck, branches[branch_idx][1])

        part1_latency = ghz_steiner_tree.balanced_fusion_tree_latency(num_edges=part1_num_edges,
                                                                      max_edge_latency=part1_bottleneck,
                                                                      fusion_success_rate=fusion_success_rate,
                                                                      fusion_time=fusion_time,
                                                                      classical_time=classical_time)
        part2_latency = ghz_steiner_tree.balanced_fusion_tree_latency(num_edges=part2_num_edges,
                                                                      max_edge_latency=part2_bottleneck,
                                                                      fusion_success_rate=fusion_success_rate,
                                                                      fusion_time=fusion_time,
                                                                      classical_time=classical_time)
        if max(part1_latency, part2_latency) < max_latency or \
                (max(part1_latency, part2_latency) == max_latency and
                 abs(part1_num_edges - part2_num_edges) < abs(best_part1_num_edges - best_part2_num_edges)):
            max_latency = max(part1_latency, part2_latency)
            best_partition = partition
            best_latency1 = part1_latency
            best_latency2 = part2_latency
            best_part1_num_edges = part1_num_edges
            best_part2_num_edges = part2_num_edges
    partition1, partition2 = [], []
    for idx in range(n):
        if best_partition[idx]:
            partition1.append(idx)
        else:
            partition2.append(idx)
    return best_latency1, partition1, best_latency2, partition2


def graph_has_cycle(edges: List[qChannel]) -> bool:
    def dfs_helper(visiting_node: int, parent: int):
        visited[visiting_node] = True
        for neighbor in adj_list[visiting_node]:
            if not visited[neighbor]:
                if dfs_helper(neighbor, visiting_node):
                    return True
            elif parent != neighbor:
                return True
        return False

    adj_list = defaultdict(set)
    for edge in edges:
        adj_list[edge.this.id].add(edge.other.id)
        adj_list[edge.other.id].add(edge.this.id)
    visited = {key: False for key in adj_list.keys()}
    for node in adj_list.keys():
        if not visited[node]:
            if dfs_helper(node, -1):
                return True
    return False


if __name__ == "__main__":
    numberss = [8]
    print(equal_edges_partitioning(numberss))
