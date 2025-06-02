'''
design heuristics to find paths faster
'''

import sys
sys.path.append('..')

import math
import numpy as np
from collections import defaultdict
from typing import List
import fibheap as fibh
from random import randint
from commons.Point import Point
from commons.qNode import BSM_TIME, qNode
from commons.qChannel import qChannel
from commons.qGraph import qGraph
from commons.tools import get_min_max, get_classical_time
from topology.tools import Topology
from commons.tree_node import tree_node
from routing_algorithms.dp_shortest_path import dp_shortest_path
import functools



class Path:
    '''encapsulate a path, i.e. a sequence of node ids (index)
    '''
    def __init__(self):
        self.bottleneck = 0   # the bottleneck capacity of a path
        self.nodes = []       # the list of nodes
        self.hop = 0          # number of nodes minus 1
        self.metric = 0
        self.fail = False

    def __str__(self):
        route = '-'.join(map(str, self.nodes))
        return f'route = {route}, bottleneck = {self.bottleneck:.3e}, metric = {self.metric:.3e}'

    def create_final_path(self, u: int, prev: List, bottleneck: List, length: List):
        '''
        Args:
            u:          the last node of the path
            prev:       the previous array
            bottleneck: bottleneck[i] the bottleneck capacity of the path from start start to node i
            length:     length[i] is the path length from the start to node i
        '''
        self.bottleneck = bottleneck[u]
        cur = u
        while cur != prev[cur]:
            self.nodes.append(cur)
            cur = prev[cur]
        self.nodes.append(cur)
        self.nodes = self.nodes[::-1]
        self.hop = len(self.nodes) - 1


class DP_Alt:
    '''DP has high time complexity, so introduce the DP alternate to reduce the time complexity by adding heuristics
    '''
    def __init__(self, G: qGraph, decoherence_time:  float, avg_classical_time=0.0):
        self.check_G(G)
        self.G = G
        self.bsm_success_rate = self.G.V[0].bsm_success_rate  # NOTE temporary solution
        self.N = len(G.V)
        self.fnodes = []                    # fibonacci nodes
        self.adj_list = defaultdict(list)   # adjacency list: u -> [v1, v2, ...]
        self.edge2qchannel = {}             # (u, v) -> qChannel
        self.init_adj_list()
        self.init_edge_qchannel()
        if avg_classical_time:
            self.classical_time = avg_classical_time
        else:
            self.classical_time = self.G.avg_classical_time
            #self.average_node_classical_time()
        self.decoherence_time = decoherence_time

    def check_G(self, G: qGraph):
        '''if the node id starts from 0 and increase by one (id equals the index), then things are simpler.
        '''
        for i, qnode in enumerate(G.V):
            if i != qnode.id:
                raise Exception("id issue, such as the qnode id doesn't start from 0")

    def init_fib_nodes(self):
        '''initialize the nodes for the fibonacci heap
        '''
        self.fnodes = []
        for qnode in self.G.V:
            self.fnodes.append(fibh.Node((float('inf'), qnode.id)))   # (-metric, id)

    def init_adj_list(self):
        '''dijkstra needs an adjacency list.
        '''
        for channel in self.G.E:
            u = channel.this.id
            v = channel.other.id
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def init_edge_qchannel(self):
        '''map an edge to the quantum channel of that edge
        '''
        for qchannel in self.G.E:
            u = qchannel.this.id
            v = qchannel.other.id
            self.edge2qchannel[(get_min_max(u, v))] = qchannel    # order the two node id, so that don't need to store twice

    def average_link_classical_time(self):
        '''the average classical time of all links
        '''
        classical_times = []
        for qchannel in self.G.E:
            distance = qchannel.this.loc.distance(qchannel.other.loc)
            classical_times.append(get_classical_time(distance))
        return sum(classical_times) / len(classical_times)

    def average_node_classical_time(self):
        '''the average classical time between all nodes
        '''
        classical_times = []
        for node1 in self.G.V:
            for node2 in self.G.V:
                #distance = self.G.distance_mat[node1.id][node2.id]
                time = self.G.time_matrix[node1.id][node2.id]
                classical_times.append(time)
        return sum(classical_times) / len(classical_times)

    def dijsktra(self, src: qNode, dst: qNode, metric: str = 'time'):
        '''the extendted dijsktra, single source, the metric is either based on capacity or time
        Args:
            src: source
            dst: destination
            metric: 'capacity' or 'time'
        Returns:
            'Path': the best path
        '''
        if metric == 'capacity':
            return self._dijkstra_capacity(src, dst)
        if metric == 'time':
            return self._dijkstra_time(src, dst)
        else:
            raise Exception('wrong input')

    def _dijkstra_capacity(self, src: qNode, dst: qNode):
        '''uses a metric based on bottleneck capacity
        Args:
            src: source
            dst: destination
        Returns:
            'Path': the best path
        '''
        E       = [float('inf')] * self.N
        prev    = [None] * self.N
        prev[src.id] = src.id             # the previous of source is itself
        visited = [0] * self.N
        bottlenecks = [float('inf')] * self.N
        lengths  = [0] * self.N
        heap     = fibh.makefheap()        # the 'q', a min fibonacci heap
        self.init_fib_nodes()
        for fnode in self.fnodes:
            heap.insert(fnode)
        E[src.id] = float('-inf')
        heap.decrease_key(self.fnodes[src.id], (float('-inf'), src.id))
        
        while heap.num_nodes:
            fnode = heap.extract_min()
            _, u = fnode.key              # (negative_metric, idx), negative metric not used for now
            if visited[u]:
                continue
            visited[u] = 1
            if u == dst.id:
                path = Path()
                path.create_final_path(u, prev, bottlenecks, lengths)
                path.metric = -E[u]
                return path
            for v in self.adj_list[u]:
                if visited[v]:
                    continue
                u_v_capacity = self.edge2qchannel[get_min_max(u, v)].residual_capacity
                bottleneck = min(bottlenecks[u], u_v_capacity)
                length = lengths[u] + 1
                metric = bottleneck * (2/3)**(math.log2(length)) * self.bsm_success_rate**(math.log2(length))
                negative_metric = -metric
                if negative_metric < E[v]:
                    bottlenecks[v] = bottleneck
                    lengths[v] = length
                    E[v] = negative_metric
                    prev[v] = u
                    heap.decrease_key(self.fnodes[v], (negative_metric, v))

    def _dijkstra_time(self, src: qNode, dst: qNode):
        '''uses a metric based on bottleneck time (average entanglement time + classical time + bsm time)
           This is the best possible approximate of a linear path to a tree structure.
        Args:
            src: source
            dst: destination
        Returns:
            'Path': the best path
        '''
        def metric(ent_time, bsm_time, classical_time, length):
            p_prime = 2.5 / (2 * self.bsm_success_rate)
            h = math.ceil(math.log2(length))
            return p_prime ** h * ent_time + ((classical_time + bsm_time) / self.bsm_success_rate) * (p_prime ** h - 1) / (p_prime - 1)

        E       = [float('inf')] * self.N
        prev    = [None] * self.N
        prev[src.id] = src.id             # the previous of source is itself
        # prev = [i for i in range(self.N)]
        visited     = [0] * self.N
        bottlenecks = [0] * self.N        # the time, the bottleneck here is the largest time (entanglement time + bsm_time + classical time)
        lengths     = [0] * self.N
        heap        = fibh.makefheap()    # the 'q', a min fibonacci heap
        self.init_fib_nodes()
        for fnode in self.fnodes:
            heap.insert(fnode)
        E[src.id] = 0
        heap.decrease_key(self.fnodes[src.id], (0, src.id))

        while heap.num_nodes:
            fnode = heap.extract_min()
            _, u = fnode.key              # (negative_metric, idx), negative metric not used for now
            if visited[u]:
                continue
            visited[u] = 1
            if u == dst.id:
                path = Path()
                try:
                    path.create_final_path(u, prev, bottlenecks, lengths)
                    path.metric = E[u]
                except Exception as e:
                    path.fail = True
                return path
            for v in self.adj_list[u]:
                if visited[v]:
                    continue
                u_v_ent_time = self.edge2qchannel[get_min_max(u, v)].avr_ent_time
                bottleneck = max(bottlenecks[u], u_v_ent_time)
                length = lengths[u] + 1
                eval = metric(bottleneck, BSM_TIME, self.classical_time, length)
                if eval < E[v]:
                    bottlenecks[v] = bottleneck
                    lengths[v] = length
                    E[v] = eval
                    prev[v] = u
                    heap.decrease_key(self.fnodes[v], (eval, v))

    def build_tree_layers_iterative(self, path: Path):
        '''build a tree iteratively, pretty sure it is balanced
        Args:
            path -- a Path object
        Return:
            List[List] -- describes the tree structure not in tree, but in lists
        '''
        entangle_link_layer = [(path.nodes[i], path.nodes[i + 1]) for i in range(len(path.nodes)-1)]
        tree_layers = [entangle_link_layer]
        left_to_right = True
        while True:
            latest_layer = tree_layers[-1]
            new_layer = []
            if left_to_right:
                i = 0
                while i < len(latest_layer) - 1:
                    new_layer.append((latest_layer[i][0], latest_layer[i][-1], latest_layer[i+1][-1]))
                    i += 2
                if i == len(latest_layer) - 1:
                    new_layer.append(latest_layer[-1])
                left_to_right = False
            else:
                i = len(latest_layer) - 1
                while i > 0:
                    new_layer.append((latest_layer[i-1][0], latest_layer[i-1][-1], latest_layer[i][-1]))
                    i -= 2
                if i == 0:
                    new_layer.append(latest_layer[0])
                new_layer = new_layer[::-1]
                left_to_right = True
            tree_layers.append(new_layer)
            if len(new_layer) == 1:
                break
        return tree_layers

    def build_tree_layers_recursive(self, path: Path):
        '''build a balanced tree recursively
        Args:
            path -- a Path object
        Return:
            List[List] -- describes the tree structure not in tree, but in lists
        '''
        entangle_link_layer = [(path.nodes[i], path.nodes[i + 1]) for i in range(len(path.nodes)-1)]   # (left, right)
        tree_layers = [entangle_link_layer]
        height = math.ceil(math.log2(len(path.nodes) - 1))
        for _ in range(height):
            tree_layers.append([])

        def helper(layer, height):
            if len(layer) == 0:
                return
            if len(layer) == 1:
                node = (layer[0][0], layer[0][1])
                tree_layers[height].append(node)
                return
            if len(layer) == 2:
                left_node = layer[0]
                right_node = layer[1]
                node = (left_node[0], left_node[-1], right_node[-1])                     # (left, middle, right)
                tree_layers[height].append(node)
                return
            left, right = 0, len(layer) - 1
            mid_left = right // 2
            mid_right = mid_left + 1
            helper(layer[left:mid_left+1], height-1)
            helper(layer[mid_right:right+1], height-1)
            left_node = layer[left][0]
            mid_node = layer[mid_left][-1]
            right_node = layer[right][-1]
            node = (left_node, mid_node, right_node)
            tree_layers[height].append(node)

        helper(entangle_link_layer, height)
        return tree_layers

    def construct_tree(self, tree_layers: List, decoherence_time: float):
        '''construct tree, the tree may not meet decoherence time contraint
        Args:
            tree_layers -- a list of tree
            decoherence_time -- decoherence time when an entanglement disappears
        Return:
            Tree
        '''
        tree_node_dict = {}
        # tree_layers[0] are the elementary links
        for u, v in tree_layers[0]:
            qchannel = self.edge2qchannel[get_min_max(u, v)]
            if qchannel.this.id != u:
                qchannel = qChannel(qchannel.other, qchannel.this, qchannel.channels_num)
            tree_node_dict[(u, v)] = tree_node(qchannel,
                                               avr_ent_time = qchannel.avr_ent_time if qchannel.avr_success_time < decoherence_time else float('inf'),
                                               one_round_ent_time = qchannel.avr_success_time if qchannel.avr_success_time < decoherence_time else float('inf'))
        # the BSM
        for tree_layer in tree_layers[1:]:
            for node in tree_layer:
                if len(node) == 3:
                    src, k, dst = node
                    src_node = self.G.V[src]
                    k_node   = self.G.V[k]   # bsm node
                    dst_node = self.G.V[dst]
                    # src_to_k_classical = get_classical_time(k_node.loc.distance(src_node.loc))
                    # k_to_dst_classical = get_classical_time(k_node.loc.distance(dst_node.loc))
                    t = 1.5*( max(tree_node_dict[(src, k)].avr_ent_time, tree_node_dict[(k, dst)].avr_ent_time) +
                        self.classical_time + k_node.bsm_time) / k_node.bsm_success_rate
                    # t_s = self.classical_time + k_node.bsm_time
                    # if tree_node_dict[(src, k)].avr_ent_time < tree_node_dict[(k, dst)].avr_ent_time:
                    #     t_s += max(tree_node_dict[(src, k)].one_round_ent_time + tree_node_dict[(k, dst)].avr_ent_time - tree_node_dict[(src, k)].avr_ent_time, 
                    #                tree_node_dict[(k, dst)].one_round_ent_time)
                    # else:
                    #     t_s += max(tree_node_dict[(src, k)].one_round_ent_time, 
                    #                tree_node_dict[(k, dst)].one_round_ent_time + tree_node_dict[(src, k)].avr_ent_time - tree_node_dict[(k, dst)].avr_ent_time)
                    # if t_s < decoherence_time:
                    left  = tree_node_dict[(src, k)]
                    right = tree_node_dict[(k, dst)]
                    src_dst_node = tree_node(data=k_node, left=left, right=right, avr_ent_time=t,
                                             one_round_ent_time=0,
                                             classical_time=self.classical_time,
                                             left_avr_ent_time=tree_node_dict[(src, k)].avr_ent_time,
                                             right_avr_ent_time=tree_node_dict[(k, dst)].avr_ent_time)
                    tree_node_dict[(src, dst)] = src_dst_node
                if len(node) == 2:  # elementary link, no bsm
                    pass
        # return the root
        root = (tree_layers[-1][0][0], tree_layers[-1][0][-1])
        return tree_node_dict[root]

    def path2tree(self, path: Path, decoherence_time: float, update_time: bool = True, NON_THROTLING:bool=False):
        '''first try balanced tree, if surpass the decoherence time, then try the dp tree
           NOTE this function is also shared by Delft-LP 
        Args:
            path -- tree object
            decoherence_time -- decoherence time when an entanglement disappears
        Return:
            tree_node
        '''
        _, tree_root = self.path2tree_balanced(path, decoherence_time, 'recursive')
        if update_time:
            _, _, new_tree_root = dp_shortest_path.update_ent_time_tree(tree_root, tree_root.avr_ent_time, NON_THROTLING)
            if new_tree_root.one_round_ent_time < decoherence_time:
                return new_tree_root   # for DP alternate, the balanced tree is valid
        if update_time:
            # if reach here, then balanced tree failed, then try dp
            tree_root = self.path2tree_dp(path, decoherence_time)
            _, _, new_tree_root = dp_shortest_path.update_ent_time_tree(tree_root, tree_root.avr_ent_time, NON_THROTLING)
            if new_tree_root.one_round_ent_time < decoherence_time:
                return new_tree_root   # for DP alternate, the dp tree is valid
            else:
                return None        # for DP alternate, the tree construction fails
        return tree_root           # for Delft LP

    def path2tree_balanced(self, path: Path, decoherence_time: float, method: str = 'recursive'):
        '''from path to balanced tree
        Args:
            path -- Path object
            decoherence_time -- decoherence time when an entanglement disappears
            method -- 'iterative' or 'recursive'
        Return:
            List[List], tree_node
        '''
        if method == 'iterative':
            tree_layers = self.build_tree_layers_iterative(path)
        elif method == 'recursive':
            tree_layers = self.build_tree_layers_recursive(path)
        else:
            raise Exception('argument method input error')
        tree_root = self.construct_tree(tree_layers, decoherence_time)
        #tree_root.avr_ent_time *=0.1*len(self.G.V)
        #if len(path.nodes)==2:
        #    tree_root.avr_ent_time = 0.0005   # TODO let it be like this for now
        return tree_layers, tree_root

    def path2tree_dp(self, path: Path, decoherence_time: float):
        '''use dp to construct a path to a tree
        Args:
            path -- a linear path
            decoherence_time -- a threshold for decoherence
        Return:
            tree_node
        '''
        n = len(path.nodes)
        dp = [[tree_node()] * n for _ in range(n)]   # reduce n^3 space complexity to n^2 by using the new_dp
        for i in range(n):
            u = path.nodes[i]
            dp[i][i] = tree_node(data=self.G.V[u], avr_ent_time=0, one_round_ent_time=0)
        for i in range(n-1):
            u, v = path.nodes[i], path.nodes[i + 1]
            qchannel = self.edge2qchannel[get_min_max(u, v)]
            dp[i][i + 1] = tree_node(qchannel, avr_ent_time = qchannel.avr_ent_time if qchannel.avr_success_time < decoherence_time else float('inf'),
                                     one_round_ent_time = qchannel.avr_success_time if qchannel.avr_success_time < decoherence_time else float('inf'))
            dp[i + 1][i] = dp[i][i + 1]
        for _ in range(n - 2):
            new_dp = [[tree_node()] * n for _ in range(n)]
            for i in range(n):
                u = path.nodes[i]
                new_dp[i][i] = tree_node(data=self.G.V[u], avr_ent_time=0, one_round_ent_time=0)
            for src in range(n):  # src id
                i = path.nodes[src]
                src_node = self.G.V[i]
                for dst in range(src + 1, n):  # dst id
                    j = path.nodes[dst]
                    dst_node = self.G.V[j]
                    if src == dst:  # dp(src, src) always zero
                        continue
                    shortest_src_dst = dp[src][dst]
                    for k in range(n):
                        if k == src or k == dst:  # one side is zero
                            continue
                        k_node = self.G.V[path.nodes[k]]  # the node bsm happens
                        # average total ent time
                        src_to_k_classical = get_classical_time(k_node.loc.distance(src_node.loc))
                        k_to_dst_classical = get_classical_time(k_node.loc.distance(dst_node.loc))
                        t = (1.5 * max(dp[src][k].avr_ent_time, dp[k][dst].avr_ent_time) + k_node.bsm_time + max(src_to_k_classical, k_to_dst_classical)) / k_node.bsm_success_rate
                                   # time to generate both left and right subtrees         # time to send BSM res back
                        # average one round of entanglement (waiting included)
                        # t_s = max(src_to_k_classical, k_to_dst_classical) + k_node.bsm_time
                        # if dp[src][k].avr_ent_time < dp[k][dst].avr_ent_time:
                        #     t_s += max(dp[src][k].one_round_ent_time + dp[k][dst].avr_ent_time - dp[src][k].avr_ent_time, dp[k][dst].one_round_ent_time)
                        # else:
                        #     t_s += max(dp[src][k].one_round_ent_time, dp[k][dst].one_round_ent_time + dp[src][k].avr_ent_time - dp[k][dst].avr_ent_time)

                        if t < shortest_src_dst.avr_ent_time: # and t_s < decoherence_time:  # found a shorter path
                            left  = dp[src][k]
                            right = dp[k][dst]
                            shortest_src_dst = tree_node(data = k_node, left = left, right = right, avr_ent_time = t, one_round_ent_time = 0,
                                                         classical_time = max(src_to_k_classical, k_to_dst_classical),
                                                         left_avr_ent_time = dp[src][k].avr_ent_time, right_avr_ent_time = dp[k][dst].avr_ent_time)
                    new_dp[src][dst] = new_dp[dst][src] = shortest_src_dst  # both ways
            dp = new_dp
        if dp[0][n-1] is not None:
            dp[0][n-1].avr_ent_time = path.metric   # TODO let it be like this for now
        return dp[0][n-1]

    def get_shortest_path(self, src: qNode, dst: qNode, NON_THROTTLING: bool = False):
        if src is dst:
            return tree_node()
        path = self.dijsktra(src, dst, metric='time')
        if path.fail:
            return None
        return self.path2tree(path, decoherence_time=self.decoherence_time, update_time=True,
                              NON_THROTLING=NON_THROTTLING)

    def get_shortest_path_by_id(self, src_id: int, dst_id: int, NON_THROTTLING: bool = False):
        return self.get_shortest_path(self.G.get_node(src_id), self.G.get_node(dst_id), NON_THROTTLING)

    def get_shortest_path_without_update_by_id(self, src_id: int, dst_id: int):
        return self.get_shortest_path_by_id(src_id, dst_id)

    def get_shortest_path_without_update(self, src: qNode, dst: qNode):
        return self.get_shortest_path(src, dst)

def create_toy_graph(area_side: int, cell_size: int, number_nodes: int, min_memory_size: int, max_memory_size: int,
                     min_num_edge: int, max_num_edge: int, min_channel_num: int, max_channel_num: int) -> qGraph:
    points = [Point(x, y) for x, y in zip([7700, 9800, 7400, 3900, 8500, 9100, 4100, 2600, 4700, 6900],
                                          [8200,  5600, 2400, 2400, 6600, 5300, 7000, 9900, 8100, 8900])]  # for test
    num_channels =  [5, 3, 1, 4, 3, 5, 4, 1, 2, 3, 3, 3, 1, 1]
    num_edges_rnd = [2, 2, 2, 3, 3, 3, 4, 2, 4, 3]                                                         # for test
    V = [qNode(randint(min_memory_size, max_memory_size), point) for point in points]
    E = []
    num_edges = [0] * len(V)  # number of edges for each node
    edges_set = set()  # a set to avoid parallel edges with key "u-v"
    for i, v in enumerate(V):
        num_v_edges = num_edges_rnd[i] - num_edges[i]                                                      # for test
        distance_to_v = [(v.loc.distance(V[k].loc), k) for k in range(i)] + \
                        [(v.loc.distance(V[k].loc), k) for k in range(i+1, len(V))]
        distance_to_v.sort(key=lambda x: x[0])  # sort based on distances
        neighbor_idx, edge_num = 0, 0
        while edge_num < num_v_edges and neighbor_idx < len(distance_to_v):
            k_id = distance_to_v[neighbor_idx][1]
            if "{u}-{v}".format(u=i, v=k_id) in edges_set:
                neighbor_idx += 1
                continue
            E.append(qChannel(v, V[k_id], num_channels.pop()))    # v -> k                                      # for test
            num_edges[k_id] += 1
            edge_num += 1
            neighbor_idx += 1
            edges_set.add("{u}-{v}".format(u=i, v=k_id))
            edges_set.add("{u}-{v}".format(u=k_id, v=i))
        num_edges[i] += num_v_edges
    return qGraph(V=V, E=E)


def test(filename):
    '''testing dijkstra
    '''
    topo = Topology()
    topo.read_file(filename)
    qnodes = []
    qchannels = []
    counter = 0
    for idx, (x, y) in topo.nodes.items():
        if counter != idx:
            raise Exception('id issue')
        counter += 1
        qnodes.append(qNode(memory = topo.memories[idx], loc = Point(x, y), node_id = idx))
    for idx1, idx2 in topo.edges:
        qchannels.append(qChannel(qnodes[idx1], qnodes[idx2], channels_num=topo.num_channels[idx1, idx2]))

    # metric = 'capacity'
    metric = 'time'
    dp_alt = DP_Alt(qGraph(qnodes, qchannels))
    src = qnodes[0]
    dst = qnodes[5]
    path = dp_alt.dijsktra(src, dst, metric=metric)
    tree_layers, tree_root = dp_alt.path2tree_balanced(path, decoherence_time=0.02)
    print(path)
    for i in range(len(tree_layers) - 1, -1, -1):
        print(tree_layers[i])
    if tree_root:
        print(tree_root)
        print(f'difference = {(tree_root.avr_ent_time - path.metric) / path.metric * 100:.3}%')
    else:
        print('balanced tree exceed the decoherene time limit')

    tree_root_dp = dp_alt.path2tree_dp(path, decoherence_time=0.02)
    if tree_root_dp:
        print(tree_root_dp)
        print(f'difference = {(tree_root_dp.avr_ent_time - path.metric) / path.metric * 100:.3}%')
    else:
        print('dp tree exceed the decoherence time limit')


def test2(a, b):
    '''difference between the path metric and balanced tree
    '''
    qNode._stat_id = 0
    graph0 = create_toy_graph(area_side=100, cell_size=1000, number_nodes=100,
                              min_memory_size=10, max_memory_size=20,
                              min_num_edge=3, max_num_edge=5, min_channel_num=1, max_channel_num=5)
    # graph0.visualize(filename='../topology/toy_a')
    metric = 'time'
    dp_alt = DP_Alt(graph0)
    src = graph0.V[a]
    dst = graph0.V[b]
    path = dp_alt.dijsktra(src, dst, metric=metric)
    tree_layers, tree_root = dp_alt.path2tree_balanced(path, decoherence_time=0.03)
    # print(path)
    # for i in range(len(tree_layers) - 1, -1, -1):
    #     print(tree_layers[i])
    if tree_root:
        pass
        # print(tree_root)
        # print(f'difference = {(path.metric - tree_root.avr_ent_time) / path.metric * 100:.3}%')
    else:
        print('exceed the decoherene time limit')
    return path.hop, (path.metric - tree_root.avr_ent_time) / path.metric * 100

def test3(a, b):
    '''difference between the path metric and dp tree
    '''
    qNode._stat_id = 0
    graph0 = create_toy_graph(area_side=100, cell_size=1000, number_nodes=100,
                              min_memory_size=10, max_memory_size=20,
                              min_num_edge=3, max_num_edge=5, min_channel_num=1, max_channel_num=5)
    # graph0.visualize(filename='../topology/toy_a')
    metric = 'time'
    dp_alt = DP_Alt(graph0)
    src = graph0.V[a]
    dst = graph0.V[b]
    path = dp_alt.dijsktra(src, dst, metric=metric)
    tree_root = dp_alt.path2tree_dp(path, decoherence_time=0.03)
    if tree_root:
        pass
        # print(tree_root)
        # print(f'difference = {(path.metric - tree_root.avr_ent_time) / path.metric * 100:.3}%')
    else:
        print('exceed the decoherene time limit')
    return path.hop, (path.metric - tree_root.avr_ent_time) / path.metric * 100


def test4(a, b):
    '''comparing the path metric, balanced tree and dp tree
    '''
    qNode._stat_id = 0
    graph0 = create_toy_graph(area_side=100, cell_size=1000, number_nodes=100,
                              min_memory_size=10, max_memory_size=20,
                              min_num_edge=3, max_num_edge=5, min_channel_num=1, max_channel_num=5)
    metric = 'time'
    dp_alt = DP_Alt(graph0)
    src = graph0.V[a]
    dst = graph0.V[b]
    path = dp_alt.dijsktra(src, dst, metric=metric)
    tree_layers, tree_root_balanced = dp_alt.path2tree_balanced(path, decoherence_time=0.03)
    tree_root_dp = dp_alt.path2tree_dp(path, decoherence_time=0.03)
    if tree_root_balanced and tree_root_dp:
        return path, tree_root_balanced, tree_root_dp
    else:
        return None, None, None


if __name__ == '__main__':
    # toy_example = '../topology/toy1'
    # toy_example = '../topology/toy2-multichannel'
    # test(toy_example)

    # toy_example = '../topology/toy3-multichannel'

    # toy_example = '../topology/toy4-multichannel'
    # test(toy_example)

    diff = defaultdict(list)
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            length, d = test2(a, b)
            diff[length].append(d)
    for length, diffs in sorted(diff.items()):
        print(f'path length = {length}, avg path.metric & balanced tree diff = {np.average(np.abs(diffs)):.2f}%, max diff = {np.max(np.abs(diffs)):.2f}%')

    diff = defaultdict(list)
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            length, d = test3(a, b)
            diff[length].append(d)
    for length, diffs in sorted(diff.items()):
        print(f'path length = {length}, avg path.metric & dp tree diff = {np.average(np.abs(diffs)):.2f}%, max diff = {np.max(np.abs(diffs)):.2f}%')

    # counter_same = Counter()
    # counter_diff = Counter()
    # diff = defaultdict(list)
    # for a in range(10):
    #     for b in range(10):
    #         if a == b:
    #             continue
    #         path, tree_root_balanced, tree_root_dp = test4(a, b)
    #         if abs(tree_root_balanced.avr_ent_time - tree_root_dp.avr_ent_time) > 10**-8:
    #             counter_diff[path.hop] += 1
    #             print(path)
    #             print('balanced tree:\n', tree_root_balanced)
    #             print('dp tree:\n', tree_root_dp)
    #             print('===================\n')
    #         else:
    #             counter_same[path.hop] += 1
    # print('\ncounter_same')
    # print(counter_same)
    # print('\ncounter_diff')
    # print(counter_diff)
