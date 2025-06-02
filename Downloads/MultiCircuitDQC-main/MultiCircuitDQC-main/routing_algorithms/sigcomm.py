'''multiple channels in each edge. current status is that the phase two of Q-CAST is complete.
'''
import sys
sys.path.append('..')

from collections import defaultdict, Counter
from typing import List
import math
import fibheap as fibh
import copy

from commons.Point import Point
from commons.qNode import qNode, GENERATION_TIME, BSM_TIME
from commons.qChannel import qChannel
from commons.qGraph import qGraph
from commons.tree_node import tree_node
from commons.tools import get_min_max, get_classical_time, time_slot_num, time_slot_to_time
from topology.tools import Topology



class Path:
    '''encapsulate a path, i.e. a sequence of node ids (index)
    '''
    def __init__(self, w: int = 1):
        self.W = w            # the width of a path
        self.nodes = []       # the list of nodes
        self.hop = 0          # number of nodes minus 1
        self.metric = 0       # the metric defined in the paper equation (2)
        self.avr_ent_time = 0 # rate = metric / super_time_slot, avr_ent_time = 1 / rate

    def __str__(self):
        route = '-'.join(map(str, self.nodes))
        return f'route = {route}, width = {self.W}, metric = {self.metric:.3e}'

    def create_tmp_path(self, u: int, v: int, prev: List, width: List, edge2qchannel: defaultdict, node_memory: Counter):
        '''
        Args:
            u: second last node of a path
            v: the last node of a path
            prev: the previous array
            width: the width array
            edge_qchannel: defaultdict(list), (u, v) -> [qchannel1, qchannel2, ...]
        '''
        self.nodes = [v]
        if u != prev[u]:  # u is not the first node of the path
            memory_limit = min(int(node_memory[u] / 2), node_memory[v])
        else:             # u is the first node of the path
            memory_limit = min(node_memory[u], node_memory[v])
        # consider both quantum channel capacity and node quantum memory capacity
        width_uv = min(edge2qchannel[(get_min_max(u, v))].channels_num, memory_limit)
        self.W = width_uv
        cur = u
        while cur != prev[cur]:
            self.nodes.append(cur)
            self.W = min(self.W, width[cur])
            cur = prev[cur]
        self.nodes.append(cur)
        self.nodes = self.nodes[::-1]
        self.hop = len(self.nodes) - 1


    def create_final_path(self, u: int, prev: List, width: List):
        '''
        Args:
            u:     the last node of the path
            prev:  the previous array
            width: the width array
        '''
        self.W = float('inf')
        cur = u
        while cur != prev[cur]:
            self.nodes.append(cur)
            self.W = min(self.W, width[cur])
            cur = prev[cur]
        self.nodes.append(cur)
        self.nodes = self.nodes[::-1]
        self.hop = len(self.nodes) - 1


class Sigcomm:
    '''This class implements the Algorithm 2 (Extended Dijsktra's) and also the phase two in the Sigcomm 2020 paper
    '''
    def __init__(self, G: qGraph):
        self.check_G(G)
        self.G = G
        self.bsm_success_rate = self.G.V[0].bsm_success_rate  # NOTE temporary solution
        self.N = len(G.V)
        self.fnodes = []                    # fibonacci nodes
        self.adj_list = defaultdict(list)   # adjacency list: u -> [v1, v2, ...]
        self.edge2qchannel = {}             # (u, v) -> qChannel
        self.node_memory = Counter()        # int -> int    -- number of memory units at a node
        self.init_adj_list()
        self.init_edge_qchannel()
        self.init_node_memory()


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
        '''dijstra needs an adjacency list.
        '''
        for channel in self.G.E:
            u = channel.this.id
            v = channel.other.id
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)


    def init_edge_qchannel(self):
        '''map an edge to all the multiple channels in that edge
        '''
        for qchannel in self.G.E:
            u = qchannel.this.id
            v = qchannel.other.id
            self.edge2qchannel[(get_min_max(u, v))] = qchannel    # order the two node id, so that don't need to store twice


    def init_node_memory(self):
        '''each node has a memory that holds some number of qubits
        '''
        for qnode in self.G.V:
            self.node_memory[qnode.id] = qnode.memory


    def routing_metric(self, path: Path):
        '''return how good/bad a path is, returns a real number and the higher the better
        TODO The metric can be optimized
        Args:
            path:  a path
            width: width of a path
        Returns:
            float
        '''
        # step 1, compute Q
        Q = [[0 for _ in range(path.W + 1)] for _ in range(path.hop + 1)]
        for k in range(1, path.hop + 1):
            u, v = get_min_max(path.nodes[k-1], path.nodes[k])
            qchannel = self.edge2qchannel[(u, v)]             # assume each qchannel in a same edge is identical
            if qchannel.channels_num:
                p_suc = qchannel.channel_success_rate
            else:
                p_suc = 0
            for i in range(1, path.W + 1):
                Q[k][i] = math.comb(path.W, i) * p_suc**i * (1 - p_suc)**(path.W - i)

        # step 2, compute P
        P = [[0 for _ in range(path.W + 1)] for _ in range(path.hop + 1)]
        P[1] = Q[1][:]
        for k in range(2, path.hop + 1):
            for i in range(1, path.W + 1):
                term1 = 0
                for l in range(i, path.W + 1):
                    term1 += Q[k][l]
                term1 *= P[k-1][i]
                term2 = 0
                for l in range(i + 1, path.W + 1):
                    term2 += P[k - 1][l]
                term2 *= Q[k][i]
                P[k][i] = term1 + term2 

        # step 3, the metric, exptected throughput
        e_t = []
        for i in range(1, path.W + 1):
            e_t.append(self.bsm_success_rate**(path.hop-1) * i * P[path.hop][i])   # I am convinced that this is correct, i.e. q^{i*h} is wrong

        return sum(e_t)


    def dijkstra(self, src: qNode, dst: qNode):
        '''the extendted dijsktra, single source
        Args:
            src: source
            dst: destination
        Returns:
            'Path': the best path
            int   : the width of the best path
        '''
        E       = [float('inf')] * self.N
        prev    = [None] * self.N
        prev[src.id] = src.id             # the previous of source is itself
        visited = [0] * self.N
        width   = [0] * self.N
        heap    = fibh.makefheap()        # the 'q', a min fibonacci heap
        self.init_fib_nodes()
        for fnode in self.fnodes:
            heap.insert(fnode)
        E[src.id] = float('-inf')
        width[src.id] = float('inf')
        heap.decrease_key(self.fnodes[src.id], (float('-inf'), src.id))
        
        while heap.num_nodes:
            fnode = heap.extract_min()
            _, u = fnode.key              # (negative_metric, idx), negative metric not used for now
            if visited[u]:
                continue
            visited[u] = 1
            if u == dst.id:
                path = Path()
                path.create_final_path(u, prev, width)
                path.metric = -E[u]
                return path
            for v in self.adj_list[u]:
                if visited[v]:
                    continue
                path = Path()
                path.create_tmp_path(u, v, prev, width, self.edge2qchannel, self.node_memory)  # W is included in class Path
                metric = self.routing_metric(path)
                negative_metric = -metric
                if negative_metric < E[v]:
                    E[v] = negative_metric
                    prev[v] = u
                    width[v] = path.W
                    heap.decrease_key(self.fnodes[v], (negative_metric, v))


    def update_residual(self, path: Path):
        '''update the resources of the network, including memories at the nodes and number of quantum channels at an edge
        Args:
            path: the path that consumes resources
        '''
        # step 1: update the quantum channels at the edge
        for i in range(path.hop):
            u, v = path.nodes[i], path.nodes[i + 1]
            channels_num = self.edge2qchannel[get_min_max(u, v)].channels_num
            self.edge2qchannel[get_min_max(u, v)].channels_num = channels_num - path.W
            if self.edge2qchannel[get_min_max(u, v)].channels_num < 0:
                raise Exception('channels number cannot be negative, somewhere is wrong. Oops!')
        # step 2: update the memories at the node
        for i, node_id in enumerate(path.nodes):
            if i == 0 or i == len(path.nodes) - 1:       # case 1: node is the first or last node of a path
                self.node_memory[node_id] -= path.W
            else:                                        # case 2: node is in the middle
                self.node_memory[node_id] -= path.W * 2


    def phase_two(self, sd_pairs: List):
        '''the phase two of the Q-CAST (the best of best)
        Args:
            sd_pairs -- a list of (src, dst), where src and dst are qNode
        Return:
            paths: List -- an element is a Path object
        '''
        selected_paths = []
        while True:
            best_path = None
            max_eps = 0
            for src, dst in sd_pairs:
                path = self.dijkstra(src, dst)
                if path.metric > max_eps:
                    max_eps = path.metric
                    best_path = path
            if best_path:
                selected_paths.append(best_path)
                self.update_residual(best_path)
            else:    # no more paths, then break
                break
        return selected_paths


    def super_time_slot(self, path: Path):
        '''A super time slot, everything is done in this one shot, either all is success or fail
        Args:
            path -- a path object
        Return:
            float
        '''
        max_length = 0
        for k in range(1, path.hop + 1):
            u, v = path.nodes[k - 1], path.nodes[k]
            qchannel = self.edge2qchannel[get_min_max(u, v)]
            max_length = max(max_length, qchannel.length)
        link_time = GENERATION_TIME + get_classical_time(max_length)  # charlie in the middle, so the classical time is get_classical_time(max_length/2) * 2
        swapping_time = BSM_TIME + 2 * get_classical_time(max_length)
        return link_time + swapping_time


    def overall_probability(self, path: Path):
        '''no memory. the overall probability for everything to success at one shot
        Args:
            path -- a Path object
        Return:
            float
        '''
        # step 1: individual entanglement links
        overall_prob = 1
        lengths = []
        for k in range(1, path.hop + 1):
            u, v = path.nodes[k - 1], path.nodes[k]
            qchannel = self.edge2qchannel[get_min_max(u, v)]
            link_prob = qchannel.link_success_rate
            overall_prob *= link_prob
            lengths.append(qchannel.length)
        # step 2: entanglement swapping
        for k in range(1, path.hop):
            u, v = path.nodes[k - 1], path.nodes[k]
            bsm_node = self.edge2qchannel[get_min_max(u, v)].other
            overall_prob *= bsm_node.bsm_success_rate
        return overall_prob


    def paths2rates(self, paths: List[Path], verbose: bool = True):
        '''given a list of paths, compute the entanglement pair rate per second
        Args:
            paths -- a list of paths
        Return:
            List -- a list of entanglement pair per second
        '''
        rates = []
        for path in paths:
            super_time_slot = self.super_time_slot(path)
            overall_probability = self.overall_probability(path)
            rate = path.W * overall_probability / super_time_slot
            rates.append(rate)
            if verbose:
                print(f'{path} super time slot is {super_time_slot:.6f}, overall probability is {overall_probability}')

        return rates


class Sigcomm_W:
    '''This class implements the Algorithm 2 (Extended Dijsktra's) and also the phase two in the Sigcomm 2020 paper
       This modified version passes an additional parameter W as the input, assume each link is identical and has W "sub-channels"
    '''
    def __init__(self, G: qGraph, W: int):
        self.check_G(G)
        self.G = copy.deepcopy(G)
        self.W = W
        self.bsm_success_rate = self.G.V[0].bsm_success_rate  # NOTE temporary solution
        self.N = len(G.V)
        self.fnodes = []                    # fibonacci nodes
        self.adj_list = defaultdict(list)   # adjacency list: u -> [v1, v2, ...]
        self.edge2qchannel = {}             # (u, v) -> qChannel
        self.node_memory = Counter()        # int -> int    -- number of memory units at a node
        self.init_adj_list()
        self.init_edge_qchannel()
        self.init_node_memory()
        self.classical_time = self.average_node_classical_time()


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
        '''dijstra needs an adjacency list.
        '''
        for channel in self.G.E:
            u = channel.this.id
            v = channel.other.id
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)


    def init_edge_qchannel(self):
        '''map an edge to all the multiple channels in that edge
        '''
        for qchannel in self.G.E:
            u = qchannel.this.id
            v = qchannel.other.id
            qchannel.channels_num = self.W                        # NOTE setting the W for channel
            self.edge2qchannel[(get_min_max(u, v))] = qchannel    # order the two node id, so that don't need to store twice


    def init_node_memory(self):
        '''each node has a memory that holds some number of qubits
        '''
        for qnode in self.G.V:
            self.node_memory[qnode.id] = qnode.memory

    def average_node_classical_time(self):
        '''the average classical time between all nodes
        '''
        classical_times = []
        for node1 in self.G.V:
            for node2 in self.G.V:
                distance = node1.loc.distance(node2.loc)
                time = get_classical_time(distance)
                classical_times.append(time)
        return sum(classical_times) / len(classical_times)


    def routing_metric(self, path: Path):
        '''return how good/bad a path is, returns a real number and the higher the better
        TODO The metric can be optimized
        Args:
            path:  a path
            width: width of a path
        Returns:
            float
        '''
        # step 1, compute Q
        Q = [[0 for _ in range(path.W + 1)] for _ in range(path.hop + 1)]
        for k in range(1, path.hop + 1):
            u, v = get_min_max(path.nodes[k-1], path.nodes[k])
            qchannel = self.edge2qchannel[(u, v)]             # assume each qchannel in a same edge is identical
            if qchannel.channels_num:
                p_suc = qchannel.link_success_rate
            else:
                p_suc = 0
            for i in range(1, path.W + 1):
                Q[k][i] = math.comb(path.W, i) * p_suc**i * (1 - p_suc)**(path.W - i)

        # step 2, compute P
        P = [[0 for _ in range(path.W + 1)] for _ in range(path.hop + 1)]
        P[1] = Q[1][:]
        for k in range(2, path.hop + 1):
            for i in range(1, path.W + 1):
                term1 = 0
                for l in range(i, path.W + 1):
                    term1 += Q[k][l]
                term1 *= P[k-1][i]
                term2 = 0
                for l in range(i + 1, path.W + 1):
                    term2 += P[k - 1][l]
                term2 *= Q[k][i]
                P[k][i] = term1 + term2 

        # step 3, the metric, exptected throughput
        e_t = []
        for i in range(1, path.W + 1):
            e_t.append(self.bsm_success_rate**(path.hop-1) * i * P[path.hop][i])   # I am convinced that this is correct, i.e. q^{i*h} is wrong

        return sum(e_t)


    def dijkstra(self, src: qNode, dst: qNode):
        '''the extendted dijsktra, single source
        Args:
            src: source
            dst: destination
        Returns:
            'Path': the best path
            int   : the width of the best path
        '''
        E       = [float('inf')] * self.N
        prev    = [None] * self.N
        prev[src.id] = src.id             # the previous of source is itself
        visited = [0] * self.N
        width   = [0] * self.N
        heap    = fibh.makefheap()        # the 'q', a min fibonacci heap
        self.init_fib_nodes()
        for fnode in self.fnodes:
            heap.insert(fnode)
        E[src.id] = float('-inf')
        width[src.id] = float('inf')
        heap.decrease_key(self.fnodes[src.id], (float('-inf'), src.id))
        
        while heap.num_nodes:
            fnode = heap.extract_min()
            _, u = fnode.key              # (negative_metric, idx), negative metric not used for now
            if visited[u]:
                continue
            visited[u] = 1
            if u == dst.id:
                path = Path()
                path.create_final_path(u, prev, width)
                path.metric = -E[u]
                return path
            for v in self.adj_list[u]:
                if visited[v]:
                    continue
                path = Path()
                path.create_tmp_path(u, v, prev, width, self.edge2qchannel, self.node_memory)  # W is included in class Path
                metric = self.routing_metric(path)
                negative_metric = -metric
                if negative_metric < E[v]:
                    E[v] = negative_metric
                    prev[v] = u
                    width[v] = path.W
                    heap.decrease_key(self.fnodes[v], (negative_metric, v))


    def update_residual(self, path: Path):
        '''update the resources of the network, including memories at the nodes and number of quantum channels at an edge
        Args:
            path: the path that consumes resources
        '''
        # step 1: update the quantum channels at the edge
        for i in range(path.hop):
            u, v = path.nodes[i], path.nodes[i + 1]
            channels_num = self.edge2qchannel[get_min_max(u, v)].channels_num
            self.edge2qchannel[get_min_max(u, v)].channels_num = channels_num - path.W
            if self.edge2qchannel[get_min_max(u, v)].channels_num < 0:
                raise Exception('channels number cannot be negative, somewhere is wrong. Oops!')
        # step 2: update the memories at the node
        for i, node_id in enumerate(path.nodes):
            if i == 0 or i == len(path.nodes) - 1:       # case 1: node is the first or last node of a path
                self.node_memory[node_id] -= path.W
            else:                                        # case 2: node is in the middle
                self.node_memory[node_id] -= path.W * 2


    def phase_two(self, sd_pairs: List):
        '''the phase two of the Q-CAST (the best of best)
        Args:
            sd_pairs -- a list of (src, dst), where src and dst are qNode
        Return:
            paths: List -- an element is a Path object
        '''
        selected_paths = []
        while True:
            best_path = None
            max_eps = 0
            for src, dst in sd_pairs:
                path = self.dijkstra(src, dst)
                if path.metric > max_eps:
                    max_eps = path.metric
                    best_path = path
            if best_path:
                selected_paths.append(best_path)
                self.update_residual(best_path)
            else:    # no more paths, then break
                break
        return selected_paths


    def super_time_slot(self, path: Path):
        '''A super time slot for each sub-channel
        Args:
            path -- a path object
        Return:
            float
        '''
        time_slots = []
        for k in range(1, path.hop + 1):
            u, v = path.nodes[k - 1], path.nodes[k]
            qchannel = self.edge2qchannel[get_min_max(u, v)]

            # time slot version 1
            node_constraint = min(qchannel.this.remaining_capacity / (2 * self.W), qchannel.other.remaining_capacity / (2 * self.W))
            channel_constraint = 1. / qchannel.tau
            time_slot = 1. / min(node_constraint, channel_constraint)
            
            # time slot version 2
            # node1, node2 = qchannel.this, qchannel.other
            # time_slot = max(node1.generation_time, node2.generation_time) + qchannel.tau

            time_slots.append(time_slot)
        super_time_slot = max(time_slots)
        return super_time_slot


    def paths2rates(self, paths: List[Path], verbose: bool = True):
        '''given a list of paths, compute the entanglement pair rate per second
        Args:
            paths -- a list of paths
        Return:
            List -- a list of entanglement pair per second
        '''
        rates = []
        for path in paths:
            super_time_slot = self.super_time_slot(path)
            rate = path.metric / super_time_slot
            rates.append(rate)
            if verbose:
                print(f'{path} metric = {path.metric:.6f}, super time slot = {super_time_slot:.6f} s, rate = {rate:.6f} ep/s')
        return rates


    def update_ent_time_tree(self, node: tree_node, avr_ent_time: float):
        '''
        Args:
            node -- the tree node who's avr_ent_time needs to be updated
            avr_ent_time -- the average entanglement time for the leaf node, every leaf is the same
        '''
        if node is None or node.data is None:
            return
        if isinstance(node.data, qChannel):
            node.avr_ent_time = avr_ent_time
        self.update_ent_time_tree(node.left, avr_ent_time)
        self.update_ent_time_tree(node.right, avr_ent_time)
        node.left_avr_ent_time = node.right_avr_ent_time = avr_ent_time


    def get_leaf_effective_rate(self, path: Path):
        '''get the effective rate for each **sub-channel** of the selected path
        Args:
            path -- Path object
        Return:
            rate -- float
        '''
        rates = []
        for k in range(1, path.hop + 1):
            u, v = path.nodes[k - 1], path.nodes[k]
            qchannel = self.edge2qchannel[get_min_max(u, v)]
            node_constraint     = min(qchannel.this.remaining_capacity / (2 * self.W), qchannel.other.remaining_capacity / (2 * self.W))
            node_constraint    *= qchannel.this.gen_success_rate * qchannel.other.gen_success_rate * qchannel.channel_success_rate ** 2 * qchannel.optical_bsm_rate
            channel_constraint  = 1. / qchannel.tau
            channel_constraint *= min(qchannel.this.gen_success_rate, qchannel.other.gen_success_rate) * qchannel.channel_success_rate ** 2 * qchannel.optical_bsm_rate
            capacity = min(node_constraint, channel_constraint)
            rates.append(capacity)   # the sub-channel's rate is operating at its full capacity
        min_rate = min(rates)
        return min_rate


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
        '''construct tree
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
                    t = (1.5 * max(tree_node_dict[(src, k)].avr_ent_time, tree_node_dict[(k, dst)].avr_ent_time) +
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
                
        # for layer in tree_layers:
        root = (tree_layers[-1][0][0], tree_layers[-1][0][-1])
        return tree_node_dict[root]

    def path2tree(self, path: Path, decoherence_time: float, update_time: bool = True):
        '''first try balanced tree, if surpass the decoherence time, then try the dp tree
        Args:
            path -- tree object
            decoherence_time -- decoherence time when an entanglement disappears
        Return:
            tree_node
        '''
        _, tree_root = self.path2tree_balanced(path, decoherence_time, 'recursive')
        # if tree_root is None:  # balanced tree failed, then try dp tree
        #     tree_root = self.path2tree_dp(path, decoherence_time)
        return tree_root

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
            raise Exception('iterative not implemented')
        elif method == 'recursive':
            tree_layers = self.build_tree_layers_recursive(path)
        else:
            raise Exception('argument method input error')
        tree_root = self.construct_tree(tree_layers, decoherence_time)
        if tree_root is not None:
            tree_root.avr_ent_time = path.avr_ent_time   # TODO let it be like this for now
        return tree_layers, tree_root

    # def path2tree_dp(self, path: Path, decoherence_time: float):
    #     '''use dp to construct a path to a tree
    #     Args:
    #         path -- a linear path
    #         decoherence_time -- a threshold for decoherence
    #     Return:
    #         tree_node
    #     '''
    #     n = len(path.nodes)
    #     dp = [[tree_node()] * n for _ in range(n)]   # reduce n^3 space complexity to n^2 by using the new_dp
    #     for i in range(n):
    #         u = path.nodes[i]
    #         dp[i][i] = tree_node(data=self.G.V[u], avr_ent_time=0, one_round_ent_time=0)
    #     for i in range(n-1):
    #         u, v = path.nodes[i], path.nodes[i + 1]
    #         qchannel = self.edge2qchannel[get_min_max(u, v)]
    #         dp[i][i + 1] = tree_node(qchannel, avr_ent_time = qchannel.avr_ent_time if qchannel.avr_success_time < decoherence_time else float('inf'),
    #                                  one_round_ent_time = qchannel.avr_success_time if qchannel.avr_success_time < decoherence_time else float('inf'))
    #         dp[i + 1][i] = dp[i][i + 1]
    #     for _ in range(n - 2):
    #         new_dp = [[tree_node()] * n for _ in range(n)]
    #         for i in range(n):
    #             u = path.nodes[i]
    #             new_dp[i][i] = tree_node(data=self.G.V[u], avr_ent_time=0, one_round_ent_time=0)
    #         for src in range(n):  # src id
    #             i = path.nodes[src]
    #             src_node = self.G.V[i]
    #             for dst in range(src + 1, n):  # dst id
    #                 j = path.nodes[dst]
    #                 dst_node = self.G.V[j]
    #                 if src == dst:  # dp(src, src) always zero
    #                     continue
    #                 shortest_src_dst = dp[src][dst]
    #                 for k in range(n):
    #                     if k == src or k == dst:  # one side is zero
    #                         continue
    #                     k_node = self.G.V[path.nodes[k]]  # the node bsm happens
    #                     # average total ent time
    #                     src_to_k_classical = get_classical_time(k_node.loc.distance(src_node.loc))
    #                     k_to_dst_classical = get_classical_time(k_node.loc.distance(dst_node.loc))
    #                     t = (1.5 * max(dp[src][k].avr_ent_time, dp[k][dst].avr_ent_time) + k_node.bsm_time + max(src_to_k_classical, k_to_dst_classical)) / k_node.bsm_success_rate
    #                                # time to generate both left and right subtrees         # time to send BSM res back
    #                     # average one round of entanglement (waiting included)
    #                     t_s = max(src_to_k_classical, k_to_dst_classical) + k_node.bsm_time
    #                     if dp[src][k].avr_ent_time < dp[k][dst].avr_ent_time:
    #                         t_s += max(dp[src][k].one_round_ent_time + dp[k][dst].avr_ent_time - dp[src][k].avr_ent_time, dp[k][dst].one_round_ent_time)
    #                     else:
    #                         t_s += max(dp[src][k].one_round_ent_time, dp[k][dst].one_round_ent_time + dp[src][k].avr_ent_time - dp[k][dst].avr_ent_time)

    #                     if t < shortest_src_dst.avr_ent_time and t_s < decoherence_time:  # found a shorter path
    #                         left  = dp[src][k]
    #                         right = dp[k][dst]
    #                         shortest_src_dst = tree_node(data = k_node, left = left, right = right, avr_ent_time = t, one_round_ent_time = t_s,
    #                                                      classical_time = max(src_to_k_classical, k_to_dst_classical),
    #                                                      left_avr_ent_time = dp[src][k].avr_ent_time, right_avr_ent_time = dp[k][dst].avr_ent_time)
    #                 new_dp[src][dst] = new_dp[dst][src] = shortest_src_dst  # both ways
    #         dp = new_dp
    #     if dp[0][n-1] is not None:
    #         dp[0][n-1].avr_ent_time = path.avr_ent_time   # TODO let it be like this for now
    #     return dp[0][n-1]



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

    sigcomm = Sigcomm(qGraph(qnodes, qchannels))
    src = qnodes[0]
    dst = qnodes[5]
    path = sigcomm.dijkstra(src, dst)
    print(path)


def test2(filename):
    '''testing phase two
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

    sigcomm = Sigcomm(qGraph(qnodes, qchannels))

    sd_pairs = [(qnodes[0], qnodes[5]), (qnodes[1], qnodes[2])]
    # sd_pairs = [(qnodes[0], qnodes[5])]
    # sd_pairs = [(qnodes[1], qnodes[2])]
    paths = sigcomm.phase_two(sd_pairs)
    print('\ninput src-dst pairs are:')
    for i, (src, dst) in enumerate(sd_pairs):
        print(f'{i}, src = {src.id}, dst = {dst.id}')
    rates = sigcomm.paths2rates(paths)
    print('\noutput selected paths are:')
    for i, path in enumerate(paths):
        print(f'{i}, {path}, rate = {rates[i]:.6f}')
    print(f'The total rate is {sum(rates):.4f} entanglement pairs per second')


def test3(filename):
    '''testing dijkstra of sigcomm_w
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

    W = 10
    sigcomm_w = Sigcomm_W(qGraph(qnodes, qchannels), W)
    src = qnodes[0]
    dst = qnodes[5]
    path = sigcomm_w.dijkstra(src, dst)
    sig_rates = sigcomm_w.paths2rates([path])
    path.metric = 1 / sig_rates[0]   # from rate to time
    print(path)


if __name__ == '__main__':
    # toy_example = '../topology/toy1'
    # toy_example = '../topology/toy2-multichannel'
    # test(toy_example)

    # toy_example = '../topology/toy3-multichannel'
    toy_example = '../topology/toy5-multichannel'
    # test2(toy_example)
    test3(toy_example)
