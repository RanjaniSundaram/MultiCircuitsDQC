'''Delft Multi-Commodity Flow LP
'''
import sys
from netsquid.components import qchannel
sys.path.append('..')

import networkx as nx
import matplotlib.pyplot as plt
import math
from typing import List
from ortools.linear_solver import pywraplp  # the Google LP solver
from collections import defaultdict
import copy
from commons.Point import Point
from commons.qNode import qNode
from commons.qChannel import qChannel
from commons.qGraph import qGraph
from commons.tools import get_min_max
from topology.tools import Topology
from commons.tree_node import tree_node
from routing_algorithms.sigcomm import Path

EPSILON = 1e-8

class modified_qGraph:
    '''Network modification
    '''
    def __init__(self, qgraph: qGraph, demands: List, F: float = 0.99):
        '''
        Args:
            qgraph  -- the original graph
            demands -- an element is (scr, dst, fidelity)
        '''
        self.V = []
        self.E = []
        self.v_map = {}
        self.edge2channel = {}
        self.qgraph = qgraph
        self.demands = demands
        self.F = F                       # the lower bound fidelity of each elementary pair
        self.demands_length = [(u, v, self.fidelity2length(d)) for u, v, d in demands]
        self.max_path_len = max([length for _, _, length in self.demands_length])
        self.modify(qgraph, self.max_path_len)
        self.adj_list = defaultdict(list)          # know where the outflow is going
        self.adj_list_reverse = defaultdict(list)  # know where the inflow comes from
        self.init_adj_list_and_reverse()
        self.bsm_success_rate = self.qgraph.V[0].bsm_success_rate

    def fidelity2length(self, fidelity_demand: float):
        '''convert fidelity to length constraint
        Args:
            fidelity_demand -- fidelity
        Return:
            float
        '''
        return math.floor(math.log((4*fidelity_demand-1)/3) / math.log((4*self.F-1)/3))

    def id_old2new(self, id: int, i: int):
        '''maps the old id (original) to a new id (modified)
        Args:
            id -- the id in the original qgraph
            i  -- the i_th copy of this node
        Return:
            int -- the id in the modified new graph
        '''
        node_copy_num = self.max_path_len + 1
        return id * node_copy_num + i

    def id_new2old(self, id: int):
        '''maps the new id (modified) to the old id (original)
        Args:
            id -- the new id
        Return:
            int, int -- the old id, the copy number
        '''
        node_copy_num = self.max_path_len + 1
        return id // node_copy_num, id % node_copy_num

    def modify(self, qgraph: qGraph, max_len: int):
        '''
        Args:
            qgraph -- original graph
            max_len -- the max length of a path
        '''
        node_copy_num = max_len + 1
        for v in qgraph.V:
            for i in range(max_len + 1):
                new_id = self.id_old2new(v.id, i)
                copy = v.clone_with_new_id(new_id)
                self.V.append(copy)
                self.v_map[copy.id] = copy
        for e in qgraph.E:
            u, v = e.this, e.other
            for i in range(node_copy_num - 1):
                u_new_id = self.id_old2new(u.id, i)
                v_new_id = self.id_old2new(v.id, i + 1)
                qchannel = qChannel(self.v_map[u_new_id], self.v_map[v_new_id], e.channels_num, optical_bsm_rate=e.optical_bsm_rate)
                self.E.append(qchannel)
                self.edge2channel[(u_new_id, v_new_id)] = qchannel
            u, v = e.other, e.this
            for i in range(node_copy_num - 1):
                u_new_id = self.id_old2new(u.id, i)
                v_new_id = self.id_old2new(v.id, i + 1)
                qchannel = qChannel(self.v_map[u_new_id], self.v_map[v_new_id], e.channels_num, optical_bsm_rate=e.optical_bsm_rate)
                self.E.append(qchannel)
                self.edge2channel[(u_new_id, v_new_id)] = qchannel

    def init_adj_list_and_reverse(self):
        '''initialize the adjacency list
        '''
        for e in self.E:
            u, v = e.this, e.other
            self.adj_list[u.id].append(v.id)
            self.adj_list_reverse[v.id].append(u.id)

    def get_range(self, axis: str):
        '''get the range of the x or y coordinates
        Args:
            axis -- 'x' or 'y'
        return:
            float, float  -- mix and max value
        '''
        minn, maxx = float('inf'), float('-inf')
        if axis == 'y':
            for v in self.V:
                minn = min(minn, v.loc.y)
                maxx = max(maxx, v.loc.y)
        if axis == 'x':
            for v in self.V:
                minn = min(minn, v.loc.x)
                maxx = max(maxx, v.loc.x)
        return minn, maxx

    def visualize(self, filename: str = ''):
        G = nx.Graph()
        nodes = {}
        edges = []
        edges2width = {}
        widths = []
        minn, maxx = self.get_range('y')
        step = (maxx - minn) / 10
        for node in self.V:
            _, copy_num = self.id_new2old(node.id)
            nodes[node.id] = node.loc.x, node.loc.y + copy_num * step
        for qchannel in self.E:
            u, v = qchannel.this.id, qchannel.other.id
            edges.append((u, v))
            edges2width[get_min_max(u, v)] = qchannel.channels_num
        G.add_edges_from(edges)
        for u, v in G.edges:
            widths.append(edges2width[get_min_max(u, v)])
        fig, ax = plt.subplots(figsize=(10,10))
        nx.draw_networkx(G, nodes, width=widths)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()


class LP:
    '''Linear programming
    '''
    def __init__(self, modified_qgraph: modified_qGraph):
        self.modified_qgraph = modified_qgraph
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
        self.g = None   # the variable,the g_ij(...) in the paper
        self.constraint_edge = None
        self.constraint_node = None
        self.constraint_conservation = None
        self.objective = None
        self.k = len(self.modified_qgraph.demands_length)   # number of demands
        self.l_max = self.modified_qgraph.max_path_len


    def set_varialbles(self):
        '''set up the variables, 3 dimension list
        '''
        self.g = [[]] * self.k
        for i in range(self.k):
            li = self.modified_qgraph.demands_length[i][2]
            self.g[i] = [[]] * li
            for j in range(li):
                self.g[i][j] = {}   # for the i,jth demand, create a variable for all edges in the modified graph
                for qchannel in self.modified_qgraph.E:
                    u, v = qchannel.this.id, qchannel.other.id
                    # max_channel_capacity = qchannel.max_channel_capacity
                    max_channel_capacity = qchannel.max_channel_capacity_delftlp()
                    self.g[i][j][(u, v)] = self.solver.NumVar(0, max_channel_capacity, f'i,j={i},{j}-edge({u},{v})')  # including equation (7) in the paper


    def set_contraints(self):
        '''set up the contraints. 
        NOTE the current constraints are sticking to the original paper, but it has one issue when fitting to our model (node capacity)
        '''
        # capacity constraint for (original) edges, equation (8) in the paper
        self.constraint_edge = {}
        for qchannel in self.modified_qgraph.qgraph.E:
            u, v = qchannel.this.id, qchannel.other.id       # the id in the original qgraph
            # max_channel_capacity = qchannel.max_channel_capacity
            max_channel_capacity = qchannel.max_channel_capacity_delftlp()
            self.constraint_edge[(u, v)] = self.solver.Constraint(0, max_channel_capacity, f'edge({u},{v})')
            for i in range(self.k):
                li = self.modified_qgraph.demands_length[i][2]
                for j in range(li):
                    for t in range(self.l_max):
                        u_new_id = self.modified_qgraph.id_old2new(u, t)
                        v_new_id = self.modified_qgraph.id_old2new(v, t + 1)
                        self.constraint_edge[(u, v)].SetCoefficient(self.g[i][j][(u_new_id, v_new_id)], 1)
            u, v = qchannel.other.id, qchannel.this.id        # undirected graph
            self.constraint_edge[(u, v)] = self.solver.Constraint(0, max_channel_capacity)
            for i in range(self.k):
                li = self.modified_qgraph.demands_length[i][2]
                for j in range(li):
                    for t in range(self.l_max):
                        u_new_id = self.modified_qgraph.id_old2new(u, t)
                        v_new_id = self.modified_qgraph.id_old2new(v, t + 1)
                        self.constraint_edge[(u, v)].SetCoefficient(self.g[i][j][(u_new_id, v_new_id)], 1)

        # capacity constraint for (original) nodes, let's say it is the new equation (10)
        self.constraint_node = {}
        for qnode in self.modified_qgraph.qgraph.V:
            v = qnode.id
            max_node_capacity = qnode.max_capacity
            self.constraint_node[v] = self.solver.Constraint(0, max_node_capacity, f'node-{v}')
            for i in range(self.k):
                li = self.modified_qgraph.demands_length[i][2]
                for j in range(li):
                    for t in range(self.l_max + 1):
                        v_new = self.modified_qgraph.id_old2new(v, t)
                        for u_new in self.modified_qgraph.adj_list_reverse[v_new]:  # inflow
                            qchannel = self.modified_qgraph.edge2channel[(u_new, v_new)]
                            # coefficient = 1. / qchannel.link_success_rate
                            coefficient = 1
                            self.constraint_node[v].SetCoefficient(self.g[i][j][(u_new, v_new)], coefficient)
                        for w_new in self.modified_qgraph.adj_list[v_new]:          # outflow
                            qchannel = self.modified_qgraph.edge2channel[(v_new, w_new)]
                            # coefficient = 1. / qchannel.link_success_rate
                            coefficient = 1
                            self.constraint_node[v].SetCoefficient(self.g[i][j][(v_new, w_new)], coefficient)

        # conservation property, for each demand (bug was here, neglecting for each demand) and for each node, inflow = outflow, equation (9) in the paper
        self.constraint_conservation = [[]] * self.k
        for i in range(self.k):
            src, dst, li = self.modified_qgraph.demands_length[i]
            self.constraint_conservation[i] = [[]] * li
            for j in range(li):
                len_j = j + 1
                src_new = self.modified_qgraph.id_old2new(src.id, 0)
                dst_new = self.modified_qgraph.id_old2new(dst.id, len_j) # bug was here, the bug was using self.l_max instead of len_j. specifing the correct destination is important
                self.constraint_conservation[i][j] = {}
                for node_v in self.modified_qgraph.V:
                    v = node_v.id
                    if v == src_new or v == dst_new:
                        continue
                    self.constraint_conservation[i][j][v] = self.solver.Constraint(0, 0, f'node-{v}')
                    for u in self.modified_qgraph.adj_list_reverse[v]:
                        if (u, v) in self.modified_qgraph.edge2channel:
                            self.constraint_conservation[i][j][v].SetCoefficient(self.g[i][j][(u, v)], 1)    # inflow
                    for w in self.modified_qgraph.adj_list[v]:
                        if (v, w) in self.modified_qgraph.edge2channel:
                            self.constraint_conservation[i][j][v].SetCoefficient(self.g[i][j][(v, w)], -1)   # outflow


    def set_objective(self):
        '''set up the objective function
        '''
        self.objective = self.solver.Objective()
        for i in range(self.k):
            src, _, li = self.modified_qgraph.demands_length[i]
            for j in range(li):
                bsm_prob = self.modified_qgraph.bsm_success_rate ** j
                s_i_0 = self.modified_qgraph.id_old2new(src.id, 0)
                for v_1 in self.modified_qgraph.adj_list[s_i_0]:
                    self.objective.SetCoefficient(self.g[i][j][(s_i_0, v_1)], bsm_prob)
        self.objective.SetMaximization()
        return self.objective

    def print_results(self):
        print('solution')
        print(f'objective value       = {self.objective.Value()}')
        print(f'number of constraints = {self.solver.NumConstraints()}')
        print(f'number of variables   = {self.solver.NumVariables()}')
        for i in range(self.k):
            li = self.modified_qgraph.demands_length[i][2]
            for j in range(li):
                for qchannel in self.modified_qgraph.E:
                    u, v = qchannel.this.id, qchannel.other.id
                    if self.g[i][j][(u, v)].solution_value() > 0:
                        u_old, u_copy = self.modified_qgraph.id_new2old(u)
                        v_old, v_copy = self.modified_qgraph.id_new2old(v)
                        print(f'demand {i}, path length {j + 1}, edge {u_old}-{v_old}, flow value = {self.g[i][j][(u, v)].solution_value()}')

    def solve(self, print=True):
        '''solve it
        '''
        self.set_varialbles()
        self.set_contraints()
        self.set_objective()
        self.solver.Solve()
        if print:
            self.print_results()

    def extract_path(self):
        '''extract the path given the LP results
        '''
        def f_sum(F):
            summ = 0
            for val in F.values():
                summ += val
            return summ

        def dfs(adj_list, s, d, stack, find):
            if s == d:
                stack.append(s)
                find[0] = True
                return
            for nxt in adj_list[s]:
                stack.append(s)            
                dfs(adj_list, nxt, d, stack, find)
                if find[0] == True:
                    return
                stack.pop()

        def copy_gij(gij):
            '''
            Args:
                gij -- a dict, (int, int) --> ortools.linear_solver.pywarplp.Variable
            Return:
                F -- a dict,   (int, int) --> float
            '''
            F = {}
            for key, var in gij.items():
                F[key] = var.solution_value()
            return F

        def analytical_rate(path, bottleneck):
            '''
            Args:
                path  -- Path -- the path that needs to compute the analytical rate
                stack -- list
            Return:
                float
            '''
            q_prob = self.modified_qgraph.bsm_success_rate ** (path.hop - 1)  # the bsm swapping probability
            p_prob = 1                                                        # the link success probability
            for k in range(path.hop):
                u, v = path.nodes[k], path.nodes[k + 1]
                qchannel = self.modified_qgraph.edge2channel[(u, v)]
                p_prob *= qchannel.link_success_rate
            return bottleneck * q_prob * p_prob

        # step 1: build a temporary graph with valid edges
        adj_list = defaultdict(list)
        for i in range(self.k):
            li = self.modified_qgraph.demands_length[i][2]
            for j in range(li):
                for qchannel in self.modified_qgraph.E:
                    u, v = qchannel.this.id, qchannel.other.id
                    if self.g[i][j][(u, v)].solution_value() > 0:
                        adj_list[u].append(v)
        # step 2: get extract the paths, the find path part is dfs
        paths = []
        for i in range(self.k):
            src, dst, li = self.modified_qgraph.demands_length[i]
            for j in range(li):
                F = copy_gij(self.g[i][j])
                while f_sum(F) > EPSILON:
                    path_len = j + 1
                    s = self.modified_qgraph.id_old2new(src.id, 0)
                    d = self.modified_qgraph.id_old2new(dst.id, path_len)
                    stack = []
                    find = [False]
                    dfs(adj_list, s, d, stack, find)
                    if find[0]:
                        path = Path(w=1)
                        path.nodes = stack
                        path.hop   = len(stack) - 1
                        rates = [self.g[i][j][(stack[k], stack[k+1])].solution_value() for k in range(path.hop)]
                        bottleneck  = min(rates)
                        # path.metric = bottleneck * self.modified_qgraph.bsm_success_rate ** (path.hop - 1)
                        path.metric = analytical_rate(path, bottleneck)
                        for k in range(path.hop):
                            u, v = stack[k], stack[k + 1]
                            F[u, v] -= bottleneck
                            if F[u, v] < EPSILON:
                                adj_list[u].remove(v)
                        paths.append(path)
        # step 3: new node id to old id
        paths2 = []
        for path in paths:
            tmp_path = copy.deepcopy(path)
            tmp_path.nodes = []
            for new in path.nodes:
                old, _ = self.modified_qgraph.id_new2old(new)
                tmp_path.nodes.append(old)
            paths2.append(tmp_path)
        return paths2


    def get_leaf_effective_rate(self, path):
        '''return the tree's leaf node effective rate, given the path
        Args:
            path -- Path
        Return:
            float
        '''
        # step 1: compute the bottleneck, the reverse process of self.extract_path.analytical_rate()
        rate = path.metric
        q_prob = self.modified_qgraph.bsm_success_rate ** (path.hop - 1)  # the bsm swapping probability
        p_prob = 1
        min_link_prob = 1                                                 # the link success probability
        for k in range(path.hop):
            u, v = path.nodes[k], path.nodes[k + 1]
            qchannel = self.modified_qgraph.qgraph.get_edge(u, v)
            p_prob *= qchannel.link_success_rate
            min_link_prob = min(min_link_prob, qchannel.link_success_rate)
        bottleneck_capacity = rate / (q_prob * p_prob)   # you don't know if this bottleneck comes from node or edge? Will there be issue here?
        bottleneck_flow = bottleneck_capacity * min_link_prob
        return bottleneck_flow

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


def test(filename):
    '''
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

    qgraph = qGraph(qnodes, qchannels)
    sd_pairs = [(qnodes[0], qnodes[5]), (qnodes[1], qnodes[2])]
    fidelities = [0.93, 0.96]
    F = 0.98
    demands = [(sd[0], sd[1], f) for sd, f in zip(sd_pairs, fidelities)]
    modify_qgraph = modified_qGraph(qgraph, demands, F=F)
    for u, v, length in modify_qgraph.demands_length:
        print(u.id, v.id, length)
    modify_qgraph.visualize(filename=filename + '-modified')

    lp = LP(modify_qgraph)
    lp.solve()
    paths = lp.extract_path()
    for path in paths:
        print(path)


if __name__ == '__main__':
    toy_example = '../topology/toy5-multichannel'
    test(toy_example)
