import math
import pickle
import time
import traceback
from typing import List

from networkx.classes import graph
from commons.qChannel import qChannel
from commons.qNode import qNode, BSM_SUCCESS_RATE
from commons.qGraph import qGraph
from commons.Point import Point
from random import randint, seed, sample, random, shuffle, choices

from predistributed_algorithms.greedy_predist import greedy_predist
from predistributed_algorithms.multi_pair_greedy_predist import multi_pair_greedy_predist
from predistributed_algorithms.cluster_predist import cluster_predist
from routing_algorithms.max_flow import max_flow
from routing_algorithms.dp_shortest_path import dp_shortest_path as dp
from routing_algorithms.sigcomm import Sigcomm_W
from routing_algorithms.dp_alternate import DP_Alt
from routing_algorithms.naive_shortest_path import hop_shortest_path as naive_dp
from networkx.generators.geometric import waxman_graph
from networkx import Graph
from routing_algorithms.delft_lp import modified_qGraph, LP
import datetime
from routing_algorithms.optimal_linear_programming import optimal_linear_programming
from protocols.network_protocol import TreeProtocol, FusionTreeProtocol, FusionRetainOnlyTreeProtocol, \
    CentralGHZTreeProtocol
from protocols.predistribution.network_protocol import TreeProtocol as PredistProtocol
from protocols.sigcomm_network_protocol import TreeProtocol as SigcommTreeProtocol
from commons.tree_node import tree_node
import os
import argparse
import time
from routing_algorithms.optimal_balanced_tree import optimal_balanced_tree
from routing_algorithms.ghz_fusion_retain import ghz_fusion_retain
from routing_algorithms.ghz_steiner_tree import ghz_steiner_tree, EQUAL_EDGES_PARTITIONING, LATENCY_PARTITIONING
from routing_algorithms.ghz_naive import ghz_star_multi_pair
from routing_algorithms.ghz_star_expansion import ghz_star_expansion


def create_toy_graph(area_side: int, cell_size: int, number_nodes: int, min_memory_size: int, max_memory_size: int,
                     min_num_edge: int, max_num_edge: int,
                     min_channel_num: int, max_channel_num: int) -> qGraph:
    """
    :param area_side: side size of a square area
    :param cell_size: size of each cell (square)
    :param number_nodes: number of nodes (qNodes) in the network
    :param min_memory_size: minimum number of memories for each node
    :param max_memory_size: maximum number of memories for each node
    :param min_num_edge: minimum number of edges for each node
    :param max_num_edge: maximum number of edges per each node
    :param min_channel_num: minimum number of channels for each edge
    :param max_channel_num: maximum number of channels for each edge
    :return:
    """
    points = [Point(x, y) for x, y in zip([7700, 9800, 7400, 3900, 8500, 9100, 4100, 2600, 4700, 6900],
                                          [8200, 5600, 2400, 2400, 6600, 5300, 7000, 9900, 8100, 8900])]  # for test
    num_channels = [5, 3, 1, 4, 3, 5, 4, 1, 2, 3, 3, 3, 1, 1]
    num_edges_rnd = [2, 2, 2, 3, 3, 3, 4, 2, 4, 3]  # for test
    V = [qNode(randint(min_memory_size, max_memory_size), point) for point in points]
    E = []
    num_edges = [0] * len(V)  # number of edges for each node
    edges_set = set()  # a set to avoid parallel edges with key "u-v"
    for i, v in enumerate(V):
        num_v_edges = num_edges_rnd[i] - num_edges[i]  # for test
        distance_to_v = [(v.loc.distance(V[k].loc), k) for k in range(i)] + \
                        [(v.loc.distance(V[k].loc), k) for k in range(i + 1, len(V))]
        distance_to_v.sort(key=lambda x: x[0])  # sort based on distances
        neighbor_idx, edge_num = 0, 0
        while edge_num < num_v_edges and neighbor_idx < len(distance_to_v):
            k_id = distance_to_v[neighbor_idx][1]
            if "{u}-{v}".format(u=i, v=k_id) in edges_set:
                neighbor_idx += 1
                continue
            E.append(qChannel(v, V[k_id], num_channels.pop()))  # v -> k                                      # for test
            num_edges[k_id] += 1
            edge_num += 1
            neighbor_idx += 1
            edges_set.add("{u}-{v}".format(u=i, v=k_id))
            edges_set.add("{u}-{v}".format(u=k_id, v=i))
        num_edges[i] += num_v_edges
    return qGraph(V=V, E=E)


def create_random_graph(area_side: int, cell_size: int, number_nodes: int, min_memory_size: int, max_memory_size: int,
                        min_node_degree: int, max_node_degree: int,
                        min_channel_num: int, max_channel_num: int) -> qGraph:
    """
    :param area_side: side size of a square area
    :param cell_size: size of each cell (square)
    :param number_nodes: number of nodes (qNodes) in the network
    :param min_memory_size: minimum number of memories for each node
    :param max_memory_size: maximum number of memories for each node
    :param min_node_degree: minimum number of edges for each node
    :param max_node_degree: maximum number of edges per each node
    :param min_channel_num: minimum number of channels for each edge
    :param max_channel_num: maximum number of channels for each edge
    :return:
    """
    points = [Point(randint(0, area_side - 1) * cell_size, randint(0, area_side - 1) * cell_size)
              for _ in range(number_nodes)]
    V = [qNode(randint(min_memory_size, max_memory_size), point) for point in points]
    E = []
    num_edges = [0] * len(V)  # number of edges for each node
    edges_set = set()  # a set to avoid parallel edges with key "u-v"
    for i, v in enumerate(V):
        num_v_edges = randint(min_node_degree, max_node_degree) - num_edges[i]  # number of edges to be created
        distance_to_v = [(v.loc.distance(V[k].loc), k) for k in range(i)] + \
                        [(v.loc.distance(V[k].loc), k) for k in range(i + 1, len(V))]
        distance_to_v.sort(key=lambda x: x[0])  # sort based on distances
        neighbor_idx, edge_num = 0, 0
        while edge_num < num_v_edges and neighbor_idx < len(distance_to_v):
            k_id = distance_to_v[neighbor_idx][1]
            if "{u}-{v}".format(u=i, v=k_id) in edges_set:
                neighbor_idx += 1
                continue
            E.append(qChannel(v, V[k_id], randint(min_channel_num, max_channel_num)))  # v -> k
            num_edges[k_id] += 1
            edge_num += 1
            neighbor_idx += 1
            edges_set.add("{u}-{v}".format(u=i, v=k_id))
            edges_set.add("{u}-{v}".format(u=k_id, v=i))
        num_edges[i] += num_v_edges
    return qGraph(V=V, E=E)


def create_random_waxman(area_side: int, cell_size: int, number_nodes: int, min_memory_size: int, max_memory_size: int,
                         min_channel_num: int, max_channel_num: int, atomic_bsm_success_rate: float,
                         edge_density: float, max_edge_size: float = 10e3, fusion_success_rate: float = -1,
                         optical_bsm_success_rate: float = -1) -> qGraph:
    dist = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
    max_number_edges = (number_nodes - 1) * number_nodes / 2
    alpha_min, alpha_max, beta_min, beta_max = 0, 1, 0, 1
    while alpha_max - alpha_min > 0.02 and beta_max - beta_min > 0.02:
        alpha = (alpha_min + alpha_max) / 2
        beta = (beta_min + beta_max) / 2
        graphx = waxman_graph(number_nodes, domain=(0, 0, area_side * cell_size, area_side * cell_size), metric=dist,
                              beta=beta, alpha=alpha, L=max_edge_size)
        if (edge_density - 0.002) * max_number_edges <= len(graphx.edges()) <= (
                edge_density + 0.002) * max_number_edges:
            break
        elif len(graphx.edges()) > (edge_density + 0.002) * max_number_edges:
            alpha_max = alpha
            beta_max = beta
        else:
            alpha_min = alpha
            beta_min = beta
    if atomic_bsm_success_rate is not None:
        V = [qNode(randint(min_memory_size, max_memory_size),
                   Point(int(graphx.nodes[i]["pos"][0]), int(graphx.nodes[i]["pos"][1])),
                   bsm_success_rate=atomic_bsm_success_rate, fusion_success_rate=fusion_success_rate)
             for i in range(number_nodes)]
    else:
        V = [qNode(randint(min_memory_size, max_memory_size),
                   Point(int(graphx.nodes[i]["pos"][0]), int(graphx.nodes[i]["pos"][1]))) for i in range(number_nodes)]
    E = []
    # if optical_bsm_rate is not None:
    #     for u, v in graphx.edges():
    #         E.append(qChannel(V[u], V[v], randint(min_channel_num, max_channel_num),
    #                           optical_bsm_rate=optical_bsm_rate))
    # else:
    for u, v in graphx.edges():
        E.append(qChannel(V[u], V[v], randint(min_channel_num, max_channel_num),
                          optical_bsm_rate=V[0].bsm_success_rate / 2 if optical_bsm_success_rate == -1 else
                          optical_bsm_success_rate))
    return qGraph(V=V, E=E)


def create_random_waxman2(area_side: int, cell_size: int, number_nodes: int, min_memory_size: int, max_memory_size: int,
                          min_channel_num: int, max_channel_num: int, atomic_bsm_success_rate: float,
                          edge_density: float) -> qGraph:
    dist = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
    alpha_min, alpha_max, beta_min, beta_max = 0, 1, 0, 1
    while alpha_max - alpha_min > 0.02 and beta_max - beta_min > 0.02:
        alpha = (alpha_min + alpha_max) / 2
        beta = (beta_min + beta_max) / 2
        graphx = waxman_graph(number_nodes, domain=(0, 0, area_side * cell_size, area_side * cell_size), metric=dist,
                              beta=beta, alpha=alpha, L=10e3)
        if (edge_density - 0.002) * number_nodes ** 2 <= len(graphx.edges()) <= (
                edge_density + 0.002) * number_nodes ** 2:
            break
        elif len(graphx.edges()) > (edge_density - 0.002) * number_nodes ** 2:
            alpha_max = alpha
            beta_min = beta
        else:
            alpha_min = alpha
            beta_max = beta
    if atomic_bsm_success_rate is not None:
        V = [qNode(randint(min_memory_size, max_memory_size),
                   Point(int(graphx.nodes[i]["pos"][0]), int(graphx.nodes[i]["pos"][1])),
                   bsm_success_rate=atomic_bsm_success_rate) for i in range(number_nodes)]
    else:
        V = [qNode(randint(min_memory_size, max_memory_size),
                   Point(int(graphx.nodes[i]["pos"][0]), int(graphx.nodes[i]["pos"][1]))) for i in range(number_nodes)]
    E = []
    # if optical_bsm_rate is not None:
    #     for u, v in graphx.edges():
    #         E.append(qChannel(V[u], V[v], randint(min_channel_num, max_channel_num),
    #                           optical_bsm_rate=optical_bsm_rate))
    # else:
    for u, v in graphx.edges():
        E.append(qChannel(V[u], V[v], randint(min_channel_num, max_channel_num),
                          optical_bsm_rate=V[0].bsm_success_rate / 2))
    return qGraph(V=V, E=E)


def create_line_graph(number_of_nodes: int, distance_between_nodes: float, min_memory_size: int, max_memory_size: int,
                      min_channel_num: int, max_channel_num: int, atomic_bsm_success_rate: float = None,
                      optical_bsm_rate: float = None):
    if atomic_bsm_success_rate is not None:
        V = [qNode(randint(min_memory_size, max_memory_size), loc=Point(x * distance_between_nodes, 0),
                   bsm_success_rate=atomic_bsm_success_rate)
             for x in range(number_of_nodes)]
    else:
        V = [qNode(randint(min_memory_size, max_memory_size), loc=Point(x * distance_between_nodes, 0))
             for x in range(number_of_nodes)]
    if optical_bsm_rate is not None:
        E = [qChannel(V[i], V[i + 1], randint(min_channel_num, max_channel_num), optical_bsm_rate=optical_bsm_rate)
             for i in range(len(V) - 1)]
    else:
        E = [qChannel(V[i], V[i + 1], randint(min_channel_num, max_channel_num))
             for i in range(len(V) - 1)]
    return qGraph(V=V, E=E)


def create_line_graph2(total_length: float, min_link_Length: float, max_link_length: float,
                       min_memory_size: int = 10, max_memory_size: int = 10,
                       min_channel_num: int = 1, max_channel_num: int = 1,
                       atomic_bsm_success_rate: float = 0.4,
                       optical_bsm_rate: float = 0.2):
    current_length = 0
    V = [qNode(randint(min_memory_size, max_memory_size), loc=Point(0, 0), bsm_success_rate=atomic_bsm_success_rate)]
    E = []
    while current_length < total_length * 1e3 and (total_length * 1e3 - current_length) > min_link_Length * 1e3 / 2:
        vleft = V[-1]
        link_length = int((min_link_Length + random() * (max_link_length - min_link_Length)) * 1e3)  # km
        vright = qNode(randint(min_memory_size, max_memory_size),
                       loc=Point(vleft.loc.x + link_length, 0), bsm_success_rate=atomic_bsm_success_rate)
        edge = qChannel(vleft, vright, channels_num=randint(min_channel_num, max_channel_num),
                        optical_bsm_rate=optical_bsm_rate)
        V.append(vright)
        E.append(edge)
        current_length += link_length
    print(f"Line Graph was created with edges between: [{min_link_Length}-{max_link_length}] km,"
          f" with total length of {current_length}")
    return qGraph(V=V, E=E)


def sigcomm_graph(graph: qGraph, sigcom_w: int, metric: str = "min"):
    """Converting a general graph to a homogeneous graph where channels' length&width, nodes memory are the same
    based on metric which is either minimum over all values or maximum"""
    links_length = [e.length for e in graph.E]
    links_subchannels = [e.channels_num for e in graph.E]
    length = sum(links_length) / len(links_length)  # min(links_length) if metric == "min" else max(links_length)
    subchannels = sigcom_w  # min(links_subchannels) if metric == "min" else max(links_subchannels)
    new_v = [qNode(memory=2 * 10, loc=v.loc, gen_success_rate=v.gen_success_rate,
                   generation_time=v.generation_time, bsm_time=v.bsm_time, bsm_success_rate=v.bsm_success_rate,
                   node_id=v.id) for v in graph.V]
    new_e = [qChannel(this=e.this, other=e.other, channels_num=subchannels,
                      optical_bsm_rate=e.optical_bsm_rate, optical_bsm_time=e.optical_bsm_time)
             for e in graph.E]
    return qGraph(V=new_v, E=new_e), length


def connected_graph(graph: qGraph, min_channel_num, max_channel_num):
    """Check if a graph is connected. IF not, make it a connected graph."""

    def dfs(src: int, adj_list, visited: list, components):
        components[-1].add(src)
        visited[src] = True
        for dst in adj_list[src]:
            if not visited[dst]:
                dfs(dst, adj_list, visited, components)

    id_to_pos = {node.id: idx for idx, node in enumerate(graph.V)}
    adj_list = [[] for _ in range(len(graph.V))]
    for e in graph.E:
        adj_list[id_to_pos[e.this.id]].append(id_to_pos[e.other.id])
        adj_list[id_to_pos[e.other.id]].append(id_to_pos[e.this.id])
    visited = [False] * len(graph.V)
    components = []

    for i in range(len(graph.V)):
        if not visited[i]:
            components.append(set())
            dfs(i, adj_list, visited, components)

    while len(components) > 1:
        min_dist = float('inf')
        selected_components = []
        selected_nodes = []
        for i in range(len(components) - 1):
            for j in range(i + 1, len(components)):
                for node_i_idx in components[i]:
                    node_i = graph.V[node_i_idx]
                    for node_j_idx in components[j]:
                        node_j = graph.V[node_j_idx]
                        if node_i.loc.distance(node_j.loc) < min_dist:
                            min_dist = node_i.loc.distance(node_j.loc)
                            selected_nodes = [node_i, node_j]
                            selected_components = [i, j]
        graph.add_edge(qChannel(this=selected_nodes[0], other=selected_nodes[1],
                                channels_num=randint(min_channel_num, max_channel_num)))
        merged_component = components[selected_components[0]].union(components[selected_components[1]])
        components = components[:selected_components[0]] + components[selected_components[0] + 1:selected_components[1]] \
                     + components[selected_components[1] + 1:]
        components.append(merged_component)


def weighted_avg(count_avg: List):
    num = sum([x[0] for x in count_avg])
    if num == 0:
        return 0, 0
    return num, sum([x[0] * x[1] for x in count_avg]) / num


def main():
    seed(10)
    # **************** init values ************
    area_side = 10  # a square side of side x side cells
    cell_size = 10000  # side value of each square side (m)
    min_memory_size, max_memory_size = 15, 20
    min_node_degree, max_node_degree = 2, 4  # number of edges from one node to its neighbors
    min_channel_num, max_channel_num = 1, 1  # number of parallel channels a connection would have
    sigcom_w = 5
    default_number_of_src_dst_pair = 3
    default_decoherence_time = 5  # seconds
    default_atomic_bsm_rate, default_optical_bsm_rate = 0.4, 0.2
    default_edge_density = 0.06
    default_num_nodes = 100
    default_line_total_length = 1000  # kms
    default_line_link_length = (30, 35)  # kms
    default_distance_range = [30, 40] if default_number_of_src_dst_pair == 1 else [10,
                                                                                   70]  # minimum distance between src-dst in km
    IS_TELEPORTING = False  # defines if we send information from Alice to Bob after tree construction in protocols

    simulation_duration = 30  # seconds
    number_of_repeat = 5
    DP_OPT = False
    NON_THROTTLING = False

    date_now = datetime.datetime.now()

    directory = "results/" + date_now.strftime('%Y_%m_%d')

    if not os.path.exists(directory):
        os.makedirs(directory)
    saving_file_name = directory + "/results" + date_now.strftime('_%H_%M_%S')

    ################################################################################

    # example: python main_quantum_routing_args.py -dp -dpa -para num_nodes

    parser = argparse.ArgumentParser(description='Quantum network routing')

    # methods
    parser.add_argument('-dp', '--dynamic_programming', action='store_true', help='If given, run the DP')
    parser.add_argument('-dpa', '--dynamic_programming_alternate', action='store_true', help='If given, run the DP_ALT')
    parser.add_argument('-n', '--naive', action='store_true', help='If given, run the NAIVE_SHORTEST_PATH')
    parser.add_argument('-dpi', '--dynamic_programming_iterative', action='store_true',
                        help='If given, run the DP Iterative')
    parser.add_argument('-ni', '--naive_iterative', action='store_true', help='If given, run the NAIVE Iterative')
    parser.add_argument('-lp', '--linear_programming', action='store_true', help='If given, run the LP')
    parser.add_argument('-lpa', '--linear_programming_alternate', action='store_true', help='If given, run the LP_ALT')
    parser.add_argument('-sigs', '--sigcomm_single', action='store_true', help='If given, run the Singcomm single')
    parser.add_argument('-sigm', '--sigcomm_multiple', action='store_true', help='If given, run the Singcomm multiple')
    parser.add_argument('-dft', '--delft_lp', action='store_true', help='If given, run the Delft LP')
    parser.add_argument('-line', '--line_graph', action='store_true', help='If given, create line graph')

    # parameter
    parser.add_argument('-para', '--parameter', type=str, nargs=1, default=['atomic_bsm'])

    args = parser.parse_args()

    # ********************** changing parameters (x-values) & algorithms used *******************
    # parameter : {"atomic_bsm", "optical_bsm", "decoherence_time", "num_nodes", "edge_density", "distance", "src_dst_pair"}
    IS_DP = args.dynamic_programming  # single   path
    IS_DP_ALT = args.dynamic_programming_alternate  # single   path
    IS_NAIVE = args.naive  # single   path
    IS_DP_ITER = args.dynamic_programming_iterative  # multiple path
    IS_NAIVE_ITER = args.naive_iterative  # multiple paths
    IS_LP = args.linear_programming  # multiple path
    IS_LP_ALT = args.linear_programming_alternate  # multiple path
    IS_SIGCOMM_SINGLE = args.sigcomm_single  # single   path
    IS_SIGCOMM_MULTI = args.sigcomm_multiple  # multiple path
    IS_DELFT_LP = args.delft_lp  # multiple path
    IS_LINE = args.line_graph

    parameter = args.parameter[0]
    if IS_LINE:
        number_of_repeat = 1
        simulation_duration = 60
        if parameter == "distance":
            values = [100, 300, 500, 700, 900, 1000]
        elif parameter == "link":
            values = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
        elif parameter == "decoherence_time":
            values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60]
        else:
            raise Exception(f'Oops! wrong parameter input: {parameter} for line_graph(only distance and link)')
    else:
        if parameter == 'atomic_bsm':
            values = [0.2, 0.3, 0.4, 0.5, 0.6]
        elif parameter == 'num_nodes':
            values = [25, 50, 75, 200, 300, 400, 500]
        elif parameter == 'edge_density':
            values = [0.02, 0.04, 0.08, 0.1]
        elif parameter == 'distance':
            values = [(10, 20), (20, 30), (40, 50), (50, 60), (60, 70)]
        elif parameter == 'src_dst_pair':
            values = [1, 2, 4, 5]
        else:
            raise Exception(f'Oops! wrong parameter input: {parameter}')
    ################################################################################

    dp_res, dp_protocol_res, dp_res_avg, dp_protocol_res_avg, dp_fidelity, dp_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    dp_alt_res, dp_alt_protocol_res, dp_alt_res_avg, dp_alt_protocol_res_avg, dp_alt_fidelity, dp_alt_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    naive_res, naive_protocol_res, naive_res_avg, naive_protocol_res_avg, naive_fidelity, naive_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    dp_iter_res, dp_iter_protocol_res, dp_iter_res_avg, dp_iter_protocol_res_avg, dp_iter_fidelity, dp_iter_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    naive_iter_res, naive_iter_protocol_res, naive_iter_res_avg, naive_iter_protocol_res_avg, naive_iter_fidelity, \
    naive_iter_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    lp_res, lp_protocol_res, lp_res_avg, lp_protocol_res_avg, lp_fidelity, lp_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    lp_alt_res, lp_alt_protocol_res, lp_alt_res_avg, lp_alt_protocol_res_avg, lp_alt_fidelity, lp_alt_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    sigcomm_single_res, sigcomm_single_protocol_res, sigcomm_single_res_avg, sigcomm_single_protocol_res_avg, \
    sigcomm_single_fidelity, sigcomm_single_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    sigcomm_multiple_res, sigcomm_multiple_protocol_res, sigcomm_multiple_res_avg, sigcomm_multiple_protocol_res_avg, \
    sigcomm_multi_fidelity, sigcomm_multi_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    delft_lp_res, delft_lp_protocol_res, delft_lp_res_avg, delft_lp_protocol_res_avg, delft_lp_fidelity, \
    delft_lp_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    plotting_date = {'x': (parameter, values)}

    maximum_distance_reached = [[] for _ in range(len(values))]

    with open(saving_file_name + ".txt", 'a') as file:
        file.write(parameter + ": " + str(values))
        if IS_LINE:
            file.write(f"\n******** LINE GRAPH *********\n")
    for value_idx, value in enumerate(values):
        teleportation_bsm_rate = value if parameter == "atomic_bsm" else BSM_SUCCESS_RATE
        print(f"\n\nTESTING {parameter}: {value}")
        with open(saving_file_name + ".txt", 'a') as file:
            file.write(f"\nTESTING {parameter}: {value}")
        # ******************* creating the network *****************
        qNode._stat_id = 0  # need to reset qNode._stat_id for every new experiment
        seed(value_idx)
        if not IS_LINE:
            graph0 = create_random_waxman(area_side=area_side, cell_size=cell_size,
                                          number_nodes=value if parameter == "num_nodes" else default_num_nodes,
                                          min_memory_size=min_memory_size, max_memory_size=max_memory_size,
                                          min_channel_num=min_channel_num, max_channel_num=max_channel_num,
                                          atomic_bsm_success_rate=value if parameter == "atomic_bsm"
                                          else default_atomic_bsm_rate,
                                          edge_density=value if parameter == "edge_density" else default_edge_density)
            connected_graph(graph0, min_channel_num=min_channel_num, max_channel_num=max_channel_num)
        else:
            if parameter != "decoherence_time":
                graph0 = create_line_graph2(total_length=value if parameter == "distance" else
                default_line_total_length, min_link_Length=value[0] if parameter == "link"
                else default_line_link_length[0], max_link_length=value[1] if parameter == "link" else
                default_line_link_length[1])
            else:
                minimum_dist, maximum_dist = 0, 10000
                while maximum_dist - minimum_dist > 5:
                    mid_dist = (minimum_dist + maximum_dist) / 2
                    qNode._stat_id = 0
                    graph0 = create_line_graph2(total_length=mid_dist,
                                                min_link_Length=default_line_link_length[0],
                                                max_link_length=default_line_link_length[1])
                    if IS_DP:
                        dp_shortest = dp(graph0, decoherence_time=value, USAGE=DP_OPT)
                        dp_shortest_path = dp_shortest.get_shortest_path(graph0.V[0], graph0.V[-1],
                                                                         NON_THROTTLING=NON_THROTTLING)
                    elif IS_DP_ALT:
                        dp_alt = DP_Alt(graph0)
                        metric = 'time'
                        src, dst = graph0.V[0], graph0.V[-1]
                        path = dp_alt.dijsktra(src, dst, metric=metric)
                        dp_shortest_path = dp_alt.path2tree(path,
                                                            decoherence_time=value if parameter == "decoherence_time"
                                                            else default_decoherence_time,
                                                            update_time=True)
                    if dp_shortest_path:
                        reached_graph = graph0.clone()
                        minimum_dist = mid_dist
                    else:
                        maximum_dist = mid_dist
                max_dist_reached = sum([e.length for e in reached_graph.E]) / 1000
                print(f"\nMaximum distance reached: {max_dist_reached} kms")
                with open(saving_file_name + ".txt", 'a') as file:
                    file.write(f"\tMaximum distance reached: {max_dist_reached} kms")
                maximum_distance_reached[value_idx].append(max_dist_reached)
                graph0 = reached_graph
        sigcomm_graph0, sigcomm_ref_length = sigcomm_graph(graph0, sigcom_w, "max")
        for repeat_idx in range(number_of_repeat):
            print(f"Repeat #{repeat_idx}:")
            # # generating random src-dst pair
            srcs, dsts = [], []
            if not IS_LINE:
                n = value if parameter == "num_nodes" else default_num_nodes
                min_distance = value[0] if parameter == "distance" else default_distance_range[0]
                max_distance = value[1] if parameter == "distance" else default_distance_range[1]
                ids_with_distance = [(x, y) for x in range(n) for y in range(x + 1, n) if
                                     (min_distance * 10 ** 3 <= graph0.V[x].loc.distance(graph0.V[y].loc) <=
                                      max_distance * 10 ** 3 and graph0.get_edge(graph0.V[x].id,
                                                                                 graph0.V[y].id) is None)]
                seed(repeat_idx)
                shuffle(ids_with_distance)
                for i in range(min((value if parameter == "src_dst_pair" else default_number_of_src_dst_pair),
                                   len(ids_with_distance))):
                    srcs.append(ids_with_distance[i][0])
                    dsts.append(ids_with_distance[i][1])
            else:
                srcs.append(0)
                dsts.append(len(graph0.V) - 1)
            print(f"srcs = {srcs}, dsts = {dsts}")
            with open(saving_file_name + ".txt", 'a') as file:
                file.write(f"\nRepeat #: {repeat_idx}")
                file.write(f"\nsrcs = {srcs}, dsts = {dsts}")

            # srcs, dsts = [16], [56]
            # while len(srcs) < number_of_src_dst_pair:
            #
            #     src, dst = sample(range(value if parameter == "num_nodes" else default_num_nodes), 2)
            #     dist = graph0.V[src].loc.distance(graph0.V[dst].loc)
            #     if str(src) + "-" + str(dst) not in current_pairs and \
            #             str(dst) + "-" + str(src) not in current_pairs and \
            #             (value if parameter == "min_distance" else default_min_distance) <= dist:
            #         srcs.append(src)
            #         dsts.append(dst)
            #         current_pairs.add(str(src) + "-" + str(dst))

            # ******* line graph
            # qNode._stat_id = 0  # need to reset qNode._stat_id for every new experiment
            # graph0 = create_line_graph(4, 10e3, 16, 16, 4, 4,
            #                            atomic_bsm_success_rate=value if parameter == "atomic_bsm" else
            #                            default_atomic_bsm_rate,
            #                            optical_bsm_rate=value if parameter == "optical_bsm" else default_optical_bsm_rate)
            # srcs, dsts = [3], [0]

            # qNode._stat_id = 0
            # graph0 = create_toy_graph(area_side=100, cell_size=1000, number_nodes=100,
            #                           min_memory_size=10, max_memory_size=20,
            #                           min_num_edge=3, max_num_edge=5, min_channel_num=1, max_channel_num=5)
            # srcs = [3, 7, 8]
            # dsts = [0, 8, 1]
            # sigcomm_graph0 = sigcomm_graph(graph0)
            # Dynamic programming for single path
            if IS_DP:
                graph_dp = graph0.clone()
                dp_shortest = dp(graph_dp, value if parameter == "decoherence_time" else default_decoherence_time,
                                 USAGE=DP_OPT)
                dp_shortest_path = dp_shortest.get_shortest_path(graph_dp.V[srcs[0]], graph_dp.V[dsts[0]],
                                                                 NON_THROTTLING=NON_THROTTLING)
                if dp_shortest_path:
                    alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / dp_shortest_path.avr_ent_time
                    print(f"\n\nDP: {alg_result:.3f} (EP/s)", ("-OPTIMAL" if DP_OPT else ""),
                          ("-NON_THROTTLED" if NON_THROTTLING else ""))
                    tree_node.print_tree(dp_shortest_path)
                    with open(saving_file_name + ".txt", 'a') as file:
                        file.write("\n**** DP ***" + ("-OPTIMAL" if DP_OPT else "") +
                                   ("-NON_THROTTLED" if NON_THROTTLING else ""))
                        file.write(tree_node.print_tree_str(dp_shortest_path))
                    network_protocol_dp = TreeProtocol(graph=graph0, trees=[dp_shortest_path],
                                                       sources=[graph_dp.V[srcs[0]]],
                                                       destinations=[graph_dp.V[dsts[0]]],
                                                       duration=simulation_duration,
                                                       IS_TELEPORTING=IS_TELEPORTING)

                    print(f"DP (protocol): {network_protocol_dp.success_rate[0]} (EP/s), avg_fidelity: "
                          f"{network_protocol_dp.avg_fidelity[0]} "
                          f"\nduration={simulation_duration}s")
                    dp_res[value_idx].append(alg_result)
                    dp_protocol_res[value_idx].append(network_protocol_dp.success_rate[0])
                    dp_fidelity[value_idx].append(network_protocol_dp.avg_fidelity[0])
                    del network_protocol_dp
                else:  # no path found
                    dp_res[value_idx].append(0)
                    dp_protocol_res[value_idx].append(0)
                dp_res_avg[value_idx] = sum(dp_res[value_idx]) / len(dp_res[value_idx])
                dp_protocol_res_avg[value_idx] = sum(dp_protocol_res[value_idx]) / len(dp_protocol_res[value_idx])
                dp_fidelity_avg[value_idx] = weighted_avg(dp_fidelity[value_idx])[1]
                plotting_date['dp'] = dp_res_avg
                plotting_date['dp_proto'] = dp_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(dp_res))
                    file.write("\nProtocol results: " + str(dp_protocol_res))
                    file.write("\nFidelity results: " + str(dp_fidelity))
                    file.write("\nAlgorithm results (average): " + str(dp_res_avg))
                    file.write("\nProtocol results (average): " + str(dp_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(dp_fidelity_avg) + "\n\n")

            # DP Alternate
            if IS_DP_ALT:
                graph_dp_alt = graph0.clone()
                dp_alt = DP_Alt(graph_dp_alt,
                                decoherence_time=value if parameter == "decoherence_time" else default_decoherence_time)
                metric = 'time'
                src, dst = graph_dp_alt.V[srcs[0]], graph_dp_alt.V[dsts[0]]
                path = dp_alt.dijsktra(src, dst, metric=metric)
                dp_alt_tree = dp_alt.path2tree(path,
                                               decoherence_time=value if parameter == "decoherence_time"
                                               else default_decoherence_time,
                                               update_time=True)
                if dp_alt_tree:
                    alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / dp_alt_tree.avr_ent_time
                    print(f"\n\n-->DP Alternate: {alg_result:.3f} (EP/s)")
                    tree_node.print_tree(dp_alt_tree)
                    with open(saving_file_name + ".txt", 'a') as file:
                        file.write("\n**** DP Alternate ***")
                        file.write(tree_node.print_tree_str(dp_alt_tree))
                    network_protocol_dp_alt = TreeProtocol(graph0, [dp_alt_tree], [src], [dst], simulation_duration,
                                                           IS_TELEPORTING=IS_TELEPORTING)
                    print(f"DP Alternate (protocol): {network_protocol_dp_alt.success_rate[0]} (EP/s), "
                          f"avg_fidelity: {network_protocol_dp_alt.avg_fidelity[0]} "
                          f"\nduration={simulation_duration}s")
                    dp_alt_res[value_idx].append(alg_result)
                    dp_alt_protocol_res[value_idx].append(network_protocol_dp_alt.success_rate[0])
                    dp_alt_fidelity[value_idx].append(network_protocol_dp_alt.avg_fidelity[0])
                    del network_protocol_dp_alt
                else:  # no path found
                    dp_alt_res[value_idx].append(0)
                    dp_alt_protocol_res[value_idx].append(0)
                dp_alt_res_avg[value_idx] = sum(dp_alt_res[value_idx]) / len(dp_alt_res[value_idx])
                dp_alt_protocol_res_avg[value_idx] = sum(dp_alt_protocol_res[value_idx]) / len(
                    dp_alt_protocol_res[value_idx])
                dp_alt_fidelity_avg[value_idx] = weighted_avg(dp_alt_fidelity[value_idx])[1]
                plotting_date['dp_alt'] = dp_alt_res_avg
                plotting_date['dp_alt_proto'] = dp_alt_protocol_res_avg

                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(dp_alt_res))
                    file.write("\nProtocol results: " + str(dp_alt_protocol_res))
                    file.write("\nFidelity results: " + str(dp_alt_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(dp_alt_res_avg))
                    file.write("\nProtocol results (Average): " + str(dp_alt_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(dp_alt_fidelity_avg) + "\n\n")

            if IS_NAIVE:
                graph_naive = graph0.clone()
                naive_shortest = naive_dp(graph_naive,
                                          value if parameter == "decoherence_time" else default_decoherence_time)
                naive_shortest_path = naive_shortest.get_shortest_path(graph_naive.V[srcs[0]],
                                                                       graph_naive.V[dsts[0]],
                                                                       NON_THROTTLING=NON_THROTTLING)
                if naive_shortest_path:
                    alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / naive_shortest_path.avr_ent_time
                    print(f"\n\nNAIVE: {alg_result:.3f} (EP/s)",
                          ("-NON_THROTTLED" if NON_THROTTLING else ""))
                    tree_node.print_tree(naive_shortest_path)
                    with open(saving_file_name + ".txt", 'a') as file:
                        file.write("\n**** NAIVE ***" +
                                   ("-NON_THROTTLED" if NON_THROTTLING else ""))
                        file.write(tree_node.print_tree_str(naive_shortest_path))
                    network_protocol_naive = TreeProtocol(graph=graph0, trees=[naive_shortest_path],
                                                          sources=[graph_naive.V[srcs[0]]],
                                                          destinations=[graph_naive.V[dsts[0]]],
                                                          duration=simulation_duration,
                                                          IS_TELEPORTING=IS_TELEPORTING)

                    print(f"NAIVE (protocol): {network_protocol_naive.success_rate[0]} (EP/s), avg_fidelity: "
                          f"{network_protocol_naive.avg_fidelity[0]} "
                          f"\nduration={simulation_duration}s")
                    naive_res[value_idx].append(alg_result)
                    naive_protocol_res[value_idx].append(network_protocol_naive.success_rate[0])
                    naive_fidelity[value_idx].append(network_protocol_naive.avg_fidelity[0])
                    del network_protocol_naive
                else:  # no path found
                    naive_res[value_idx].append(0)
                    naive_protocol_res[value_idx].append(0)
                naive_res_avg[value_idx] = sum(naive_res[value_idx]) / len(naive_res[value_idx])
                naive_protocol_res_avg[value_idx] = sum(naive_protocol_res[value_idx]) / len(
                    naive_protocol_res[value_idx])
                naive_fidelity_avg[value_idx] = weighted_avg(naive_fidelity[value_idx])[1]
                plotting_date['naive'] = naive_res_avg
                plotting_date['naive_proto'] = naive_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(naive_res))
                    file.write("\nProtocol results: " + str(naive_protocol_res))
                    file.write("\nFidelity results: " + str(naive_fidelity))
                    file.write("\nAlgorithm results (average): " + str(naive_res_avg))
                    file.write("\nProtocol results (average): " + str(naive_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(naive_fidelity_avg) + "\n\n")

            # max flow
            if IS_DP_ITER:
                print(f"-->\n\n(DP Iterative)", ("-OPTIMAL" if DP_OPT else ""),
                      ("-NON_THROTTLED" if NON_THROTTLING else ""))
                start_time = datetime.datetime.now()
                graph_ours = graph0.clone()
                ours_srcs = [graph_ours.V[src] for src in srcs]
                ours_dsts = [graph_ours.V[dst] for dst in dsts]
                max_flow_ours = max_flow(graph_ours, ours_srcs, ours_dsts,
                                         decoherence_time=value if parameter == "decoherence_time" else
                                         default_decoherence_time
                                         , multiple_pairs_search='best_paths',
                                         NON_THROTTLE=NON_THROTTLING)
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) * max_flow_ours.get_flow_total
                print("Algorithm (max-flow) results")
                print("Total flow algorithm (max-flow): {:.2f} q/s\ntime:".format(alg_result),
                      datetime.datetime.now() - start_time)
                if max_flow_ours.get_flow_total > 0:
                    ours_all_trees = []
                    ours_protocol_all_srcs, ours_protocol_all_dsts = [], []
                    tree_num = 0
                    with open(saving_file_name + ".txt", 'a') as file:
                        file.write(f"\n**** DP Iterative ***" + ("-OPTIMAL" if DP_OPT else "") +
                                   ("-NON_THROTTLED" if NON_THROTTLING else ""))
                    for pair_id in range(len(ours_dsts)):
                        pair_res = max_flow_ours.get_flows(pair_id)
                        for _, tree, _ in pair_res:
                            ours_all_trees.append(tree)
                            ours_protocol_all_srcs.append(ours_srcs[pair_id])
                            ours_protocol_all_dsts.append(ours_dsts[pair_id])

                            print(f"Tree{tree_num} (rate:"
                                  f" {(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time} EPs/s), "
                                  f"between nodes {ours_srcs[pair_id].id} (loc: {ours_srcs[pair_id].loc}) and "
                                  f"{ours_dsts[pair_id].id} (loc: {ours_dsts[pair_id].loc})")
                            tree_node.print_tree(tree)
                            with open(saving_file_name + ".txt", 'a') as file:
                                file.write(f"\nTree{tree_num} (rate:"
                                           f" {(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time}"
                                           f" EPs/s), "
                                           f"between nodes {ours_srcs[pair_id].id} (loc: {ours_srcs[pair_id].loc}) and "
                                           f"{ours_dsts[pair_id].id} (loc: {ours_dsts[pair_id].loc})")
                                file.write(tree_node.print_tree_str(tree))
                            tree_num += 1
                    start_time = datetime.datetime.now()
                    ours_protocol = TreeProtocol(graph=graph0, trees=ours_all_trees,
                                                 sources=ours_protocol_all_srcs, destinations=ours_protocol_all_dsts,
                                                 duration=simulation_duration, IS_TELEPORTING=IS_TELEPORTING)
                    valid_fidelities = [ours_protocol.avg_fidelity[i] for i in range(len(ours_protocol.avg_fidelity))]
                    print("\nProtocol(max-flow) Results:")
                    print(f"Total flow algorithm (max-flow): {sum(ours_protocol.success_rate):.2f} ep/s, avg_fidelity: "
                          f"{weighted_avg(valid_fidelities)[1] if weighted_avg(valid_fidelities)[0] > 0 else -1}"
                          f"\nduration: {simulation_duration}s")
                    for tree_id, rate in enumerate(ours_protocol.success_rate):
                        print(f"Tree{tree_id}, rate: {rate} EPs/s, avg_fidelity: {ours_protocol.avg_fidelity[tree_id]}")
                    dp_iter_res[value_idx].append(alg_result)
                    dp_iter_protocol_res[value_idx].append(sum(ours_protocol.success_rate))
                    dp_iter_fidelity[value_idx].append(weighted_avg(valid_fidelities))
                else:  # no path found
                    dp_iter_res[value_idx].append(0)
                    dp_iter_protocol_res[value_idx].append(0)
                dp_iter_res_avg[value_idx] = sum(dp_iter_res[value_idx]) / len(dp_iter_res[value_idx])
                dp_iter_protocol_res_avg[value_idx] = sum(dp_iter_protocol_res[value_idx]) / len(
                    dp_iter_protocol_res[value_idx])
                dp_iter_fidelity_avg[value_idx] = weighted_avg(dp_iter_fidelity[value_idx])[1]
                plotting_date['dp_iter'] = dp_iter_res_avg
                plotting_date['dp_iter_proto'] = dp_iter_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(dp_iter_res))
                    file.write("\nProtocol results: " + str(dp_iter_protocol_res))
                    file.write("\nFidelity results: " + str(dp_iter_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(dp_iter_res_avg))
                    file.write("\nProtocol results (Average): " + str(dp_iter_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(dp_iter_fidelity_avg) + "\n\n")
                del ours_protocol, max_flow_ours

            # DP Alternate + max_flow
            if IS_LP_ALT:
                print('\n\n-->Ours (LP Alternate)')
                with open(saving_file_name + ".txt", 'a') as file:
                    file.write("*\n*** LP Alternate ***")
                start_time = datetime.datetime.now()
                graph_lp_alt = graph0.clone()
                lp_alt_srcs = [graph_lp_alt.V[src] for src in srcs]
                lp_alt_dsts = [graph_lp_alt.V[dst] for dst in dsts]
                max_flow_alt = max_flow(graph_lp_alt, lp_alt_srcs, lp_alt_dsts,
                                        decoherence_time=value if parameter == "decoherence_time" else
                                        default_decoherence_time,
                                        multiple_pairs_search='best_paths', alternate=True)
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) * max_flow_alt.get_flow_total
                print("Algorithm (LP alternate) results")
                print("Total flow algorithm (LP alternate): {:.2f} q/s\ntime:".format(alg_result),
                      datetime.datetime.now() - start_time)
                if max_flow_alt.get_flow_total > 0:
                    lp_alt_all_trees = []
                    lp_alt_protocol_all_srcs, lp_alt_protocol_all_dsts = [], []
                    tree_num = 0
                    for pair_id in range(len(lp_alt_dsts)):
                        pair_res = max_flow_alt.get_flows(pair_id)
                        for _, tree, _ in pair_res:
                            lp_alt_all_trees.append(tree)
                            lp_alt_protocol_all_srcs.append(lp_alt_srcs[pair_id])
                            lp_alt_protocol_all_dsts.append(lp_alt_dsts[pair_id])

                            print(f"Tree{tree_num} (rate: "
                                  f"{(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time} EPs/s), "
                                  f"between nodes {lp_alt_srcs[pair_id].id} (loc: {lp_alt_srcs[pair_id].loc}) and "
                                  f"{lp_alt_dsts[pair_id].id} (loc: {lp_alt_dsts[pair_id].loc})")
                            tree_node.print_tree(tree)
                            with open(saving_file_name + ".txt", 'a') as file:
                                file.write(f"\nTree{tree_num} (rate: "
                                           f"{(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time} "
                                           f"EPs/s), "
                                           f"between nodes {lp_alt_srcs[pair_id].id} (loc: {lp_alt_srcs[pair_id].loc}) and "
                                           f"{lp_alt_dsts[pair_id].id} (loc: {lp_alt_dsts[pair_id].loc})")
                                file.write(tree_node.print_tree_str(tree))
                            tree_num += 1
                    start_time = datetime.datetime.now()
                    lp_alt_protocol = TreeProtocol(graph=graph0, trees=lp_alt_all_trees,
                                                   sources=lp_alt_protocol_all_srcs,
                                                   destinations=lp_alt_protocol_all_dsts,
                                                   duration=simulation_duration, IS_TELEPORTING=IS_TELEPORTING)
                    valid_fidelities = [lp_alt_protocol.avg_fidelity[i] for i in
                                        range(len(lp_alt_protocol.avg_fidelity))]
                    print("\nProtocol (LP alternate) Results:")
                    print(f"Total flow algorithm (LP alternate): {sum(lp_alt_protocol.success_rate):.2f} eps/s, "
                          f"avg_fidelity: {weighted_avg(valid_fidelities)[1] if weighted_avg(valid_fidelities)[0] > 0 else -1}"
                          f"\nduration: {simulation_duration}s")
                    for tree_id, rate in enumerate(lp_alt_protocol.success_rate):
                        print(f"Tree{tree_id}, rate: {rate} EPs/s, fidelity: {lp_alt_protocol.avg_fidelity[tree_id]}")
                    lp_alt_res[value_idx].append(alg_result)
                    lp_alt_protocol_res[value_idx].append(sum(lp_alt_protocol.success_rate))
                    lp_alt_fidelity[value_idx].append(weighted_avg(valid_fidelities))
                else:  # no path found
                    lp_alt_res[value_idx].append(0)
                    lp_alt_protocol_res[value_idx].append(0)
                lp_alt_res_avg[value_idx] = sum(lp_alt_res[value_idx]) / len(lp_alt_res[value_idx])
                lp_alt_protocol_res_avg[value_idx] = sum(lp_alt_protocol_res[value_idx]) / len(
                    lp_alt_protocol_res[value_idx])
                lp_alt_fidelity_avg[value_idx] = weighted_avg(lp_alt_fidelity[value_idx])[1]
                plotting_date['lp_alt'] = lp_alt_res_avg
                plotting_date['lp_alt_proto'] = lp_alt_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(lp_alt_res))
                    file.write("\nProtocol results: " + str(lp_alt_protocol_res))
                    file.write("\nFidelity results: " + str(lp_alt_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(lp_alt_res_avg))
                    file.write("\nProtocol results (Average): " + str(lp_alt_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(lp_alt_fidelity_avg) + "\n\n")
                del lp_alt_protocol

            # Naive + max_flow
            if IS_NAIVE_ITER:
                print(f"-->\n\n(NAIVE Iterative)", ("-NON_THROTTLED" if NON_THROTTLING else ""))
                start_time = datetime.datetime.now()
                graph_naive_iter = graph0.clone()
                ours_srcs = [graph_naive_iter.V[src] for src in srcs]
                ours_dsts = [graph_naive_iter.V[dst] for dst in dsts]
                max_flow_naive = max_flow(graph_naive_iter, ours_srcs, ours_dsts,
                                          decoherence_time=value if parameter == "decoherence_time" else
                                          default_decoherence_time,
                                          multiple_pairs_search='best_paths',
                                          NON_THROTTLE=NON_THROTTLING,
                                          IS_NAIVE=True)
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) * max_flow_naive.get_flow_total
                print("Algorithm (max-flow-naive) results")
                print("Total flow algorithm (max-flow-naive): {:.2f} q/s\ntime:".format(alg_result),
                      datetime.datetime.now() - start_time)
                if max_flow_naive.get_flow_total > 0:
                    naive_iter_all_trees = []
                    naive_iter_protocol_all_srcs, naive_iter_protocol_all_dsts = [], []
                    tree_num = 0
                    with open(saving_file_name + ".txt", 'a') as file:
                        file.write(f"\n**** NAIVE Iterative ***" + ("-NON_THROTTLED" if NON_THROTTLING else ""))
                    for pair_id in range(len(ours_dsts)):
                        pair_res = max_flow_naive.get_flows(pair_id)
                        for _, tree, _ in pair_res:
                            naive_iter_all_trees.append(tree)
                            naive_iter_protocol_all_srcs.append(ours_srcs[pair_id])
                            naive_iter_protocol_all_dsts.append(ours_dsts[pair_id])

                            print(f"Tree{tree_num} (rate:"
                                  f" {(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time} EPs/s), "
                                  f"between nodes {ours_srcs[pair_id].id} (loc: {ours_srcs[pair_id].loc}) and "
                                  f"{ours_dsts[pair_id].id} (loc: {ours_dsts[pair_id].loc})")
                            tree_node.print_tree(tree)
                            with open(saving_file_name + ".txt", 'a') as file:
                                file.write(f"\nTree{tree_num} (rate:"
                                           f" {(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time}"
                                           f" EPs/s), "
                                           f"between nodes {ours_srcs[pair_id].id} (loc: {ours_srcs[pair_id].loc}) and "
                                           f"{ours_dsts[pair_id].id} (loc: {ours_dsts[pair_id].loc})")
                                file.write(tree_node.print_tree_str(tree))
                            tree_num += 1
                    start_time = datetime.datetime.now()
                    naive_iter_protocol = TreeProtocol(graph=graph0, trees=naive_iter_all_trees,
                                                       sources=naive_iter_protocol_all_srcs,
                                                       destinations=naive_iter_protocol_all_dsts,
                                                       duration=simulation_duration, IS_TELEPORTING=IS_TELEPORTING)
                    valid_fidelities = [naive_iter_protocol.avg_fidelity[i]
                                        for i in range(len(naive_iter_protocol.avg_fidelity))]
                    print("\nProtocol(max-flow-naive) Results:")
                    print(f"Total flow algorithm (max-flow-naive): {sum(naive_iter_protocol.success_rate):.2f} ep/s, "
                          f"avg_fidelity: "
                          f"{weighted_avg(valid_fidelities)[1] if weighted_avg(valid_fidelities)[0] > 0 else -1}"
                          f"\nduration: {simulation_duration}s")
                    for tree_id, rate in enumerate(naive_iter_protocol.success_rate):
                        print(f"Tree{tree_id}, rate: {rate} EPs/s, avg_fidelity: "
                              f"{naive_iter_protocol.avg_fidelity[tree_id]}")
                    naive_iter_res[value_idx].append(alg_result)
                    naive_iter_protocol_res[value_idx].append(sum(naive_iter_protocol.success_rate))
                    naive_iter_fidelity[value_idx].append(weighted_avg(valid_fidelities))
                    del naive_iter_protocol
                else:  # no path found
                    naive_iter_res[value_idx].append(0)
                    naive_iter_protocol_res[value_idx].append(0)
                naive_iter_res_avg[value_idx] = sum(naive_iter_res[value_idx]) / len(naive_iter_res[value_idx])
                naive_iter_protocol_res_avg[value_idx] = sum(naive_iter_protocol_res[value_idx]) / len(
                    naive_iter_protocol_res[value_idx])
                naive_iter_fidelity_avg[value_idx] = weighted_avg(naive_iter_fidelity[value_idx])[1]
                plotting_date['dp_iter'] = naive_iter_res_avg
                plotting_date['dp_iter_proto'] = naive_iter_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(naive_iter_res))
                    file.write("\nProtocol results: " + str(naive_iter_protocol_res))
                    file.write("\nFidelity results: " + str(naive_iter_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(naive_iter_res_avg))
                    file.write("\nProtocol results (Average): " + str(naive_iter_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(naive_iter_fidelity_avg) + "\n\n")
                del max_flow_naive

            # Linear Programming
            if IS_LP:
                print("\n\n-->LP")
                with open(saving_file_name + ".txt", 'a') as file:
                    file.write("*\n*** LP ***")
                graph_lp = graph0.clone()
                start_time = datetime.datetime.now()
                lp_srcs = [graph_lp.V[src] for src in srcs]
                lp_dsts = [graph_lp.V[dst] for dst in dsts]
                lp = optimal_linear_programming(graph_lp, lp_srcs, lp_dsts)
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) * lp.max_flow
                lp_all_trees = []
                lp_protocol_all_srcs, lp_protocol_all_dsts = [], []
                tree_num = 0
                print("Algorithm (LP) results")
                print("Total flow algorithm (LP): {:.3f} q/s\ntime:".format(alg_result),
                      datetime.datetime.now() - start_time)
                for pair_id, trees in enumerate(lp.trees):
                    for tree in trees:
                        lp_all_trees.append(tree)
                        lp_protocol_all_srcs.append(lp_srcs[pair_id])
                        lp_protocol_all_dsts.append(lp_dsts[pair_id])

                        print(f"Tree{tree_num} (rate:"
                              f" {(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time} EPs/s), "
                              f"between nodes {lp_srcs[pair_id].id} (loc: {lp_srcs[pair_id].loc}) and "
                              f"{lp_dsts[pair_id].id} (loc: {lp_dsts[pair_id].loc})")
                        tree_node.print_tree(tree)
                        with open(saving_file_name + ".txt", 'a') as file:
                            file.write(f"\nTree{tree_num} (rate:"
                                       f" {(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time} "
                                       f"EPs/s), "
                                       f"between nodes {lp_srcs[pair_id].id} (loc: {lp_srcs[pair_id].loc}) and "
                                       f"{lp_dsts[pair_id].id} (loc: {lp_dsts[pair_id].loc})")
                            file.write(tree_node.print_tree_str(tree))
                        tree_num += 1

                start_time = datetime.datetime.now()
                lp_protocol = TreeProtocol(graph=graph0, trees=lp_all_trees,
                                           sources=lp_protocol_all_srcs, destinations=lp_protocol_all_dsts,
                                           duration=simulation_duration, IS_TELEPORTING=IS_TELEPORTING)
                valid_fidelities = [lp_protocol.avg_fidelity[i] for i in range(len(lp_protocol.avg_fidelity))]
                print("\nProtocol (LP) Results:")
                print(f"Total flow Protocol (LP): {sum(lp_protocol.success_rate):.2f} eps/s, "
                      f"avg_fidelity: {weighted_avg(valid_fidelities)[1]}"
                      f"\nduration: {simulation_duration}")
                for tree_id, rate in enumerate(lp_protocol.success_rate):
                    print(f"Tree{tree_id}, rate: {rate} EPs/s, avg_fidelity: {lp_protocol.avg_fidelity[tree_id]}")
                lp_res[value_idx].append(alg_result)
                lp_protocol_res[value_idx].append(sum(lp_protocol.success_rate))
                lp_fidelity[value_idx].append(weighted_avg(valid_fidelities))
                lp_res_avg[value_idx] = sum(lp_res[value_idx]) / len(lp_res[value_idx])
                lp_protocol_res_avg[value_idx] = sum(lp_protocol_res[value_idx]) / len(lp_protocol_res[value_idx])
                lp_fidelity_avg[value_idx] = weighted_avg(lp_fidelity[value_idx])[1]
                plotting_date['lp'] = lp_res_avg
                plotting_date['lp_proto'] = lp_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(lp_res))
                    file.write("\nProtocol results: " + str(lp_protocol_res))
                    file.write("\nFidelity results: " + str(lp_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(lp_res_avg))
                    file.write("\nProtocol results (Average): " + str(lp_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(lp_fidelity_avg) + "\n\n")
                del lp, lp_protocol

            # SIGCOMM 20 single path
            if IS_SIGCOMM_SINGLE:
                print(f'\n\n-->SIGCOMM 20 Single W={sigcom_w}')
                graph_sigcomm = sigcomm_graph0.clone()
                src, dst = graph_sigcomm.V[srcs[0]], graph_sigcomm.V[dsts[0]]
                W = graph_sigcomm.E[0].channels_num
                sigcomm_w = Sigcomm_W(graph_sigcomm, W)
                sig_path = sigcomm_w.dijkstra(src, dst)
                sig_rates = sigcomm_w.paths2rates([sig_path])
                sig_path.avr_ent_time = 1 / sig_rates[0]
                sig_tree = sigcomm_w.path2tree(sig_path, decoherence_time=default_decoherence_time, update_time=False)
                leaf_effective_rate = sigcomm_w.get_leaf_effective_rate(sig_path)
                leaf_ent_time = 1. / leaf_effective_rate
                sigcomm_w.update_ent_time_tree(sig_tree, leaf_ent_time)
                print(f"SIGCOMM 20 single path: {sig_rates[0]} (EP/s)")
                tree_node.print_tree(sig_tree)
                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\n***** SIGCOMM 20 Single  W={sigcom_w} ****")
                    file.write(tree_node.print_tree_str(sig_tree))
                valid_fidelities = -1
                if sig_rates[0] > 0.01:
                    network_protocol_sigcomm = SigcommTreeProtocol(sigcomm_graph0, [(sig_tree, sig_path.W)], [src],
                                                                   [dst],
                                                                   2 * simulation_duration,
                                                                   swapping_time_window=0,
                                                                   IS_TELEPORTING=IS_TELEPORTING,
                                                                   ref_length=sigcomm_ref_length)
                    sigcomm_protocol_rate = network_protocol_sigcomm.success_rate[0]
                    valid_fidelities = network_protocol_sigcomm.avg_fidelity[0]
                    sigcomm_single_fidelity[value_idx].append(valid_fidelities)

                else:
                    sigcomm_protocol_rate = sig_rates[0]
                print(f"Sigcomm single (protocol): {sigcomm_protocol_rate} (EP/s), avg_fidelity: {valid_fidelities}"
                      f"\nduration={2 * simulation_duration}s")
                sigcomm_single_res[value_idx].append(
                    (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / sig_tree.avr_ent_time)
                sigcomm_single_protocol_res[value_idx].append(sigcomm_protocol_rate)
                if weighted_avg(sigcomm_single_fidelity[value_idx])[0] > 0:
                    sigcomm_single_fidelity_avg[value_idx] = weighted_avg(sigcomm_single_fidelity[value_idx])[1]
                sigcomm_single_res_avg[value_idx] = sum(sigcomm_single_res[value_idx]) / len(
                    sigcomm_single_res[value_idx])
                sigcomm_single_protocol_res_avg[value_idx] = sum(sigcomm_single_protocol_res[value_idx]) / len(
                    sigcomm_single_protocol_res[value_idx])

                plotting_date['sig'] = sigcomm_single_res_avg
                plotting_date['sig_proto'] = sigcomm_single_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(sigcomm_single_res))
                    file.write("\nProtocol results: " + str(sigcomm_single_protocol_res))
                    file.write("\nFidelity results: " + str(sigcomm_single_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(sigcomm_single_res_avg))
                    file.write("\nProtocol results (Average): " + str(sigcomm_single_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(sigcomm_single_fidelity_avg) + "\n\n")

            # SIGCOMM 20 multiple path
            if IS_SIGCOMM_MULTI:
                print(f'\n\n-->SIGCOMM 20 Multiple  W={sigcom_w}')
                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\n***** SIGCOMM 20 Multiple  W={sigcom_w} ****")
                graph_sigcomm = sigcomm_graph0.clone()
                sig_srcs = [graph_sigcomm.V[src] for src in srcs]
                sig_dsts = [graph_sigcomm.V[dst] for dst in dsts]
                sd_pairs = [[src, dst] for src, dst in zip(sig_srcs, sig_dsts)]
                print('\ninput src-dst pairs are:')
                for i, (src, dst) in enumerate(sd_pairs):
                    print(f'{i}, src = {src.id}, dst = {dst.id}')
                start = time.time()
                W = graph_sigcomm.E[0].channels_num
                sigcomm_w = Sigcomm_W(graph_sigcomm, W)
                sig_paths = sigcomm_w.phase_two(sd_pairs)
                sig_rates = sigcomm_w.paths2rates(sig_paths)
                end = time.time()
                sigcomm_all_trees = []
                sigcomm_all_srcs, sigcomm_all_dsts = [], []
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) * sum(sig_rates)
                print("Algorithm (SIGCOMM20) results")
                print("Total flow algorithm (SIGCOMM20): {:.3f} EPs/s\ntime:".format(alg_result), end - start)
                for path, rate in zip(sig_paths, sig_rates):
                    if rate < 0.01:
                        continue
                    path.avr_ent_time = 1 / rate
                    tree = sigcomm_w.path2tree(path, decoherence_time=default_decoherence_time)
                    leaf_effective_rate = sigcomm_w.get_leaf_effective_rate(path)
                    leaf_ent_time = 1. / leaf_effective_rate
                    sigcomm_w.update_ent_time_tree(tree, leaf_ent_time)
                    src, dst = path.nodes[0], path.nodes[-1]
                    sigcomm_all_trees.append((tree, path.W))
                    sigcomm_all_srcs.append(graph_sigcomm.V[src])
                    sigcomm_all_dsts.append(graph_sigcomm.V[dst])
                    print(f"SIGCOMM 20 path: {rate} (EP/s)")
                    tree_node.print_tree(tree)
                    with open(saving_file_name + ".txt", 'a') as file:
                        file.write(tree_node.print_tree_str(tree))

                if len(sigcomm_all_trees) > 0:
                    sigcomm_protocol = SigcommTreeProtocol(graph=sigcomm_graph0, trees=sigcomm_all_trees,
                                                           sources=sigcomm_all_srcs, destinations=sigcomm_all_dsts,
                                                           duration=2 * simulation_duration, swapping_time_window=1,
                                                           IS_TELEPORTING=IS_TELEPORTING,
                                                           ref_length=sigcomm_ref_length)
                    print("\nProtocol (SIGCOMM) Results:")
                    valid_fidelities = [sigcomm_protocol.avg_fidelity[i] for i in
                                        range(len(sigcomm_protocol.avg_fidelity))]
                    weighted_fidelity = weighted_avg(valid_fidelities)
                    print(f"Total flow Protocol (SIGCOMM): {sum(sigcomm_protocol.success_rate)} (EP/s), "
                          f"Avg_fidelity:{weighted_fidelity[1] if weighted_fidelity[0] > 0 else -1}, "
                          f"\nduration={2 * simulation_duration}s")
                    for tree_id, rate in enumerate(sigcomm_protocol.success_rate):
                        print(
                            f"Tree{tree_id}, rate: {rate} EPs/s, Avg_fidelity: {sigcomm_protocol.avg_fidelity[tree_id]}")
                    sigcomm_multiple_protocol_res[value_idx].append(sum(sigcomm_protocol.success_rate))
                    sigcomm_multi_fidelity[value_idx].append(weighted_fidelity)
                    del sigcomm_protocol
                else:
                    sigcomm_multiple_protocol_res[value_idx].append(alg_result)
                sigcomm_multiple_res[value_idx].append(alg_result)
                sigcomm_multiple_res_avg[value_idx] = sum(sigcomm_multiple_res[value_idx]) / \
                                                      len(sigcomm_multiple_res[value_idx])
                sigcomm_multiple_protocol_res_avg[value_idx] = sum(sigcomm_multiple_protocol_res[value_idx]) / \
                                                               len(sigcomm_multiple_protocol_res[value_idx])
                if weighted_avg(sigcomm_multi_fidelity[value_idx])[0] > 0:
                    sigcomm_multi_fidelity_avg[value_idx] = weighted_avg(sigcomm_multi_fidelity[value_idx])[1]
                plotting_date['sig_multi'] = sigcomm_multiple_res_avg
                plotting_date['sig_multi_proto'] = sigcomm_multiple_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(sigcomm_multiple_res))
                    file.write("\nProtocol results: " + str(sigcomm_multiple_protocol_res))
                    file.write("\nFidelity results: " + str(sigcomm_multi_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(sigcomm_multiple_res_avg))
                    file.write("\nProtocol results (Average): " + str(sigcomm_multiple_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(sigcomm_multi_fidelity_avg) + "\n\n")

            # Delft LP
            if IS_DELFT_LP:
                def fidelity_generator(param1: float = 0.85, param2: float = 0.05):
                    '''
                    Args:
                        param -- control the fidelity
                    Yields:
                        fidelities
                    '''
                    seed(param1)
                    while True:
                        yield param1 + (random() * 2 - 1) * param2  # with default param, the yield range is 0.8 ~ 0.9

                graph_delft_lp = graph0.clone()
                print('\n\n-->Delft LP')
                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\n***** Delft LP ****")
                F = 0.97
                fidelities = fidelity_generator()
                sig_srcs = [graph_delft_lp.V[src] for src in srcs]
                sig_dsts = [graph_delft_lp.V[dst] for dst in dsts]
                sd_pairs = [[src, dst] for src, dst in zip(sig_srcs, sig_dsts)]
                demands = [(sd[0], sd[1], f) for sd, f in zip(sd_pairs, fidelities)]
                start = time.time()
                modified_qgraph = modified_qGraph(graph_delft_lp, demands, F=F)
                for u, v, length in modified_qgraph.demands_length:
                    print(f'src = {u.id}, dst = {v.id}, max path length = {length}')
                # modified_qgraph.visualize(filename=filename + '-modified')
                lp = LP(modified_qgraph)
                lp.solve(print=False)
                paths = lp.extract_path()
                # use the DP_Alt to get the trees
                graph_dp_alt = graph0.clone()
                dp_alt = DP_Alt(graph_dp_alt)
                delft_lp_all_trees = []
                delft_lp_all_srcs, delft_lp_all_dsts = [], []
                rates = []
                for path in paths:
                    rate = path.metric
                    rates.append(rate)
                    leaf_effective_rate = lp.get_leaf_effective_rate(path)
                    try:
                        leaf_ent_time = 1. / leaf_effective_rate
                    except Exception as _:
                        continue
                    path.metric = 1 / path.metric  # from rate to avg_ent_time, for the dp_alt.path2tree method
                    tree = dp_alt.path2tree(path, decoherence_time=default_decoherence_time, update_time=False)
                    lp.update_ent_time_tree(tree, leaf_ent_time)
                    if rate < 0.01:
                        continue
                    src, dst = path.nodes[0], path.nodes[-1]
                    delft_lp_all_trees.append((tree, 1))
                    delft_lp_all_srcs.append(graph_delft_lp.V[src])
                    delft_lp_all_dsts.append(graph_delft_lp.V[dst])
                    print(f"Delft LP path: {rate} (EP/s)")
                    tree_node.print_tree(tree)
                    with open(saving_file_name + ".txt", "a") as file:
                        file.write(tree_node.print_tree_str(tree))
                # print(f'Delft LP time = {time.time() - start}')
                if len(delft_lp_all_trees) > 0:
                    delft_lp_protocol = SigcommTreeProtocol(graph=sigcomm_graph0, trees=delft_lp_all_trees,
                                                            sources=delft_lp_all_srcs, destinations=delft_lp_all_dsts,
                                                            duration=2 * simulation_duration, swapping_time_window=1,
                                                            IS_TELEPORTING=IS_TELEPORTING,
                                                            ref_length=sigcomm_ref_length)
                    print("\nProtocol (DELFT) Results:")
                    valid_fidelities = [delft_lp_protocol.avg_fidelity[i] for i in
                                        range(len(delft_lp_protocol.avg_fidelity))]
                    weighted_fidelity = weighted_avg(valid_fidelities)
                    print(f"Total flow Protocol (DELFT): {sum(delft_lp_protocol.success_rate)} (EP/s), "
                          f"Avg_fidelity:{weighted_fidelity[1] if weighted_fidelity[0] > 0 else -1}, "
                          f"\nduration={2 * simulation_duration}s")
                    for tree_id, rate in enumerate(delft_lp_protocol.success_rate):
                        print(f"Tree{tree_id}, rate: {rate} EPs/s, "
                              f"Avg_fidelity: {delft_lp_protocol.avg_fidelity[tree_id]}")
                    delft_lp_protocol_res[value_idx].append(sum(delft_lp_protocol.success_rate))
                    delft_lp_fidelity[value_idx].append(weighted_fidelity)
                    del delft_lp_protocol
                else:
                    delft_lp_protocol_res[value_idx].append(
                        (teleportation_bsm_rate if IS_TELEPORTING else 1.0) * sum(rates))
                delft_lp_res[value_idx].append((teleportation_bsm_rate if IS_TELEPORTING else 1.0) * sum(rates))
                delft_lp_res_avg[value_idx] = sum(delft_lp_res[value_idx]) / len(delft_lp_res[value_idx])
                delft_lp_protocol_res_avg[value_idx] = sum(delft_lp_protocol_res[value_idx]) / len(
                    delft_lp_protocol_res[value_idx])
                if weighted_avg(delft_lp_fidelity[value_idx])[0] > 0:
                    delft_lp_fidelity_avg[value_idx] = weighted_avg(delft_lp_fidelity[value_idx])[1]
                plotting_date['delft_lp'] = delft_lp_res_avg
                plotting_date['delft_lp_proto'] = delft_lp_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\nAlgorithm results: " + str(delft_lp_res))
                    file.write("\nProtocol results: " + str(delft_lp_protocol_res))
                    file.write("\nFidelity results: " + str(delft_lp_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(delft_lp_res_avg))
                    file.write("\nProtocol results (Average): " + str(delft_lp_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(delft_lp_fidelity_avg) + "\n\n")
                del lp

    with open(saving_file_name + ".txt", 'a') as file:
        file.write(f"\n\n\n*********Summary*******")
        file.write(f"\n\nTESTING {parameter}: {values}")

    if IS_LINE and parameter == "decoherence_time":
        with open(saving_file_name + ".txt", 'a') as file:
            file.write(f"\n\nMaximum distance reached: {maximum_distance_reached}")

    if IS_DP:
        with open(saving_file_name + ".txt", "a") as file:
            file.write("\n**** DP ***" + ("-OPTIMAL" if DP_OPT else "") +
                       ("-NON_THROTTLED" if NON_THROTTLING else ""))
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in dp_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in dp_protocol_res]))
            file.write("\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in dp_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in dp_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in dp_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in dp_fidelity_avg]) + "\n\n")

    if IS_DP_ALT:
        with open(saving_file_name + ".txt", "a") as file:
            file.write("\n**** DP-Alternate (Balanced-Tree) ***")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in dp_alt_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in dp_alt_protocol_res]))
            file.write("\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in dp_alt_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in dp_alt_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in dp_alt_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in dp_alt_fidelity_avg]) + "\n\n")

    if IS_NAIVE:
        with open(saving_file_name + ".txt", "a") as file:
            file.write("\n**** Naive (single) ***")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in naive_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in naive_protocol_res]))
            file.write("\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in naive_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in naive_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in naive_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in naive_fidelity_avg]) + "\n\n")

    if IS_DP_ITER:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n**** DP Iterative ***" + ("-OPTIMAL" if DP_OPT else "") +
                       ("-NON_THROTTLED" if NON_THROTTLING else ""))
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in dp_iter_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in dp_iter_protocol_res]))
            file.write(
                "\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in dp_iter_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in dp_iter_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in dp_iter_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in dp_iter_fidelity_avg]) + "\n\n")

    if IS_NAIVE_ITER:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n**** NAIVE Iterative ***" + ("-NON_THROTTLED" if NON_THROTTLING else ""))
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in naive_iter_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in naive_iter_protocol_res]))
            file.write(
                "\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in naive_iter_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in naive_iter_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in naive_iter_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in naive_iter_fidelity_avg]) + "\n\n")

    if IS_LP_ALT:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n**** LP Alternate (Balanced-Tree) ***")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in lp_alt_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in lp_alt_protocol_res]))
            file.write("\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in lp_alt_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in lp_alt_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in lp_alt_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in lp_alt_fidelity_avg]) + "\n\n")

    if IS_LP:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n**** LP ***")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in lp_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in lp_protocol_res]))
            file.write("\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in lp_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in lp_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in lp_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in lp_fidelity_avg]) + "\n\n")

    if IS_SIGCOMM_SINGLE:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n***** SIGCOMM 20 Single  W={sigcom_w} ****")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in sigcomm_single_res]))
            file.write(
                "\nProtocol results: " + str([[round(x, 3) for x in row] for row in sigcomm_single_protocol_res]))
            file.write("\nFidelity results: " + str(
                [[(x[0], round(x[1], 3)) for x in row] for row in sigcomm_single_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in sigcomm_single_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in sigcomm_single_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in sigcomm_single_fidelity_avg]) +
                       "\n\n")

    if IS_SIGCOMM_MULTI:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n***** SIGCOMM 20 Multiple  W={sigcom_w} ****")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in sigcomm_multiple_res]))
            file.write(
                "\nProtocol results: " + str([[round(x, 3) for x in row] for row in sigcomm_multiple_protocol_res]))
            file.write(
                "\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in sigcomm_multi_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in sigcomm_multiple_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in sigcomm_multiple_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in sigcomm_multi_fidelity_avg])
                       + "\n\n")

    if IS_DELFT_LP:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n***** Delft LP ****")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in delft_lp_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in delft_lp_protocol_res]))
            file.write(
                "\nFidelity results: " + str([[(x[0], round(x[1], 3)) for x in row] for row in delft_lp_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in delft_lp_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in delft_lp_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in delft_lp_fidelity_avg]) + "\n\n")

    # Plot.graph(plotting_data=plotting_date, filename=saving_file_name + ".png")


def main_pre():
    seed(10)
    # **************** init values ************
    area_side = 10  # a square side of side x side cells
    cell_size = 10000  # side value of each square side (m)
    min_memory_size, max_memory_size = 15, 20
    min_node_degree, max_node_degree = 2, 4  # number of edges from one node to its neighbors
    min_channel_num, max_channel_num = 1, 1  # number of parallel channels a connection would have
    default_number_of_src_dst_pair = 12
    default_decoherence_time = 2  # seconds
    default_atomic_bsm_rate, default_optical_bsm_rate = 0.4, 0.2
    default_edge_density = 0.08
    default_num_nodes = 100
    default_distance_range = [30, 120]
    default_cost_budget = 20e3
    default_line_total_length = 100  # kms
    default_line_link_length = (30, 35)  # kms
    default_timeslot = 4.0  # (seconds) the interval of demands
    # [15, 20] if default_number_of_src_dst_pair == 1 else [5, 100]
    IS_LATENCY_OBJECTIVE = True
    FIXED_LATENCY, LATENCY_THRESHOLD_DEFAULT = False, 0.3
    RUNNING_PROTOCO = True

    number_of_repeat = 5

    date_now = datetime.datetime.now()

    directory = "results/pre_distribution/" + date_now.strftime('%Y_%m_%d')

    if not os.path.exists(directory):
        os.makedirs(directory)
    saving_file_name = directory + "/results" + date_now.strftime('_%H_%M_%S')

    ################################################################################

    # example: python main_quantum_routing_args.py -dp -dpa -para num_nodes

    parser = argparse.ArgumentParser(description='Quantum network routing')

    # methods
    parser.add_argument('-s', '--single_path', action='store_true', help='If given, run the Single Greedy')
    parser.add_argument('-sns', '--single_non_shortest_sl', action='store_true',
                        help='If given, run the non-shortest Single')
    parser.add_argument('-snd', '--single_path_no_deletion', action='store_true',
                        help='If given, run the Single Greedy where no deletion (update) happens. (latency/cost)')
    parser.add_argument('-sndl', '--single_path_no_deletion_latency', action='store_true',
                        help='If given, run the Single Greedy where no deletion (update) happens. (latency)')
    parser.add_argument('-sn', '--single_path_naive', action='store_true', help='If given, run the Single_Naive')
    parser.add_argument('-sc', '--single_path_cluster', action='store_true', help='If given, run the Single Cluster')
    parser.add_argument('-m', '--multi_path_iter', action='store_true', help='If given, run the Multi_Iter')
    parser.add_argument('-mns', '--multi_non_shortest_sls_path_iter', action='store_true',
                        help='If given, run the Multi_Iter where SLs can also be non-shortest paths')
    parser.add_argument('-mn', '--multi_path_naive', action='store_true', help='If given, run the Multi_Naive')
    parser.add_argument('-line', '--line_graph', action='store_true', help='If given, create line graph')

    # parameter
    parser.add_argument('-para', '--parameter', type=str, nargs=1, default=['atomic_bsm'])

    args = parser.parse_args()

    # ********************** changing parameters (x-values) & algorithms used *******************
    # parameter : {"atomic_bsm", "optical_bsm", "decoherence_time", "num_nodes", "edge_density", "distance", "src_dst_pair"}
    IS_SINGLE = args.single_path  # single   path
    IS_NON_SHORTEST_SLS = args.single_non_shortest_sl
    IS_SINGLE_NO_DELETION = args.single_path_no_deletion
    IS_SINGLE_NO_DELETION_LATENCY = args.single_path_no_deletion_latency
    IS_SINGLE_NAIVE = args.single_path_naive  # single   path
    IS_SINGLE_CLUSTER = args.single_path_cluster  # multiple path
    IS_MULTI_ITER = args.multi_path_iter  # multiple path
    IS_MULTI_NON_SHORTEST_ITER = args.multi_non_shortest_sls_path_iter  # multiple path
    IS_MULTI_NAIVE = args.multi_path_naive  # multiple path
    IS_LINE = args.line_graph

    parameter = args.parameter[0]
    if IS_LINE:
        number_of_repeat = 1
        simulation_duration = 20
        if parameter == "distance":
            values = [300]  # [100, 300, 500, 700, 900, 1000]
        elif parameter == "link":
            values = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
        elif parameter == "cost_budget":
            values = [5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3]
        else:
            raise Exception(f'Oops! wrong parameter input: {parameter} for line_graph(only distance and link)')
    else:
        if parameter == 'atomic_bsm':
            values = [0.2, 0.3, 0.5, 0.6]
        elif parameter == 'num_nodes':
            values = [50, 75, 150, 200, 300]
        elif parameter == 'edge_density':
            values = [0.04, 0.06, 0.1, 0.12]
        elif parameter == 'distance':
            values = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 50), (50, 60)]
        elif parameter == 'src_dst_pair':
            values = [4, 8, 16, 20]
        elif parameter == "latency_threshold":
            values = [0.1, 0.2, 0.4, 0.5]
        elif parameter == "cost_budget":
            values = [5e3, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3]
        elif parameter == "decoherence_time":
            values = [0.1, 0.2, 0.3, 0.4]
        elif parameter == "timeslot":
            values = [0.1, 0.2, 0.6, 0.8]
        else:
            raise Exception(f'Oops! wrong parameter input: {parameter}')
    ################################################################################

    # ********************** changing parameters (x-values) & algorithms used *******************
    # parameter : {"atomic_bsm", "optical_bsm", "decoherence_time", "num_nodes", "edge_density", "distance", "src_dst_pair"}

    single_cost, single_satisfied_sd_num, single_latency_avg, single_latency_max, single_latency_avg_pro, \
    single_latency_max_pro, single_all_Latencies, single_sl_latency = \
        [[] for _ in range(len(values))], [0] * len(values), [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))]
    single_naive_cost, single_naive_satisfied_sd_num, single_naive_latency_avg, single_naive_latency_max, \
    single_naive_latency_avg_pro, single_naive_latency_max_pro, single_naive_all_Latencies, single_naive_sl_latency = \
        [[] for _ in range(len(values))], [0] * len(values), [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))]
    single_cluster_cost, single_cluster_satisfied_sd_num, single_cluster_latency_avg, single_cluster_latency_max, \
    single_cluster_latency_avg_pro, single_cluster_latency_max_pro, single_cluster_all_latencies, single_cluster_sl_latency = \
        [[] for _ in range(len(values))], [0] * len(values), [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))]
    multi_iter_cost, multi_iter_satisfied_sd_num, multi_iter_latency_avg, multi_iter_latency_max, \
    multi_iter_latency_avg_pro, multi_iter_latency_max_pro, multi_iter_all_latencies, multi_iter_sl_latency = \
        [[] for _ in range(len(values))], [0] * len(values), [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))]
    multi_naive_cost, multi_naive_satisfied_sd_num, multi_naive_latency_avg, multi_naive_latency_max, \
    multi_naive_latency_avg_pro, multi_naive_latency_max_pro, multi_naive_all_latencies, multi_naive_sl_latency = \
        [[] for _ in range(len(values))], [0] * len(values), [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [[] for _ in range(len(values))], \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))]

    with open(saving_file_name + ".txt", 'a') as file:
        file.write(parameter + ": " + str(values))
        if IS_LINE:
            file.write(f"\n******** LINE GRAPH *********\n")

    for value_idx, value in enumerate(values):
        if parameter == "num_nodes" and value == 100:
            continue
        seed(value_idx)
        print(f"\n\nTESTING {parameter}: {value}")
        with open(saving_file_name + ".txt", 'a') as file:
            file.write(f"\nTESTING {parameter}: {value}")
        # ******************* creating the network *****************

        total_number_of_sd_pairs = (value if parameter == "src_dst_pair" else default_number_of_src_dst_pair) * \
                                   number_of_repeat
        number_of_sl_memories = int(1.0 / value ** 2) if parameter == "atomic_bsm" else \
            int(1.0 / default_atomic_bsm_rate ** 2)
        max_sl_latency = ((value if parameter == "timeslot" else default_timeslot) / number_of_sl_memories) * 1.5
        simulation_duration = math.ceil((value if parameter == "timeslot" else default_timeslot)) * 100  # seconds

        for repeat_idx in range(number_of_repeat):
            seed(repeat_idx)
            print(f"\nRepeat #{repeat_idx}:")

            # # generating random src-dst pair
            srcs, dsts = [], []
            qNode._stat_id = 0  # need to reset qNode._stat_id for every new experiment
            if IS_LINE:
                graph0 = create_line_graph2(
                    total_length=value if parameter == "distance" else default_line_total_length,
                    min_link_Length=value[0] if parameter == "link"
                    else default_line_link_length[0],
                    max_link_length=value[1] if parameter == "link" else default_line_link_length[
                        1])
                srcs, dsts = [0], [len(graph0.V) - 1]
            else:
                graph0 = create_random_waxman(area_side=area_side, cell_size=cell_size,
                                              number_nodes=value if parameter == "num_nodes" else default_num_nodes,
                                              min_memory_size=min_memory_size, max_memory_size=max_memory_size,
                                              min_channel_num=min_channel_num, max_channel_num=max_channel_num,
                                              atomic_bsm_success_rate=value if parameter == "atomic_bsm"
                                              else default_atomic_bsm_rate,
                                              edge_density=value if parameter == "edge_density" else default_edge_density)
                connected_graph(graph0, min_channel_num=min_channel_num, max_channel_num=max_channel_num)
                # current_pairs = set()
                n = value if parameter == "num_nodes" else default_num_nodes
                min_distance = value[0] if parameter == "distance" else default_distance_range[0]
                max_distance = value[1] if parameter == "distance" else default_distance_range[1]
                ids_with_distance = [(x, y) for x in range(n) for y in range(x + 1, n) if
                                     (min_distance * 10 ** 3 <= graph0.V[x].loc.distance(graph0.V[y].loc) <=
                                      max_distance * 10 ** 3 and graph0.get_edge(graph0.V[x].id,
                                                                                 graph0.V[y].id) is None)]
                shuffle(ids_with_distance)
                # for i in range(min((value if parameter == "src_dst_pair" else default_number_of_src_dst_pair),
                #                    len(ids_with_distance))):
                for i in range(min((value if parameter == "src_dst_pair" else default_number_of_src_dst_pair),
                                   len(ids_with_distance))):
                    srcs.append(ids_with_distance[i][0])
                    dsts.append(ids_with_distance[i][1])
            # # generating random src-dst pair
            print(f"srcs = {srcs}, dsts = {dsts}")
            with open(saving_file_name + ".txt", 'a') as file:
                file.write(f"\nRepeat #: {repeat_idx}")
                file.write(f"\nsrcs = {srcs}, dsts = {dsts}")

            # print(f"srcs = {srcs}, dsts = {dsts}")

            # predist algo
            if IS_SINGLE or IS_NON_SHORTEST_SLS or IS_SINGLE_NO_DELETION or IS_SINGLE_NO_DELETION_LATENCY:
                if not IS_SINGLE_NO_DELETION and not IS_SINGLE_NO_DELETION_LATENCY:
                    print(f"\n***** SINGLE {'- (Non-Shortest SLs) ' if IS_NON_SHORTEST_SLS else ''}****")
                else:
                    print(f"\n***** SINGLE - (No-Deletion) - {'latency/cost ' if IS_SINGLE_NO_DELETION else 'latency'}"
                          f"****")
                graph_single = graph0.clone()
                start = datetime.datetime.now()
                single = greedy_predist(G=graph_single, srcs=[graph_single.V[src] for src in srcs],
                                        dsts=[graph_single.V[dst] for dst in dsts],
                                        latency_threshold=value if parameter == "latency_threshold"
                                        else LATENCY_THRESHOLD_DEFAULT,
                                        decoherence_time=value if parameter == "decoherence_time" else
                                        default_decoherence_time,
                                        FIXED_LATENCY=FIXED_LATENCY,
                                        NAIVE=False, max_sl_latency=max_sl_latency,
                                        IS_LATENCY_OBJECTIVE=IS_LATENCY_OBJECTIVE,
                                        cost_budget=value if parameter == "cost_budget" else default_cost_budget,
                                        NON_SHORTEST_SLS=IS_NON_SHORTEST_SLS,
                                        NO_DELETION_GREEDY=IS_SINGLE_NO_DELETION or IS_SINGLE_NO_DELETION_LATENCY,
                                        no_deletion_version='latency' if IS_SINGLE_NO_DELETION_LATENCY else 'latency/cost')
                stop = datetime.datetime.now()
                print(f"Running Time: {stop - start}")
                single_latency_avg[value_idx].append(single.average_latency)
                single_cost[value_idx].append(sum([sl.cost for sl in single.get_sls]))
                single_sl_latency[value_idx].append(sum([sl.tree.avr_ent_time for sl in single.get_sls]))
                single_satisfied_sd_num[value_idx] += len(srcs) / total_number_of_sd_pairs
                if RUNNING_PROTOCO:
                    single_protocol = PredistProtocol(graph=graph0, trees=single.get_final_paths,
                                                      SLs=single.get_sls,
                                                      sources=[graph_single.V[src] for src in srcs],
                                                      destinations=[graph_single.V[dst] for dst in dsts],
                                                      duration=simulation_duration,
                                                      NUMBER_OF_SL_MEMORIES=number_of_sl_memories + 1,
                                                      TIMESLOT=(value if parameter == "timeslot" else default_timeslot),
                                                      max_sl_latency=max_sl_latency)
                    sum_latencies = [sum(latency) / len(latency) for latency in single_protocol.latencies
                                     if len(latency) > 0]
                    max_latencies = [max(latency) for latency in single_protocol.latencies if len(latency) > 0]
                    single_latency_avg_pro[value_idx].append(sum(sum_latencies) / len(sum_latencies))
                    single_latency_max_pro[value_idx].append(max(sum_latencies))
                    single_all_Latencies[value_idx].append(single_protocol.latencies)
                    del single_protocol, single
                with open(saving_file_name + ".txt", "a") as file:
                    if not IS_SINGLE_NO_DELETION and not IS_SINGLE_NO_DELETION_LATENCY:
                        file.write(f"\n***** SINGLE {'- (Non-Shortest SLs) ' if IS_NON_SHORTEST_SLS else ''}****")
                    else:
                        file.write(
                            f"\n***** SINGLE - (No-Deletion) - {'latency/cost ' if IS_SINGLE_NO_DELETION else 'latency'}"
                            f"****")
                    file.write(f"\nRunning Time: {stop - start}")
                    file.write("\nSL costs: " + str(single_cost))
                    file.write("\nSL latencies: " + str(single_sl_latency))
                    file.write("\nSatisfied #: " + str(single_satisfied_sd_num))
                    file.write("\nAll latencies: " + str(single_all_Latencies))
                    file.write("\nAverage Latency (algorithm): " + str(single_latency_avg))
                    file.write("\nAverage Latency (protocol): " + str(single_latency_avg_pro))
                    file.write("\nMax Latency (Protocol): " + str(single_latency_max_pro) + "\n\n")

            if IS_SINGLE_NAIVE:
                print(f"\n***** SINGLE-NAIVE ****")
                graph_single_naive = graph0.clone()
                start = datetime.datetime.now()
                single_naive = greedy_predist(G=graph_single_naive, srcs=[graph_single_naive.V[src] for src in srcs],
                                              dsts=[graph_single_naive.V[dst] for dst in dsts],
                                              latency_threshold=value if parameter == "latency_threshold"
                                              else LATENCY_THRESHOLD_DEFAULT,
                                              decoherence_time=value if parameter == "decoherence_time" else
                                              default_decoherence_time,
                                              FIXED_LATENCY=FIXED_LATENCY,
                                              NAIVE=True, max_sl_latency=max_sl_latency,
                                              IS_LATENCY_OBJECTIVE=IS_LATENCY_OBJECTIVE,
                                              cost_budget=value if parameter == "cost_budget" else default_cost_budget)

                stop = datetime.datetime.now()
                print(f"Running Time: {stop - start}")
                single_naive_latency_avg[value_idx].append(single_naive.average_latency)
                single_naive_cost[value_idx].append(sum([sl.cost for sl in single_naive.get_sls]))
                single_naive_sl_latency[value_idx].append(sum([sl.tree.avr_ent_time for sl in
                                                               single_naive.get_sls]))
                single_naive_satisfied_sd_num[value_idx] += len(srcs) / total_number_of_sd_pairs
                if RUNNING_PROTOCO:
                    single_naive_protocol = PredistProtocol(graph=graph0, trees=single_naive.get_final_paths,
                                                            SLs=single_naive.get_sls,
                                                            sources=[graph_single_naive.V[src] for src in srcs],
                                                            destinations=[graph_single_naive.V[dst] for dst in dsts],
                                                            duration=simulation_duration,
                                                            NUMBER_OF_SL_MEMORIES=number_of_sl_memories + 1,
                                                            TIMESLOT=(value if parameter == "timeslot" else
                                                                      default_timeslot), max_sl_latency=max_sl_latency)
                    sum_latencies = [sum(latency) / len(latency) for latency in single_naive_protocol.latencies
                                     if len(latency) > 0]
                    max_latencies = [max(latency) for latency in single_naive_protocol.latencies if len(latency) > 0]
                    single_naive_all_Latencies[value_idx].append(single_naive_protocol.latencies)
                    single_naive_latency_avg_pro[value_idx].append(sum(sum_latencies) / len(sum_latencies))
                    single_naive_latency_max_pro[value_idx].append(max(sum_latencies))
                    del single_naive, single_naive_protocol
                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\n***** SINGLE-NAIVE ****")
                    file.write(f"\nRunning Time: {stop - start}")
                    file.write("\nSL costs: " + str(single_naive_cost))
                    file.write("\nSL latencies: " + str(single_naive_sl_latency))
                    file.write("\nSatisfied #: " + str(single_naive_satisfied_sd_num))
                    file.write("\nAll latencies: " + str(single_naive_all_Latencies))
                    file.write("\nAverage Latency (algorithm): " + str(single_naive_latency_avg))
                    file.write("\nAverage Latency (protocol): " + str(single_naive_latency_avg_pro))
                    file.write("\nMax Latency (Protocol): " + str(single_naive_latency_max_pro) + "\n\n")

            if IS_SINGLE_CLUSTER:
                print(f"\n***** SINGLE-CLUSTER ****")
                graph_single_cluster = graph0.clone()
                start = datetime.datetime.now()
                single_cluster = cluster_predist(G=graph_single_cluster,
                                                 srcs=[graph_single_cluster.V[src] for src in srcs],
                                                 dsts=[graph_single_cluster.V[dst] for dst in dsts],
                                                 latency_threshold=value if parameter == "latency_threshold"
                                                 else LATENCY_THRESHOLD_DEFAULT, FIXED_LATENCY=FIXED_LATENCY,
                                                 max_sl_latency=max_sl_latency,
                                                 IS_LATENCY_OBJECTIVE=IS_LATENCY_OBJECTIVE,
                                                 cost_budget=value if parameter == "cost_budget" else
                                                 default_cost_budget,
                                                 decoherence_time=value if parameter == "decoherence_time" else
                                                 default_decoherence_time)
                stop = datetime.datetime.now()
                print(f"Running Time: {stop - start}")
                single_cluster_latency_avg[value_idx].append(single_cluster.average_latency)
                single_cluster_cost[value_idx].append(sum([sl.cost for sl in single_cluster.get_sls]))
                single_cluster_sl_latency[value_idx].append(sum([sl.tree.avr_ent_time for sl in
                                                                 single_cluster.get_sls]))
                single_cluster_satisfied_sd_num[value_idx] += len(srcs) / total_number_of_sd_pairs
                if len(single_cluster.get_sls) != 0 or True:
                    single_cluster_protocol = PredistProtocol(graph=graph0, trees=single_cluster.get_final_paths,
                                                              SLs=single_cluster.get_sls,
                                                              sources=[graph_single_cluster.V[src] for src in srcs],
                                                              destinations=[graph_single_cluster.V[dst] for dst in
                                                                            dsts],
                                                              duration=simulation_duration,
                                                              TIMESLOT=(value if parameter == "timeslot" else
                                                                        default_timeslot),
                                                              NUMBER_OF_SL_MEMORIES=number_of_sl_memories + 1,
                                                              max_sl_latency=max_sl_latency)
                    sum_latencies = [sum(latency) / len(latency) for latency in single_cluster_protocol.latencies
                                     if len(latency) > 0]
                    max_latencies = [max(latency) for latency in single_cluster_protocol.latencies if len(latency) > 0]
                    single_cluster_all_latencies[value_idx].append(single_cluster_protocol.latencies)
                    single_cluster_latency_avg_pro[value_idx].append(sum(sum_latencies) / len(sum_latencies))
                    single_cluster_latency_max_pro[value_idx].append(max(sum_latencies))
                    del single_cluster, single_cluster_protocol
                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\n***** SINGLE-CLUSTER ****")
                    file.write(f"\nRunning Time: {stop - start}")
                    file.write("\nSL costs: " + str(single_cluster_cost))
                    file.write("\nSL latencies: " + str(single_cluster_sl_latency))
                    file.write("\nSatisfied #: " + str(single_cluster_satisfied_sd_num))
                    file.write("\nAll latencies: " + str(single_cluster_all_latencies))
                    file.write("\nAverage Latency (algorithm): " + str(single_cluster_latency_avg))
                    file.write("\nAverage Latency (protocol): " + str(single_cluster_latency_avg_pro))
                    file.write("\nMax Latency (Protocol): " + str(single_cluster_latency_max_pro) + "\n\n")

            if IS_MULTI_ITER or IS_MULTI_NON_SHORTEST_ITER:
                print(f"\n***** MULTI-ITER  {'- (Non-Shortest SLs) ' if IS_MULTI_NON_SHORTEST_ITER else ''}****")
                graph_multi_iter = graph0.clone()
                start = datetime.datetime.now()
                multi_iter = multi_pair_greedy_predist(G=graph_multi_iter,
                                                       srcs=[graph_multi_iter.V[src] for src in srcs],
                                                       dsts=[graph_multi_iter.V[dst] for dst in dsts],
                                                       latency_threshold=value if parameter == "latency_threshold"
                                                       else LATENCY_THRESHOLD_DEFAULT,
                                                       FIXED_LATENCY=FIXED_LATENCY, IS_NAIVE=False,
                                                       max_sl_latency=max_sl_latency,
                                                       IS_LATENCY_OBJECTIVE=IS_LATENCY_OBJECTIVE,
                                                       cost_budget=value if parameter == "cost_budget" else
                                                       default_cost_budget,
                                                       decoherence_time=value if parameter == "decoherence_time" else
                                                       default_decoherence_time,
                                                       NON_SHORTEST_SLS=IS_MULTI_NON_SHORTEST_ITER)
                stop = datetime.datetime.now()
                print(f"Running Time: {stop - start}")
                multi_iter_latency_avg[value_idx].append(multi_iter.average_latency)
                multi_iter_cost[value_idx].append(sum([sl.cost for sl in multi_iter.get_sls]))
                multi_iter_sl_latency[value_idx].append(sum([sl.tree.avr_ent_time for sl in multi_iter.get_sls]))
                multi_iter_satisfied_sd_num[value_idx] += len(srcs) / total_number_of_sd_pairs
                if RUNNING_PROTOCO:
                    multi_iter_protocol = PredistProtocol(graph=graph0, trees=multi_iter.get_final_paths,
                                                          SLs=multi_iter.get_sls,
                                                          sources=[graph_multi_iter.V[src] for src in srcs],
                                                          destinations=[graph_multi_iter.V[dst] for dst in dsts],
                                                          duration=simulation_duration,
                                                          TIMESLOT=(value if parameter == "timeslot" else
                                                                    default_timeslot),
                                                          NUMBER_OF_SL_MEMORIES=number_of_sl_memories + 1,
                                                          max_sl_latency=max_sl_latency)
                    sum_latencies = [sum(latency) / len(latency) for latency in multi_iter_protocol.latencies
                                     if len(latency) > 0]
                    max_latencies = [max(latency) for latency in multi_iter_protocol.latencies if len(latency) > 0]
                    multi_iter_all_latencies[value_idx].append(multi_iter_protocol.latencies)
                    multi_iter_latency_avg_pro[value_idx].append(sum(sum_latencies) / len(sum_latencies))
                    multi_iter_latency_max_pro[value_idx].append(max(sum_latencies))
                    del multi_iter, multi_iter_protocol
                with open(saving_file_name + ".txt", "a") as file:
                    file.write(
                        f"\n***** MULTI-ITER  {'- (Non-Shortest SLs) ' if IS_MULTI_NON_SHORTEST_ITER else ''}****")
                    file.write(f"\nRunning Time: {stop - start}")
                    file.write("\nSL costs: " + str(multi_iter_cost))
                    file.write("\nSL latencies: " + str(multi_iter_sl_latency))
                    file.write("\nSatisfied #: " + str(multi_iter_satisfied_sd_num))
                    file.write("\nAll latencies: " + str(multi_iter_all_latencies))
                    file.write("\nAverage Latency (algorithm): " + str(multi_iter_latency_avg))
                    file.write("\nAverage Latency (protocol): " + str(multi_iter_latency_avg_pro))
                    file.write("\nMax Latency (Protocol): " + str(multi_iter_latency_max_pro) + "\n\n")

            if IS_MULTI_NAIVE:
                print(f"\n***** MULTI-NAIVE ****")
                graph_multi_naive = graph0.clone()
                start = datetime.datetime.now()
                multi_naive = multi_pair_greedy_predist(G=graph_multi_naive,
                                                        srcs=[graph_multi_naive.V[src] for src in srcs],
                                                        dsts=[graph_multi_naive.V[dst] for dst in dsts],
                                                        latency_threshold=value if parameter == "latency_threshold"
                                                        else LATENCY_THRESHOLD_DEFAULT,
                                                        decoherence_time=value if parameter == "decoherence_time" else
                                                        default_decoherence_time,
                                                        FIXED_LATENCY=FIXED_LATENCY, IS_NAIVE=True,
                                                        max_sl_latency=max_sl_latency,
                                                        IS_LATENCY_OBJECTIVE=IS_LATENCY_OBJECTIVE,
                                                        cost_budget=value if parameter == "cost_budget" else
                                                        default_cost_budget)
                stop = datetime.datetime.now()
                print(f"Running Time: {stop - start}")
                multi_naive_latency_avg[value_idx].append(multi_pair_greedy_predist.average_latency)
                multi_naive_cost[value_idx].append(sum([sl.cost for sl in multi_naive.get_sls]))
                multi_naive_sl_latency[value_idx].append(sum([sl.tree.avr_ent_time for sl in multi_naive.get_sls]))
                multi_naive_satisfied_sd_num[value_idx] += len(srcs) / total_number_of_sd_pairs
                if RUNNING_PROTOCO:
                    multi_naive_protocol = PredistProtocol(graph=graph0, trees=multi_naive.get_final_paths,
                                                           SLs=multi_naive.get_sls,
                                                           sources=[graph_multi_naive.V[src] for src in srcs],
                                                           destinations=[graph_multi_naive.V[dst] for dst in dsts],
                                                           duration=simulation_duration,
                                                           TIMESLOT=(value if parameter == "timeslot" else
                                                                     default_timeslot),
                                                           NUMBER_OF_SL_MEMORIES=number_of_sl_memories + 1,
                                                           max_sl_latency=max_sl_latency)
                    sum_latencies = [sum(latency) / len(latency) for latency in multi_naive_protocol.latencies
                                     if len(latency) > 0]
                    max_latencies = [max(latency) for latency in multi_naive_protocol.latencies if len(latency) > 0]
                    multi_naive_all_latencies[value_idx].append(multi_naive_protocol.latencies)
                    multi_naive_latency_avg_pro[value_idx].append(sum(sum_latencies) / len(sum_latencies))
                    multi_naive_latency_max_pro[value_idx].append(max(sum_latencies))
                    del multi_naive, multi_naive_protocol
                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\n***** Multi-Naive ****")
                    file.write(f"\nRunning Time: {stop - start}")
                    file.write("\nSL costs: " + str(multi_naive_cost))
                    file.write("\nSL latencies: " + str(multi_naive_sl_latency))
                    file.write("\nSatisfied #: " + str(multi_naive_satisfied_sd_num))
                    file.write("\nAll latencies: " + str(multi_naive_all_latencies))
                    file.write("\nAverage Latency (algorithm): " + str(multi_naive_latency_avg))
                    file.write("\nAverage Latency (protocol): " + str(multi_naive_latency_avg_pro))
                    file.write("\nMax Latency (Protocol): " + str(multi_naive_latency_max_pro) + "\n\n")

    # SUMMARY
    with open(saving_file_name + ".txt", 'a') as file:
        file.write(f"\n\n\n*********Summary*******")
        file.write(f"\n\nTESTING {parameter}: {values}")

    if IS_SINGLE or IS_NON_SHORTEST_SLS or IS_SINGLE_NO_DELETION or IS_SINGLE_NO_DELETION_LATENCY:
        with open(saving_file_name + ".txt", "a") as file:
            if not IS_SINGLE_NO_DELETION and not IS_SINGLE_NO_DELETION_LATENCY:
                file.write(f"\n***** SINGLE {'- (Non-Shortest SLs) ' if IS_NON_SHORTEST_SLS else ''}****")
            else:
                file.write(
                    f"\n***** SINGLE - (No-Deletion) - {'latency/cost ' if IS_SINGLE_NO_DELETION else 'latency'}"
                    f"****")
            file.write("\nSL costs: " + str([round(sum(cost) / len(cost), 3) for cost in single_cost if len(cost) > 0]))
            file.write("\nSL latencies: " + str([round(sum(sl_latency) / len(sl_latency), 3) for sl_latency in
                                                 single_sl_latency if len(sl_latency) > 0]))
            file.write("\nSatisfied #: " + str(single_satisfied_sd_num))
            file.write("\nAverage Latency (algorithm): " + str([round(sum(avg) / len(avg), 3) for avg in
                                                                single_latency_avg if len(avg) > 0]))
            file.write("\nAverage Latency (protocol): " + str([round(sum(avg) / (len(avg) * 1e9), 3) for avg in
                                                               single_latency_avg_pro if len(avg) > 0]))
            file.write("\nMax Latency (Protocol): " + str([round(sum(max_) / (len(max_) * 1e9), 3) for max_ in
                                                           single_latency_max_pro if len(max_) > 0]))

    if IS_SINGLE_NAIVE:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n\n***** SINGLE-NAIVE ****")
            file.write("\nSL costs: " + str([round(sum(cost) / len(cost), 3) for cost in single_naive_cost
                                             if len(cost) > 0]))
            file.write("\nSL latencies: " + str([round(sum(sl_latency) / len(sl_latency), 3) for sl_latency in
                                                 single_naive_sl_latency if len(sl_latency) > 0]))
            file.write("\nSatisfied #: " + str(single_naive_satisfied_sd_num))
            file.write("\nAverage Latency (algorithm): " + str([round(sum(avg) / len(avg), 3) for avg in
                                                                single_naive_latency_avg if len(avg) > 0]))
            file.write("\nAverage Latency (protocol): " + str([round(sum(avg) / (len(avg) * 1e9), 3) for avg in
                                                               single_naive_latency_avg_pro if len(avg) > 0]))
            file.write("\nMax Latency (Protocol): " + str([round(sum(max_) / (len(max_) * 1e9), 3) for max_ in
                                                           single_naive_latency_max_pro if len(max_) > 0]))

    if IS_SINGLE_CLUSTER:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n\n***** SINGLE-CLUSTER ****")
            file.write("\nSL costs: " + str([round(sum(cost) / len(cost), 3) for cost in single_cluster_cost
                                             if len(cost) > 0]))
            file.write("\nSL latencies: " + str([round(sum(sl_latency) / len(sl_latency), 3) for sl_latency in
                                                 single_cluster_sl_latency if len(sl_latency) > 0]))
            file.write("\nSatisfied #: " + str(single_naive_satisfied_sd_num))
            file.write("\nAverage Latency (algorithm): " + str([round(sum(avg) / len(avg), 3) for avg in
                                                                single_cluster_latency_avg if len(avg) > 0]))
            file.write("\nAverage Latency (protocol): " + str([round(sum(avg) / (len(avg) * 1e9), 3) for avg in
                                                               single_cluster_latency_avg_pro if len(avg) > 0]))
            file.write("\nMax Latency (Protocol): " + str([round(sum(max_) / (len(max_) * 1e9), 3) for max_ in
                                                           single_cluster_latency_max_pro if len(max_) > 0]))

    if IS_MULTI_ITER or IS_MULTI_NON_SHORTEST_ITER:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n***** MULTI-ITER  {'- (Non-Shortest SLs) ' if IS_MULTI_NON_SHORTEST_ITER else ''}****")
            file.write("\nSL costs: " + str([round(sum(cost) / len(cost), 3) for cost in multi_iter_cost
                                             if len(cost) > 0]))
            file.write("\nSL latencies: " + str([round(sum(sl_latency) / len(sl_latency), 3) for sl_latency in
                                                 multi_iter_sl_latency if len(sl_latency) > 0]))
            file.write("\nSatisfied #: " + str(multi_iter_satisfied_sd_num))
            file.write("\nAverage Latency (algorithm): " + str([round(sum(avg) / len(avg), 3) for avg in
                                                                multi_iter_latency_avg if len(avg) > 0]))
            file.write("\nAverage Latency (protocol): " + str([round(sum(avg) / (len(avg) * 1e9), 3) for avg in
                                                               multi_iter_latency_avg_pro if len(avg) > 0]))
            file.write("\nMax Latency (Protocol): " + str([round(sum(max_) / (len(max_) * 1e9), 3) for max_ in
                                                           multi_iter_latency_max_pro if len(max_) > 0]))

    if IS_MULTI_NAIVE:
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"\n\n***** MULTI-NAIVE ****")
            file.write("\nSL costs: " + str([round(sum(cost) / len(cost), 3) for cost in multi_naive_cost
                                             if len(cost) > 0]))
            file.write("\nSL latencies: " + str([round(sum(sl_latency) / len(sl_latency), 3) for sl_latency in
                                                 multi_naive_sl_latency if len(sl_latency) > 0]))
            file.write("\nSatisfied #: " + str(multi_naive_satisfied_sd_num))
            file.write("\nAverage Latency (algorithm): " + str([round(sum(avg) / len(avg), 3) for avg in
                                                                multi_naive_latency_avg if len(avg) > 0]))
            file.write("\nAverage Latency (protocol): " + str([round(sum(avg) / (len(avg) * 1e9), 3) for avg in
                                                               multi_naive_latency_avg_pro if len(avg) > 0]))
            file.write("\nMax Latency (Protocol): " + str([round(sum(max_) / (len(max_) * 1e9), 3) for max_ in
                                                           multi_naive_latency_max_pro if len(max_) > 0]))


def main_calefi():
    # **************** init values ************
    area_side = 50  # a square side of side x side cells
    cell_size = 1000  # side value of each square side (m)
    min_memory_size, max_memory_size = 15, 20
    min_channel_num, max_channel_num = 1, 1  # number of parallel channels a connection would have
    default_number_of_src_dst_pair = 1
    default_decoherence_time = 5  # seconds
    default_atomic_bsm_rate, default_optical_bsm_rate = 0.4, 0.2
    default_edge_density = 0.40
    default_num_nodes = 15
    default_distance_range = [30, 100]
    default_line_total_length = 500  # kms
    default_line_link_length = (30, 35)  # kms
    IS_TELEPORTING = False  # defines if we send information from Alice to Bob after tree construction in protocols

    simulation_duration = 30  # seconds
    number_of_repeat = 10
    DP_OPT = True
    NON_THROTTLING = False
    RUNNING_PROTOCOL = True

    date_now = datetime.datetime.now()

    directory = "results/" + date_now.strftime('%Y_%m_%d')

    if not os.path.exists(directory):
        os.makedirs(directory)
    saving_file_name = directory + "/results" + date_now.strftime('_%H_%M_%S')

    ################################################################################

    # example: python main_quantum_routing_args.py -dp -dpa -para num_nodes

    parser = argparse.ArgumentParser(description='Quantum network routing')

    # methods
    parser.add_argument('-dp', '--dynamic_programming', action='store_true', help='If given, run the DP')
    parser.add_argument('-dpa', '--dynamic_programming_alternate', action='store_true', help='If given, run the DP_ALT')
    parser.add_argument('-c', '--calefi', action='store_true', help='If given, run the CAL')
    parser.add_argument('-line', '--line_graph', action='store_true', help='If given, create line graph')

    # parameter
    parser.add_argument('-para', '--parameter', type=str, nargs=1, default=['atomic_bsm'])

    args = parser.parse_args()

    # ********************** changing parameters (x-values) & algorithms used *******************
    # parameter : {"atomic_bsm", "optical_bsm", "decoherence_time", "num_nodes", "edge_density", "distance", "src_dst_pair"}
    IS_DP = args.dynamic_programming  # single   path
    IS_DP_ALT = args.dynamic_programming_alternate  # single   path
    IS_LINE = args.line_graph
    IS_CALEFI = args.calefi

    parameter = args.parameter[0]
    if IS_LINE:
        number_of_repeat = 1
        simulation_duration = 30
        if parameter == "distance":
            values = [100, 300, 500, 700, 900, 1000]
        elif parameter == "link":
            values = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
        else:
            raise Exception(f'Oops! wrong parameter input: {parameter} for line_graph(only distance and link)')
    else:
        if parameter == 'atomic_bsm':
            values = [0.2, 0.3, 0.4, 0.5, 0.6]
        elif parameter == 'num_nodes':
            values = [20]
        elif parameter == 'edge_density':
            values = [0.2, 0.3, 0.5]
        elif parameter == 'distance':
            values = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 50), (50, 60)]
        elif parameter == 'src_dst_pair':
            values = [1, 2, 4, 5]
        else:
            raise Exception(f'Oops! wrong parameter input: {parameter}')
    ################################################################################

    dp_res, dp_protocol_res, dp_res_avg, dp_protocol_res_avg, dp_fidelity, dp_fidelity_avg, dp_time = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values), [[] for _ in range(len(values))]
    dp_alt_res, dp_alt_protocol_res, dp_alt_res_avg, dp_alt_protocol_res_avg, dp_alt_fidelity, dp_alt_fidelity_avg, \
    dp_alt_time = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values), [[] for _ in range(len(values))]
    calefi_res, calefi_protocol_res, calefi_res_avg, calefi_protocol_res_avg, calefi_fidelity, calefi_fidelity_avg, \
    calefi_time = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values), [[] for _ in range(len(values))]

    with open(saving_file_name + ".txt", 'a') as file:
        file.write(parameter + ": " + str(values))
        if IS_LINE:
            file.write(f"\n******** LINE GRAPH *********\n")
    for value_idx, value in enumerate(values):
        seed(value_idx)
        teleportation_bsm_rate = value if parameter == "atomic_bsm" else BSM_SUCCESS_RATE
        print(f"\n\nTESTING {parameter}: {value}")
        with open(saving_file_name + ".txt", 'a') as file:
            file.write(f"\nTESTING {parameter}: {value}")

        for repeat_idx in range(number_of_repeat):
            seed(repeat_idx)
            print(f"Repeat #{repeat_idx}:")
            # ******************* creating the network *****************
            qNode._stat_id = 0  # need to reset qNode._stat_id for every new experiment
            if not IS_LINE:
                graph0 = create_random_waxman(area_side=area_side, cell_size=cell_size,
                                              number_nodes=value if parameter == "num_nodes" else default_num_nodes,
                                              min_memory_size=min_memory_size, max_memory_size=max_memory_size,
                                              min_channel_num=min_channel_num, max_channel_num=max_channel_num,
                                              atomic_bsm_success_rate=value if parameter == "atomic_bsm"
                                              else default_atomic_bsm_rate,
                                              edge_density=value if parameter == "edge_density" else default_edge_density)
                connected_graph(graph0, min_channel_num=min_channel_num, max_channel_num=max_channel_num)
            else:
                graph0 = create_line_graph2(
                    total_length=value if parameter == "distance" else default_line_total_length,
                    min_link_Length=value[0] if parameter == "link"
                    else default_line_link_length[0],
                    max_link_length=value[1] if parameter == "link" else default_line_link_length[
                        1])
            # # generating random src-dst pair
            srcs, dsts = [], []
            if not IS_LINE:
                n = value if parameter == "num_nodes" else default_num_nodes
                min_distance = value[0] if parameter == "distance" else default_distance_range[0]
                max_distance = value[1] if parameter == "distance" else default_distance_range[1]
                ids_with_distance = [(x, y) for x in range(n) for y in range(x + 1, n) if
                                     (min_distance * 10 ** 3 <= graph0.V[x].loc.distance(graph0.V[y].loc) <=
                                      max_distance * 10 ** 3 and graph0.get_edge(graph0.V[x].id,
                                                                                 graph0.V[y].id) is None)]
                shuffle(ids_with_distance)
                for i in range(min((value if parameter == "src_dst_pair" else default_number_of_src_dst_pair),
                                   len(ids_with_distance))):
                    srcs.append(ids_with_distance[i][0])
                    dsts.append(ids_with_distance[i][1])
            else:
                srcs.append(0)
                dsts.append(len(graph0.V) - 1)
            print(f"srcs = {srcs}, dsts = {dsts}")
            with open(saving_file_name + ".txt", 'a') as file:
                file.write(f"\nRepeat #: {repeat_idx}")
                file.write(f"\nsrcs = {srcs}, dsts = {dsts}")

            if IS_DP:
                graph_dp = graph0.clone()
                start = time.time()
                dp_shortest = dp(graph_dp, value if parameter == "decoherence_time" else default_decoherence_time,
                                 USAGE=DP_OPT)
                dp_shortest_path = dp_shortest.get_shortest_path(graph_dp.V[srcs[0]], graph_dp.V[dsts[0]],
                                                                 NON_THROTTLING=NON_THROTTLING)
                dp_time[value_idx].append(time.time() - start)
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / dp_shortest_path.avr_ent_time
                print(f"\n\nDP: {alg_result:.3f} (EP/s)", ("-OPTIMAL" if DP_OPT else ""),
                      ("-NON_THROTTLED" if NON_THROTTLING else ""), f"Time: {round(dp_time[value_idx][-1])} s")
                tree_node.print_tree(dp_shortest_path)
                with open(saving_file_name + ".txt", 'a') as file:
                    file.write("\n**** DP ***" + ("-OPTIMAL" if DP_OPT else "") +
                               ("-NON_THROTTLED" if NON_THROTTLING else ""))
                    file.write(tree_node.print_tree_str(dp_shortest_path))
                if RUNNING_PROTOCOL:
                    network_protocol_dp = TreeProtocol(graph=graph0, trees=[dp_shortest_path],
                                                       sources=[graph_dp.V[srcs[0]]],
                                                       destinations=[graph_dp.V[dsts[0]]],
                                                       duration=simulation_duration,
                                                       IS_TELEPORTING=IS_TELEPORTING)

                    print(f"DP (protocol): {network_protocol_dp.success_rate[0]} (EP/s), avg_fidelity: "
                          f"{network_protocol_dp.avg_fidelity[0]} "
                          f"\nduration={simulation_duration}s")
                    dp_protocol_res[value_idx].append(network_protocol_dp.success_rate[0])
                    dp_fidelity[value_idx].append(network_protocol_dp.avg_fidelity[0])
                    dp_protocol_res_avg[value_idx] = sum(dp_protocol_res[value_idx]) / len(dp_protocol_res[value_idx])
                    dp_fidelity_avg[value_idx] = weighted_avg(dp_fidelity[value_idx])[1]
                    del network_protocol_dp
                dp_res[value_idx].append(alg_result)
                dp_res_avg[value_idx] = sum(dp_res[value_idx]) / len(dp_res[value_idx])
                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\nTime: {str(dp_time)}")
                    file.write("\nAlgorithm results: " + str(dp_res))
                    file.write("\nProtocol results: " + str(dp_protocol_res))
                    file.write("\nFidelity results: " + str(dp_fidelity))
                    file.write("\nAlgorithm results (average): " + str(dp_res_avg))
                    file.write("\nProtocol results (average): " + str(dp_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(dp_fidelity_avg) + "\n\n")

            # DP Alternate
            if IS_DP_ALT:
                graph_dp_alt = graph0.clone()
                dp_alt = DP_Alt(graph_dp_alt)
                metric = 'time'
                src, dst = graph_dp_alt.V[srcs[0]], graph_dp_alt.V[dsts[0]]
                start_time = time.time()
                path = dp_alt.dijsktra(src, dst, metric=metric)
                dp_alt_tree = dp_alt.path2tree(path,
                                               decoherence_time=value if parameter == "decoherence_time" else default_decoherence_time,
                                               update_time=True)
                dp_alt_time[value_idx].append(time.time() - start_time)
                if dp_alt_tree is None:
                    raise Exception('Oops! DP alternate tree construction failed :(')

                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / dp_alt_tree.avr_ent_time
                print(f"\n\n-->DP Alternate: {alg_result:.3f} (EP/s), Time: {round(dp_time[value_idx][-1])} s")
                tree_node.print_tree(dp_alt_tree)
                with open(saving_file_name + ".txt", 'a') as file:
                    file.write("\n**** DP Alternate ***")
                    file.write(tree_node.print_tree_str(dp_alt_tree))
                if RUNNING_PROTOCOL:
                    network_protocol_dp_alt = TreeProtocol(graph0, [dp_alt_tree], [src], [dst], simulation_duration,
                                                           IS_TELEPORTING=IS_TELEPORTING, CHECK_CAPACITIES=True)
                    print(f"DP Alternate (protocol): {network_protocol_dp_alt.success_rate[0]} (EP/s), "
                          f"avg_fidelity: {network_protocol_dp_alt.avg_fidelity[0]} "
                          f"\nduration={simulation_duration}s")
                    dp_alt_protocol_res[value_idx].append(network_protocol_dp_alt.success_rate[0])
                    dp_alt_fidelity[value_idx].append(network_protocol_dp_alt.avg_fidelity[0])
                    dp_alt_protocol_res_avg[value_idx] = sum(dp_alt_protocol_res[value_idx]) / len(
                        dp_alt_protocol_res[value_idx])
                    dp_alt_fidelity_avg[value_idx] = weighted_avg(dp_alt_fidelity[value_idx])[1]
                    del network_protocol_dp_alt
                dp_alt_res[value_idx].append(alg_result)
                dp_alt_res_avg[value_idx] = sum(dp_alt_res[value_idx]) / len(dp_alt_res[value_idx])

                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\nTime: {str(dp_alt_time)}")
                    file.write("\nAlgorithm results: " + str(dp_alt_res))
                    file.write("\nProtocol results: " + str(dp_alt_protocol_res))
                    file.write("\nFidelity results: " + str(dp_alt_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(dp_alt_res_avg))
                    file.write("\nProtocol results (Average): " + str(dp_alt_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(dp_alt_fidelity_avg) + "\n\n")

            if IS_CALEFI:
                graph_calefi = graph0.clone()
                start = time.time()
                calefi_optimal = optimal_balanced_tree(graph_calefi,
                                                       value if parameter == "decoherence_time" else
                                                       default_decoherence_time)
                calefi_best_tree = calefi_optimal.get_best_tree(graph_calefi.V[srcs[0]], graph_calefi.V[dsts[0]])
                calefi_time[value_idx].append(time.time() - start)
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / calefi_best_tree.avr_ent_time
                print(f"\n\nCalefi: {alg_result:.3f} (EP/s), Time: {round(calefi_time[value_idx][-1])} s")
                tree_node.print_tree(calefi_best_tree)
                with open(saving_file_name + ".txt", 'a') as file:
                    file.write("\n**** CALEFI ***")
                    file.write(tree_node.print_tree_str(calefi_best_tree))
                if RUNNING_PROTOCOL:
                    network_protocol_calefi = TreeProtocol(graph=graph0, trees=[calefi_best_tree],
                                                           sources=[graph_calefi.V[srcs[0]]],
                                                           destinations=[graph_calefi.V[dsts[0]]],
                                                           duration=simulation_duration,
                                                           IS_TELEPORTING=IS_TELEPORTING)

                    print(f"CALEFI (protocol): {network_protocol_calefi.success_rate[0]} (EP/s), avg_fidelity: "
                          f"{network_protocol_calefi.avg_fidelity[0]} "
                          f"\nduration={simulation_duration}s")
                    calefi_protocol_res[value_idx].append(network_protocol_calefi.success_rate[0])
                    calefi_fidelity[value_idx].append(network_protocol_calefi.avg_fidelity[0])
                    calefi_protocol_res_avg[value_idx] = sum(calefi_protocol_res[value_idx]) / len(
                        calefi_protocol_res[value_idx])
                    calefi_fidelity_avg[value_idx] = weighted_avg(calefi_fidelity[value_idx])[1]
                    del network_protocol_calefi
                calefi_res[value_idx].append(alg_result)

                calefi_res_avg[value_idx] = sum(calefi_res[value_idx]) / len(calefi_res[value_idx])

                with open(saving_file_name + ".txt", "a") as file:
                    file.write(f"\nTime: {str(calefi_time)}")
                    file.write("\nAlgorithm results: " + str(calefi_res))
                    file.write("\nProtocol results: " + str(calefi_protocol_res))
                    file.write("\nFidelity results: " + str(calefi_fidelity))
                    file.write("\nAlgorithm results (average): " + str(calefi_res_avg))
                    file.write("\nProtocol results (average): " + str(calefi_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(calefi_fidelity_avg) + "\n\n")


def main_ghz():
    seed(10)
    # **************** init values ************
    area_side = 10000  # a square side of side x side cells
    cell_size = 10  # side value of each square side (m)
    min_memory_size, max_memory_size = 15, 20
    min_node_degree, max_node_degree = 2, 4  # number of edges from one node to its neighbors
    min_channel_num, max_channel_num = 1, 1  # number of parallel channels a connection would have

    default_num_end_nodes = 5
    default_decoherence_time = 5  # seconds
    default_atomic_bsm_rate, default_optical_bsm_rate = 0.4, 0.2
    default_fusion_success_rate = 0.4
    default_edge_density = 0.08
    default_num_nodes = 100
    default_line_total_length = 1000  # kms
    default_line_link_length = (30, 35)  # kms
    # default_distance_range = [30, 40] if default_number_of_src_dst_pair == 1 else [10,
    #                                                                                70]  # minimum distance between src-dst in km
    # IS_TELEPORTING = False  # defines if we send information from Alice to Bob after tree construction in protocols

    simulation_duration = 40  # seconds
    IS_PROTOCOL = False
    number_of_repeat = 10
    NON_THROTTLING = True

    date_now = datetime.datetime.now()

    directory = "results/ghz/" + date_now.strftime('%Y_%m_%d')

    if not os.path.exists(directory):
        os.makedirs(directory)
    saving_file_name = directory + "/results" + date_now.strftime('_%H_%M_%S')

    ################################################################################

    # example: python main_quantum_routing_args.py -dp -dpa -para num_nodes

    parser = argparse.ArgumentParser(description='Quantum network routing')

    # methods
    parser.add_argument('-fro', '--fusion_retain_only', action='store_true', help='If given, run FUSION_RETAIN')
    parser.add_argument('-fst', '--fusion_steiner_tree', action='store_true', help='If given, run GENERAL_STEINER_TREE')
    parser.add_argument('-ns', '--naive_star', action='store_true', help='If given, run STAR_MULTI_PAIR')
    parser.add_argument('-sx', '--star_expansion', action='store_true', help='If given, run STAR_EXPANSION')

    # parameter
    parser.add_argument('-para', '--parameter', type=str, nargs=1, default=['atomic_bsm'])

    args = parser.parse_args()

    # ********************** changing parameters (x-values) & algorithms used *******************
    # parameter : {"atomic_bsm", "num_end_nodes", "decoherence_time", "num_nodes", "edge_density"}
    IS_FRO = args.fusion_retain_only
    IS_FST = args.fusion_steiner_tree
    IS_NS = args.naive_star
    IS_SX = args.star_expansion
    IS_LINE = False  # args.line_graph

    parameter = args.parameter[0]
    if IS_LINE:
        number_of_repeat = 1
        simulation_duration = 60
        if parameter == "distance":
            values = [100, 300, 500, 700, 900, 1000]
        elif parameter == "link":
            values = [(20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 50)]
        elif parameter == "decoherence_time":
            values = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 5, 10, 15, 20, 25, 30, 35, 45, 50, 55, 60]
        else:
            raise Exception(f'Oops! wrong parameter input: {parameter} for line_graph(only distance and link)')
    else:
        if parameter == 'atomic_fusion':
            values = [0.2, 0.3, 0.4, 0.5, 0.6]
        elif parameter == 'atomic_bsm':
            values = [0.2, 0.3, 0.4, 0.5, 0.6]
        elif parameter == 'num_nodes':
            values = [50, 75, 200, 300]
        elif parameter == 'edge_density':
            values = [0.04, 0.06, 0.1, 0.12]
        elif parameter == 'num_end_nodes':
            values = [3, 4, 6, 7]
        else:
            raise Exception(f'Oops! wrong parameter input: {parameter}')
    ################################################################################

    fro_res, fro_protocol_res, fro_res_avg, fro_protocol_res_avg, fro_fidelity, fro_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    fst_edge_res, fst_edge_protocol_res, fst_edge_res_avg, fst_edge_protocol_res_avg, fst_edge_fidelity, \
    fst_edge_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    fst_latency_res, fst_latency_protocol_res, fst_latency_res_avg, fst_latency_protocol_res_avg, fst_latency_fidelity, \
    fst_latency_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    ns_res, ns_protocol_res, ns_res_avg, ns_protocol_res_avg, ns_fidelity, ns_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    sx_res, sx_protocol_res, sx_res_avg, sx_protocol_res_avg, sx_fidelity, sx_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)

    # maximum_distance_reached = [[] for _ in range(len(values))]

    with open(saving_file_name + ".txt", 'a') as file:
        file.write(parameter + ": " + str(values))
    for value_idx, value in enumerate(values):
        teleportation_bsm_rate = value if parameter == "atomic_fusion" else BSM_SUCCESS_RATE
        print(f"\n\nTESTING {parameter}: {value}")
        with open(saving_file_name + ".txt", 'a') as file:
            file.write(f"\n\nTESTING {parameter}: {value}")
        seed(value_idx)

        for repeat_idx in range(number_of_repeat):
            # if value_idx == 0 or (value_idx == 1 and repeat_idx < 2):
            #     continue
            print(f"\nRepeat #{repeat_idx}:")
            seed(repeat_idx)
            # ******************* creating the network *****************
            qNode._stat_id = 0  # need to reset qNode._stat_id for every new experiment
            graph0 = create_random_waxman(area_side=area_side, cell_size=cell_size,
                                          number_nodes=value if parameter == "num_nodes" else default_num_nodes,
                                          min_memory_size=min_memory_size, max_memory_size=max_memory_size,
                                          min_channel_num=min_channel_num, max_channel_num=max_channel_num,
                                          atomic_bsm_success_rate=value if (parameter == "atomic_bsm" or
                                                                            parameter == "atomic_fusion")
                                          else default_atomic_bsm_rate,
                                          edge_density=value if parameter == "edge_density" else default_edge_density,
                                          max_edge_size=area_side * cell_size / 10,
                                          fusion_success_rate=value if (parameter == "atomic_bsm" or
                                                                        parameter == "atomic_fusion")else
                                          default_fusion_success_rate,
                                          optical_bsm_success_rate=default_optical_bsm_rate)
            connected_graph(graph0, min_channel_num=min_channel_num, max_channel_num=max_channel_num)

            # # generating random end nodes to creat GHZ-n state
            n = value if parameter == "num_nodes" else default_num_nodes
            num_end_nodes = value if parameter == "num_end_nodes" else default_num_end_nodes

            end_nodes = choices(range(n), k=num_end_nodes)
            print(f"end_nodes = {end_nodes}")
            with open(saving_file_name + ".txt", 'a') as file:
                file.write(f"\n\nRepeat #: {repeat_idx}")
                file.write(f"\nend_nodes = {end_nodes}")

            if IS_FRO:
                graph_fro = graph0.clone()
                dump_filename = f"saved_trees/fro_parameter_{parameter}_value_idx_{value_idx}_" \
                                f"repeat_{repeat_idx}_max_min"
                try:
                    ghz_fro = pickle.load(open(dump_filename, "rb"))
                except Exception as ee:
                    continue
                    try:
                        ghz_fro = ghz_fusion_retain(graph_fro, end_nodes=set([graph_fro.V[idx] for idx in end_nodes]),
                                                    decoherence_time=value if parameter == "decoherence_time" else
                                                    default_decoherence_time)
                        pickle.dump(ghz_fro, open(dump_filename, "wb"))
                    except Exception as e:
                        print(e)
                        continue
                bi_trees = ghz_fro.bi_trees
                fro_fusion_tree = ghz_fro.ghz_fusion_tree
                fro_res[value_idx].append(fro_fusion_tree.avr_ent_time)
                fro_res_avg[value_idx] = sum(fro_res[value_idx]) / len(fro_res[value_idx])

                if IS_PROTOCOL:
                    try:
                        fro_network = FusionRetainOnlyTreeProtocol(qgraph=graph_fro,
                                                                   fusion_tree=fro_fusion_tree,
                                                                   swapping_trees=ghz_fro.bi_trees,
                                                                   simulation_duration=simulation_duration)
                    except Exception as e:
                        print(f"Protocol for FRO failed. Error: {e}")
                        continue

                    fro_protocol_res[value_idx].append(float('inf') if fro_network.success_rate == 0 else
                                                       1.0 / fro_network.success_rate)
                    fro_protocol_res_avg[value_idx] = average(fro_protocol_res[value_idx])
                    fro_fidelity[value_idx].append(fro_network.fidelity)
                    fro_fidelity_avg[value_idx] = average(fro_fidelity[value_idx])
                    del fro_network

                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\n******* FUSION RETAIN ONLY *******")
                    file.write("\nAlgorithm results: " + str(fro_res))
                    file.write("\nProtocol results: " + str(fro_protocol_res))
                    file.write("\nFidelity results: " + str(fro_fidelity))
                    file.write("\nAlgorithm results (average): " + str(fro_res_avg))
                    file.write("\nProtocol results (average): " + str(fro_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(fro_fidelity_avg))
                    file.write("\n\n")

            if IS_FST:
                graph_fst = graph0.clone()
                dump_filename = f"saved_trees/steiner_tree_parameter_{parameter}_value_idx_{value_idx}_" \
                                f"repeat_{repeat_idx}"
                try:
                    ghz_fst = pickle.load(open(dump_filename, "rb"))

                except Exception as ee:
                    continue
                    try:
                        ghz_fst = ghz_steiner_tree(graph0, end_nodes=set([graph_fst.V[idx] for idx in end_nodes]),
                                                   decoherence_time=value if parameter == "decoherence_time" else
                                                   default_decoherence_time)
                        pickle.dump(ghz_fst, open(dump_filename, "wb"))
                    except Exception as e:
                        print(e)
                        continue
                bi_trees = ghz_fst.best_steiner_tree
                edge_partitioned_fusion_tree = ghz_fst.fusion_tree(EQUAL_EDGES_PARTITIONING)
                fst_edge_res[value_idx].append(edge_partitioned_fusion_tree.avr_ent_time)
                fst_edge_res_avg[value_idx] = sum(fst_edge_res[value_idx]) / len(fst_edge_res[value_idx])

                if IS_PROTOCOL:

                    edge_fst_network = FusionTreeProtocol(graph_fst, trees=[edge_partitioned_fusion_tree],
                                                          simulation_duration=simulation_duration)
                    fst_edge_protocol_res[value_idx].append(float('inf') if edge_fst_network.success_rate[0] == 0 else
                                                            1.0 / edge_fst_network.success_rate[0])
                    fst_edge_protocol_res_avg[value_idx] = average(fst_edge_protocol_res[value_idx])
                    fst_edge_fidelity[value_idx].append(edge_fst_network.fidelity[0])
                    fst_edge_fidelity_avg[value_idx] = average(fst_edge_fidelity[value_idx])
                    del edge_fst_network

                latency_partitioned_fusion_tree = ghz_fst.fusion_tree(LATENCY_PARTITIONING)
                fst_latency_res[value_idx].append(latency_partitioned_fusion_tree.avr_ent_time)
                fst_latency_res_avg[value_idx] = sum(fst_latency_res[value_idx]) / len(fst_latency_res[value_idx])

                if IS_PROTOCOL:
                    latency_fst_network = FusionTreeProtocol(graph_fst, trees=[latency_partitioned_fusion_tree],
                                                             simulation_duration=simulation_duration)
                    fst_latency_protocol_res[value_idx].append(float('inf') if latency_fst_network.success_rate[0] == 0
                                                               else 1.0 / latency_fst_network.success_rate[0])
                    fst_latency_protocol_res_avg[value_idx] = average(fst_latency_protocol_res[value_idx])
                    fst_latency_fidelity[value_idx].append(latency_fst_network.fidelity[0])
                    fst_latency_fidelity_avg[value_idx] = average(fst_latency_fidelity[value_idx])
                    del latency_fst_network

                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\n******* FUSION STEINER TREE - Edge Balanced *******")
                    file.write("\nAlgorithm results: " + str(fst_edge_res))
                    file.write("\nProtocol results: " + str(fst_edge_protocol_res))
                    file.write("\nFidelity results: " + str(fst_edge_fidelity))
                    file.write("\nAlgorithm results (average): " + str(fst_edge_res_avg))
                    file.write("\nProtocol results (average): " + str(fst_edge_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(fst_edge_fidelity_avg))
                    file.write("\n\n")

                    file.write("\n******* FUSION STEINER TREE - Latency Balanced *******")
                    file.write("\nAlgorithm results: " + str(fst_latency_res))
                    file.write("\nProtocol results: " + str(fst_latency_protocol_res))
                    file.write("\nFidelity results: " + str(fst_latency_fidelity))
                    file.write("\nAlgorithm results (average): " + str(fst_latency_res_avg))
                    file.write("\nProtocol results (average): " + str(fst_latency_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(fst_latency_fidelity_avg))
                    file.write("\n\n")

            if IS_NS:
                graph_ns = graph0.clone()
                dump_filename = f"saved_trees/central_ghz_trees_parameter_{parameter}_value_idx_{value_idx}_" \
                                f"repeat_{repeat_idx}_DP"
                try:
                    ghz_ns = pickle.load(open(dump_filename, "rb"))

                except Exception as ee:
                    try:
                        ghz_ns = ghz_star_multi_pair(graph0, end_nodes=set([graph0.V[idx] for idx in end_nodes]),
                                                     decoherence_time=value if parameter == "decoherence_time" else
                                                     default_decoherence_time)
                        pickle.dump(ghz_ns, open(dump_filename, "wb"))
                    except Exception as e:
                        print(f"Error happened running Central GHZ, error: {e}].\n "
                              f"Error Traceback\n{traceback.format_exc()}")
                        continue
                bi_trees = ghz_ns.bi_trees
                ns_res[value_idx].append(ghz_ns.avr_ent_time)
                ns_res_avg[value_idx] = average(ns_res[value_idx])

                if IS_PROTOCOL:
                    try:
                        ns_network = CentralGHZTreeProtocol(graph_ns, set(end_nodes), swapping_trees=ghz_ns.bi_trees,
                                                            simulation_duration=simulation_duration)
                    except Exception as e:
                        print(f"CentralGHZ protocol failed. Error: {e}.\n full traceback:\n{traceback.format_exc()}")
                        continue

                    ns_protocol_res[value_idx].append(float('inf') if ns_network.success_rate == 0
                                                      else 1.0 / ns_network.success_rate)
                    ns_protocol_res_avg[value_idx] = average(ns_protocol_res[value_idx])
                    ns_fidelity[value_idx].append(ns_network.fidelity)
                    ns_fidelity_avg[value_idx] = average(ns_fidelity[value_idx])
                    del ns_network

                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\n******* NAIVE - STAR GRAPH *******")
                    file.write("\nAlgorithm results: " + str(ns_res))
                    file.write("\nProtocol results: " + str(ns_protocol_res))
                    file.write("\nFidelity results: " + str(ns_fidelity))
                    file.write("\nAlgorithm results (average): " + str(ns_res_avg))
                    file.write("\nProtocol results (average): " + str(ns_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(ns_fidelity_avg))
                    file.write("\n\n")

            if IS_SX:
                graph_sx = graph0.clone()
                dump_filename = f"saved_trees/star_expansion_trees_parameter_{parameter}_value_idx_{value_idx}_" \
                                f"repeat_{repeat_idx}"
                try:
                    ghz_sx = pickle.load(open(dump_filename, "rb"))

                except Exception as ee:
                    try:
                        ghz_sx = ghz_star_expansion(graph0, end_nodes=set([graph0.V[idx] for idx in end_nodes]),
                                                    decoherence_time=value if parameter == "decoherence_time" else
                                                    default_decoherence_time)
                        pickle.dump(ghz_sx, open(dump_filename, "wb"))
                    except Exception as e:
                        print(f"Error happened running Central GHZ, error: {e}].\n "
                              f"Error Traceback\n{traceback.format_exc()}")
                        continue
                steiner_trees = ghz_sx.steiner_tree
                star_expansion_fusion_tree = ghz_sx.fusion_tree
                sx_res[value_idx].append(star_expansion_fusion_tree.avr_ent_time)
                sx_res_avg[value_idx] = average(sx_res[value_idx])

                if IS_PROTOCOL:
                    try:
                        sx_network = CentralGHZTreeProtocol(graph_sx, set(end_nodes), swapping_trees=ghz_ns.bi_trees,
                                                            simulation_duration=simulation_duration)
                    except Exception as e:
                        print(f"CentralGHZ protocol failed. Error: {e}.\n full traceback:\n{traceback.format_exc()}")
                        continue

                    sx_protocol_res[value_idx].append(float('inf') if sx_network.success_rate == 0
                                                      else 1.0 / sx_network.success_rate)
                    sx_protocol_res_avg[value_idx] = average(sx_protocol_res[value_idx])
                    sx_fidelity[value_idx].append(sx_network.fidelity)
                    sx_fidelity_avg[value_idx] = average(sx_fidelity[value_idx])
                    del sx_network

                with open(saving_file_name + ".txt", "a") as file:
                    file.write("\n******* STAR EXPANSION *******")
                    file.write("\nAlgorithm results: " + str(sx_res))
                    file.write("\nProtocol results: " + str(sx_protocol_res))
                    file.write("\nFidelity results: " + str(sx_fidelity))
                    file.write("\nAlgorithm results (average): " + str(sx_res_avg))
                    file.write("\nProtocol results (average): " + str(sx_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(sx_fidelity_avg))
                    file.write("\n\n")


    with open(saving_file_name + ".txt", 'a') as file:
        file.write(f"\n\n\n*********Summary*******")
        file.write(f"\n\nTESTING {parameter}: {values}")

    if IS_FRO:
        with open(saving_file_name + ".txt", "a") as file:
            file.write("\n**** FUSION RETAIN ONLY ****")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in fro_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in fro_protocol_res]))
            file.write("\nFidelity results: " + str([[round(x, 3)for x in row] for row in fro_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in fro_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in fro_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in fro_fidelity_avg]) + "\n\n")

    if IS_FST:
        with open(saving_file_name + ".txt", "a") as file:
            file.write("\n**** FUSION STEINER TREE - EDGE BALANCED ****")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in fst_edge_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in fst_edge_protocol_res]))
            file.write("\nFidelity results: " + str([[round(x, 3)for x in row] for row in fst_edge_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in fst_edge_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in fst_edge_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in fst_edge_fidelity_avg]) + "\n\n")

            file.write("\n**** FUSION STEINER TREE - LATENCY BALANCED ****")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in fst_latency_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in fst_latency_protocol_res]))
            file.write("\nFidelity results: " + str([[round(x, 3) for x in row] for row in fst_latency_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in fst_latency_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in fst_latency_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in fst_latency_fidelity_avg]) + "\n\n")

    if IS_NS:
        with open(saving_file_name + ".txt", "a") as file:
            file.write("\n**** NAIVE - STAR GRAPH ****")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in ns_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in ns_protocol_res]))
            file.write("\nFidelity results: " + str([[round(x, 3) for x in row] for row in ns_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in ns_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in ns_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in ns_fidelity_avg]) + "\n\n")

    if IS_SX:
        with open(saving_file_name + ".txt", "a") as file:
            file.write("\n**** STAR EXPANSION ****")
            file.write("\nAlgorithm results: " + str([[round(x, 3) for x in row] for row in sx_res]))
            file.write("\nProtocol results: " + str([[round(x, 3) for x in row] for row in sx_protocol_res]))
            file.write("\nFidelity results: " + str([[round(x, 3) for x in row] for row in sx_fidelity]))
            file.write("\nAlgorithm results (average): " + str([round(x, 3) for x in sx_res_avg]))
            file.write("\nProtocol results (average): " + str([round(x, 3) for x in sx_protocol_res_avg]))
            file.write("\nFidelity results (average): " + str([round(x, 3) for x in sx_fidelity_avg]) + "\n\n")

def average(vars: List):
    if len([var for var in vars if var != float('inf')]) == 0:
        return float('inf')
    return sum([var for var in vars if var != float('inf')]) / len([var for var in vars if var != float('inf')])

if __name__ == "__main__":
    # main_calefi()
    # main()
    # main_pre()
    main_ghz()
