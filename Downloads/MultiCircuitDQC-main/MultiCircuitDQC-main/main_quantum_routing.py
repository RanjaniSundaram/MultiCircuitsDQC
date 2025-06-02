import time

from networkx.classes import graph
from commons.qChannel import qChannel
from commons.qNode import qNode, BSM_SUCCESS_RATE
from commons.qGraph import qGraph
from commons.Point import Point
from random import randint, seed, sample, random, shuffle
from routing_algorithms.max_flow import max_flow
from routing_algorithms.dp_shortest_path import dp_shortest_path as dp
from routing_algorithms.sigcomm import Sigcomm_W
from routing_algorithms.dp_alternate import DP_Alt
from networkx.generators.geometric import waxman_graph
from networkx import Graph
from routing_algorithms.delft_lp import modified_qGraph, LP
import datetime
from routing_algorithms.optimal_linear_programming import optimal_linear_programming
from protocols.predistribution.network_protocol import TreeProtocol as PredistProtocol
from protocols.network_protocol import TreeProtocol
from protocols.sigcomm_network_protocol import TreeProtocol as SigcommTreeProtocol
from commons.tree_node import tree_node
from plot_results import Plot
import os
from predistributed_algorithms.greedy_predist import greedy_predist
from predistributed_algorithms.multi_pair_greedy_predist import multi_pair_greedy_predist


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
                       min_memory_size: int = 2, max_memory_size: int = 2,
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
    new_v = [qNode(memory=2 * subchannels, loc=v.loc, gen_success_rate=v.gen_success_rate,
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


# continuous (s-d) pairs EP generation
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
    default_decoherence_time = 60  # seconds
    default_atomic_bsm_rate, default_optical_bsm_rate = 0.4, 0.2
    default_edge_density = 0.03
    default_num_nodes = 10
    default_distance_range = [15, 20] if default_number_of_src_dst_pair == 1 else [10,
                                                                                   30]  # minimum distance between src-dst in km
    IS_TELEPORTING = False  # defines if we send information from Alice to Bob after tree construction in protocols

    simulation_duration = 3  # seconds
    number_of_repeat = 1

    date_now = datetime.datetime.now()

    directory = "results/" + date_now.strftime('%Y_%m_%d')

    if not os.path.exists(directory):
        os.makedirs(directory)
    saving_file_name = directory + "/results" + date_now.strftime('_%H_%M_%S')

    # ********************** changing parameters (x-values) & algorithms used *******************
    # parameter : {"atomic_bsm", "optical_bsm", "decoherence_time", "num_nodes", "edge_density", "distance", "src_dst_pair"}
    parameter = "atomic_bsm"
    values = [0.6]
    IS_DP = False  # single path
    IS_DP_ALT = False  # single path
    IS_DP_ITER = True  # multiple path
    IS_LP = False  # multiple path
    IS_LP_ALT = False  # multiple path
    IS_SIGCOMM_SINGLE = False  # single path
    IS_SIGCOMM_MULTI = False  # multiple path
    IS_DELFT_LP = False  # multiple path
    dp_res, dp_protocol_res, dp_res_avg, dp_protocol_res_avg, dp_fidelity, dp_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    dp_alt_res, dp_alt_protocol_res, dp_alt_res_avg, dp_alt_protocol_res_avg, dp_alt_fidelity, dp_alt_fidelity_avg = \
        [[] for _ in range(len(values))], [[] for _ in range(len(values))], [0] * len(values), [0] * len(values), \
        [[] for _ in range(len(values))], [0] * len(values)
    dp_iter_res, dp_iter_protocol_res, dp_iter_res_avg, dp_iter_protocol_res_avg, dp_iter_fidelity, dp_iter_fidelity_avg = \
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

    for value_idx, value in enumerate(values):
        teleportation_bsm_rate = value if parameter == "atomic_bsm" else BSM_SUCCESS_RATE
        print(f"\n\nTESTING {parameter}: {value}")
        with open(saving_file_name + ".txt", 'a') as file:
            file.write(parameter + ": " + str(values) + "\n\n")
        # ******************* creating the network *****************
        qNode._stat_id = 0  # need to reset qNode._stat_id for every new experiment
        seed(10)
        graph0 = create_random_waxman(area_side=area_side, cell_size=cell_size,
                                      number_nodes=value if parameter == "num_nodes" else default_num_nodes,
                                      min_memory_size=min_memory_size, max_memory_size=max_memory_size,
                                      min_channel_num=min_channel_num, max_channel_num=max_channel_num,
                                      atomic_bsm_success_rate=value if parameter == "atomic_bsm"
                                      else default_atomic_bsm_rate,
                                      edge_density=value if parameter == "edge_density" else default_edge_density)
        connected_graph(graph0, min_channel_num=min_channel_num, max_channel_num=max_channel_num)
        sigcomm_graph0, sigcomm_ref_length = sigcomm_graph(graph0, sigcom_w, "max")
        for repeat_idx in range(number_of_repeat):
            print(f"Repeat #{repeat_idx}:")
            # # generating random src-dst pair
            srcs, dsts = [], []
            # current_pairs = set()
            n = value if parameter == "num_nodes" else default_num_nodes
            min_distance = value[0] if parameter == "distance" else default_distance_range[0]
            max_distance = value[1] if parameter == "distance" else default_distance_range[1]
            ids_with_distance = [(x, y) for x in range(n) for y in range(x + 1, n) if
                                 (min_distance * 10 ** 3 <= graph0.V[x].loc.distance(graph0.V[y].loc) <=
                                  max_distance * 10 ** 3 and graph0.get_edge(graph0.V[x].id, graph0.V[y].id) is None)]
            shuffle(ids_with_distance)
            for i in range(min((value if parameter == "src_dst_pair" else default_number_of_src_dst_pair),
                               len(ids_with_distance))):
                srcs.append(ids_with_distance[i][0])
                dsts.append(ids_with_distance[i][1])

            print(f"srcs = {srcs}, dsts = {dsts}")

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
                dp_shortest = dp(graph_dp, value if parameter == "decoherence_time" else default_decoherence_time)
                dp_shortest_path = dp_shortest.get_shortest_path(graph_dp.V[srcs[0]], graph_dp.V[dsts[0]])
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / dp_shortest_path.avr_ent_time
                print(f"\n\nDP: {alg_result:.3f} (EP/s)")
                tree_node.print_tree(dp_shortest_path)
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
                dp_res_avg[value_idx] += dp_res[value_idx][-1] / number_of_repeat
                dp_protocol_res_avg[value_idx] += dp_protocol_res[value_idx][-1] / number_of_repeat
                dp_fidelity_avg[value_idx] += dp_fidelity[value_idx][-1] / number_of_repeat
                plotting_date['dp'] = dp_res_avg
                plotting_date['dp_proto'] = dp_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("**** DP ***")
                    file.write("\nAlgorithm results: " + str(dp_res))
                    file.write("\nProtocol results: " + str(dp_protocol_res))
                    file.write("\nFidelity results: " + str(dp_fidelity))
                    file.write("\nAlgorithm results (average): " + str(dp_res_avg))
                    file.write("\nProtocol results (average): " + str(dp_protocol_res_avg))
                    file.write("\nFidelity results (average): " + str(dp_fidelity_avg) + "\n\n")
                del network_protocol_dp

            # DP Alternate
            if IS_DP_ALT:
                graph_dp_alt = graph0.clone()
                dp_alt = DP_Alt(graph_dp_alt)
                metric = 'time'
                src, dst = graph_dp_alt.V[srcs[0]], graph_dp_alt.V[dsts[0]]
                path = dp_alt.dijsktra(src, dst, metric=metric)
                dp_alt_tree = dp_alt.path2tree(path,
                                               decoherence_time=value if parameter == "decoherence_time" else default_decoherence_time,
                                               update_time=True)
                if dp_alt_tree is None:
                    raise Exception('Oops! DP alternate tree construction failed :(')

                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / dp_alt_tree.avr_ent_time
                print(f"\n\n-->DP Alternate: {alg_result:.3f} (EP/s)")
                tree_node.print_tree(dp_alt_tree)
                network_protocol_dp_alt = TreeProtocol(graph0, [dp_alt_tree], [src], [dst], simulation_duration,
                                                       IS_TELEPORTING=IS_TELEPORTING)
                print(f"DP Alternate (protocol): {network_protocol_dp_alt.success_rate[0]} (EP/s), "
                      f"avg_fidelity: {network_protocol_dp_alt.avg_fidelity[0]} "
                      f"\nduration={simulation_duration}s")
                dp_alt_res[value_idx].append(alg_result)
                dp_alt_protocol_res[value_idx].append(network_protocol_dp_alt.success_rate[0])
                dp_alt_fidelity[value_idx].append(network_protocol_dp_alt.avg_fidelity[0])
                dp_alt_res_avg[value_idx] += dp_alt_res[value_idx][-1] / number_of_repeat
                dp_alt_protocol_res_avg[value_idx] += dp_alt_protocol_res[value_idx][-1] / number_of_repeat
                dp_alt_fidelity_avg[value_idx] += dp_alt_fidelity[value_idx][-1] / number_of_repeat
                plotting_date['dp_alt'] = dp_alt_res_avg
                plotting_date['dp_alt_proto'] = dp_alt_protocol_res_avg

                with open(saving_file_name + ".txt", "a") as file:
                    file.write("**** DP Alternate ***")
                    file.write("\nAlgorithm results: " + str(dp_alt_res))
                    file.write("\nProtocol results: " + str(dp_alt_protocol_res))
                    file.write("\nFidelity results: " + str(dp_alt_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(dp_alt_res_avg))
                    file.write("\nProtocol results (Average): " + str(dp_alt_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(dp_alt_fidelity_avg) + "\n\n")
                del network_protocol_dp_alt

            # max flow
            if IS_DP_ITER:
                print('-->\n\n(DP Iterative)')
                start_time = datetime.datetime.now()
                graph_ours = graph0.clone()
                ours_srcs = [graph_ours.V[src] for src in srcs]
                ours_dsts = [graph_ours.V[dst] for dst in dsts]
                max_flow_ours = max_flow(graph_ours, ours_srcs, ours_dsts,
                                         decoherence_time=value if parameter == "decoherence_time" else
                                         default_decoherence_time
                                         , multiple_pairs_search='best_paths')
                alg_result = (teleportation_bsm_rate if IS_TELEPORTING else 1.0) * max_flow_ours.get_flow_total
                print("Algorithm (max-flow) results")
                print("Total flow algorithm (max-flow): {:.2f} q/s\ntime:".format(alg_result),
                      datetime.datetime.now() - start_time)
                ours_all_trees = []
                ours_protocol_all_srcs, ours_protocol_all_dsts = [], []
                tree_num = 0
                for pair_id in range(len(ours_dsts)):
                    pair_res = max_flow_ours.get_flows(pair_id)
                    for _, tree, _ in pair_res:
                        ours_all_trees.append(tree)
                        ours_protocol_all_srcs.append(ours_srcs[pair_id])
                        ours_protocol_all_dsts.append(ours_dsts[pair_id])

                        print(f"Tree{tree_num} (rate:"
                              f" {(teleportation_bsm_rate if IS_TELEPORTING else 1.0) / tree.avr_ent_time} EPs/s), "
                              f"between nodes {ours_srcs[pair_id].id} (loc: {ours_srcs[pair_id].loc}) and "
                              f"{ours_srcs[pair_id].id} (loc: {ours_dsts[pair_id].loc})")
                        tree_node.print_tree(tree)
                        tree_num += 1
                start_time = datetime.datetime.now()
                ours_protocol = TreeProtocol(graph=graph0, trees=ours_all_trees,
                                             sources=ours_protocol_all_srcs, destinations=ours_protocol_all_dsts,
                                             duration=simulation_duration, IS_TELEPORTING=IS_TELEPORTING)
                print("\nProtocol(max-flow) Results:")
                valid_fidelities = [ours_protocol.avg_fidelity[i] for i in range(len(ours_protocol.avg_fidelity)) if
                                    ours_protocol.avg_fidelity[i] != -1]
                print(valid_fidelities)
                print(f"Total flow algorithm (max-flow): {sum(ours_protocol.success_rate):.2f} ep/s, avg_fidelity: "
                      f"{sum([fid for gar,fid in valid_fidelities]) / len(valid_fidelities)}"
                      f"\nduration: {simulation_duration}s")
                for tree_id, rate in enumerate(ours_protocol.success_rate):
                    print(f"Tree{tree_id}, rate: {rate} EPs/s, avg_fidelity: {ours_protocol.avg_fidelity[tree_id]}")
                dp_iter_res[value_idx].append(alg_result)
                dp_iter_protocol_res[value_idx].append(sum(ours_protocol.success_rate))
                dp_iter_fidelity[value_idx].append(sum([fid for gar,fid in valid_fidelities]) / len(valid_fidelities))
                dp_iter_res_avg[value_idx] += dp_iter_res[value_idx][-1] / number_of_repeat
                dp_iter_protocol_res_avg[value_idx] += dp_iter_protocol_res[value_idx][-1] / number_of_repeat
                dp_iter_fidelity_avg[value_idx] += dp_iter_fidelity[value_idx][-1] / number_of_repeat
                plotting_date['dp_iter'] = dp_iter_res_avg
                plotting_date['dp_iter_proto'] = dp_iter_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("**** DP Iterative ***")
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
                        tree_num += 1
                start_time = datetime.datetime.now()
                lp_alt_protocol = TreeProtocol(graph=graph0, trees=lp_alt_all_trees,
                                               sources=lp_alt_protocol_all_srcs, destinations=lp_alt_protocol_all_dsts,
                                               duration=simulation_duration, IS_TELEPORTING=IS_TELEPORTING)
                print("\nProtocol (LP alternate) Results:")
                valid_fidelities = [lp_alt_protocol.avg_fidelity[i] for i in range(len(lp_alt_protocol.avg_fidelity)) if
                                    lp_alt_protocol.avg_fidelity[i] != -1]
                print(f"Total flow algorithm (LP alternate): {sum(lp_alt_protocol.success_rate):.2f} eps/s, "
                      f"avg_fidelity: {sum(valid_fidelities) / len(valid_fidelities)}"
                      f"\nduration: {simulation_duration}s")
                for tree_id, rate in enumerate(lp_alt_protocol.success_rate):
                    print(f"Tree{tree_id}, rate: {rate} EPs/s, fidelity: {lp_alt_protocol.avg_fidelity[tree_id]}")
                lp_alt_res[value_idx].append(alg_result)
                lp_alt_protocol_res[value_idx].append(sum(lp_alt_protocol.success_rate))
                lp_alt_fidelity[value_idx].append(sum(valid_fidelities) / len(valid_fidelities))
                lp_alt_res_avg[value_idx] += lp_alt_res[value_idx][-1] / number_of_repeat
                lp_alt_protocol_res_avg[value_idx] += lp_alt_protocol_res[value_idx][-1] / number_of_repeat
                lp_alt_fidelity_avg[value_idx] += lp_alt_fidelity[value_idx][-1] / number_of_repeat
                plotting_date['lp_alt'] = lp_alt_res_avg
                plotting_date['lp_alt_proto'] = lp_alt_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("**** LP Alternate ***")
                    file.write("\nAlgorithm results: " + str(lp_alt_res))
                    file.write("\nProtocol results: " + str(lp_alt_protocol_res))
                    file.write("\nProtocol results: " + str(lp_alt_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(lp_alt_res_avg))
                    file.write("\nProtocol results (Average): " + str(lp_alt_protocol_res_avg))
                    file.write("\nFidelity results (Average): " + str(lp_alt_fidelity_avg) + "\n\n")
                del lp_alt_protocol

            # Linear Programming
            if IS_LP:
                print("\n\n-->LP")
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
                        tree_num += 1

                start_time = datetime.datetime.now()
                lp_protocol = TreeProtocol(graph=graph0, trees=lp_all_trees,
                                           sources=lp_protocol_all_srcs, destinations=lp_protocol_all_dsts,
                                           duration=simulation_duration, IS_TELEPORTING=IS_TELEPORTING)
                print("\nProtocol (LP) Results:")
                print(f"Total flow Protocol (LP): {sum(lp_protocol.success_rate):.2f} eps/s, "
                      f"avg_fidelity: {sum(lp_protocol.avg_fidelity) / len(lp_protocol.avg_fidelity)}"
                      f"\nduration: {simulation_duration}")
                for tree_id, rate in enumerate(lp_protocol.success_rate):
                    print(f"Tree{tree_id}, rate: {rate} EPs/s, avg_fidelity: {lp_protocol.avg_fidelity[tree_id]}")
                lp_res[value_idx].append(alg_result)
                lp_protocol_res[value_idx].append(sum(lp_protocol.success_rate))
                lp_fidelity[value_idx].append(sum(lp_protocol.avg_fidelity) / len(lp_protocol.avg_fidelity))
                lp_res_avg[value_idx] += lp_res[value_idx][-1] / number_of_repeat
                lp_protocol_res_avg[value_idx] += lp_protocol_res[value_idx][-1] / number_of_repeat
                lp_fidelity_avg[value_idx] += lp_fidelity[value_idx][-1] / number_of_repeat
                plotting_date['lp'] = lp_res_avg
                plotting_date['lp_proto'] = lp_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("**** LP ***")
                    file.write("\nAlgorithm results: " + str(lp_res))
                    file.write("\nProtocol results: " + str(lp_protocol_res))
                    file.write("\nFidelity results: " + str(lp_fidelity))
                    file.write("\nAlgorithm results (Average): " + str(lp_res_avg))
                    file.write("\nProtocol results (Average): " + str(lp_protocol_res_avg))
                    file.write("\nProtocol results (Average): " + str(lp_fidelity_avg) + "\n\n")
                del lp, lp_protocol

            # SIGCOMM 20 single path
            if IS_SIGCOMM_SINGLE:
                print('\n\n-->SIGCOMM 20 Single')
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

                if sig_rates[0] > 0.01:
                    network_protocol_sigcomm = SigcommTreeProtocol(sigcomm_graph0, [(sig_tree, sig_path.W)], [src],
                                                                   [dst],
                                                                   3 * simulation_duration,
                                                                   swapping_time_window=0,
                                                                   IS_TELEPORTING=IS_TELEPORTING,
                                                                   ref_length=sigcomm_ref_length)
                    sigcomm_protocol_rate = network_protocol_sigcomm.success_rate[0]
                else:
                    sigcomm_protocol_rate = sig_rates[0]
                print(f"Sigcomm single (protocol): {sigcomm_protocol_rate} (EP/s)"
                      f"\nduration={3 * simulation_duration}s")
                sigcomm_single_res[value_idx].append(
                    (teleportation_bsm_rate if IS_TELEPORTING else 1.0) / sig_tree.avr_ent_time)
                sigcomm_single_protocol_res[value_idx].append(sigcomm_protocol_rate)
                sigcomm_single_res_avg[value_idx] += sigcomm_single_res[value_idx][-1] / number_of_repeat
                sigcomm_single_protocol_res_avg[value_idx] += sigcomm_single_protocol_res[value_idx][
                                                                  -1] / number_of_repeat
                plotting_date['sig'] = sigcomm_single_res_avg
                plotting_date['sig_proto'] = sigcomm_single_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("**** SIGCOMM 20 Single ****")
                    file.write("\nAlgorithm results: " + str(sigcomm_single_res))
                    file.write("\nProtocol results: " + str(sigcomm_single_protocol_res))
                    file.write("\nAlgorithm results (Average): " + str(sigcomm_single_res_avg))
                    file.write("\nProtocol results (Average): " + str(sigcomm_single_protocol_res_avg) + "\n\n")

            # SIGCOMM 20 multiple path
            if IS_SIGCOMM_MULTI:
                print('\n\n-->SIGCOMM 20 Multiple')
                graph_sigcomm = graph0.clone()
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

                sigcomm_protocol = SigcommTreeProtocol(graph=sigcomm_graph0, trees=sigcomm_all_trees,
                                                       sources=sigcomm_all_srcs, destinations=sigcomm_all_dsts,
                                                       duration=3 * simulation_duration, swapping_time_window=1,
                                                       IS_TELEPORTING=IS_TELEPORTING,
                                                       ref_length=sigcomm_ref_length)
                print("\nProtocol (SIGCOMM) Results:")
                print(f"Total flow Protocol (SIGCOMM): {sum(sigcomm_protocol.success_rate)} (EP/s)"
                      f"\nduration={3 * simulation_duration}s")
                for tree_id, rate in enumerate(sigcomm_protocol.success_rate):
                    print(f"Tree{tree_id}, rate: {rate} EPs/s")
                sigcomm_multiple_res[value_idx].append(alg_result)
                sigcomm_multiple_protocol_res[value_idx].append(sum(sigcomm_protocol.success_rate))
                sigcomm_multiple_res_avg[value_idx] += sigcomm_multiple_res[value_idx][-1] / number_of_repeat
                sigcomm_multiple_protocol_res_avg += sigcomm_multiple_protocol_res[value_idx][-1] / number_of_repeat
                plotting_date['sig_multi'] = sigcomm_multiple_res_avg
                plotting_date['sig_multi_proto'] = sigcomm_multiple_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("**** SIGCOMM 20 Multiple ****")
                    file.write("\nAlgorithm results: " + str(sigcomm_multiple_res))
                    file.write("\nProtocol results: " + str(sigcomm_multiple_protocol_res))
                    file.write("\nAlgorithm results (Average): " + str(sigcomm_multiple_res_avg))
                    file.write("\nProtocol results (Average): " + str(sigcomm_multiple_protocol_res_avg) + "\n\n")
                del sigcomm_protocol

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
                    leaf_ent_time = 1. / leaf_effective_rate
                    path.metric = 1 / path.metric  # from rate to avg_ent_time, for the dp_alt.path2tree method
                    tree = dp_alt.path2tree(path, decoherence_time=default_decoherence_time, update_time=False)
                    lp.update_ent_time_tree(tree, leaf_ent_time)
                    src, dst = path.nodes[0], path.nodes[-1]
                    delft_lp_all_trees.append(tree)
                    delft_lp_all_srcs.append(graph_delft_lp.V[src])
                    delft_lp_all_dsts.append(graph_delft_lp.V[dst])
                    print(f"Delft LP path: {rate} (EP/s)")
                    tree_node.print_tree(tree)
                print(f'Delft LP time = {time.time() - start}')

                delft_lp_protocol = SigcommTreeProtocol(graph=graph0, trees=delft_lp_all_trees,
                                                        sources=delft_lp_all_srcs, destinations=delft_lp_all_dsts,
                                                        duration=simulation_duration, swapping_time_window=1,
                                                        IS_TELEPORTING=IS_TELEPORTING,
                                                        ref_length=sigcomm_ref_length)

                print("\nProtocol (Delft LP) Results:")
                print("Total flow Protocol (Delft LP): {:.2f} q/s\ntime:".format(sum(delft_lp_protocol.success_rate)),
                      time.time() - start)
                for tree_id, rate in enumerate(delft_lp_protocol.success_rate):
                    print(f"Tree{tree_id}, rate: {rate} EPs/s")
                delft_lp_res[value_idx].append((teleportation_bsm_rate if IS_TELEPORTING else 1.0) * sum(rates))
                delft_lp_protocol_res[value_idx].append(sum(delft_lp_protocol.success_rate))
                delft_lp_res_avg[value_idx] += delft_lp_res[value_idx][-1] / number_of_repeat
                delft_lp_protocol_res_avg[value_idx] += delft_lp_protocol_res[value_idx][-1] / number_of_repeat
                plotting_date['delft_lp'] = delft_lp_res_avg
                plotting_date['delft_lp_proto'] = delft_lp_protocol_res_avg
                with open(saving_file_name + ".txt", "a") as file:
                    file.write("**** Delft LP ****")
                    file.write("\nAlgorithm results: " + str(delft_lp_res))
                    file.write("\nProtocol results: " + str(delft_lp_protocol_res))
                    file.write("\nAlgorithm results (Average): " + str(delft_lp_res_avg))
                    file.write("\nProtocol results (Average): " + str(delft_lp_protocol_res_avg) + "\n\n")
                del delft_lp_protocol, lp

    Plot.graph(plotting_data=plotting_date, filename=saving_file_name + ".png")


# online demand-based (s-d) pairs EP generation
def main_pre():
    seed(10)
    # **************** init values ************
    area_side = 10  # a square side of side x side cells
    cell_size = 10000  # side value of each square side (m)
    min_memory_size, max_memory_size = 15, 20
    min_node_degree, max_node_degree = 2, 4  # number of edges from one node to its neighbors
    min_channel_num, max_channel_num = 1, 1  # number of parallel channels a connection would have
    default_number_of_src_dst_pair = 5
    default_decoherence_time = 600  # seconds
    default_atomic_bsm_rate, default_optical_bsm_rate = 0.4, 0.2
    default_edge_density = 0.03
    default_num_nodes = 20
    default_distance_range = [15, 20] if default_number_of_src_dst_pair == 1 else [5, 100]
    IS_LINE = True

    simulation_duration = 150  # seconds
    number_of_repeat = 50

    FIXED_LATENCY, LATENCY_THRESHOLD = False, 0.50

    date_now = datetime.datetime.now()

    directory = "results/pre_distribution/" + date_now.strftime('%Y_%m_%d')

    if not os.path.exists(directory):
        os.makedirs(directory)
    saving_file_name = directory + "/results" + date_now.strftime('_%H_%M_%S')

    # ********************** changing parameters (x-values) & algorithms used *******************
    # parameter : {"atomic_bsm", "optical_bsm", "decoherence_time", "num_nodes", "edge_density", "distance", "src_dst_pair"}
    parameter = "num_nodes"
    values = [20, 50, 70, 100, 150, 200, 250]

    num_accepted_req = [0] * len(values)
    num_req_full_sl = [0] * len(values)

    for value_idx, value in enumerate(values):
        print(f"\n\nTESTING {parameter}: {value}")
        with open(saving_file_name + ".txt", 'a') as file:
            file.write(parameter + ": " + str(values) + "\n\n")
        # ******************* creating the network *****************
        qNode._stat_id = 0  # need to reset qNode._stat_id for every new experiment
        seed(value_idx)

        for repeat_idx in range(number_of_repeat):
            seed(repeat_idx)
            print(f"Repeat #{repeat_idx}:")
            # # generating random src-dst pair
            srcs, dsts = [], []
            if IS_LINE:
                graph0 = create_line_graph2(total_length=110,  min_link_Length=10,
                                            max_link_length=15)
                srcs += [0]
                dsts += [8]
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
                for i in range(int(value ** 2 / 100)):
                    srcs.append(ids_with_distance[i][0])
                    dsts.append(ids_with_distance[i][1])

            # print(f"srcs = {srcs}, dsts = {dsts}")

            # predist algo
            graph_pre_greedy = graph0.clone()
            if False:
                greedy_pre = greedy_predist(G=graph_pre_greedy, srcs=[graph_pre_greedy.V[src] for src in srcs],
                                            dsts=[graph_pre_greedy.V[dst] for dst in dsts],
                                            latency_threshold=LATENCY_THRESHOLD, FIXED_LATENCY=FIXED_LATENCY,
                                            NAIVE=True)
            else:
                greedy_pre = multi_pair_greedy_predist(G=graph_pre_greedy,
                                                       srcs=[graph_pre_greedy.V[src] for src in srcs],
                                                       dsts=[graph_pre_greedy.V[dst] for dst in dsts],
                                                       latency_threshold=LATENCY_THRESHOLD, FIXED_LATENCY=FIXED_LATENCY,
                                                       IS_NAIVE=False)
            protocol = PredistProtocol(graph=graph0, trees=greedy_pre.get_final_paths, SLs=greedy_pre.get_sls,
                                       sources=[graph0.V[src] for src in srcs],
                                       destinations=[graph0.V[dst] for dst in dsts],
                                       duration=simulation_duration)


            # print("*******SUPER-LINKS************")
            # for sl_idx, sl in enumerate(greedy_sls):
            #     print(f"SL #{sl_idx}: src{sl.src.id} -> dst{sl.dst.id}")
            #     tree_node.print_tree(sl.tree)
            # print("New Trees (with SLs)")
            # shortest_path = dp(G=graph_pre_greedy, decoherence_time=6000, superlinks=greedy_sls)
            # for src, dst in zip(srcs, dsts):
            #     src_node, dst_node = graph_pre_greedy.V[src], graph_pre_greedy.V[dst]
            #     sd_shortest_tree = shortest_path.get_shortest_path(src_node, dst_node)
            #     print(f"src: {src}(loc={src_node.loc}), dst: {dst} (loc={dst_node.loc})\n"
            #           f"latency={sd_shortest_tree.avr_ent_time}")
            #     tree_node.print_tree(sd_shortest_tree)
        if num_accepted_req[value_idx] != 0:
            num_req_full_sl[value_idx] /= (int(value ** 2 / 100) * number_of_repeat * num_accepted_req[value_idx])
        with open(saving_file_name + ".txt", "a") as file:
            file.write(f"Accepted requests:  {str(num_accepted_req)}\n")
            file.write(f"Full SL paths: {str(num_req_full_sl)}\n")
        print(f"Results:{str(num_accepted_req)}")


if __name__ == "__main__":
    main()
