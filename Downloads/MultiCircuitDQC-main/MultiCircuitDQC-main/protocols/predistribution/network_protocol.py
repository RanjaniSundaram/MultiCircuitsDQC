import random
import sys

sys.path.append("..")

from commons.qNode import qNode
from commons.qGraph import qGraph
from commons.qChannel import qChannel
from commons.tree_node import tree_node
from protocols.predistribution.protocol_bases import *
from netsquid.nodes import Network
from protocols.predistribution.component_bases import ClassicalConnection, qPNode, \
    Charlie, QuantumConnection
from typing import List

from netsquid.components import QuantumMemory
from commons.Point import Point
from netsquid.nodes import Node
from random import seed
from collections import defaultdict
from predistributed_algorithms.super_link import SL

PORT_NAMES_CLASSICAL = ["c" + inOrOut + "_" + direction for inOrOut in ["in", "out"]
                        for direction in ["left", "right", "parent"]]
PORT_NAMES_QUANTUM = ["q" + inOrOut + "_" + direction for inOrOut in ["in"]
                      for direction in ["left", "right"]]


class TreeProtocol:

    def __init__(self, graph: qGraph, trees: List[List[List]], SLs: List[SL],
                 sources: List[qNode], destinations: List[qNode], duration: int, max_sl_latency: float,
                 swapping_time_window: float = float('inf'), IS_TELEPORTING: bool = False,
                 CHECK_CAPACITIES: bool = False, NUMBER_OF_SL_MEMORIES: int = 4, TIMESLOT: float = 2.0):
        """duration(s) to run the simulation"""
        ns.set_qstate_formalism(ns.QFormalism.DM)
        ns.set_random_state(42)
        ns.sim_reset()
        # get only those nodes and edges that are going to be used to save memory and efficiency
        used_nodes, used_edges = set(), set()
        for sd_trees in trees:
            for tree in sd_trees:
                for sub_tree in tree:
                    if isinstance(sub_tree, tree_node):
                        TreeProtocol._used_nodes_edges(sub_tree, used_nodes, used_edges)
        for sl_tree in SLs:
            TreeProtocol._used_nodes_edges(sl_tree.tree, used_nodes, used_edges)

        nodes = [qPNode(node) for node in graph.V if node.id in used_nodes]
        network = Network("network", nodes)

        self._src_protocols = []  # protocols running on sources (Alices)
        self._dst_protocols = []  # protocols running on destinations (Bobs)
        self._fidelity_collector = []
        self._duration = duration
        self._IS_TELEPORTING = IS_TELEPORTING  # defines if we send qubit information from ALice to bob after the trees
        # are built
        # setup classical connections between each pair of nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                TreeProtocol._add_classical_connections(network, nodes[i], nodes[j])
        # adding manager node
        manager_node = Node("manager_node")
        network.add_node(manager_node)
        self._add_manager_connections(network, manager_node, nodes)
        unique_src_ids = set([src.id for src in sources])
        self._add_sources_to_manager_connections(network, manager_node,
                                                 [network.get_node(f"Node{src_id}") for src_id in unique_src_ids])

        messaging_protocols = {node.ID: MessagingProtocol(node) for node in nodes}  # one for each node for all trees

        # defining protocols link_correction and charlie protocols
        charlie_protocols = []
        link_correction_protocols = {}
        entangling_protocols = {}
        correction_protocols = {}
        swapping_protocols = {}
        teleportation_protocols = {}
        all_protocols = []

        # link_layer_max_capacity = {}
        # setup quantum connection for each edge in G.E,
        for e in graph.E:
            if "{this}-{other}".format(this=e.this.id, other=e.other.id) in used_edges:
                charlie = TreeProtocol._add_quantum_connections(network=network, e=e)
                charlie_protocols.append(CharlieProtocol(charlie))
                charlie_id = "{this}_{other}".format(this=e.this.id, other=e.other.id)
                if e.other.id not in link_correction_protocols:  # add link correction protocol if not existed
                    link_correction_protocols[e.other.id] = LinkCorrectionProtocol(network.get_node(f"Node{e.other.id}")
                                                                                   , messaging_protocols[e.other.id])
                # link_layer_max_capacity[charlie_id] = e

        # add all protocols
        for node in nodes:
            entangling_protocol = EntanglingProtocol(node, messaging_protocols[node.ID],
                                                     link_correction_protocols[node.ID]
                                                     if node.ID in link_correction_protocols else None)
            correction_protocol = CorrectionProtocol(node, messaging_protocol=messaging_protocols[node.ID],
                                                     entangling_protocol=entangling_protocol,
                                                     NUMBER_OF_SL_MEMORIES=NUMBER_OF_SL_MEMORIES)
            swapping_protocol = SwappingProtocol(node, entangling_protocol, correction_protocol,
                                                 messaging_protocols[node.ID], NUMBER_OF_SL_MEMORIES)
            entangling_protocols[node.ID] = entangling_protocol
            correction_protocols[node.ID] = correction_protocol
            swapping_protocols[node.ID] = swapping_protocol

        # teleportation protocols for source node
        for src in sources:
            if src.id not in teleportation_protocols:
                teleportation_protocols[src.id] = TeleportationProtocol(network.get_node(f"Node{src.id}"),
                                                                        messaging_protocols[src.id],
                                                                        entangling_protocols[src.id],
                                                                        NUMBER_OF_SL_MEMORIES)
        manager_messaging_protocol = ManagerMessagingProtocol(manager_node)
        self._manager_protocol = ManagerProtocol(manager_node, manager_messaging_protocol, time_period=TIMESLOT,
                                                 max_sl_latency=max_sl_latency)

        link_layer_requests = defaultdict(list)
        for sl_tree_id, sl_tree in enumerate(SLs):
            sd_id = 0
            sl_used_nodes, sl_used_edges = set(), set()
            self._used_nodes_edges(sl_tree.tree, sl_used_nodes, sl_used_edges)
            protocols = []
            sl_src, sl_dst, _, _, is_path_one_hop, dst_correction_protocol = \
                TreeProtocol._setup_tree_protocols(network=network, node=sl_tree.tree, sd_id=sd_id,
                                                   tree_id=sl_tree_id,
                                                   protocols=protocols,
                                                   messaging_protocols=messaging_protocols,
                                                   entangling_protocols=entangling_protocols,
                                                   correction_protocols=correction_protocols,
                                                   link_correction_protocols=link_correction_protocols,
                                                   swapping_protocols=swapping_protocols,
                                                   swapping_time_window=swapping_time_window,
                                                   link_layer_requests=link_layer_requests,
                                                   original_graph=graph,
                                                   ACCEPTING_EPS=True, NUMBER_OF_SL_MEMORIES=NUMBER_OF_SL_MEMORIES,
                                                   src_id=sl_tree.src.id, dst_id=sl_tree.dst.id)
            # correction_protocols[sl_dst.ID].add_sl_information(sd_id=sd_id, tree_id=sl_tree_id)
            swapping_protocols[sl_dst.ID].update_sl_information(sd_id=sd_id, tree_id=sl_tree_id,
                                                                other_node_id=sl_src.ID, IS_LEFT_SL=True,
                                                                is_path_one_hop=is_path_one_hop)
            swapping_protocols[sl_src.ID].update_sl_information(sd_id=sd_id, tree_id=sl_tree_id,
                                                                other_node_id=sl_dst.ID, IS_LEFT_SL=False,
                                                                is_path_one_hop=is_path_one_hop,
                                                                sl_nodes_id=sl_used_nodes)
            messaging_protocols[sl_src.ID].add_sl_end_signals(sd_id, sl_tree_id)
            messaging_protocols[sl_dst.ID].add_sl_end_signals(sd_id, sl_tree_id)
            all_protocols += protocols
            for node_id in sl_used_nodes:  # add signal for (sd, tree)
                messaging_protocols[node_id].add_signals(sd_id, sl_tree_id)
            TreeProtocol._add_classical_sl_zero_connections(network, sl_used_nodes, sl_src, sl_dst)

        for sd_id_tmp, sd_trees in enumerate(trees):
            sd_id = sd_id_tmp + 1
            src_id = sources[sd_id_tmp].id
            dst_id = destinations[sd_id_tmp].id
            sd_used_nodes, sd_used_edges = set(), set()  # only add appropriate signal and labels that are supposed to be used
            for tree_id, sd_tree in enumerate(sd_trees):
                path_key = f"{sd_id}_{tree_id}"
                tree_used_nodes, tree_used_edges = set(), set()
                path_one_hop_info = []
                for sub_tre_idx, sub_tree in enumerate(sd_tree):
                    if isinstance(sub_tree, tree_node):
                        self._used_nodes_edges(sub_tree, tree_used_nodes, tree_used_edges)

                        protocols = []
                        _, _, _, _, is_path_one_hop, dst_correction_protocol = \
                            TreeProtocol._setup_tree_protocols(network=network, node=sub_tree, sd_id=sd_id,
                                                               tree_id=tree_id,
                                                               protocols=protocols,
                                                               messaging_protocols=messaging_protocols,
                                                               entangling_protocols=entangling_protocols,
                                                               correction_protocols=correction_protocols,
                                                               link_correction_protocols=link_correction_protocols,
                                                               swapping_protocols=swapping_protocols,
                                                               swapping_time_window=swapping_time_window,
                                                               link_layer_requests=link_layer_requests,
                                                               original_graph=graph)
                        path_one_hop_info.append(is_path_one_hop)
                        all_protocols += protocols

                if len(sd_tree) > 1:  # path have SL  and the last two(or one) BSM should be taken care of
                    if len(sd_tree) == 3:  # SL in the middle
                        sl_tree_id = sd_tree[1][0]
                        if not sd_tree[1][2]:
                            left_sl_node, right_sl_node = SLs[sl_tree_id].src, SLs[sl_tree_id].dst
                        else:  # use SL inversely
                            left_sl_node, right_sl_node = SLs[sl_tree_id].dst, SLs[sl_tree_id].src
                        LEFT_FIRST = sd_tree[1][1]
                        left_sl_qpnode = network.get_node(f"Node{left_sl_node.id}")
                        right_sl_qpnode = network.get_node(f"Node{right_sl_node.id}")
                        left_sl_qpnode.left_height[f"{sd_id}_{tree_id}"] = 1 if path_one_hop_info[0] else 2
                        swapping_protocols[left_sl_node.id].link_path_to_sl(path_sd_id=sd_id, path_tree_id=tree_id,
                                                                            sl_tree_id=sl_tree_id, IS_SL_LEFT=False,
                                                                            IS_FIRST_BSM=LEFT_FIRST,
                                                                            path_root_id=
                                                                            sd_tree[0].data.id if not
                                                                            path_one_hop_info[0] else -1)
                        right_sl_qpnode.right_height[f"{sd_id}_{tree_id}"] = 1 if path_one_hop_info[1] else 2
                        swapping_protocols[right_sl_node.id].link_path_to_sl(path_sd_id=sd_id, path_tree_id=tree_id,
                                                                             sl_tree_id=sl_tree_id, IS_SL_LEFT=True,
                                                                             IS_FIRST_BSM=not LEFT_FIRST,
                                                                             path_root_id=
                                                                             sd_tree[2].data.id if not
                                                                             path_one_hop_info[1] else -1)
                        left_sl_qpnode.leftest_id[path_key], left_sl_qpnode.rightest_id[path_key] = src_id, dst_id
                        right_sl_qpnode.leftest_id[path_key], right_sl_qpnode.rightest_id[path_key] = src_id, dst_id
                        # FIX CORRECTION PROTOCOLS
                        correction_protocols[dst_id].add_signal_label(sd_id, tree_id)
                        if LEFT_FIRST:
                            # first correction on right side
                            correction_protocols[right_sl_node.id].add_signal_label(sd_id, tree_id, False)
                            correction_protocols[right_sl_node.id].add_main_root_id(sd_id, tree_id, left_sl_node.id)
                            # right side send BSM res to dst
                            correction_protocols[dst_id].add_main_root_id(sd_id, tree_id, right_sl_node.id)
                        else:
                            # fno correction on either side of SLs
                            correction_protocols[dst_id].add_main_root_id(sd_id, tree_id, left_sl_node.id)
                    else:  # SL is located in left-most or right-most of the path (one BSM)
                        if isinstance(sd_tree[0], tree_node):  # right-most
                            sl_tree_id = sd_tree[1][0]
                            if not sd_tree[1][2]:
                                left_sl_node = SLs[sl_tree_id].src
                            else:  # use SL inversely
                                left_sl_node = SLs[sl_tree_id].dst
                            left_sl_qpnode = network.get_node(f"Node{left_sl_node.id}")
                            left_sl_qpnode.left_height[f"{sd_id}_{tree_id}"] = 1 if path_one_hop_info[0] else 2
                            swapping_protocols[left_sl_node.id].link_path_to_sl(path_sd_id=sd_id, path_tree_id=tree_id,
                                                                                sl_tree_id=sl_tree_id, IS_SL_LEFT=False,
                                                                                IS_FIRST_BSM=True,
                                                                                path_root_id=
                                                                                sd_tree[0].data.id if not
                                                                                path_one_hop_info[0] else -1)
                            network.get_node(f"Node{dst_id}").left_sl_id[f"{sd_id}_{tree_id}"] = f"0_{sl_tree_id}"
                            left_sl_qpnode.leftest_id[path_key], left_sl_qpnode.rightest_id[path_key] = src_id, dst_id
                            # dst == right_sl
                            correction_protocols[dst_id].add_signal_label(sd_id, tree_id, False)
                            correction_protocols[dst_id].add_main_root_id(sd_id, tree_id, left_sl_node.id)
                            correction_protocols[dst_id].node_is_dst(sd_id, tree_id, sl_tree_id)
                            messaging_protocols[dst_id].add_swapping_correction_signal(sd_id, tree_id)

                        else:  # left-most
                            sl_tree_id = sd_tree[0][0]
                            if not sd_tree[0][2]:
                                right_sl_node = SLs[sl_tree_id].dst
                            else:
                                right_sl_node = SLs[sl_tree_id].src
                            right_sl_qpnode = network.get_node(f"Node{right_sl_node.id}")
                            right_sl_qpnode.right_height[f"{sd_id}_{tree_id}"] = 1 if path_one_hop_info[0] else 2
                            swapping_protocols[right_sl_node.id].link_path_to_sl(path_sd_id=sd_id, path_tree_id=tree_id,
                                                                                 sl_tree_id=sl_tree_id, IS_SL_LEFT=True,
                                                                                 IS_FIRST_BSM=True,
                                                                                 path_root_id=
                                                                                 sd_tree[1].data.id if not
                                                                                 path_one_hop_info[0] else -1)
                            right_sl_qpnode.leftest_id[path_key], right_sl_qpnode.rightest_id[path_key] = src_id, dst_id
                            correction_protocols[dst_id].add_signal_label(sd_id, tree_id)
                            correction_protocols[dst_id].add_main_root_id(sd_id, tree_id, right_sl_node.id)
                            messaging_protocols[src_id].add_correction_ack_signal(sd_id, tree_id)
                            teleportation_protocols[src_id].node_is_src_left_sl(sd_id, tree_id, sl_tree_id)
                elif len(sd_tree) == 1 and not isinstance(sd_tree[0], tree_node):
                    sl_tree_id = sd_tree[0][0]
                    self._manager_protocol.update_sd_whole_sl(sd_id, SLs[sl_tree_id].tree.avr_ent_time)

                if len(path_one_hop_info) > 0:
                    teleportation_protocols[src_id].add_protocol_label(sd_id=sd_id, tree_id=tree_id,
                                                                       dst_id=destinations[sd_id_tmp].id,
                                                                       is_path_one_hop=True if len(sd_tree) == 1 and
                                                                                               path_one_hop_info[0]
                                                                       else False)
                for node_id in tree_used_nodes:  # add signal for (sd, tree)
                    messaging_protocols[node_id].add_signals(sd_id, tree_id)
                sd_used_nodes = sd_used_nodes.union(tree_used_nodes)

            for node_id in sd_used_nodes:  # add sd-specific signals
                messaging_protocols[node_id].add_non_sl_signals(sd_id)

            manager_messaging_protocol.add_signals(sd_id)
            self._manager_protocol.add_sd_information(sd_id=sd_id, nodes_id=sd_used_nodes)

        if CHECK_CAPACITIES:
            # check if link-layer requests is not greater than edge maximum capacity
            for link_id, requests in link_layer_requests.items():
                sum_requests = sum([request[0] for request in requests])
                node1, node2 = link_id.split('_')
                e = graph.get_edge(int(node1), int(node2))
                if sum_requests > e.max_channel_capacity:
                    effective_reduce_amount_per_request = (sum_requests - e.max_channel_capacity) / \
                                                          len(requests)
                    reduce_amount = TreeProtocol._effective_rate_to_success_node_rate(
                        effective_reduce_amount_per_request, e)
                    updated_requests = []
                    for prev_rate, left_request, right_request in requests:
                        left_request._rate -= reduce_amount
                        right_request._rate -= reduce_amount
                        updated_requests.append((prev_rate - effective_reduce_amount_per_request, left_request,
                                                 right_request))
                    link_layer_requests[link_id] = updated_requests
            # check if requests do not violate node's generation capacity
            node_link_layer_requests = defaultdict(lambda: defaultdict(list))
            for link_id, requests in link_layer_requests.items():
                node1, node2 = link_id.split('_')
                orig_e = graph.get_edge(int(node1), int(node2))
                for effective_rate, left_request, right_request in requests:
                    node_generation_rate = TreeProtocol._effective_rate_to_node_rate(effective_rate, orig_e)
                    node_link_layer_requests[int(node1)][int(node2)].append([node_generation_rate, left_request,
                                                                             right_request, orig_e])
                    node_link_layer_requests[int(node2)][int(node1)].append([node_generation_rate, left_request,
                                                                             right_request, orig_e])
            for node_id, node_requests in node_link_layer_requests.items():
                sum_requests = sum([request[0] for requests in node_requests.values() for request in requests])
                node_max_capacity = graph.get_node(node_id).max_capacity
                if sum_requests > node_max_capacity:
                    reduce_amount = (sum_requests - node_max_capacity) / \
                                    len([request[0] for requests in node_requests.values() for request in requests])
                    for other_node_id, requests in node_requests.items():
                        idx = 0
                        for _, left_request, right_request, orig_e in requests:
                            node_reduced_rate = TreeProtocol._node_rate_to_success_rate(reduce_amount, orig_e)
                            left_request._rate -= node_reduced_rate
                            right_request._rate -= node_reduced_rate
                            node_link_layer_requests[node_id][other_node_id][idx][0] -= reduce_amount
                            idx += 1
                        # update the other side
                        for req in node_link_layer_requests[other_node_id][node_id]:
                            req[0] -= reduce_amount

        for messaging_protocol in messaging_protocols.values():
            messaging_protocol.start()
        for proto in all_protocols:
            proto.start()
        for ch_proto in charlie_protocols:
            ch_proto.start()
        for link_correction_protocol in link_correction_protocols.values():
            link_correction_protocol.start()
        for entangling_protocol in entangling_protocols.values():
            entangling_protocol.start()
        for correction_protocol in correction_protocols.values():
            correction_protocol.start()
        for swapping_protocol in swapping_protocols.values():
            swapping_protocol.start()
        for teleportation_protocol in teleportation_protocols.values():
            teleportation_protocol.start()

        manager_messaging_protocol.start()
        self._manager_protocol.start()
        self._stats = ns.sim_run(duration=duration * 1e9)

    @property
    def latencies(self):
        return self._manager_protocol.latencies

    @property
    def failed_tries(self):
        return self._manager_protocol.failed_tries

    @property
    def success_rate(self) -> List[float]:
        """return a list of rates, each for a pair of (src, dst)"""
        if self._IS_TELEPORTING:
            # get the total number of successful transmission of qubits from Alices' to Bobs'.
            return [self._dst_protocols[i].successful_count / self._duration for i in range(len(self._dst_protocols))]
        # get the total number of successful tree constructions (only care about EPs'/s)
        return [self._src_protocols[i].tree_success_count / self._duration for i in range(len(self._src_protocols))]

    @property
    def avg_fidelity(self) -> List[float]:
        return [self._fidelity_collector[i].average_fidelity for i in range(len(self._fidelity_collector))]

    @property
    def statistics(self):
        return self._stats

    @staticmethod
    def _used_nodes_edges(node: tree_node, used_nodes: set, used_edges: set):
        if isinstance(node.data, qChannel):
            used_nodes.add(node.data.this.id)
            used_nodes.add(node.data.other.id)
            used_edges.add("{this}-{other}".format(this=node.data.this.id, other=node.data.other.id))
            used_edges.add("{this}-{other}".format(this=node.data.other.id, other=node.data.this.id))
            return
        TreeProtocol._used_nodes_edges(node.left, used_nodes, used_edges)
        TreeProtocol._used_nodes_edges(node.right, used_nodes, used_edges)

    @staticmethod
    def _add_quantum_connections(network: Network, e: qChannel) -> Charlie:
        node_left = network.get_node("Node" + str(e.this.id))
        node_right = network.get_node("Node" + str(e.other.id))
        charlie_id = str(node_left.ID) + "_" + str(node_right.ID)
        # add ports
        node_left.add_ports(["qout_right" + charlie_id, "cfrom_neighbor" + str(node_right.ID),
                             "cto_neighbor" + str(node_right.ID)])
        node_right.add_ports(["qout_left" + charlie_id, "cfrom_neighbor" + str(node_left.ID),
                              "cto_neighbor" + str(node_left.ID),
                              "cfrom_charlie" + charlie_id])
        # create charlie in the middle
        charlie = Charlie(left_node_id=node_left.ID, right_node_id=node_right.ID, bsm_success_rate=e.optical_bsm_rate,
                          bsm_time=e.optical_bsm_time)
        network.add_node(charlie)

        # quantum connections between charlie and left & right
        left_to_charlie_quantum = QuantumConnection(length=node_left.loc.distance(node_right.loc) / 2e3,  # half in km
                                                    channel_success_rate=e.channel_success_rate,
                                                    name=node_left.name + "_to_charlie")
        network.add_connection(node_left, charlie, connection=left_to_charlie_quantum,
                               port_name_node1="qout_right" + charlie_id, port_name_node2="qin_left",
                               label="Quantum")
        right_to_charlie_quantum = QuantumConnection(length=node_left.loc.distance(node_right.loc) / 2e3,
                                                     channel_success_rate=e.channel_success_rate,
                                                     name=node_right.name + "_to_charlie")
        network.add_connection(node_right, charlie, connection=right_to_charlie_quantum,
                               port_name_node1="qout_left" + charlie_id, port_name_node2="qin_right",
                               label="Quantum")

        # add classical connection to send BSM results to right node
        charlie_to_right_classic = ClassicalConnection(length=node_left.loc.distance(node_right.loc) / 2e3)
        network.add_connection(charlie, node_right, connection=charlie_to_right_classic,
                               port_name_node1="cout_right", port_name_node2="cfrom_charlie" + charlie_id,
                               label="Classical")

        # add zero latency classical connections in purpose to let each other know when to restart entangling again
        # node1 - node2
        network.add_connection(node_left, node_right,
                               connection=ClassicalConnection(0, node_left.name + "->" + node_right.name +
                                                              "|ZeroLatency"),
                               port_name_node1="cto_neighbor" + str(node_right.ID),
                               port_name_node2="cfrom_neighbor" + str(node_left.ID),
                               label="classicalZeroLatency")
        # node2 -> node1
        network.add_connection(node_right, node_left,
                               connection=ClassicalConnection(0, node_right.name + "->" + node_left.name +
                                                              "|ZeroLatency"),
                               port_name_node1="cto_neighbor" + str(node_left.ID),
                               port_name_node2="cfrom_neighbor" + str(node_right.ID),
                               label="classicalZeroLatency")
        return charlie

    @staticmethod
    def _add_classical_connections(network: Network, node1: qPNode, node2: qPNode):
        node1.add_ports(["cto_node" + str(node2.ID), "cfrom_node" + str(node2.ID)])
        node2.add_ports(["cto_node" + str(node1.ID), "cfrom_node" + str(node1.ID)])

        network.add_connection(node1, node2,
                               connection=ClassicalConnection(length=node1.loc.distance(node2.loc) / 1e3,  # km
                                                              name=node1.name + "->" + node2.name),
                               port_name_node1="cto_node" + str(node2.ID),
                               port_name_node2="cfrom_node" + str(node1.ID),
                               label="classical")
        network.add_connection(node2, node1,
                               connection=ClassicalConnection(length=node1.loc.distance(node2.loc) / 1e3,
                                                              name=node2.name + "->" + node1.name),
                               port_name_node1="cto_node" + str(node1.ID),
                               port_name_node2="cfrom_node" + str(node2.ID),
                               label="classical")

    @staticmethod
    def _add_classical_sl_zero_connections(network: Network, nodes: set[int], src_node: qPNode, dst_node: qPNode):
        for node_id in nodes: # from left to all
            node = network.get_node(f"Node{node_id}")
            if node_id == src_node.ID:
                continue
            src_node.add_ports(["cto_sl_node" + str(node.ID)])
            node.add_ports(["cfrom_sl_node" + str(src_node.ID)])

            network.add_connection(src_node, node,
                                   connection=ClassicalConnection(length=0,  # km
                                                                  name=src_node.name + "->" + node.name + "|SL"),
                                   port_name_node1="cto_sl_node" + str(node.ID),
                                   port_name_node2="cfrom_sl_node" + str(src_node.ID),
                                   label="SL_classical")
        # from right to left
        # if not network.get_connection(dst_node, src_node, "SL_classical"):
        src_node.add_ports(["cfrom_sl_node" + str(dst_node.ID)])
        dst_node.add_ports(["cto_sl_node" + str(src_node.ID)])
        network.add_connection(dst_node, src_node,
                               connection=ClassicalConnection(length=0,
                                                              name=dst_node.name + "->" + src_node.name + "|SL"),
                               port_name_node1="cto_sl_node" + str(src_node.ID),
                               port_name_node2="cfrom_sl_node" + str(dst_node.ID),
                               label="SL_classical")

    @staticmethod
    def _add_manager_connections(network: Network, manager_node: Node, other_nodes: List[qPNode]):
        for node in other_nodes:
            manager_node.add_ports(["cto_node" + str(node.ID)])
            node.add_ports(["cfrom_manager"])

            network.add_connection(manager_node, node,
                                   connection=ClassicalConnection(length=0,  # km
                                                                  name=manager_node.name + "->" + node.name),
                                   port_name_node1="cto_node" + str(node.ID),
                                   port_name_node2="cfrom_manager",
                                   label="classical")

    @staticmethod
    def _add_sources_to_manager_connections(network: Network, manager_node: Node, sources: List[qPNode]):
        for node in sources:
            manager_node.add_ports(["cfrom_node" + str(node.ID)])
            node.add_ports(["cto_manager"])

            network.add_connection(node, manager_node,
                                   connection=ClassicalConnection(length=0,  # km
                                                                  name=node.name + "->" + manager_node.name),
                                   port_name_node1="cto_manager",
                                   port_name_node2="cfrom_node" + str(node.ID),
                                   label="classical")

    @staticmethod
    def _setup_tree_protocols(network: Network, protocols: List, messaging_protocols: dict[int, MessagingProtocol],
                              node: tree_node, link_layer_requests: dict, original_graph: qGraph,
                              sd_id: int, tree_id: int,
                              correction_protocols: dict[int, CorrectionProtocol],
                              entangling_protocols: dict[int, EntanglingProtocol],
                              swapping_protocols: dict[int, SwappingProtocol],
                              link_correction_protocols: dict[int, LinkCorrectionProtocol],
                              swapping_time_window: float = float('inf'), ACCEPTING_EPS: bool = False,
                              NUMBER_OF_SL_MEMORIES: int = 1, src_id: int = -1, dst_id: int = -1):
        key = f"{sd_id}_{tree_id}"
        if isinstance(node.data, qChannel):
            # get qNodes based on name
            qpnode1 = network.get_node("Node" + str(node.data.this.id))
            qpnode2 = network.get_node("Node" + str(node.data.other.id))

            qpnode1.accepting_eps[key] = ACCEPTING_EPS
            qpnode2.accepting_eps[key] = ACCEPTING_EPS

            # set heights
            qpnode1.left_height[key], qpnode1.right_height[key] = 0, 0
            qpnode2.left_height[key], qpnode2.right_height[key] = 0, 0

            # assigning qubit memories
            qpnode1_right_link_qubit_pos, qpnode1_right_qubit_pos = qpnode1.avail_mem_pos(sd_id), \
                                                                    qpnode1.avail_mem_pos(sd_id,
                                                                                          1 + NUMBER_OF_SL_MEMORIES
                                                                                          if sd_id == 0 and
                                                                                             qpnode1.ID == src_id else 1)
            if key in qpnode1.qubit_mapping:
                # left qubits were assigned
                qpnode1.qubit_mapping[key] = ((qpnode1.qubit_mapping[key][0][0], qpnode1_right_qubit_pos),
                                              (qpnode1.qubit_mapping[key][1][0], qpnode1_right_link_qubit_pos))
            else:
                qpnode1.qubit_mapping[key] = ((-1, qpnode1_right_qubit_pos),
                                              (-1, qpnode1_right_link_qubit_pos))
            qpnode2_left_link_qubit_pos, qpnode2_left_qubit_pos = qpnode2.avail_mem_pos(sd_id), \
                                                                  qpnode2.avail_mem_pos(sd_id,
                                                                                        1 + NUMBER_OF_SL_MEMORIES
                                                                                        if sd_id == 0 and
                                                                                           qpnode2.ID == dst_id else 1)
            if tree_id in qpnode2.qubit_mapping:
                # left qubits were assigned
                qpnode2.qubit_mapping[key] = ((qpnode2_left_qubit_pos, qpnode2.qubit_mapping[key][0][1]),
                                              (qpnode2_left_link_qubit_pos, qpnode2.qubit_mapping[key][1][1]))
            else:
                qpnode2.qubit_mapping[key] = ((qpnode2_left_qubit_pos, -1),
                                              (qpnode2_left_link_qubit_pos, -1))

            if key in qpnode1.neighbors:
                # left neighbor of qpnode1 is already set
                qpnode1.neighbors[key] = (qpnode1.neighbors[key][0], qpnode2.ID)
                qpnode1.entangling[key] = (qpnode1.entangling[key][0], True)
            else:
                qpnode1.neighbors[key] = (-1, qpnode2.ID)
                qpnode1.entangling[key] = (False, True)

            if tree_id in qpnode2.neighbors:
                # right neighbor of qpnode2 is already set
                qpnode2.neighbors[key] = (qpnode1.ID, qpnode2.neighbors[key][1])
                qpnode2.entangling[key] = (True, qpnode2.entangling[key][1])
            else:
                qpnode2.neighbors[key] = (qpnode1.ID, -1)
                qpnode2.entangling[key] = (True, False)

            # create empty set for children
            qpnode1.children_id[key] = set()
            qpnode2.children_id[key] = set()

            # make link layer request here
            left_node, right_node = qpnode1, qpnode2
            left_node_link_pos, right_node_link_pos = qpnode1_right_link_qubit_pos, qpnode2_left_link_qubit_pos
            if f"Charlie_{node.data.this.id}_{node.data.other.id}" in network.nodes:
                charlie_id = f"{node.data.this.id}_{node.data.other.id}"
            else:
                left_node, right_node = qpnode2, qpnode1
                left_node_link_pos, right_node_link_pos = qpnode2_left_link_qubit_pos, qpnode1_right_link_qubit_pos
                charlie_id = f"{node.data.other.id}_{node.data.this.id}"
            effective_rate = 1.0 / node.avr_ent_time
            orig_e = original_graph.get_edge(left_node.ID, right_node.ID)
            # effective_rate = orig_e.residual_capacity  # make it non-throttled
            rate = TreeProtocol._effective_rate_to_success_node_rate(effective_rate, orig_e)
            rndm_val = random()
            protocols.append(EntanglementGenerator(node=left_node, sd_id=sd_id, tree_id=tree_id, rate=rate + rndm_val,
                                                   isLeft=False, IS_SL=sd_id == 0,
                                                   messaging_protocol=messaging_protocols[left_node.ID],
                                                   charlie_id=charlie_id, link_layer_qmem_pos=left_node_link_pos))
            protocols.append(EntanglementGenerator(node=right_node, sd_id=sd_id, tree_id=tree_id, rate=rate + rndm_val,
                                                   isLeft=True, IS_SL=sd_id == 0,
                                                   messaging_protocol=messaging_protocols[right_node.ID],
                                                   charlie_id=charlie_id, link_layer_qmem_pos=right_node_link_pos))
            link_layer_requests[charlie_id].append((effective_rate, protocols[-2], protocols[-1]))
            link_correction_protocols[right_node.ID].add_signal_label(sd_id=sd_id, tree_id=tree_id,
                                                                      link_qmem_pos=right_node_link_pos,
                                                                      left_node_id=left_node.ID)
            entangling_protocols[left_node.ID].add_signal_label(sd_id, tree_id, isMessaging=True,
                                                                isLeft=False if qpnode1 is left_node else True)
            entangling_protocols[right_node.ID].add_signal_label(sd_id, tree_id, isMessaging=False,
                                                                 isLeft=True if qpnode1 is left_node else False)
            messaging_protocols[right_node.ID].add_link_correction_signal(sd_id, tree_id, left_node_id=left_node.ID)

            return qpnode1, qpnode2, qpnode1, qpnode2, True, None

        nodeqp = network.get_node("Node" + str(node.data.id))

        # child_ent_time = (average_ent_time * node.data.bsm_success_rate -
        #                   node.data.bsm_time - node.classical_time) / 1.5
        left_left, _, left, _, _, _ = TreeProtocol._setup_tree_protocols(network=network, protocols=protocols,
                                                                         messaging_protocols=messaging_protocols,
                                                                         node=node.left,
                                                                         tree_id=tree_id, sd_id=sd_id,
                                                                         swapping_protocols=swapping_protocols,
                                                                         entangling_protocols=entangling_protocols,
                                                                         correction_protocols=correction_protocols,
                                                                         link_correction_protocols=link_correction_protocols,
                                                                         link_layer_requests=link_layer_requests,
                                                                         swapping_time_window=swapping_time_window,
                                                                         original_graph=original_graph,
                                                                         ACCEPTING_EPS=ACCEPTING_EPS,
                                                                         NUMBER_OF_SL_MEMORIES=NUMBER_OF_SL_MEMORIES,
                                                                         src_id=src_id, dst_id=dst_id)
        _, right_right, _, right, _, _ = TreeProtocol._setup_tree_protocols(network=network, protocols=protocols,
                                                                            messaging_protocols=messaging_protocols,
                                                                            node=node.right,
                                                                            tree_id=tree_id, sd_id=sd_id,
                                                                            swapping_protocols=swapping_protocols,
                                                                            entangling_protocols=entangling_protocols,
                                                                            correction_protocols=correction_protocols,
                                                                            link_correction_protocols=link_correction_protocols,
                                                                            link_layer_requests=link_layer_requests,
                                                                            swapping_time_window=swapping_time_window,
                                                                            original_graph=original_graph,
                                                                            ACCEPTING_EPS=ACCEPTING_EPS,
                                                                            NUMBER_OF_SL_MEMORIES=NUMBER_OF_SL_MEMORIES,
                                                                            src_id=src_id, dst_id=dst_id)
        # set left & right height, leftest & rightest ids, and children
        nodeqp.left_height[key] = max(left.left_height[key], left.right_height[key]) + 1
        nodeqp.right_height[key] = max(right.left_height[key], right.right_height[key]) + 1

        nodeqp.leftest_id[key], nodeqp.rightest_id[key] = left_left.ID, right_right.ID

        nodeqp.children_id[key] = left.children_id[key].union(right.children_id[key])
        nodeqp.children_id[key].add(left.ID)
        nodeqp.children_id[key].add(right.ID)
        if nodeqp.ID in nodeqp.children_id[key]:  # remove node itself from its children
            nodeqp.children_id[key].remove(nodeqp.ID)

        swapping_protocols[nodeqp.ID].add_protocol_label(sd_id=sd_id, tree_id=tree_id)
        correction_protocols[right_right.ID].add_signal_label(sd_id=sd_id, tree_id=tree_id)
        correction_protocols[right_right.ID].add_main_root_id(sd_id=sd_id, tree_id=tree_id, root_id=nodeqp.ID)

        return left_left, right_right, nodeqp, nodeqp, False, correction_protocols[right_right.ID]

    @staticmethod
    def _effective_rate_to_success_node_rate(effective_rate: float, e: qChannel):
        return effective_rate / (e.channel_success_rate ** 2 * e.optical_bsm_rate)

    @staticmethod
    def _effective_rate_to_node_rate(effective_rate: float, e: qChannel):
        return TreeProtocol._effective_rate_to_success_node_rate(effective_rate, e) / (e.this.gen_success_rate *
                                                                                       e.other.gen_success_rate)

    @staticmethod
    def _node_rate_to_success_rate(node_rate: float, e: qChannel):
        return node_rate * e.this.gen_success_rate * e.other.gen_success_rate


def add_classical_connections(network: Network, node1: qPNode, node2: qPNode):
    if network.get_connection(node1, node2, "classical") is None and \
            not node1.ports["cout_right"].is_connected and \
            not node2.ports["cin_left"].is_connected:
        left_right_cchannel = ClassicalConnection(node1.loc.distance(node2.loc),
                                                  node1.name + "->" + node2.name + "|classical")
        network.add_connection(node1, node2, connection=left_right_cchannel,
                               port_name_node1="cout_right", port_name_node2="cin_left", label="classical")
    if network.get_connection(node2, node1, "classical") is None and \
            not node2.ports["cout_left"].is_connected and \
            not node1.ports["cin_right"].is_connected:
        right_left_cchannel = ClassicalConnection(node1.loc.distance(node2.loc),
                                                  node2.name + "->" + node1.name + "|classical")
        network.add_connection(node2, node1, connection=right_left_cchannel,
                               port_name_node1="cout_left", port_name_node2="cin_right")


if __name__ == "__main__1":
    mem = QuantumMemory("test", num_positions=2)
    qubit = ns.qubits.create_qubits(1)
    mem.put(qubit, positions=[0])
    print(mem.peek(0))
    print(mem.peek(1))
    mem.put(mem.pop(positions=[0]), positions=[1])
    print(mem.peek(0))
    print(mem.peek(1))
    network = Network("test")
    port_names = ["cin_left", "cout_left", "cin_right", "cout_right"]
    node1 = Node("node1", port_names=port_names)
    node2 = Node("node2", port_names=port_names)
    node3 = Node("node3", port_names=port_names)
    network.add_nodes([node1, node2, node3])
    node12_classic = ClassicalConnection(1e4, "1>2|classical")
    node21_classic = ClassicalConnection(1e4, "2>1|classical")
    network.add_connection(node1, node2, connection=node12_classic, port_name_node1="cout_right",
                           port_name_node2="cin_left", label="classical")
    if network.get_connection(node1, node2, "classica") is None and not node1.ports["cout_right"].is_connected and \
            not node2.ports["cin_left"].is_connected:
        network.add_connection(node1, node2, connection=node12_classic, port_name_node1="cout_right",
                               port_name_node2="cin_left")
    network.add_connection(node2, node1, node21_classic, port_name_node1="cout_left", port_name_node2="cin_right")

    # network.add_connection(node1, node2, node12_classic_2way)
    node32_classic = ClassicalConnection(1e5, "3>2|classical")
    node23_classic = ClassicalConnection(1e5, "2>3|classical")
    network.add_connection(node3, node2, node32_classic, port_name_node1="cout_left", port_name_node2="cin_right")
    network.add_connection(node2, node3, node23_classic, port_name_node1="cout_right", port_name_node2="cin_left")

if __name__ == "__main__":
    seed(10)
    nodes = []
    for i in range(100):
        nodes.append(qNode(20, Point(0, i * 10e3)))
    qnodes = [qPNode(node) for node in nodes]
    network = Network("net", qnodes)
    for i in range(len(nodes) - 1):
        edge = qChannel(nodes[i], nodes[i + 1])
        CharlieProtocol(TreeProtocol._add_quantum_connections(network, edge)).start()
    mess_proto = [MessagingProtocol(qnode) for qnode in qnodes]
    for mess in mess_proto:
        mess.start()

    ns.sim_run(duration=100e9)
    # nodes = [qNode(20, Point(0, 50000)), qNode(20, Point(0, 100000)), qNode(20, Point(0, 150000)),
    #          qNode(20, Point(0, 200000)), qNode(20, Point(0, 250000))]
    # edges = [qChannel(nodes[0], nodes[1]), qChannel(nodes[1], nodes[2]), qChannel(nodes[2], nodes[3]),
    #          qChannel(nodes[3], nodes[4])]
    # graph = qGraph(V=nodes, E=edges)
    # root = [tree_node(graph.V[2],
    #                   tree_node(graph.V[1],
    #                             tree_node(graph.E[0], avr_ent_time=0.05),
    #                             tree_node(graph.E[1], avr_ent_time=0.05)),
    #                   tree_node(graph.V[3],
    #                             tree_node(graph.E[2], avr_ent_time=0.05),
    #                             tree_node(graph.E[3], avr_ent_time=0.05))
    #                   ),
    #         tree_node(graph.V[1],
    #                   tree_node(graph.E[0], avr_ent_time=0.05),
    #                   tree_node(graph.E[1], avr_ent_time=0.05))
    #         ]
    #
    # network_protocol = TreeProtocol(graph, root, [graph.V[0], graph.V[0]], [graph.V[4], graph.V[2]], 5)
    # for idx, success_rate in enumerate(network_protocol.success_rate):
    #     print(f"Tree{str(idx)}: {success_rate:.4f} (EPs/s)")

if __name__ == "__main__1":
    # test entanglement link layer
    nodes = [qNode(20, Point(0, 50000)), qNode(20, Point(0, 100000)), qNode(20, Point(0, 150000))]
    qpnodes = [qPNode(node) for node in nodes]

    charlies = [Charlie(str(i) + str(i + 1)) for i in range(len(nodes) - 1)]

    network = Network("link_layer", qpnodes + charlies)

    # add port & connections
    for qpnode in qpnodes:
        qpnode.add_ports(["qout_left", "qout_right", "cfrom_charlie"])

    qpnodes[0].qubit_mapping[0] = ((-1, 0), (-1, 1))
    qpnodes[1].qubit_mapping[0] = ((0, 2), (1, 3))
    qpnodes[2].qubit_mapping[0] = ((0, -1), (1, 0))
    qpnodes[0].neighbors[0] = (-1, 1)
    qpnodes[1].neighbors[0] = (0, 2)
    qpnodes[2].neighbors[0] = (1, -1)

    messaging_protocols = [MessagingProtocol(node) for node in qpnodes]
    entanglment_protocol = [EntanglingProtocol(node, 0) for node in qpnodes]
    protocols = []

    # quantum channel
    for i in range(len(nodes) - 1):
        i_to_charlie_quantum = QuantumConnection(qpnodes[i].loc.distance(qpnodes[i + 1].loc) / 2e3, 0.2,
                                                 "node" + str(i) + "_to_charlie")
        network.add_connection(qpnodes[i], charlies[i], connection=i_to_charlie_quantum,
                               port_name_node1="qout_right", port_name_node2="qin_left", label="Quantum")
        i_plus_to_charlie_quantum = QuantumConnection(qpnodes[i].loc.distance(qpnodes[i + 1].loc) / 2e3, 0.2,
                                                      "node" + str(i + 1) + "_to_charlie")
        network.add_connection(qpnodes[i + 1], charlies[i], connection=i_plus_to_charlie_quantum,
                               port_name_node1="qout_left", port_name_node2="qin_right", label="Quantum")
        charlie_to_right_classic = ClassicalConnection(qpnodes[i].loc.distance(qpnodes[i + 1].loc) / 2e3)
        network.add_connection(charlies[i], qpnodes[i + 1], connection=charlie_to_right_classic,
                               port_name_node1="cout_right", port_name_node2="cfrom_charlie")
        qpnodes[i].add_ports(["cfrom_neighbor" + str(i + 1), "cto_neighbor" + str(i + 1)])
        qpnodes[i + 1].add_ports(["cto_neighbor" + str(i), "cfrom_neighbor" + str(i)])
        i_to_plus_i = ClassicalConnection(0)
        i_plus_to_i = ClassicalConnection(0)
        network.add_connection(qpnodes[i], qpnodes[i + 1], connection=i_to_plus_i,
                               port_name_node1="cto_neighbor" + str(i + 1), port_name_node2="cfrom_neighbor" + str(i),
                               label="Classical")
        network.add_connection(qpnodes[i + 1], qpnodes[i], connection=i_plus_to_i,
                               port_name_node1="cto_neighbor" + str(i), port_name_node2="cfrom_neighbor" + str(i + 1),
                               label="Classical")
        if 0 in qpnodes[i].entangling:
            qpnodes[i].entangling[0] = (qpnodes[i].entangling[0][0], True)
        else:
            qpnodes[i].entangling[0] = (False, True)
        if 0 in qpnodes[i + 1].entangling:
            qpnodes[i + 1].entangling[0] = (True, qpnodes[i + 1].entangling[0][1])
        else:
            qpnodes[i + 1].entangling[0] = (True, False)
        protocols.append(EntanglementGenerator(qpnodes[i], 0, 10, False))
        protocols.append(EntanglementGenerator(qpnodes[i + 1], 0, 10, True))
        protocols.append(CharlieProtocol(charlies[i]))
        protocols.append(LinkCorrectionProtocol(qpnodes[i + 1], 0, messaging_protocols[i + 1]))
        entanglment_protocol[i].add_protocol(messaging_protocols[i], False)
        entanglment_protocol[i + 1].add_protocol(protocols[-1], True)

    for protocol in messaging_protocols:
        protocol.start()
    for protocol in entanglment_protocol:
        protocol.start()
    for protocol in protocols:
        protocol.start()

    ns.sim_run(duration=10e10)
