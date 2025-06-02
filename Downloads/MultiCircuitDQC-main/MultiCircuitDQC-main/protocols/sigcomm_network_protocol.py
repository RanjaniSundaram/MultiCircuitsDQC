import math
import random
import sys

from commons.tools import ATT_LENGTH

sys.path.append("..")

from commons.qNode import qNode
from commons.qGraph import qGraph
from commons.qChannel import qChannel
from commons.tree_node import tree_node
import netsquid as ns
from protocols.sigcomm_protocol_bases import *
from netsquid.nodes import Network
from protocols.sigcomm_component_bases import ClassicalConnection, EntanglingConnection, qPNode, Charlie, QuantumConnection
from typing import List

from netsquid.components import QuantumMemory
from commons.Point import Point
from netsquid.nodes import Node
from random import seed
from collections import defaultdict

PORT_NAMES_CLASSICAL = ["c" + inOrOut + "_" + direction for inOrOut in ["in", "out"]
                        for direction in ["left", "right", "parent"]]
PORT_NAMES_QUANTUM = ["q" + inOrOut + "_" + direction for inOrOut in ["in"]
                      for direction in ["left", "right"]]


class TreeProtocol:

    def __init__(self, graph: qGraph, trees: List,
                 sources: List[qNode], destinations: List[qNode], duration: int, ref_length: float,
                 swapping_time_window: float = float('inf'), IS_TELEPORTING: bool = False):
        """duration(s) to run the simulation
        trees: list of tuples (tree, w')"""
        ns.set_qstate_formalism(ns.QFormalism.DM)
        ns.set_random_state(42)
        ns.sim_reset()
        # get only those nodes and edges that are going to be used to save memory and efficiency
        used_nodes, used_edges = set(), set()
        for tree, _ in trees:
            TreeProtocol._used_nodes_edges(tree, used_nodes, used_edges)

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

        # setup quantum connection for each edge in G.E
        charlie_protocols = []
        # link_layer_max_capacity = {}
        for e in graph.E:
            if "{this}-{other}".format(this=e.this.id, other=e.other.id) in used_edges:
                charlie = TreeProtocol._add_quantum_connections(network=network, e=e, ref_length=ref_length)
                charlie_protocols.append(CharlieProtocol(charlie))
                charlie_id = "{this}_{other}".format(this=e.this.id, other=e.other.id)
                # link_layer_max_capacity[charlie_id] = e

        messaging_protocols = {node.ID: MessagingProtocol(node) for node in nodes if node.ID in used_nodes}  # one for each node for all trees
        all_protocols = []
        link_layer_requests = defaultdict(list)
        for tree_id, tree_root in enumerate(trees):
            for id, messaging_protocol in messaging_protocols.items():
                messaging_protocol.add_signals(tree_id)
            protocols = []
            entagling_protocols = {}
            _, _, _, _, is_path_one_hop, dst_correction_protocol = \
                TreeProtocol._setup_tree_protocols(network=network, node=tree_root[0], tree_id=tree_id,
                                                   protocols=protocols,
                                                   messaging_protocols=messaging_protocols,
                                                   entangling_protocols=entagling_protocols,
                                                   correction_protocols={},
                                                   swapping_time_window=swapping_time_window,
                                                   link_layer_requests=link_layer_requests,
                                                   original_graph=graph,
                                                   width=tree_root[1],
                                                   ref_length=ref_length)

            srcQPNode = network.get_node("Node" + str(sources[tree_id].id))
            dstQPNOde = network.get_node("Node" + str(destinations[tree_id].id))

            all_tree_nodes_id = set()
            if not is_path_one_hop:
                rootqp = network.get_node("Node" + str(trees[tree_id][0].data.id))
                all_tree_nodes_id = rootqp.children_id[tree_id].union({rootqp.ID})
                # all_tree_nodes_id.remove(dstQPNOde.ID)
                # TreeProtocol.add_node_to_children_classical_connections(network, dstQPNOde, all_tree_nodes_id)
            # add protocols
            protocols.append(TeleportationProtocol(node=srcQPNode, qubit_protocol=QubitGenerationProtocol(srcQPNode),
                                                   dst_id=dstQPNOde.ID,
                                                   is_path_one_hop=is_path_one_hop, tree_id=tree_id,
                                                   tree_nodes_id=all_tree_nodes_id,
                                                   messaging_protocol=messaging_protocols[srcQPNode.ID],
                                                   IS_TELEPORTING=self._IS_TELEPORTING,
                                                   entangling_protocol=entagling_protocols[srcQPNode.ID]))
            protocols.append(TeleportationCorrectionProtocol(node=dstQPNOde, src_id=srcQPNode.ID, tree_id=tree_id,
                                                             messaging_protocol=messaging_protocols[dstQPNOde.ID],
                                                             is_path_one_hop=is_path_one_hop,
                                                             correction_protocol=dst_correction_protocol,
                                                             tree_nodes_id=all_tree_nodes_id,
                                                             entangling_protocol=entagling_protocols[dstQPNOde.ID]))
            self._src_protocols.append(protocols[-2])
            self._dst_protocols.append(protocols[-1])
            protocols.append(FidelityCalculatorProtocol(tree_id, src=srcQPNode, dst=dstQPNOde,
                                                        correction_protocol=dst_correction_protocol,
                                                        is_path_one_hop=is_path_one_hop,
                                                        teleportation_protocol=self._src_protocols[tree_id]))
            self._fidelity_collector.append(protocols[-1])
            # to keep track of successful EPs rate for each pair of (src, dst)
            all_protocols += protocols

        # check if link-layer requests is not greater than edge maximum capacity
        # for link_id, requests in link_layer_requests.items():
        #     sum_requests = sum([request[0] for request in requests])
        #     node1, node2 = link_id.split('_')
        #     e = graph.get_edge(int(node1), int(node2))
        #     if sum_requests > e.max_channel_capacity:
        #         effective_reduce_amount_per_request = (sum_requests - e.max_channel_capacity) / \
        #                                               len(requests)
        #         reduce_amount = TreeProtocol._effective_rate_to_success_node_rate(effective_reduce_amount_per_request, e)
        #         updated_requests = []
        #         for prev_rate, left_request, right_request in requests:
        #             left_request._rate -= reduce_amount
        #             right_request._rate -= reduce_amount
        #             updated_requests.append((prev_rate - effective_reduce_amount_per_request, left_request,
        #                                      right_request))
        #         link_layer_requests[link_id] = updated_requests
        # # check if requests do not violate node's generation capacity
        # node_link_layer_requests = defaultdict(lambda: defaultdict(list))
        # for link_id, requests in link_layer_requests.items():
        #     node1, node2 = link_id.split('_')
        #     orig_e = graph.get_edge(int(node1), int(node2))
        #     for effective_rate, left_request, right_request in requests:
        #         node_generation_rate = TreeProtocol._effective_rate_to_node_rate(effective_rate, orig_e, ref_length)
        #         node_link_layer_requests[int(node1)][int(node2)].append([node_generation_rate, left_request,
        #                                                                  right_request, orig_e])
        #         node_link_layer_requests[int(node2)][int(node1)].append([node_generation_rate, left_request,
        #                                                                  right_request, orig_e])
        # for node_id, node_requests in node_link_layer_requests.items():
        #     sum_requests = sum([request[0] for requests in node_requests.values() for request in requests])
        #     node_max_capacity = graph.get_node(node_id).max_capacity
        #     if sum_requests > node_max_capacity:
        #         reduce_amount = (sum_requests - node_max_capacity) / \
        #                         len([request[0] for requests in node_requests.values() for request in requests])
        #         for other_node_id, requests in node_requests.items():
        #             idx = 0
        #             for _, left_request, right_request, orig_e in requests:
        #                 node_reduced_rate = TreeProtocol._node_rate_to_success_rate(reduce_amount, orig_e)
        #                 left_request._rate -= node_reduced_rate
        #                 right_request._rate -= node_reduced_rate
        #                 node_link_layer_requests[node_id][other_node_id][idx][0] -= reduce_amount
        #                 idx += 1
        #             # update the other side
        #             for req in node_link_layer_requests[other_node_id][node_id]:
        #                 req[0] -= reduce_amount


        for key, messaging_protocol in messaging_protocols.items():
            messaging_protocol.start()
        for protocol in charlie_protocols:
            protocol.start()
        for protocol in all_protocols:
            protocol.start()
        self._stats = ns.sim_run(duration=duration * 1e9)

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
    def _get_ref_length(node: tree_node):
        if isinstance(node.data, qChannel):
            return node.data.length
        return max(TreeProtocol._get_ref_length(node.left), TreeProtocol._get_ref_length(node.right))

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
    def _add_quantum_connections(network: Network, e: qChannel, ref_length: float):
        node_left = network.get_node("Node" + str(e.this.id))
        node_right = network.get_node("Node" + str(e.other.id))
        charlie_id = str(node_left.ID) + "_" + str(node_right.ID)
        # add ports
        node_left.add_ports(["qout_right" + str(idx) + "_" + charlie_id for idx in range(e.channels_num)]
                            + ["cfrom_neighbor" + str(node_right.ID), "cto_neighbor" + str(node_right.ID)])
        node_right.add_ports(["qout_left" + str(idx) + "_" + charlie_id for idx in range(e.channels_num)]
                             + ["cfrom_neighbor" + str(node_left.ID), "cto_neighbor" + str(node_left.ID),
                               "cfrom_charlie" + charlie_id])
        # create charlie in the middle
        charlie = Charlie(left_node_id=node_left.ID, right_node_id=node_right.ID, bsm_success_rate=e.optical_bsm_rate,
                          bsm_time=e.optical_bsm_time, number_of_subchannels=e.channels_num)
        network.add_node(charlie)

        # quantum connections between charlie and left & right
        for idx in range(e.channels_num):
            left_to_charlie_quantum = QuantumConnection(length=0 / 2e3,  # half in km
                                                        channel_success_rate=e.channel_success_rate,
                                                        name=node_left.name + "_to_charlie" + str(idx))
            network.add_connection(node_left, charlie, connection=left_to_charlie_quantum,
                                   port_name_node1="qout_right" + str(idx) + "_" + charlie_id,
                                   port_name_node2="qin_left" + str(idx),
                                   label="Quantum" + str(idx))
            right_to_charlie_quantum = QuantumConnection(length=0 / 2e3,
                                                         channel_success_rate=e.channel_success_rate,
                                                         name=node_right.name + "_to_charlie" + str(idx))
            network.add_connection(node_right, charlie, connection=right_to_charlie_quantum,
                                   port_name_node1="qout_left" + str(idx) + "_" + charlie_id,
                                   port_name_node2="qin_right" + str(idx),
                                   label="Quantum" + str(idx))

        # add classical connection to send BSM results to right node
        charlie_to_right_classic = ClassicalConnection(length=0)  # node_left.loc.distance(node_right.loc) / 2e3)
        network.add_connection(charlie, node_right, connection=charlie_to_right_classic,
                               port_name_node1="cout_right", port_name_node2="cfrom_charlie" + charlie_id)

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
        network.add_node(Node(name=charlie.name + "Invoker"))
        return charlie

    @staticmethod
    def _add_classical_connections(network: Network, node1: qPNode, node2: qPNode):
        node1.add_ports(["cto_node" + str(node2.ID), "cfrom_node" + str(node2.ID)])
        node2.add_ports(["cto_node" + str(node1.ID), "cfrom_node" + str(node1.ID)])

        network.add_connection(node1, node2,
                               connection=ClassicalConnection(length=0,  # node1.loc.distance(node2.loc) / 1e3,  # km
                                                              name=node1.name + "->" + node2.name),
                               port_name_node1="cto_node" + str(node2.ID),
                               port_name_node2="cfrom_node" + str(node1.ID),
                               label="classical")
        network.add_connection(node2, node1,
                               connection=ClassicalConnection(length=0, # node1.loc.distance(node2.loc) / 1e3,
                                                              name=node2.name + "->" + node1.name),
                               port_name_node1="cto_node" + str(node1.ID),
                               port_name_node2="cfrom_node" + str(node2.ID),
                               label="classical")

    @staticmethod
    def _setup_tree_protocols(network: Network, protocols: List, messaging_protocols: dict,
                              node: tree_node, link_layer_requests: dict, original_graph: qGraph, ref_length: float,
                              tree_id: int = 0, correction_protocols={}, entangling_protocols={},
                              swapping_time_window: float = float('inf'),
                              width: int = 1):
        if isinstance(node.data, qChannel):
            # get qNodes based on name
            qpnode1 = network.get_node("Node" + str(node.data.this.id))
            qpnode2 = network.get_node("Node" + str(node.data.other.id))

            if tree_id in qpnode1.neighbors:
                # left neighbor of qpnode1 is already set
                qpnode1.neighbors[tree_id] = (qpnode1.neighbors[tree_id][0], qpnode2.ID)
            else:
                qpnode1.neighbors[tree_id] = (-1, qpnode2.ID)

            if tree_id in qpnode2.neighbors:
                # right neighbor of qpnode2 is already set
                qpnode2.neighbors[tree_id] = (qpnode1.ID, qpnode2.neighbors[tree_id][1])
            else:
                qpnode2.neighbors[tree_id] = (qpnode1.ID, -1)

            if qpnode1.ID not in entangling_protocols:
                protocols.append(EntanglingProtocol(qpnode1, tree_id))
                entangling_protocols[qpnode1.ID] = protocols[-1]
            if qpnode2.ID not in entangling_protocols:
                protocols.append(EntanglingProtocol(qpnode2, tree_id))
                entangling_protocols[qpnode2.ID] = protocols[-1]

            # set heights
            qpnode1.left_height[tree_id], qpnode1.right_height[tree_id] = 0, 0
            qpnode2.left_height[tree_id], qpnode2.right_height[tree_id] = 0, 0

            # create empty set for children
            qpnode1.children_id[tree_id] = set()
            qpnode2.children_id[tree_id] = set()

            if tree_id not in qpnode1.link_to_network_pos:
                qpnode1.link_to_network_pos[tree_id] = {}
            if tree_id not in qpnode1.entangling:
                qpnode1.entangling[tree_id] = {}
            if tree_id not in qpnode1.qubit_to_port:
                qpnode1.qubit_to_port[tree_id] = {}
            if tree_id not in qpnode2.link_to_network_pos:
                qpnode2.link_to_network_pos[tree_id] = {}
            if tree_id not in qpnode2.entangling:
                qpnode2.entangling[tree_id] = {}
            if tree_id not in qpnode2.qubit_to_port:
                qpnode2.qubit_to_port[tree_id] = {}

            left_node, right_node = qpnode1, qpnode2
            if "Charlie_" + str(node.data.this.id) + "_" + str(node.data.other.id) in network.nodes:
                charlie_id = str(node.data.this.id) + "_" + str(node.data.other.id)
            else:
                left_node, right_node = qpnode2, qpnode1
                charlie_id = str(node.data.other.id) + "_" + str(node.data.this.id)
            charlie = network.get_node('Charlie_' + charlie_id)
            charlie_invoker = network.get_node(charlie.name + "Invoker")

            # attempting rate
            effective_rate = 1.0 / node.avr_ent_time
            orig_e = original_graph.get_edge(left_node.ID, right_node.ID)
            # rate = TreeProtocol._effective_rate_to_success_node_rate(effective_rate, orig_e)
            rate = TreeProtocol._effective_rate_to_node_rate(effective_rate, orig_e, ref_length=ref_length)

            # invoker protocol
            charlie.add_ports(["cin_tree" + str(tree_id)])
            charlie_invoker.add_ports(["cout_tree" + str(tree_id)])
            invoker_to_charlie_connection = ClassicalConnection(length=0 / 2e3,
                                                                name="Invoker->" + charlie.name + str(tree_id))
            network.add_connection(charlie_invoker, charlie, connection=invoker_to_charlie_connection,
                                   port_name_node1="cout_tree" + str(tree_id),
                                   port_name_node2="cin_tree" + str(tree_id),
                                   label="Classical" + str(tree_id))
            protocols.append(CharlieInvoker(charlie_invoker, rate=rate, tree_id=tree_id))
            for w in range(width):
                channel_idx = charlie.avail_channels()
                left_node_right_qubit_pos, left_node_right_link_qubit_pos = left_node.avail_mem_pos(), \
                                                                            left_node.avail_mem_pos()
                right_node_left_qubit_pos, right_node_left_link_qubit_pos = right_node.avail_mem_pos(), \
                                                                            right_node.avail_mem_pos()
                #paired qubits

                if left_node == qpnode1:
                    if tree_id not in left_node.right_paired_with:
                        left_node.right_paired_with[tree_id] = {}
                        left_node.inverse_right_paired_with[tree_id] = {}
                    if tree_id not in right_node.left_paired_with:
                        right_node.left_paired_with[tree_id] = {}
                        right_node.inverse_left_paired_with[tree_id] = {}
                    left_node.right_paired_with[tree_id][left_node_right_qubit_pos] = right_node_left_qubit_pos
                    left_node.inverse_right_paired_with[tree_id][right_node_left_qubit_pos] = left_node_right_qubit_pos

                    right_node.left_paired_with[tree_id][right_node_left_qubit_pos] = left_node_right_qubit_pos
                    right_node.inverse_left_paired_with[tree_id][left_node_right_qubit_pos] = right_node_left_qubit_pos
                else:
                    if tree_id not in left_node.left_paired_with:
                        left_node.left_paired_with[tree_id] = {}
                        left_node.inverse_left_paired_with[tree_id] = {}
                    if tree_id not in right_node.right_paired_with:
                        right_node.right_paired_with[tree_id] = {}
                        right_node.inverse_right_paired_with[tree_id] = {}
                    left_node.left_paired_with[tree_id][left_node_right_qubit_pos] = right_node_left_qubit_pos
                    left_node.inverse_left_paired_with[tree_id][right_node_left_qubit_pos] = left_node_right_qubit_pos

                    right_node.right_paired_with[tree_id][right_node_left_qubit_pos] = left_node_right_qubit_pos
                    right_node.inverse_right_paired_with[tree_id][left_node_right_qubit_pos] = right_node_left_qubit_pos

                charlie.add_request(tree_id=tree_id, left_qubit_pos=left_node_right_link_qubit_pos,
                                    right_qubit_pos=right_node_left_link_qubit_pos)

                left_node.link_to_network_pos[tree_id][left_node_right_link_qubit_pos] = left_node_right_qubit_pos
                left_node.entangling[tree_id][left_node_right_qubit_pos] = True
                left_node.qubit_to_port[tree_id][left_node_right_link_qubit_pos] = channel_idx
                right_node.link_to_network_pos[tree_id][right_node_left_link_qubit_pos] = right_node_left_qubit_pos
                right_node.entangling[tree_id][right_node_left_qubit_pos] = True
                right_node.qubit_to_port[tree_id][right_node_left_link_qubit_pos] = channel_idx
                # make link layer request here

                protocols.append(EntanglementGenerator(node=left_node, tree_id=tree_id, rate=rate, isLeft=False,
                                                       charlie_id=charlie_id,
                                                       link_layer_qmem_pos=left_node_right_link_qubit_pos,
                                                       gen_success_rate=left_node.gen_success_rate))
                protocols.append(EntanglementGenerator(node=right_node, tree_id=tree_id, rate=rate, isLeft=True,
                                                       charlie_id=charlie_id,
                                                       link_layer_qmem_pos=right_node_left_link_qubit_pos,
                                                       gen_success_rate=right_node.gen_success_rate))
                link_layer_requests[charlie_id].append((effective_rate, protocols[-2], protocols[-1]))
            messaging_protocols[right_node.ID].add_link_correction_signal(tree_id=tree_id, left_node_id=left_node.ID)
            protocols.append(LinkCorrectionProtocol(node=right_node, tree_id=tree_id,
                                                    messaging_protocol=messaging_protocols[right_node.ID],
                                                    left_node_id=left_node.ID))
            if qpnode1 == left_node:
                entangling_protocols[left_node.ID].add_protocol(messaging_protocols[left_node.ID], False)
                entangling_protocols[right_node.ID].add_protocol(protocols[-1], True)
            else:
                entangling_protocols[left_node.ID].add_protocol(messaging_protocols[left_node.ID], True)
                entangling_protocols[right_node.ID].add_protocol(protocols[-1], False)

            return qpnode1, qpnode2, qpnode1, qpnode2, True, None

        nodeqp = network.get_node("Node" + str(node.data.id))

        # child_ent_time = (average_ent_time * node.data.bsm_success_rate -
        #                   node.data.bsm_time - node.classical_time) / 1.5
        left_left, _, left, _, _, _ = TreeProtocol._setup_tree_protocols(network=network, protocols=protocols,
                                                                         messaging_protocols=messaging_protocols,
                                                                         node=node.left,
                                                                         tree_id=tree_id,
                                                                         entangling_protocols=entangling_protocols,
                                                                         correction_protocols=correction_protocols,
                                                                         link_layer_requests=link_layer_requests,
                                                                         original_graph=original_graph,
                                                                         swapping_time_window=swapping_time_window,
                                                                         width=width,
                                                                         ref_length=ref_length)
        _, right_right, _, right, _, _ = TreeProtocol._setup_tree_protocols(network=network, protocols=protocols,
                                                                            messaging_protocols=messaging_protocols,
                                                                            node=node.right, tree_id=tree_id,
                                                                            entangling_protocols=entangling_protocols,
                                                                            correction_protocols=correction_protocols,
                                                                            link_layer_requests=link_layer_requests,
                                                                            original_graph=original_graph,
                                                                            swapping_time_window=swapping_time_window,
                                                                            width=width,
                                                                            ref_length=ref_length)
        # set left & right height, leftest & rightest ids, and children
        nodeqp.left_height[tree_id] = max(left.left_height[tree_id], left.right_height[tree_id]) + 1
        nodeqp.right_height[tree_id] = max(right.left_height[tree_id], right.right_height[tree_id]) + 1

        nodeqp.leftest_id[tree_id], nodeqp.rightest_id[tree_id] = left_left.ID, right_right.ID

        nodeqp.children_id[tree_id] = left.children_id[tree_id].union(right.children_id[tree_id])
        nodeqp.children_id[tree_id].add(left.ID)
        nodeqp.children_id[tree_id].add(right.ID)
        if nodeqp.ID in nodeqp.children_id[tree_id]:  # remove node itself from its children
            nodeqp.children_id[tree_id].remove(nodeqp.ID)

        # adding & starting protocol    z
        # subtree (left_left -- nodeqp -- right_right
        correction_protocol, messaging_protocol = None, None
        if nodeqp.left_height[tree_id] > 1:
            # node should wait to its left subtrees to be completed
            correction_protocol = correction_protocols[nodeqp.ID]
        if nodeqp.right_height[tree_id] > 1:
            # node should wait to its right subtree to be completed;
            messaging_protocol = messaging_protocols[nodeqp.ID]

        protocols.append(SwappingProtocol(nodeqp, tree_id=tree_id,
                                          correction_protocol=correction_protocol,
                                          messaging_protocol=messaging_protocol,  # none means directly from link layer
                                          swapping_time_window=swapping_time_window,
                                          entangling_protocol=entangling_protocols[nodeqp.ID]))
        if right_right.ID not in correction_protocols:
            protocols.append(CorrectionProtocol(right_right, messaging_protocol=messaging_protocols[right_right.ID],
                                                tree_id=tree_id, main_root_id=nodeqp.ID))
            correction_protocols[right_right.ID] = protocols[-1]
        else:
            correction_protocols[right_right.ID].main_root_id = nodeqp.ID

        return left_left, right_right, nodeqp, nodeqp, False, correction_protocols[right_right.ID]

    @staticmethod
    def _effective_rate_to_success_node_rate(effective_rate: float, e: qChannel, ref_length):
        channel_success_rate = math.exp(-ref_length / (2 * ATT_LENGTH))
        return effective_rate / (channel_success_rate ** 2 * e.optical_bsm_rate)

    @staticmethod
    def _effective_rate_to_node_rate(effective_rate: float, e: qChannel, ref_length):
        return TreeProtocol._effective_rate_to_success_node_rate(effective_rate, e, ref_length) / \
               (e.this.gen_success_rate * e.other.gen_success_rate)

    @staticmethod
    def _node_rate_to_success_rate(node_rate: float, e: qChannel):
        return node_rate * e.this.gen_success_rate * e.other.gen_success_rate

    # @staticmethod
    # def _setup_tree_connections(network: Network, protocols: List, messaging_protocols: dict, node: tree_node,
    #                             tree_id: int = 0, correction_protocols: dict = {}, entangling_protocols=set()):
    #     if isinstance(node.data, qChannel):
    #         # get qNodes based on name
    #         qpnode1 = network.get_node("Node" + str(node.data.this.id))
    #         qpnode2 = network.get_node("Node" + str(node.data.other.id))
    #         if qpnode1.ID not in entangling_protocols:
    #             entangling_protocols.add(qpnode1.ID)
    #             protocols.append(EntanglingProtocol(qpnode1, tree_id))
    #         if qpnode2.ID not in entangling_protocols:
    #             entangling_protocols.add(qpnode2.ID)
    #             protocols.append(EntanglingProtocol(qpnode2, tree_id))
    #         qpnode1.left_height[tree_id], qpnode1.right_height[tree_id] = 0, 0
    #         qpnode2.left_height[tree_id], qpnode2.right_height[tree_id] = 0, 0
    #
    #         # add ports for this quantum channel
    #         qpnode1.add_ports(PORT_NAMES_QUANTUM)
    #         qpnode2.add_ports(PORT_NAMES_QUANTUM)
    #
    #         # assigning qubit memories
    #         qpnode1_right_qubit_pos, qpnode1_right_link_qubit_pos = qpnode1.avail_mem_pos(), qpnode1.avail_mem_pos()
    #         if tree_id in qpnode1.qubit_mapping:
    #             # left qubits were assigned
    #             qpnode1.qubit_mapping[tree_id] = ((qpnode1.qubit_mapping[tree_id][0][0], qpnode1_right_qubit_pos),
    #                                               (qpnode1.qubit_mapping[tree_id][1][0], qpnode1_right_link_qubit_pos))
    #         else:
    #             qpnode1.qubit_mapping[tree_id] = ((-1, qpnode1_right_qubit_pos),
    #                                               (-1, qpnode1_right_link_qubit_pos))
    #         qpnode2_left_qubit_pos, qpnode2_left_link_qubit_pos = qpnode2.avail_mem_pos(), qpnode2.avail_mem_pos()
    #         if tree_id in qpnode2.qubit_mapping:
    #             # left qubits were assigned
    #             qpnode2.qubit_mapping[tree_id] = ((qpnode2_left_qubit_pos, qpnode2.qubit_mapping[tree_id][0][1]),
    #                                               (qpnode2_left_link_qubit_pos, qpnode2.qubit_mapping[tree_id][1][1]))
    #         else:
    #             qpnode2.qubit_mapping[tree_id] = ((qpnode2_left_qubit_pos, -1),
    #                                               (qpnode2_left_link_qubit_pos, -1))
    #
    #         # quantum entangling
    #         qconn = EntanglingConnection(node.data.this.loc.distance(node.data.other.loc), 0.01,
    #                                      qpnode1.name + "-" + qpnode2.name + "|Entangling")  # frequency should be based on demand
    #         port_ac, port_bc = network.add_connection(qpnode1, qpnode2, connection=qconn,
    #                                                   label="quantum",
    #                                                   port_name_node1="qin_right",
    #                                                   port_name_node2="qin_left")
    #         qpnode1.ports[port_ac].forward_input(qpnode1.qmemory.ports["qin" + str(qpnode1_right_link_qubit_pos)])
    #         qpnode2.ports[port_bc].forward_input(qpnode2.qmemory.ports["qin" + str(qpnode2_left_link_qubit_pos)])
    #
    #         # classical channel between two neighbors to talk to each other to resume ACCEPTING EQs
    #         # due to synchronization issues, length of the channels are assumed to be zero
    #         # set neighbors
    #         if tree_id in qpnode1.neighbors:
    #             # left neighbor of qpnode1 is already set
    #             qpnode1.neighbors[tree_id] = (qpnode1.neighbors[tree_id][0], qpnode2.ID)
    #             qpnode1.entangling[tree_id] = (qpnode1.entangling[tree_id][0], True)
    #         else:
    #             qpnode1.neighbors[tree_id] = (-1, qpnode2.ID)
    #             qpnode1.entangling[tree_id] = (False, True)
    #
    #         if tree_id in qpnode2.neighbors:
    #             # right neighbor of qpnode2 is already set
    #             qpnode2.neighbors[tree_id] = (qpnode1.ID, qpnode2.neighbors[tree_id][1])
    #             qpnode2.entangling[tree_id] = (True, qpnode2.entangling[tree_id][1])
    #         else:
    #             qpnode2.neighbors[tree_id] = (qpnode1.ID, -1)
    #             qpnode2.entangling[tree_id] = (True, False)
    #
    #         qpnode1.add_ports(["cto_neighbor" + str(qpnode2.ID), "cfrom_neighbor" + str(qpnode2.ID)])
    #         qpnode2.add_ports(["cto_neighbor" + str(qpnode1.ID), "cfrom_neighbor" + str(qpnode1.ID)])
    #         # qpnode1 -> qpnode2
    #         qpnode1_qpnode2_cchannel = ClassicalConnection(0, qpnode1.name + "->" + qpnode2.name)
    #         network.add_connection(qpnode1, qpnode2, connection=qpnode1_qpnode2_cchannel,
    #                                port_name_node1="cto_neighbor" + str(qpnode2.ID),
    #                                port_name_node2="cfrom_neighbor" + str(qpnode1.ID),
    #                                label="classicalZeroLatency")
    #         # qpnode2 -> qpnode1
    #         qpnode2_qpnode1_cchannel = ClassicalConnection(0, qpnode2.name + "->" + qpnode1.name)
    #         network.add_connection(qpnode2, qpnode1, connection=qpnode2_qpnode1_cchannel,
    #                                port_name_node1="cto_neighbor" + str(qpnode1.ID),
    #                                port_name_node2="cfrom_neighbor" + str(qpnode2.ID),
    #                                label="classicalZeroLatency")
    #
    #         # create empty set for children
    #         qpnode1.children_id[tree_id] = set()
    #         qpnode2.children_id[tree_id] = set()
    #
    #         return qpnode1, qpnode2, qpnode1, qpnode2, True, None
    #     nodeqp = network.get_node("Node" + str(node.data.id))
    #     left_left, _, left, _, _, _ = TreeProtocol._setup_tree_connections(network=network, protocols=protocols,
    #                                                                        messaging_protocols=messaging_protocols,
    #                                                                        node=node.left, tree_id=tree_id,
    #                                                                        entangling_protocols=entangling_protocols)
    #     _, right_right, _, right, _, _ = TreeProtocol._setup_tree_connections(network=network, protocols=protocols,
    #                                                                           messaging_protocols=messaging_protocols,
    #                                                                           node=node.right, tree_id=tree_id,
    #                                                                           entangling_protocols=entangling_protocols)
    #     # set left & right height, leftest & rightest ids, and children
    #     nodeqp.left_height[tree_id] = max(left.left_height[tree_id], left.right_height[tree_id]) + 1
    #     nodeqp.right_height[tree_id] = max(right.left_height[tree_id], right.right_height[tree_id]) + 1
    #
    #     nodeqp.leftest_id[tree_id], nodeqp.rightest_id[tree_id] = left_left.ID, right_right.ID
    #
    #     nodeqp.children_id[tree_id] = left.children_id[tree_id].union(right.children_id[tree_id])
    #     nodeqp.children_id[tree_id].add(left.ID)
    #     nodeqp.children_id[tree_id].add(right.ID)
    #     if nodeqp.ID in nodeqp.children_id[tree_id]:  # remove node itself from its children
    #         nodeqp.children_id[tree_id].remove(nodeqp.ID)
    #
    #     # classical connections
    #     TreeProtocol._add_tree_classical_connection(network, nodeqp, tree_id)
    #
    #     # adding & starting protocol    z
    #     # subtree (left_left -- nodeqp -- right_right
    #     correction_protocol, messaging_protocol = None, None
    #     if nodeqp.left_height[tree_id] > 1:
    #         # node should wait to its left subtrees to be completed
    #         correction_protocol = correction_protocols[nodeqp.ID]
    #     if nodeqp.right_height[tree_id] > 1:
    #         # node should wait to its right subtree to be completed;
    #         messaging_protocol = messaging_protocols[nodeqp.ID]
    #     protocols.append(SwappingProtocol(nodeqp, tree_id=tree_id,
    #                                       correction_protocol=correction_protocol,
    #                                       messaging_protocol=messaging_protocol))  # none means directly from link layer
    #     if right_right.ID not in correction_protocols:
    #         protocols.append(CorrectionProtocol(right_right, messaging_protocol=messaging_protocols[right_right.ID],
    #                                             tree_id=tree_id, main_root_id=nodeqp.ID))
    #         correction_protocols[right_right.ID] = protocols[-1]
    #     else:
    #         correction_protocols[right_right.ID].main_root_id = nodeqp.ID
    #     # protocols.append(CorrectionAckProtocol(left_left, parent_id=nodeqp.ID, rightest_id=right_right.ID))
    #
    #     return left_left, right_right, nodeqp, nodeqp, False, correction_protocols[right_right.ID]

    @staticmethod
    def _add_node_to_children_classical_connections(network: Network, node: qPNode, children_id: set):
        for child_id in children_id:
            child_node = network.get_node("Node" + str(child_id))
            node.add_ports(["cto_node" + str(child_id)])
            child_node.add_ports(["cfrom_node" + str(node.ID)])
            if not network.get_connection(node, child_node, "classical") or \
                    node.name != network.get_connection(node, child_node, 'classical').name.split("->")[0]:
                node_child_cchannel = ClassicalConnection(node.loc.distance(child_node.loc),
                                                          node.name + "->" + child_node.name)
                network.add_connection(node, child_node, connection=node_child_cchannel,
                                       port_name_node1="cto_node" + str(child_id),
                                       port_name_node2="cfrom_node" + str(node.ID),
                                       label="classical")

    @staticmethod
    def _add_tree_classical_connection(network: Network, node: qPNode, tree_id: int):
        # add ports & connections from node to its children including rightest
        TreeProtocol._add_node_to_children_classical_connections(network, node, node.children_id[tree_id])

        # rightest to leftest connections to send CORRECTION_ACK
        rightest_node = network.get_node("Node" + str(node.rightest_id[tree_id]))
        rightest_node.add_ports(["cto_node" + str(node.leftest_id[tree_id])])
        leftest_node = network.get_node("Node" + str(node.leftest_id[tree_id]))
        leftest_node.add_ports(["cfrom_node" + str(node.rightest_id[tree_id])])
        if not network.get_connection(rightest_node, leftest_node, "classical") or \
                rightest_node.name != network.get_connection(rightest_node, leftest_node).name.split("->")[0]:
            right_left_cchannel = ClassicalConnection(rightest_node.loc.distance(leftest_node.loc),
                                                      rightest_node.name + "->" + leftest_node.name)
            network.add_connection(rightest_node, leftest_node, connection=right_left_cchannel,
                                   port_name_node1="cto_node" + str(node.leftest_id[tree_id]),
                                   port_name_node2="cfrom_node" + str(node.rightest_id[tree_id]),
                                   label="classical")


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


def add_tree_classical_connection(network: Network, node: qPNode, left: qPNode, right: qPNode):
    # node to left connections
    if network.get_connection(node, left, "classical") is None and \
            not node.ports["cout_left"].is_connected and \
            not left.ports["cin_parent"].is_connected:
        node_left_cchannel = ClassicalConnection(node.loc.distance(left.loc),
                                                 node.name + "->" + left.name + "|classical")
        network.add_connection(node, left, connection=node_left_cchannel,
                               port_name_node1="cout_left", port_name_node2="cin_parent", label="classical")
    if network.get_connection(left, node, "classical") is None and \
            not left.ports["cout_parent"].is_connected and \
            not node.ports["cin_left"].is_connected:
        left_parent_cchannel = ClassicalConnection(left.loc.distance(node.loc),
                                                   left.name + "->" + node.name + "|classical")
        network.add_connection(left, node, connection=left_parent_cchannel,
                               port_name_node1="cout_parent", port_name_node2="cin_left")

    # node to right connections
    if network.get_connection(node, right, "classical") is None and \
            not node.ports["cout_right"].is_connected and \
            not right.ports["cin_parent"].is_connected:
        node_right_cchannel = ClassicalConnection(node.loc.distance(right.loc),
                                                  node.name + "->" + right.name + "|classical")
        network.add_connection(node, right, connection=node_right_cchannel,
                               port_name_node1="cout_right", port_name_node2="cin_parent", label="classical")
    if network.get_connection(right, node, "classical") is None and \
            not right.ports["cout_parent"].is_connected and \
            not node.ports["cin_right"].is_connected:
        right_parent_cchannel = ClassicalConnection(right.loc.distance(node.loc),
                                                    right.name + "->" + node.name + "|classical")
        network.add_connection(right, node, connection=right_parent_cchannel,
                               port_name_node1="cout_parent", port_name_node2="cin_right")


def setup_tree_connections(network: Network, node: tree_node, parent: tree_node = None, tree_id: int = 0):
    if isinstance(node.data, qChannel):
        # get qNodes based on name
        qpnode1 = network.get_node("ProtocolNode" + str(node.data.this.id))
        qpnode2 = network.get_node("ProtocolNode" + str(node.data.other.id))

        # add ports for this tree
        qpnode1.add_ports(PORT_NAMES_QUANTUM + PORT_NAMES_CLASSICAL)
        qpnode2.add_ports(PORT_NAMES_QUANTUM + PORT_NAMES_CLASSICAL)

        # assigning qubits
        qpnode1_qubit_pos = qpnode1.avail_mem_pos()
        qpnode2_qubit_pos = qpnode2.avail_mem_pos()

        # this (qpnode1) as left, other (qpnode2) as right
        qpnode1.rightest_qubit_pos = qpnode1_qubit_pos
        qpnode1.init_right_qubit_pos = qpnode1_qubit_pos
        qpnode2.leftest_qubit_pos = qpnode2_qubit_pos
        qpnode2.init_left_qubit_pos = qpnode2_qubit_pos

        # quantum entangling
        qconn = EntanglingConnection(node.data.this.loc.distance(node.data.other.loc), 1e6,
                                     qpnode1.name + "-" + qpnode2.name + "|Entangling")  # frequency should be based on demand
        port_ac, port_bc = network.add_connection(qpnode1, qpnode2, connection=qconn,
                                                  label="quantum",
                                                  port_name_node1="qin_right",
                                                  port_name_node2="qin_left")
        qpnode1.ports[port_ac].forward_input(qpnode1.qmemory.ports["qin" + str(qpnode1_qubit_pos)])
        qpnode2.ports[port_bc].forward_input(qpnode2.qmemory.ports["qin" + str(qpnode2_qubit_pos)])

        # # classical connections
        # add_classical_connections(network, qpnode1, qpnode2)
        return qpnode1, qpnode2, qpnode1, qpnode2
    nodeqp = network.get_node("ProtocolNode" + str(node.data.id))
    left_left, _, left, _ = setup_tree_connections(network, node.left, node)
    _, right_right, _, right = setup_tree_connections(network, node.right, node)
    # classical connections
    add_tree_classical_connection(network, nodeqp, left, right)
    nodeqp.leftest_id = left_left.ID
    nodeqp.rightest_id = right_right.ID
    # update leftest, rightest qubit position for correct bsm ops
    # leftest node
    leftest_old_rightest, leftest_new_rightest = left_left.rightest_qubit_pos, left_left.avail_mem_pos()
    left_left.qubit_mapping[nodeqp.ID] = (leftest_old_rightest, leftest_new_rightest)
    left_left.rightest_qubit_pos = leftest_new_rightest
    # rightest node
    rightest_old_leftest, rightest_new_leftest = right_right.leftest_qubit_pos, right_right.avail_mem_pos()
    right_right.qubit_mapping[nodeqp.ID] = (rightest_old_leftest, rightest_new_leftest)
    right_right.leftest_qubit_pos = rightest_new_leftest

    return left_left, right_right, nodeqp, nodeqp


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
    nodes = [qNode(20, Point(0, 50000)), qNode(20, Point(0, 100000)), qNode(20, Point(0, 150000)),
             qNode(20, Point(0, 200000)), qNode(20, Point(0, 250000))]
    edges = [qChannel(nodes[0], nodes[1]), qChannel(nodes[1], nodes[2]), qChannel(nodes[2], nodes[3]),
             qChannel(nodes[3], nodes[4])]
    graph = qGraph(V=nodes, E=edges)
    root = [tree_node(graph.V[2],
                      tree_node(graph.V[1],
                                tree_node(graph.E[0], avr_ent_time=0.05),
                                tree_node(graph.E[1], avr_ent_time=0.05)),
                      tree_node(graph.V[3],
                                tree_node(graph.E[2], avr_ent_time=0.05),
                                tree_node(graph.E[3], avr_ent_time=0.05))
                      ),
            tree_node(graph.V[1],
                      tree_node(graph.E[0], avr_ent_time=0.05),
                      tree_node(graph.E[1], avr_ent_time=0.05))
            ]

    network_protocol = TreeProtocol(graph, root, [graph.V[0], graph.V[0]], [graph.V[4], graph.V[2]], 5)
    for idx, success_rate in enumerate(network_protocol.success_rate):
        print(f"Tree{str(idx)}: {success_rate:.4f} (EPs/s)")

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
