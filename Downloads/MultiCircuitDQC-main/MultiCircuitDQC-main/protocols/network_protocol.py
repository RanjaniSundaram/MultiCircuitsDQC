import random
import sys

sys.path.append("..")

from commons.qNode import qNode
from commons.qGraph import qGraph
from commons.qChannel import qChannel
from commons.tree_node import tree_node
import netsquid as ns
from protocols.protocol_bases import *
from netsquid.nodes import Network
from protocols.component_bases import ClassicalConnection, EntanglingConnection, qPNode, Charlie, QuantumConnection
from typing import List, Dict

from netsquid.components import QuantumMemory
from commons.Point import Point
from netsquid.nodes import Node
from protocols.link_layer_protocol import HeraldedConnection
from random import seed
from collections import defaultdict
from commons.ghz_fusion_nodes import general_fusion_node, fusion_retain_node, used_resources

PORT_NAMES_CLASSICAL = ["c" + inOrOut + "_" + direction for inOrOut in ["in", "out"]
                        for direction in ["left", "right", "parent"]]
PORT_NAMES_QUANTUM = ["q" + inOrOut + "_" + direction for inOrOut in ["in"]
                      for direction in ["left", "right"]]


class TreeProtocol:

    def __init__(self, graph: qGraph, trees: List[tree_node],
                 sources: List[qNode], destinations: List[qNode], duration: int,
                 swapping_time_window: float = float('inf'), IS_TELEPORTING: bool = False,
                 CHECK_CAPACITIES: bool = False):
        """duration(s) to run the simulation"""
        ns.set_qstate_formalism(ns.QFormalism.DM)
        ns.set_random_state(42)
        ns.sim_reset()
        # get only those nodes and edges that are going to be used to save memory and efficiency
        used_nodes, used_edges = set(), set()
        for tree in trees:
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
                TreeProtocol.add_classical_connections(network, nodes[i], nodes[j])

        # setup quantum connection for each edge in G.E
        charlie_protocols = []
        # link_layer_max_capacity = {}
        for e in graph.E:
            if "{this}-{other}".format(this=e.this.id, other=e.other.id) in used_edges:
                charlie = TreeProtocol.add_quantum_connections(network=network, e=e)
                charlie_protocols.append(CharlieProtocol(charlie))
                charlie_id = "{this}_{other}".format(this=e.this.id, other=e.other.id)
                # link_layer_max_capacity[charlie_id] = e

        messaging_protocols = {node.ID: MessagingProtocol(node) for node in nodes if
                               node.ID in used_nodes}  # one for each node for all trees
        all_protocols = []
        link_layer_requests = defaultdict(list)
        for tree_id, tree_root in enumerate(trees):
            for id, messaging_protocol in messaging_protocols.items():
                messaging_protocol.add_signals(tree_id)
            protocols = []
            _, _, _, _, is_path_one_hop, dst_correction_protocol = \
                TreeProtocol._setup_tree_protocols(network=network, node=tree_root, tree_id=tree_id,
                                                   protocols=protocols,
                                                   messaging_protocols=messaging_protocols,
                                                   entangling_protocols={}, correction_protocols={},
                                                   swapping_time_window=swapping_time_window,
                                                   link_layer_requests=link_layer_requests,
                                                   original_graph=graph)

            srcQPNode = network.get_node("Node" + str(sources[tree_id].id))
            dstQPNOde = network.get_node("Node" + str(destinations[tree_id].id))

            all_tree_nodes_id = set()
            if not is_path_one_hop:
                rootqp = network.get_node("Node" + str(trees[tree_id].data.id))
                all_tree_nodes_id = rootqp.children_id[tree_id].union({rootqp.ID})
                # all_tree_nodes_id.remove(dstQPNOde.ID)
                # TreeProtocol.add_node_to_children_classical_connections(network, dstQPNOde, all_tree_nodes_id)
            # add protocols
            protocols.append(TeleportationProtocol(node=srcQPNode, qubit_protocol=QubitGenerationProtocol(srcQPNode),
                                                   dst_id=dstQPNOde.ID,
                                                   is_path_one_hop=is_path_one_hop, tree_id=tree_id,
                                                   tree_nodes_id=all_tree_nodes_id,
                                                   messaging_protocol=messaging_protocols[srcQPNode.ID],
                                                   IS_TELEPORTING=self._IS_TELEPORTING))
            protocols.append(TeleportationCorrectionProtocol(node=dstQPNOde, src_id=srcQPNode.ID, tree_id=tree_id,
                                                             messaging_protocol=messaging_protocols[dstQPNOde.ID],
                                                             is_path_one_hop=is_path_one_hop,
                                                             correction_protocol=dst_correction_protocol,
                                                             tree_nodes_id=all_tree_nodes_id))
            self._src_protocols.append(protocols[-2])
            self._dst_protocols.append(protocols[-1])
            protocols.append(FidelityCalculatorProtocol(tree_id, src=srcQPNode, dst=dstQPNOde,
                                                        correction_protocol=dst_correction_protocol,
                                                        is_path_one_hop=is_path_one_hop,
                                                        teleportation_protocol=self._src_protocols[tree_id]))
            self._fidelity_collector.append(protocols[-1])
            # to keep track of successful EPs rate for each pair of (src, dst)
            all_protocols += protocols

        if CHECK_CAPACITIES:
            # check if link-layer requests is not greater than edge maximum capacity
            for link_id, requests in link_layer_requests.items():
                sum_requests = sum([request[0] for request in requests])
                node1, node2 = link_id.split('_')
                e = graph.get_edge(int(node1), int(node2))
                if sum_requests > e.max_channel_capacity:
                    effective_reduce_amount_per_request = (sum_requests - e.max_channel_capacity) / \
                                                          len(requests)
                    reduce_amount = TreeProtocol.effective_rate_to_success_node_rate(
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
    def add_quantum_connections(network: Network, e: qChannel):
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
        return charlie

    @staticmethod
    def add_classical_connections(network: Network, node1: qPNode, node2: qPNode):
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
    def _setup_tree_protocols(network: Network, protocols: List, messaging_protocols: dict,
                              node: tree_node, link_layer_requests: dict, original_graph: qGraph,
                              tree_id: int = 0, correction_protocols={}, entangling_protocols={},
                              swapping_time_window: float = float('inf')):
        if isinstance(node.data, qChannel):
            # get qNodes based on name
            qpnode1 = network.get_node("Node" + str(node.data.this.id))
            qpnode2 = network.get_node("Node" + str(node.data.other.id))

            if qpnode1.ID not in entangling_protocols:
                protocols.append(EntanglingProtocol(qpnode1, tree_id))
                entangling_protocols[qpnode1.ID] = protocols[-1]
            if qpnode2.ID not in entangling_protocols:
                protocols.append(EntanglingProtocol(qpnode2, tree_id))
                entangling_protocols[qpnode2.ID] = protocols[-1]

            # set heights
            qpnode1.left_height[tree_id], qpnode1.right_height[tree_id] = 0, 0
            qpnode2.left_height[tree_id], qpnode2.right_height[tree_id] = 0, 0

            # assigning qubit memories
            qpnode1_right_qubit_pos, qpnode1_right_link_qubit_pos = qpnode1.avail_mem_pos(), qpnode1.avail_mem_pos()
            if tree_id in qpnode1.qubit_mapping:
                # left qubits were assigned
                qpnode1.qubit_mapping[tree_id] = ((qpnode1.qubit_mapping[tree_id][0][0], qpnode1_right_qubit_pos),
                                                  (qpnode1.qubit_mapping[tree_id][1][0], qpnode1_right_link_qubit_pos))
            else:
                qpnode1.qubit_mapping[tree_id] = ((-1, qpnode1_right_qubit_pos),
                                                  (-1, qpnode1_right_link_qubit_pos))
            qpnode2_left_qubit_pos, qpnode2_left_link_qubit_pos = qpnode2.avail_mem_pos(), qpnode2.avail_mem_pos()
            if tree_id in qpnode2.qubit_mapping:
                # left qubits were assigned
                qpnode2.qubit_mapping[tree_id] = ((qpnode2_left_qubit_pos, qpnode2.qubit_mapping[tree_id][0][1]),
                                                  (qpnode2_left_link_qubit_pos, qpnode2.qubit_mapping[tree_id][1][1]))
            else:
                qpnode2.qubit_mapping[tree_id] = ((qpnode2_left_qubit_pos, -1),
                                                  (qpnode2_left_link_qubit_pos, -1))

            if tree_id in qpnode1.neighbors:
                # left neighbor of qpnode1 is already set
                qpnode1.neighbors[tree_id] = (qpnode1.neighbors[tree_id][0], qpnode2.ID)
                qpnode1.entangling[tree_id] = (qpnode1.entangling[tree_id][0], True)
            else:
                qpnode1.neighbors[tree_id] = (-1, qpnode2.ID)
                qpnode1.entangling[tree_id] = (False, True)

            if tree_id in qpnode2.neighbors:
                # right neighbor of qpnode2 is already set
                qpnode2.neighbors[tree_id] = (qpnode1.ID, qpnode2.neighbors[tree_id][1])
                qpnode2.entangling[tree_id] = (True, qpnode2.entangling[tree_id][1])
            else:
                qpnode2.neighbors[tree_id] = (qpnode1.ID, -1)
                qpnode2.entangling[tree_id] = (True, False)

            # create empty set for children
            qpnode1.children_id[tree_id] = set()
            qpnode2.children_id[tree_id] = set()

            # make link layer request here
            left_node, right_node = qpnode1, qpnode2
            left_node_link_pos, right_node_link_pos = qpnode1_right_link_qubit_pos, qpnode2_left_link_qubit_pos
            if "Charlie_" + str(node.data.this.id) + "_" + str(node.data.other.id) in network.nodes:
                charlie_id = str(node.data.this.id) + "_" + str(node.data.other.id)
            else:
                left_node, right_node = qpnode2, qpnode1
                left_node_link_pos, right_node_link_pos = qpnode2_left_link_qubit_pos, qpnode1_right_link_qubit_pos
                charlie_id = str(node.data.other.id) + "_" + str(node.data.this.id)
            effective_rate = 1.0 / node.avr_ent_time
            orig_e = original_graph.get_edge(left_node.ID, right_node.ID)
            # effective_rate = orig_e.residual_capacity  # make it non-throttled
            rate = TreeProtocol.effective_rate_to_success_node_rate(effective_rate, orig_e)
            protocols.append(EntanglementGenerator(node=left_node, tree_id=tree_id, rate=rate, isLeft=False,
                                                   charlie_id=charlie_id, link_layer_qmem_pos=left_node_link_pos))
            protocols.append(EntanglementGenerator(node=right_node, tree_id=tree_id, rate=rate, isLeft=True,
                                                   charlie_id=charlie_id, link_layer_qmem_pos=right_node_link_pos))
            link_layer_requests[charlie_id].append((effective_rate, protocols[-2], protocols[-1]))
            protocols.append(LinkCorrectionProtocol(node=right_node, tree_id=tree_id,
                                                    messaging_protocol=messaging_protocols[right_node.ID],
                                                    link_qmem_pos=right_node_link_pos,
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
                                                                         swapping_time_window=swapping_time_window,
                                                                         original_graph=original_graph)
        _, right_right, _, right, _, _ = TreeProtocol._setup_tree_protocols(network=network, protocols=protocols,
                                                                            messaging_protocols=messaging_protocols,
                                                                            node=node.right, tree_id=tree_id,
                                                                            entangling_protocols=entangling_protocols,
                                                                            correction_protocols=correction_protocols,
                                                                            link_layer_requests=link_layer_requests,
                                                                            swapping_time_window=swapping_time_window,
                                                                            original_graph=original_graph)
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
                                          swapping_time_window=swapping_time_window))
        if right_right.ID not in correction_protocols:
            protocols.append(CorrectionProtocol(right_right, messaging_protocol=messaging_protocols[right_right.ID],
                                                tree_id=tree_id, main_root_id=nodeqp.ID))
            correction_protocols[right_right.ID] = protocols[-1]
        else:
            correction_protocols[right_right.ID].main_root_id = nodeqp.ID

        return left_left, right_right, nodeqp, nodeqp, False, correction_protocols[right_right.ID]

    @staticmethod
    def effective_rate_to_success_node_rate(effective_rate: float, e: qChannel):
        return effective_rate / (e.channel_success_rate ** 2 * e.optical_bsm_rate)

    @staticmethod
    def _effective_rate_to_node_rate(effective_rate: float, e: qChannel):
        return TreeProtocol.effective_rate_to_success_node_rate(effective_rate, e) / (e.this.gen_success_rate *
                                                                                      e.other.gen_success_rate)

    @staticmethod
    def _node_rate_to_success_rate(node_rate: float, e: qChannel):
        return node_rate * e.this.gen_success_rate * e.other.gen_success_rate

    @staticmethod
    def _setup_tree_connections(network: Network, protocols: List, messaging_protocols: dict, node: tree_node,
                                tree_id: int = 0, correction_protocols: dict = {}, entangling_protocols=set()):
        if isinstance(node.data, qChannel):
            # get qNodes based on name
            qpnode1 = network.get_node("Node" + str(node.data.this.id))
            qpnode2 = network.get_node("Node" + str(node.data.other.id))
            if qpnode1.ID not in entangling_protocols:
                entangling_protocols.add(qpnode1.ID)
                protocols.append(EntanglingProtocol(qpnode1, tree_id))
            if qpnode2.ID not in entangling_protocols:
                entangling_protocols.add(qpnode2.ID)
                protocols.append(EntanglingProtocol(qpnode2, tree_id))
            qpnode1.left_height[tree_id], qpnode1.right_height[tree_id] = 0, 0
            qpnode2.left_height[tree_id], qpnode2.right_height[tree_id] = 0, 0

            # add ports for this quantum channel
            qpnode1.add_ports(PORT_NAMES_QUANTUM)
            qpnode2.add_ports(PORT_NAMES_QUANTUM)

            # assigning qubit memories
            qpnode1_right_qubit_pos, qpnode1_right_link_qubit_pos = qpnode1.avail_mem_pos(), qpnode1.avail_mem_pos()
            if tree_id in qpnode1.qubit_mapping:
                # left qubits were assigned
                qpnode1.qubit_mapping[tree_id] = ((qpnode1.qubit_mapping[tree_id][0][0], qpnode1_right_qubit_pos),
                                                  (qpnode1.qubit_mapping[tree_id][1][0], qpnode1_right_link_qubit_pos))
            else:
                qpnode1.qubit_mapping[tree_id] = ((-1, qpnode1_right_qubit_pos),
                                                  (-1, qpnode1_right_link_qubit_pos))
            qpnode2_left_qubit_pos, qpnode2_left_link_qubit_pos = qpnode2.avail_mem_pos(), qpnode2.avail_mem_pos()
            if tree_id in qpnode2.qubit_mapping:
                # left qubits were assigned
                qpnode2.qubit_mapping[tree_id] = ((qpnode2_left_qubit_pos, qpnode2.qubit_mapping[tree_id][0][1]),
                                                  (qpnode2_left_link_qubit_pos, qpnode2.qubit_mapping[tree_id][1][1]))
            else:
                qpnode2.qubit_mapping[tree_id] = ((qpnode2_left_qubit_pos, -1),
                                                  (qpnode2_left_link_qubit_pos, -1))

            # quantum entangling
            qconn = EntanglingConnection(node.data.this.loc.distance(node.data.other.loc), 0.01,
                                         qpnode1.name + "-" + qpnode2.name + "|Entangling")  # frequency should be based on demand
            port_ac, port_bc = network.add_connection(qpnode1, qpnode2, connection=qconn,
                                                      label="quantum",
                                                      port_name_node1="qin_right",
                                                      port_name_node2="qin_left")
            qpnode1.ports[port_ac].forward_input(qpnode1.qmemory.ports["qin" + str(qpnode1_right_link_qubit_pos)])
            qpnode2.ports[port_bc].forward_input(qpnode2.qmemory.ports["qin" + str(qpnode2_left_link_qubit_pos)])

            # classical channel between two neighbors to talk to each other to resume ACCEPTING EQs
            # due to synchronization issues, length of the channels are assumed to be zero
            # set neighbors
            if tree_id in qpnode1.neighbors:
                # left neighbor of qpnode1 is already set
                qpnode1.neighbors[tree_id] = (qpnode1.neighbors[tree_id][0], qpnode2.ID)
                qpnode1.entangling[tree_id] = (qpnode1.entangling[tree_id][0], True)
            else:
                qpnode1.neighbors[tree_id] = (-1, qpnode2.ID)
                qpnode1.entangling[tree_id] = (False, True)

            if tree_id in qpnode2.neighbors:
                # right neighbor of qpnode2 is already set
                qpnode2.neighbors[tree_id] = (qpnode1.ID, qpnode2.neighbors[tree_id][1])
                qpnode2.entangling[tree_id] = (True, qpnode2.entangling[tree_id][1])
            else:
                qpnode2.neighbors[tree_id] = (qpnode1.ID, -1)
                qpnode2.entangling[tree_id] = (True, False)

            qpnode1.add_ports(["cto_neighbor" + str(qpnode2.ID), "cfrom_neighbor" + str(qpnode2.ID)])
            qpnode2.add_ports(["cto_neighbor" + str(qpnode1.ID), "cfrom_neighbor" + str(qpnode1.ID)])
            # qpnode1 -> qpnode2
            qpnode1_qpnode2_cchannel = ClassicalConnection(0, qpnode1.name + "->" + qpnode2.name)
            network.add_connection(qpnode1, qpnode2, connection=qpnode1_qpnode2_cchannel,
                                   port_name_node1="cto_neighbor" + str(qpnode2.ID),
                                   port_name_node2="cfrom_neighbor" + str(qpnode1.ID),
                                   label="classicalZeroLatency")
            # qpnode2 -> qpnode1
            qpnode2_qpnode1_cchannel = ClassicalConnection(0, qpnode2.name + "->" + qpnode1.name)
            network.add_connection(qpnode2, qpnode1, connection=qpnode2_qpnode1_cchannel,
                                   port_name_node1="cto_neighbor" + str(qpnode1.ID),
                                   port_name_node2="cfrom_neighbor" + str(qpnode2.ID),
                                   label="classicalZeroLatency")

            # create empty set for children
            qpnode1.children_id[tree_id] = set()
            qpnode2.children_id[tree_id] = set()

            return qpnode1, qpnode2, qpnode1, qpnode2, True, None
        nodeqp = network.get_node("Node" + str(node.data.id))
        left_left, _, left, _, _, _ = TreeProtocol._setup_tree_connections(network=network, protocols=protocols,
                                                                           messaging_protocols=messaging_protocols,
                                                                           node=node.left, tree_id=tree_id,
                                                                           entangling_protocols=entangling_protocols)
        _, right_right, _, right, _, _ = TreeProtocol._setup_tree_connections(network=network, protocols=protocols,
                                                                              messaging_protocols=messaging_protocols,
                                                                              node=node.right, tree_id=tree_id,
                                                                              entangling_protocols=entangling_protocols)
        # set left & right height, leftest & rightest ids, and children
        nodeqp.left_height[tree_id] = max(left.left_height[tree_id], left.right_height[tree_id]) + 1
        nodeqp.right_height[tree_id] = max(right.left_height[tree_id], right.right_height[tree_id]) + 1

        nodeqp.leftest_id[tree_id], nodeqp.rightest_id[tree_id] = left_left.ID, right_right.ID

        nodeqp.children_id[tree_id] = left.children_id[tree_id].union(right.children_id[tree_id])
        nodeqp.children_id[tree_id].add(left.ID)
        nodeqp.children_id[tree_id].add(right.ID)
        if nodeqp.ID in nodeqp.children_id[tree_id]:  # remove node itself from its children
            nodeqp.children_id[tree_id].remove(nodeqp.ID)

        # classical connections
        TreeProtocol._add_tree_classical_connection(network, nodeqp, tree_id)

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
                                          messaging_protocol=messaging_protocol))  # none means directly from link layer
        if right_right.ID not in correction_protocols:
            protocols.append(CorrectionProtocol(right_right, messaging_protocol=messaging_protocols[right_right.ID],
                                                tree_id=tree_id, main_root_id=nodeqp.ID))
            correction_protocols[right_right.ID] = protocols[-1]
        else:
            correction_protocols[right_right.ID].main_root_id = nodeqp.ID
        # protocols.append(CorrectionAckProtocol(left_left, parent_id=nodeqp.ID, rightest_id=right_right.ID))

        return left_left, right_right, nodeqp, nodeqp, False, correction_protocols[right_right.ID]

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


class FusionTreeProtocol:
    def __init__(self, qgraph: qGraph, trees: List[general_fusion_node], simulation_duration: int,
                 fusion_time_window: float = float('inf')):
        """duration(s) to run the simulation"""
        ns.set_qstate_formalism(ns.QFormalism.KET)
        ns.set_random_state(42)
        ns.sim_reset()
        self._tree_controller_protocols = []
        self._duration = simulation_duration
        # get only those nodes and edges that are going to be used to save memory and efficiency
        used_nodes, used_edges = set(), set()
        for tree in trees:
            tree_nodes, tree_edges = used_resources(tree)
            used_nodes = used_nodes.union(tree_nodes)
            used_edges = used_edges.union(tree_edges)

        nodes = [qPNode(node) for node in qgraph.V if node.id in used_nodes]
        network = Network("network", nodes)
        # setup classical connections between each pair of nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                TreeProtocol.add_classical_connections(network, nodes[i], nodes[j])

        # create protocols that have one instance/node for all trees
        messaging_protocols = {node.ID: MessagingProtocolV2(node) for node in nodes}
        link_layer_corrections = {node.ID: LinkCorrectionProtocolV2(node, messaging_protocols[node.ID])
                                  for node in nodes}
        entangling_protocols = {node.ID: EntanglingProtocolV2(node, messaging_protocols[node.ID],
                                                              link_layer_corrections[node.ID]) for node in nodes}
        fusion_correction_protocols = {node.ID: FusionCorrectionProtocol(node, messaging_protocols[node.ID])
                                       for node in nodes}

        # create one CharlieManager/edge (only once)
        charlie_manager_protocols = defaultdict(CharlieManager)
        for e in qgraph.E:
            charlie_id = "{this}_{other}".format(this=e.this.id, other=e.other.id)
            if charlie_id in used_edges:
                charlie = TreeProtocol.add_quantum_connections(network=network, e=e)
                charlie_manager_protocols[charlie_id] = CharlieManager(charlie)

        all_other_protocols = []
        for tree_id, fusion_tree in enumerate(trees):
            tree_fusion_protocols = {}
            tree_other_protocols, _ = FusionTreeProtocol.setup_fusion_tree_protocols(
                network, qgraph, fusion_tree, tree_id, messaging_protocols, entangling_protocols,
                fusion_correction_protocols, link_layer_corrections, charlie_manager_protocols,
                fusion_time_window=fusion_time_window, fusion_protocols=tree_fusion_protocols)
            all_other_protocols += tree_other_protocols
            self._tree_controller_protocols.append(
                FusionTreeController(
                    root_node=network.get_node(f"Node{fusion_tree.node.id}"),
                    tree_id=tree_id,
                    fusion_protocol=tree_fusion_protocols["-".join([str(num)
                                                                    for num in sorted(list(fusion_tree.end_nodes))])],
                    fusion_correction_protocols=[fusion_correction_protocols[node_id] for node_id in
                                                 fusion_tree.end_nodes if node_id != fusion_tree.node.id],
                    end_nodes=fusion_tree.end_nodes, all_nodes=fusion_tree.all_nodes,
                    end_qPNodes=[network.get_node(f"Node{idd}") for idd in fusion_tree.end_nodes]))
            all_other_protocols.append(self._tree_controller_protocols[-1])
            all_other_protocols += [fusion_protocol for fusion_protocol in tree_fusion_protocols.values()]

            # add proper signals to messaging protocols
            for messaging_protocol in messaging_protocols.values():
                messaging_protocol.add_signals(tree_id=tree_id)
                messaging_protocol.add_links_signals(tree_id=tree_id)

        # starting all protocols
        for messaging_protocol in messaging_protocols.values():
            messaging_protocol.start()
        for link_layer_correction in link_layer_corrections.values():
            link_layer_correction.start()
        for entangling_protocol in entangling_protocols.values():
            entangling_protocol.start()
        for fusion_correction_protocol in fusion_correction_protocols.values():
            fusion_correction_protocol.start()
        for charlie_manager_protocol in charlie_manager_protocols.values():
            charlie_manager_protocol.start()

        # all_other_protocols.append(ProtocolTest(network.get_node("Node15")))
        # all_other_protocols[-1].start()
        # all_other_protocols.append(ProtocolTest(network.get_node("Node16")))
        # all_other_protocols[-1].start()
        for protocol in all_other_protocols:
            # if (isinstance(protocol, EntanglementGeneratorV2) and protocol.node.ID in [15, 16]) or \
            #         (isinstance(protocol, CharlieProtocolV2) and protocol.node.name == "Charlie_15_16"):
            protocol.start()

        self._stats = ns.sim_run(duration=simulation_duration * 1e9)

    @property
    def success_rate(self) -> List[float]:
        """return a list of rates, each for one GHZ state"""

        # get the total number of successful GHZ constructions (only care about EPs'/s)
        return [tree_controller_proto.successful_counts / self._duration for tree_controller_proto
                in self._tree_controller_protocols]

    @property
    def fidelity(self) -> List[float]:
        """return a list of fidelity, each for one GHZ state"""

        # get the average latency of successful GHZ constructions
        return [float('inf') if len(tree_controller_proto.fidelity) == 0 else sum(tree_controller_proto.fidelity) /
                                                                              len(tree_controller_proto.fidelity) for
                tree_controller_proto in self._tree_controller_protocols]

    @staticmethod
    def setup_fusion_tree_protocols(network: Network, qgraph: qGraph, tree_node: general_fusion_node, tree_id: int,
                                    messaging_protocols: Dict[int, MessagingProtocolV2],
                                    entangling_protocols: Dict[int, EntanglingProtocolV2],
                                    fusion_correction_protocols: Dict[int, FusionCorrectionProtocol],
                                    link_layer_corrections: Dict[int, LinkCorrectionProtocolV2],
                                    charlie_manager_protocols: Dict[str, CharlieManager],
                                    fusion_time_window: float = float('inf'),
                                    ghz_parent_list: List[str] = [],
                                    fusion_protocols: Dict[str, FusionProtocol] = {}) -> (List, int):
        if tree_node is None:
            return [], 0
        if tree_node.node is None:
            # reached to edge, assigning qubits and creating edge/link protocols
            node1_id, node2_id = list(tree_node.all_nodes)
            e = qgraph.get_edge(node1_id, node2_id)
            edge_key = f"{e.this.id}_{e.other.id}"
            # this as left, other as right
            this_node, other_node = network.get_node(f"Node{e.this.id}"), network.get_node(f"Node{e.other.id}")
            # assign new qubit positions, one line and one network for each node
            this_node.qubit_mapping[f"{tree_id}_{other_node.ID}"] = (this_node.avail_mem_pos(),
                                                                     this_node.avail_mem_pos())
            other_node.qubit_mapping[f"{tree_id}_{this_node.ID}"] = (other_node.avail_mem_pos(),
                                                                     other_node.avail_mem_pos())
            # set entangling to True at the beginning
            this_node.entangling[f"{tree_id}_{other_node.ID}"] = True
            other_node.entangling[f"{tree_id}_{this_node.ID}"] = True
            # add neighborhoods
            if tree_id not in this_node.neighbors:
                this_node.neighbors[tree_id] = set()
            this_node.neighbors[tree_id].add(e.other.id)
            if tree_id not in other_node.neighbors:
                other_node.neighbors[tree_id] = set()
            other_node.neighbors[tree_id].add(e.this.id)

            # updating list of neighbors for the list of all GHZ states this edge is part of
            for ghz in ghz_parent_list:
                this_node.ghz_edges[f"{tree_id}_{ghz}"].add(other_node.ID)
                other_node.ghz_edges[f"{tree_id}_{ghz}"].add(this_node.ID)

            # add tree signals to charlie managers
            charlie_manager_protocols[edge_key].add_signals(tree_id=tree_id)

            effective_rate = 1.0 / tree_node.avr_ent_time  # e.avr_ent_time  # TODO e here means non-throttles
            # effective_rate = orig_e.residual_capacity  # make it non-throttled
            rate = TreeProtocol.effective_rate_to_success_node_rate(effective_rate, e)

            # create EntanglementGenerator and Charlie protocols
            charlie_node = network.get_node(f"Charlie_{edge_key}")
            this_ent_gen_proto = \
                EntanglementGeneratorV2(node=this_node, tree_id=tree_id, rate=rate,
                                        isLeft=False, charlie_id=edge_key,
                                        link_layer_qmem_pos=this_node.qubit_mapping[f"{tree_id}_{other_node.ID}"][1],
                                        other_node_id=other_node.ID,
                                        messaging_protocol=messaging_protocols[this_node.ID])
            other_ent_gen_proto = \
                EntanglementGeneratorV2(node=other_node, tree_id=tree_id, rate=rate,
                                        isLeft=True, charlie_id=edge_key,
                                        link_layer_qmem_pos=other_node.qubit_mapping[f"{tree_id}_{this_node.ID}"][1],
                                        other_node_id=this_node.ID,
                                        messaging_protocol=messaging_protocols[other_node.ID])
            charlie_protocol = CharlieProtocolV2(node=charlie_node, tree_id=tree_id,
                                                 charlie_manager_protocol=charlie_manager_protocols[edge_key])
            # add edge (for this tree) signal to right node link correction protocol
            link_layer_corrections[e.other.id].add_tree_edge_signal(tree_id=tree_id, left_node_id=e.this.id)

            # add appropriate signals to listen to EntanglingProtocols
            entangling_protocols[e.this.id].add_protocol(tree_id=tree_id, other_id=e.other.id, isLeft=True)
            entangling_protocols[e.other.id].add_protocol(tree_id=tree_id, other_id=e.this.id, isLeft=False)

            return [this_ent_gen_proto, other_ent_gen_proto, charlie_protocol], 1

        end_nodes_key = '-'.join([str(end_id) for end_id in sorted(list(tree_node.end_nodes))])

        left_protocols, left_height = FusionTreeProtocol.setup_fusion_tree_protocols(
            network=network, qgraph=qgraph, tree_node=tree_node.sub1, tree_id=tree_id,
            messaging_protocols=messaging_protocols,
            entangling_protocols=entangling_protocols, fusion_correction_protocols=fusion_correction_protocols,
            link_layer_corrections=link_layer_corrections, charlie_manager_protocols=charlie_manager_protocols,
            fusion_time_window=fusion_time_window, ghz_parent_list=ghz_parent_list + [end_nodes_key],
            fusion_protocols=fusion_protocols)
        right_protocols, right_height = FusionTreeProtocol.setup_fusion_tree_protocols(
            network=network, qgraph=qgraph, tree_node=tree_node.sub2, tree_id=tree_id,
            messaging_protocols=messaging_protocols,
            entangling_protocols=entangling_protocols, fusion_correction_protocols=fusion_correction_protocols,
            link_layer_corrections=link_layer_corrections, charlie_manager_protocols=charlie_manager_protocols,
            fusion_time_window=fusion_time_window, ghz_parent_list=ghz_parent_list + [end_nodes_key],
            fusion_protocols=fusion_protocols)

        end_nodes_left_key = '-'.join([str(end_id) for end_id in sorted(list(tree_node.sub1.end_nodes))])
        end_nodes_right_key = '-'.join([str(end_id) for end_id in sorted(list(tree_node.sub2.end_nodes))])

        # FUSION protocol and adding qubit mappings and FUSION_CORRECTION signals
        node_p = network.get_node(f"Node{tree_node.node.id}")
        node_p.left_height[f"{tree_id}_{end_nodes_key}"] = left_height
        node_p.right_height[f"{tree_id}_{end_nodes_key}"] = right_height
        node_fusion_protocol = None
        end_nodes_qpnodes = [network.get_node(f'Node{node_num}') for node_num in list(tree_node.end_nodes)]
        if left_height == 1 and right_height == 1:
            # must be three nodes
            for qpnode in end_nodes_qpnodes:
                if qpnode.ID == tree_node.node.id and tree_node.is_retain:
                    # one of the qubit (left) in this node remains for larger states, fetched from link-level qubit
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = qpnode.qubit_mapping[
                        f"{tree_id}_{tree_node.sub1.end_nodes.difference({tree_node.node.id}).pop()}"][0]
                else:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{tree_node.node.id}"][0]

        elif left_height == 1:
            for qpnode in end_nodes_qpnodes:
                if qpnode.ID == tree_node.node.id and tree_node.is_retain:
                    # one of the qubit (left) in this node remains for larger states, fetched from link-level qubit
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = qpnode.qubit_mapping[
                        f"{tree_id}_{tree_node.sub1.end_nodes.difference({tree_node.node.id}).pop()}"][0]
                elif qpnode.ID in tree_node.sub1.end_nodes:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{tree_node.node.id}"][0]
                else:
                    # qubit must be fetched from a tree information
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_right_key}"]

        elif right_height == 1:
            for qpnode in end_nodes_qpnodes:
                if qpnode.ID == tree_node.node.id and tree_node.is_retain:
                    # one of the qubit (left) in this node remains for larger states, fetched from tree-level qubit
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_left_key}"]
                elif qpnode.ID in tree_node.sub2.end_nodes:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{tree_node.node.id}"][0]
                else:
                    # qubit must be fetched from a tree information
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_left_key}"]
        else:
            for qpnode in end_nodes_qpnodes:
                if qpnode.ID == tree_node.node.id and tree_node.is_retain:
                    # one of the qubit (left) in this node remains for larger states, fetched from tree-level qubit
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_left_key}"]
                elif qpnode.ID in tree_node.sub1.end_nodes:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_left_key}"]
                else:
                    # qubit must be fetched from a tree information
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_right_key}"]
        # adding signals to fusion correction protocols
        node_fusion_protocol = FusionProtocol(node=node_p, tree_id=tree_id,
                                              end_nodes_left=tree_node.sub1.end_nodes,
                                              end_nodes_right=tree_node.sub2.end_nodes,
                                              all_nodes_left=tree_node.sub1.all_nodes,
                                              all_nodes_right=tree_node.sub2.all_nodes,
                                              messaging_protocol=messaging_protocols[tree_node.node.id],
                                              fusion_correction_protocol=
                                              fusion_correction_protocols[tree_node.node.id],
                                              retain=tree_node.is_retain,
                                              fusion_time_window=fusion_time_window)
        messaging_protocols[tree_node.node.id].add_ghz_fusion_correction_ack_signals(tree_id=tree_id,
                                                                                     end_nodes_key=end_nodes_left_key)
        messaging_protocols[tree_node.node.id].add_ghz_fusion_correction_ack_signals(tree_id=tree_id,
                                                                                     end_nodes_key=end_nodes_right_key)
        for node_id in tree_node.end_nodes:
            if node_id != tree_node.node.id:
                messaging_protocols[node_id].add_ghz_signals(tree_id=tree_id, end_nodes_key=end_nodes_key)
                fusion_correction_protocols[node_id].add_signals(tree_id=tree_id, end_nodes_key=end_nodes_key)

        fusion_protocols[end_nodes_key] = node_fusion_protocol
        # updating parent dict
        if f"{tree_id}_{end_nodes_key}" not in qPNode.parents:
            qPNode.parents[f"{tree_id}_{end_nodes_key}"] = (None, None)
        qPNode.parents[f"{tree_id}_{end_nodes_key}"] = (tree_node.node.id,
                                                        qPNode.parents[f"{tree_id}_{end_nodes_key}"][1])
        if f"{tree_id}_{end_nodes_left_key}" not in qPNode.parents:
            qPNode.parents[f"{tree_id}_{end_nodes_left_key}"] = (None, None)
        qPNode.parents[f"{tree_id}_{end_nodes_left_key}"] = (qPNode.parents[f"{tree_id}_{end_nodes_left_key}"][0],
                                                             tree_node.node.id)
        if f"{tree_id}_{end_nodes_right_key}" not in qPNode.parents:
            qPNode.parents[f"{tree_id}_{end_nodes_right_key}"] = (None, None)
        qPNode.parents[f"{tree_id}_{end_nodes_right_key}"] = (qPNode.parents[f"{tree_id}_{end_nodes_right_key}"][0],
                                                              tree_node.node.id)

        return left_protocols + right_protocols, max(left_height, right_height) + 1


class FusionRetainOnlyTreeProtocol:
    """A class designed to perform FRO algorithm. The class takes a fusion tree and a set of swapping-trees.
    Upon successful swapping-trees construction, fusion tree (only retain) will be performed."""

    def __init__(self, qgraph: qGraph, fusion_tree: fusion_retain_node,
                 swapping_trees: Dict[str, List[tree_node]],
                 simulation_duration: float, fusion_time_window: float = float('inf'),
                 swapping_time_window: float = float('inf')):
        ns.set_qstate_formalism(ns.QFormalism.KET)
        ns.set_random_state(42)
        ns.sim_reset()
        self._duration = simulation_duration
        # get only those nodes and edges that are going to be used to save memory and efficiency
        used_nodes, used_edges = set(), set()
        for terminal_pair_trees in swapping_trees.values():
            for tree in terminal_pair_trees:
                tree_nodes, tree_edges = tree_node.used_nodes_edges(tree, "_")
                used_nodes = used_nodes.union(tree_nodes)
                used_edges = used_edges.union(tree_edges)

        nodes = [qPNode(node) for node in qgraph.V if node.id in used_nodes]
        network = Network("network", nodes)
        # setup classical connections between each pair of nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                TreeProtocol.add_classical_connections(network, nodes[i], nodes[j])

        # create protocols that have one instance/node for all trees
        messaging_protocols = {node.ID: MessagingProtocolV2(node) for node in nodes}
        link_layer_corrections = {node.ID: LinkCorrectionProtocolV2(node, messaging_protocols[node.ID])
                                  for node in nodes}
        entangling_protocols = {node.ID: EntanglingProtocolV2(node, messaging_protocols[node.ID],
                                                              link_layer_corrections[node.ID]) for node in nodes}
        fusion_correction_protocols = {node.ID: FusionCorrectionProtocol(node, messaging_protocols[node.ID])
                                       for node in nodes if node.ID in fusion_tree.end_nodes}
        swapping_tree_correction_protocols = {node.ID: CorrectionProtocolV2(node, messaging_protocols[node.ID])
                                              for node in nodes}

        # create one CharlieManager/edge (only once)
        charlie_manager_protocols = defaultdict(CharlieManager)
        for e in qgraph.E:
            charlie_id = f"{e.this.id}_{e.other.id}"
            if charlie_id in used_edges:
                charlie = TreeProtocol.add_quantum_connections(network=network, e=e)
                charlie_manager_protocols[charlie_id] = CharlieManager(charlie)

        all_other_protocols, swapping_trees_and_nodes, _, _ =\
            FusionRetainOnlyTreeProtocol.setup_fro_fusion_tree_protocols(
                network=network, qgraph=qgraph, tree_node=fusion_tree, tree_id=0, swapping_trees=swapping_trees,
                swapping_tree_id_offset=1, messaging_protocols=messaging_protocols,
                entangling_protocols=entangling_protocols,
                swapping_corrections_protocols=swapping_tree_correction_protocols,
                fusion_correction_protocols=fusion_correction_protocols,
                link_layer_corrections=link_layer_corrections, charlie_manager_protocols=charlie_manager_protocols,
                fusion_time_window=fusion_time_window, swapping_time_window=swapping_time_window)

        self._tree_controller_protocols = FusionTreeController(
            root_node=network.get_node(f"Node{fusion_tree.node.id}"),
            tree_id=0,
            fusion_protocol=all_other_protocols[-1],
            fusion_correction_protocols=[fusion_correction_protocols[node_id] for node_id in
                                         fusion_tree.end_nodes if node_id != fusion_tree.node.id],
            end_nodes=fusion_tree.end_nodes, all_nodes=set(),
            end_qPNodes=[network.get_node(f"Node{idd}") for idd in fusion_tree.end_nodes],
            fro=True, trees_and_nodes=swapping_trees_and_nodes)

        # starting all protocols
        for messaging_protocol in messaging_protocols.values():
            messaging_protocol.start()
        for link_layer_correction in link_layer_corrections.values():
            link_layer_correction.start()
        for entangling_protocol in entangling_protocols.values():
            entangling_protocol.start()
        for fusion_correction_protocol in fusion_correction_protocols.values():
            fusion_correction_protocol.start()
        for charlie_manager_protocol in charlie_manager_protocols.values():
            charlie_manager_protocol.start()
        for swapping_correction_protocol in swapping_tree_correction_protocols.values():
            swapping_correction_protocol.start()

        for protocol in all_other_protocols:
            protocol.start()

        self._tree_controller_protocols.start()

        self._stats = ns.sim_run(duration=simulation_duration * 1e9)

    @property
    def success_rate(self) -> float:
        return self._tree_controller_protocols.successful_counts / self._duration

    @property
    def fidelity(self) -> float:
        """return average fidelity of the final GHZ state"""
        return float('inf') if len(self._tree_controller_protocols.fidelity) == 0 else \
            sum(self._tree_controller_protocols.fidelity) / len(self._tree_controller_protocols.fidelity)

    @staticmethod
    def setup_fro_fusion_tree_protocols(network: Network, qgraph: qGraph, tree_node: fusion_retain_node, tree_id: int,
                                        swapping_trees: Dict[str, List[tree_node]], swapping_tree_id_offset: int,
                                        messaging_protocols: Dict[int, MessagingProtocolV2],
                                        entangling_protocols: Dict[int, EntanglingProtocolV2],
                                        swapping_corrections_protocols: Dict[int, CorrectionProtocolV2],
                                        fusion_correction_protocols: Dict[int, FusionCorrectionProtocol],
                                        link_layer_corrections: Dict[int, LinkCorrectionProtocolV2],
                                        charlie_manager_protocols: Dict[str, CharlieManager],
                                        fusion_time_window: float = float('inf'),
                                        swapping_time_window: float = float('inf')) -> (List, int, int):
        if tree_node is None:
            return [], 0, 0
        if tree_node.node is None:
            # reached to edge which itself is a set of swapping trees
            src_id, dst_id = list(tree_node.end_nodes)
            if not f"{src_id}_{dst_id}" in swapping_trees:
                src_id, dst_id = dst_id, src_id
            return *FusionRetainOnlyTreeProtocol.setup_fro_underlying_swapping_trees(
                network, qgraph, src_id, dst_id,
                trees=swapping_trees[f"{src_id}_{dst_id}"], tree_id_offset=swapping_tree_id_offset,
                messaging_protocols=messaging_protocols, entangling_protocols=entangling_protocols,
                link_layer_corrections=link_layer_corrections, charlie_manager_protocols=charlie_manager_protocols,
                swapping_correction_protocols=swapping_corrections_protocols,
                swapping_time_window=swapping_time_window), 1, \
                   len(swapping_trees[f"{src_id}_{dst_id}"]) + swapping_tree_id_offset

        end_nodes_key = '-'.join([str(end_id) for end_id in sorted(list(tree_node.end_nodes))])

        left_protocols, left_swapping_trees, left_height, new_swapping_tree_id_offset = \
            FusionRetainOnlyTreeProtocol.setup_fro_fusion_tree_protocols(
                network=network, qgraph=qgraph, tree_node=tree_node.sub1, tree_id=tree_id,
                swapping_trees=swapping_trees, swapping_tree_id_offset=swapping_tree_id_offset,
                messaging_protocols=messaging_protocols, entangling_protocols=entangling_protocols,
                swapping_corrections_protocols=swapping_corrections_protocols,
                fusion_correction_protocols=fusion_correction_protocols,
                link_layer_corrections=link_layer_corrections,
                charlie_manager_protocols=charlie_manager_protocols,
                fusion_time_window=fusion_time_window, swapping_time_window=swapping_time_window)
        right_protocols, right_swapping_trees, right_height, new_swapping_tree_id_offset = \
            FusionRetainOnlyTreeProtocol.setup_fro_fusion_tree_protocols(
                network=network, qgraph=qgraph, tree_node=tree_node.sub2, tree_id=tree_id,
                swapping_trees=swapping_trees, swapping_tree_id_offset=new_swapping_tree_id_offset,
                messaging_protocols=messaging_protocols, entangling_protocols=entangling_protocols,
                swapping_corrections_protocols=swapping_corrections_protocols,
                fusion_correction_protocols=fusion_correction_protocols,
                link_layer_corrections=link_layer_corrections,
                charlie_manager_protocols=charlie_manager_protocols,
                fusion_time_window=fusion_time_window, swapping_time_window=swapping_time_window)

        end_nodes_left_key = '-'.join([str(end_id) for end_id in sorted(list(tree_node.sub1.end_nodes))])
        end_nodes_right_key = '-'.join([str(end_id) for end_id in sorted(list(tree_node.sub2.end_nodes))])

        # FUSION protocol and adding qubit mappings and FUSION_CORRECTION signals
        node_p = network.get_node(f"Node{tree_node.node.id}")
        node_p.left_height[f"{tree_id}_{end_nodes_key}"] = left_height
        node_p.right_height[f"{tree_id}_{end_nodes_key}"] = right_height
        end_nodes_qpnodes = [network.get_node(f'Node{node_num}') for node_num in list(tree_node.end_nodes)]

        if left_height == 1 and right_height == 1:
            for qpnode in end_nodes_qpnodes:
                if qpnode.ID == tree_node.node.id:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = qpnode.qubit_mapping[
                        f"#{tree_node.sub1.end_nodes.difference({tree_node.node.id}).pop()}"]
                else:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = qpnode.qubit_mapping[f"#{tree_node.node.id}"]

        elif left_height == 1:
            for qpnode in end_nodes_qpnodes:
                if qpnode.ID == tree_node.node.id:
                    # one of the qubit (left) in this node remains for larger states, fetched from link-level qubit
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = qpnode.qubit_mapping[
                        f"#{tree_node.sub1.end_nodes.difference({tree_node.node.id}).pop()}"]
                elif qpnode.ID in tree_node.sub1.end_nodes:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"#{tree_node.node.id}"]
                else:
                    # qubit must be fetched from a tree information
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_right_key}"]

        elif right_height == 1:
            for qpnode in end_nodes_qpnodes:
                if qpnode.ID == tree_node.node.id:
                    # one of the qubit (left) in this node remains for larger states, fetched from tree-level qubit
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_left_key}"]
                elif qpnode.ID in tree_node.sub2.end_nodes:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"#{tree_node.node.id}"]
                else:
                    # qubit must be fetched from a tree information
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_left_key}"]

        else:
            for qpnode in end_nodes_qpnodes:
                if qpnode.ID == tree_node.node.id:
                    # one of the qubit (left) in this node remains for larger states, fetched from tree-level qubit
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_left_key}"]
                elif qpnode.ID in tree_node.sub1.end_nodes:
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_left_key}"]
                else:
                    # qubit must be fetched from a tree information
                    qpnode.qubit_mapping[f"{tree_id}_{end_nodes_key}"] = \
                        qpnode.qubit_mapping[f"{tree_id}_{end_nodes_right_key}"]

        # fusion protocol and its signals
        node_fusion_protocol = FusionProtocol(node=node_p, tree_id=tree_id,
                                              end_nodes_left=tree_node.sub1.end_nodes,
                                              end_nodes_right=tree_node.sub2.end_nodes,
                                              all_nodes_left=set(),
                                              all_nodes_right=set(),
                                              messaging_protocol=messaging_protocols[tree_node.node.id],
                                              fusion_correction_protocol=
                                              fusion_correction_protocols[tree_node.node.id],
                                              retain=True, fro=True,
                                              trees_and_nodes=left_swapping_trees + right_swapping_trees,
                                              fusion_time_window=fusion_time_window)
        messaging_protocols[tree_node.node.id].add_ghz_fusion_correction_ack_signals(tree_id=tree_id,
                                                                                     end_nodes_key=end_nodes_left_key)
        messaging_protocols[tree_node.node.id].add_ghz_fusion_correction_ack_signals(tree_id=tree_id,
                                                                                     end_nodes_key=end_nodes_right_key)

        for node_id in tree_node.end_nodes:
            if node_id != tree_node.node.id:
                messaging_protocols[node_id].add_ghz_signals(tree_id=tree_id, end_nodes_key=end_nodes_key)
                fusion_correction_protocols[node_id].add_signals(tree_id=tree_id, end_nodes_key=end_nodes_key)

        # updating parent dict
        if f"{tree_id}_{end_nodes_key}" not in qPNode.parents:
            qPNode.parents[f"{tree_id}_{end_nodes_key}"] = (None, None)
        qPNode.parents[f"{tree_id}_{end_nodes_key}"] = (tree_node.node.id,
                                                        qPNode.parents[f"{tree_id}_{end_nodes_key}"][1])
        if f"{tree_id}_{end_nodes_left_key}" not in qPNode.parents:
            qPNode.parents[f"{tree_id}_{end_nodes_left_key}"] = (None, None)
        qPNode.parents[f"{tree_id}_{end_nodes_left_key}"] = (qPNode.parents[f"{tree_id}_{end_nodes_left_key}"][0],
                                                             tree_node.node.id)
        if f"{tree_id}_{end_nodes_right_key}" not in qPNode.parents:
            qPNode.parents[f"{tree_id}_{end_nodes_right_key}"] = (None, None)
        qPNode.parents[f"{tree_id}_{end_nodes_right_key}"] = (qPNode.parents[f"{tree_id}_{end_nodes_right_key}"][0],
                                                              tree_node.node.id)

        return left_protocols + right_protocols + [node_fusion_protocol], left_swapping_trees + right_swapping_trees,\
               max(left_height, right_height) + 1, new_swapping_tree_id_offset

    @staticmethod
    def setup_fro_underlying_swapping_trees(network: Network, qgraph: qGraph, src_id: int, dst_id: int,
                                            trees: List[tree_node], tree_id_offset: int,
                                            messaging_protocols: Dict[int, MessagingProtocolV2],
                                            entangling_protocols: Dict[int, EntanglingProtocolV2],
                                            link_layer_corrections: Dict[int, LinkCorrectionProtocolV2],
                                            charlie_manager_protocols: Dict[str, CharlieManager],
                                            swapping_correction_protocols: Dict[int, CorrectionProtocolV2],
                                            swapping_time_window: float = float('inf')) -> (List, List):
        def helper(tree: tree_node, tree_id: int) -> (List, int, set, int, int):
            if tree is None:
                return []
            if isinstance(tree.data, qChannel):
                # this as left, other as right
                this_node, other_node = network.get_node(f"Node{tree.data.this.id}"), \
                                        network.get_node(f"Node{tree.data.other.id}")

                # assigning new qubit memory locations, (network, link)
                this_node.qubit_mapping[f"{tree_id}_{other_node.ID}"] = (this_node.avail_mem_pos(),
                                                                         this_node.avail_mem_pos())
                other_node.qubit_mapping[f"{tree_id}_{this_node.ID}"] = (other_node.avail_mem_pos(),
                                                                         other_node.avail_mem_pos())
                # set entangling to True at the beginning
                this_node.entangling[f"{tree_id}_{other_node.ID}"] = True
                other_node.entangling[f"{tree_id}_{this_node.ID}"] = True

                if tree_id in this_node.neighbors:
                    # left neighbor of qpnode1 is already set
                    this_node.neighbors[tree_id] = (this_node.neighbors[tree_id][0], other_node.ID)
                else:
                    this_node.neighbors[tree_id] = (-1, other_node.ID)

                if tree_id in other_node.neighbors:
                    # right neighbor of qpnode2 is already set
                    other_node.neighbors[tree_id] = (this_node.ID, other_node.neighbors[tree_id][1])
                else:
                    other_node.neighbors[tree_id] = (this_node.ID, -1)

                # add proper messaging signals
                messaging_protocols[this_node.ID].add_link_signals(tree_id, neighbor_id=other_node.ID)
                messaging_protocols[other_node.ID].add_link_signals(tree_id, neighbor_id=this_node.ID)

                # make link layer request here
                left_node, right_node = this_node, other_node
                if f"{this_node.ID}_{other_node.ID}" in charlie_manager_protocols:
                    edge_key = f"{this_node.ID}_{other_node.ID}"
                else:
                    left_node, right_node = other_node, this_node
                    edge_key = f"{other_node.ID}_{this_node.ID}"

                # add tree signals to charlie managers
                charlie_manager_protocols[edge_key].add_signals(tree_id=tree_id)

                effective_rate = 1.0 / tree.avr_ent_time
                e = qgraph.get_edge(left_node.ID, right_node.ID)
                rate = TreeProtocol.effective_rate_to_success_node_rate(effective_rate, e)

                # create EntanglementGenerator and Charlie protocols
                charlie_node = network.get_node(f"Charlie_{edge_key}")
                left_ent_gen_proto = \
                    EntanglementGeneratorV2(node=left_node, tree_id=tree_id, rate=rate,
                                            isLeft=False, charlie_id=edge_key,
                                            link_layer_qmem_pos=left_node.qubit_mapping[f"{tree_id}_{right_node.ID}"][
                                                1],
                                            other_node_id=right_node.ID,
                                            messaging_protocol=messaging_protocols[left_node.ID])
                right_ent_gen_proto = \
                    EntanglementGeneratorV2(node=right_node, tree_id=tree_id, rate=rate,
                                            isLeft=True, charlie_id=edge_key,
                                            link_layer_qmem_pos=right_node.qubit_mapping[f"{tree_id}_{left_node.ID}"][
                                                1],
                                            other_node_id=left_node.ID,
                                            messaging_protocol=messaging_protocols[right_node.ID])
                charlie_protocol = CharlieProtocolV2(node=charlie_node, tree_id=tree_id,
                                                     charlie_manager_protocol=charlie_manager_protocols[edge_key])

                # add edge (for this tree) signal to right node link correction protocol
                link_layer_corrections[right_node.ID].add_tree_edge_signal(tree_id=tree_id, left_node_id=left_node.ID)

                # add appropriate signals to listen to EntanglingProtocols
                entangling_protocols[left_node.ID].add_protocol(tree_id=tree_id, other_id=right_node.ID, isLeft=True)
                entangling_protocols[right_node.ID].add_protocol(tree_id=tree_id, other_id=left_node.ID, isLeft=False)

                return [left_ent_gen_proto, right_ent_gen_proto, charlie_protocol], 1, {left_node.ID, right_node.ID}, \
                       this_node.ID, other_node.ID

            left_protocols, left_height, left_nodes, left_left, _ = helper(tree.left, tree_id)
            right_protocols, right_height, right_nodes, _, right_right = helper(tree.right, tree_id)

            swapping_node = network.get_node(f"Node{tree.data.id}")
            swapping_node.left_height[tree_id] = left_height
            swapping_node.right_height[tree_id] = right_height
            swapping_node.leftest_id[tree_id], swapping_node.rightest_id[tree_id] = left_left, right_right
            swapping_node.children_id[tree_id] = left_nodes.union(right_nodes)
            if swapping_node.ID in swapping_node.children_id[tree_id]:  # remove node itself from its children
                swapping_node.children_id[tree_id].remove(swapping_node.ID)

            swapping_protocol = SwappingProtocolV2(swapping_node, tree_id,
                                                   correction_protocol=swapping_correction_protocols[swapping_node.ID],
                                                   messaging_protocol=messaging_protocols[swapping_node.ID],
                                                   swapping_time_window=swapping_time_window)
            messaging_protocols[swapping_node.ID].add_correction_ack_signals(tree_id,
                                                                             swapping_node.rightest_id[tree_id])
            messaging_protocols[swapping_node.leftest_id[tree_id]].add_correction_ack_signals(
                tree_id, swapping_node.rightest_id[tree_id])
            swapping_correction_protocols[right_right].add_trees(tree_id)
            swapping_correction_protocols[right_right].add_main_root_id(tree_id, swapping_node.ID)

            return left_protocols + right_protocols + [swapping_protocol], max(left_height, right_height) + 1, \
                   left_nodes.union(right_nodes), left_left, right_right

        src_node, dst_node = network.get_node(f"Node{src_id}"), network.get_node(f"Node{dst_id}")
        source_swapping_trees_controller = SwappingTreeControllerProtocol(
            node=src_node, is_source=True, other_end_id=dst_id, messaging_protocol=messaging_protocols[src_id],
            entangling_protocol=entangling_protocols[src_id], correction_protocol=swapping_correction_protocols[src_id])
        dest_swapping_trees_controller = SwappingTreeControllerProtocol(
            node=dst_node, is_source=False, other_end_id=src_id, messaging_protocol=messaging_protocols[dst_id],
            entangling_protocol=entangling_protocols[dst_id], correction_protocol=swapping_correction_protocols[dst_id])
        src_node.src_dst_ep_active[f"#{dst_id}"] = False
        src_node.acceptable_time[f"#{dst_id}"] = 0
        dst_node.src_dst_ep_active[f"#{src_id}"] = False
        dst_node.acceptable_time[f"#{src_id}"] = 0

        all_src_dst_protocols = []
        trees_and_nodes = []
        for tree_id, tree in enumerate(trees):
            tree_protocols, tree_height, all_tree_nodes, _, _ = helper(tree, tree_id + tree_id_offset)
            messaging_protocols[src_id].add_correction_ack_signals(tree_id=tree_id + tree_id_offset,
                                                                   rightest_node_id=dst_id)
            source_swapping_trees_controller.add_tree(tree_id=tree_id + tree_id_offset,
                                                      is_one_hop=tree_height == 1,
                                                      all_nodes=all_tree_nodes)
            dest_swapping_trees_controller.add_tree(tree_id=tree_id + tree_id_offset,
                                                    is_one_hop=tree_height == 1,
                                                    all_nodes=all_tree_nodes)
            all_src_dst_protocols += tree_protocols
            trees_and_nodes.append((tree_id + tree_id_offset, all_tree_nodes, src_id, dst_id))
            for node in all_tree_nodes:
                messaging_protocols[node].add_signals(tree_id=tree_id + tree_id_offset)
        src_node.qubit_mapping[f"#{dst_id}"] = src_node.avail_mem_pos()
        dst_node.qubit_mapping[f"#{src_id}"] = dst_node.avail_mem_pos()
        return all_src_dst_protocols + [source_swapping_trees_controller, dest_swapping_trees_controller], \
               trees_and_nodes


class CentralGHZTreeProtocol:
    """A class designed to perform ghz generation and distribution via a central node. The class takes a set of
    swapping-trees. Upon successful swapping-trees construction, some teleportation will be performed to distribute
    the ghz. state"""

    def __init__(self, qgraph: qGraph, end_nodes: set[int],
                 swapping_trees: List[List[tree_node]],
                 simulation_duration: float,
                 swapping_time_window: float = float('inf')):
        if len(swapping_trees) == 0:
            return
        ns.set_qstate_formalism(ns.QFormalism.KET)
        ns.set_random_state(42)
        ns.sim_reset()
        central_node_id = swapping_trees[0][0].leftest_node.id
        self._duration = simulation_duration
        # get only those nodes and edges that are going to be used to save memory and efficiency
        used_nodes, used_edges = set(), set()
        for terminal_pair_trees in swapping_trees:
            for tree in terminal_pair_trees:
                tree_nodes, tree_edges = tree_node.used_nodes_edges(tree, "_")
                used_nodes = used_nodes.union(tree_nodes)
                used_edges = used_edges.union(tree_edges)

        nodes = [qPNode(node) for node in qgraph.V if node.id in used_nodes]
        network = Network("network", nodes)
        # setup classical connections between each pair of nodes
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                TreeProtocol.add_classical_connections(network, nodes[i], nodes[j])

        # create protocols that have one instance/node for all trees
        messaging_protocols = {node.ID: MessagingProtocolV2(node) for node in nodes}
        link_layer_corrections = {node.ID: LinkCorrectionProtocolV2(node, messaging_protocols[node.ID])
                                  for node in nodes}
        entangling_protocols = {node.ID: EntanglingProtocolV2(node, messaging_protocols[node.ID],
                                                              link_layer_corrections[node.ID]) for node in nodes}
        swapping_tree_correction_protocols = {node.ID: CorrectionProtocolV2(node, messaging_protocols[node.ID])
                                              for node in nodes}

        # create one CharlieManager/edge (only once)
        charlie_manager_protocols = defaultdict(CharlieManager)
        for e in qgraph.E:
            charlie_id = f"{e.this.id}_{e.other.id}"
            if charlie_id in used_edges:
                charlie = TreeProtocol.add_quantum_connections(network=network, e=e)
                charlie_manager_protocols[charlie_id] = CharlieManager(charlie)

        tree_id_offset = 0
        all_other_protocols = []
        source_swapping_tree_controller_protocols = {}
        trees_and_nodes = []
        for terminal_pair_trees in swapping_trees:
            dst_id = terminal_pair_trees[0].rightest_node.id
            terminal_pair_protocols, terminal_trees_and_nodes = \
                FusionRetainOnlyTreeProtocol.setup_fro_underlying_swapping_trees(
                    network, qgraph, central_node_id, dst_id=dst_id,
                    trees=terminal_pair_trees, tree_id_offset=tree_id_offset,
                    messaging_protocols=messaging_protocols, entangling_protocols=entangling_protocols,
                    link_layer_corrections=link_layer_corrections, charlie_manager_protocols=charlie_manager_protocols,
                    swapping_correction_protocols=swapping_tree_correction_protocols,
                    swapping_time_window=swapping_time_window)
            source_swapping_tree_controller_protocols[dst_id] = terminal_pair_protocols[-2]
            all_other_protocols += terminal_pair_protocols
            trees_and_nodes += terminal_trees_and_nodes
            tree_id_offset += len(terminal_pair_trees)

        # add new qubit to central node if it's part of GHZ
        central_node = network.get_node(f'Node{central_node_id}')
        central_node.qubit_mapping[f"#{tree_id_offset}"] = central_node.avail_mem_pos()
        # add new tree (qubit_mapping, neighbors, and signals to messaging and correction protocols)
        for dst_id in source_swapping_tree_controller_protocols.keys():
            dst_node = network.get_node(f"Node{dst_id}")
            dst_node.neighbors[tree_id_offset] = (central_node_id, central_node_id)
            dst_node.qubit_mapping[f"{tree_id_offset}_{central_node_id}"] = \
                (dst_node.qubit_mapping[f"#{central_node_id}"], dst_node.qubit_mapping[f"#{central_node_id}"])
            messaging_protocols[dst_id].add_signals(tree_id=tree_id_offset)
            swapping_tree_correction_protocols[dst_id].add_trees(tree_id=tree_id_offset)
            swapping_tree_correction_protocols[dst_id].add_main_root_id(tree_id=tree_id_offset,
                                                                        main_root_id=central_node_id)
            messaging_protocols[central_node_id].add_correction_ack_signals(tree_id=tree_id_offset,
                                                                            rightest_node_id=dst_id)
        central_ghz_teleportation_protocol = CentralGHZTeleportation(
            node=central_node, tree_id=tree_id_offset,
            swapping_trees_controllers=source_swapping_tree_controller_protocols,
            is_terminal=len(swapping_trees) != len(end_nodes), trees_and_nodes=trees_and_nodes)
        self._central_ghz_controller = CentralGHZController(
            central_node=central_node, tree_id=tree_id_offset,
            central_ghz_teleportation=central_ghz_teleportation_protocol,
            teleportation_correction_protocols=[swapping_tree_correction_protocols[node_id] for node_id in end_nodes
                                                if node_id != central_node_id],
            end_nodes=end_nodes, trees_and_nodes=trees_and_nodes,
            end_qpNodes=[network.get_node(f"Node{node_id}") for node_id in end_nodes if node_id != central_node_id])



        # starting all protocols
        for messaging_protocol in messaging_protocols.values():
            messaging_protocol.start()
        for link_layer_correction in link_layer_corrections.values():
            link_layer_correction.start()
        for entangling_protocol in entangling_protocols.values():
            entangling_protocol.start()
        for charlie_manager_protocol in charlie_manager_protocols.values():
            charlie_manager_protocol.start()
        for swapping_correction_protocol in swapping_tree_correction_protocols.values():
            swapping_correction_protocol.start()

        for protocol in all_other_protocols:
            protocol.start()

        central_ghz_teleportation_protocol.start()
        self._central_ghz_controller.start()

        self._stats = ns.sim_run(duration=simulation_duration * 1e9)

    @property
    def success_rate(self) -> float:
        return self._central_ghz_controller.successful_counts / self._duration

    @property
    def fidelity(self) -> float:
        """return average fidelity of the final GHZ state"""
        return float('inf') if len(self._central_ghz_controller.fidelity) == 0 else \
            sum(self._central_ghz_controller.fidelity) / len(self._central_ghz_controller.fidelity)


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
