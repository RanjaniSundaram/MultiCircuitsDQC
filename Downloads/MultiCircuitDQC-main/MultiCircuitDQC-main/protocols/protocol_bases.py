import operator
import sys

import netsquid.components.qmemory

sys.path.append("..")

from netsquid.protocols import NodeProtocol, Signals, Protocol
from protocols.component_bases import qPNode, Charlie
from commons.tools import bell_measurement, fusion
import netsquid as ns
from functools import reduce
from collections import namedtuple
from enum import Enum
from random import random
from netsquid.components import QuantumMemory
import logging
import datetime
import os
from netsquid.util.datacollector import DataCollector
from typing import List, Dict

ProtocolMessage = namedtuple("ProtocolMessage", "type, tree_id, data, next_dst_id, gen_time")

if not os.path.exists("protocol_logs"):
    os.makedirs("protocol_logs")
logging.basicConfig(filename="protocol_logs/log" + datetime.datetime.now().strftime('_%Y%m_%d%H_%M') + ".log",
                    level=logging.INFO)


class MessageType(Enum):
    SWAPPING_CORRECTION = 1
    START_ENTANGLING = 2
    CORRECTION_ACK = 3
    TELEPORTATION_CORRECTION = 4
    LINK_CORRECTION = 5
    ENTANGLEMENT_READY = 6  # when link layer entanglement is ready
    CORRECTION_FAILURE = 7
    PHOTON_ARRIVED = 8
    START_ENTANGLEMENT_GENERATING = 9  # a signal to request link layer to START entanglement generating
    STOP_ENTANGLEMENT_GENERATING = 10  # a signal to request link layer to STOP entanglement generating
    FUSION_CORRECTION = 11
    FUSION_CORRECTION_ACK = 12
    FUSION_CORRECTION_FAILURE = 13
    STOP_ENTANGLING = 14
    SRC_DST_RESET = 15  # used in cases where there are multiple trees for (s,d) like fro after an ep is being consumed
    RESET_SWAPPING = 16


class tools:
    @staticmethod
    def tree_failure(node: qPNode, tree_id: int):
        logging.info(f"{ns.sim_time():.1f}; (TREE{tree_id}) swapping was UNSUCCESSFUL"
                     f" at node{node.ID}.")
        node.entangling[tree_id] = (True, True)  # node itself should start entangling
        # send instant message to its neighbors
        for neighbor in node.neighbors[tree_id]:
            node.ports["cto_neighbor" + str(neighbor)].tx_output(ProtocolMessage(
                MessageType.START_ENTANGLING, tree_id, None, -1, ns.sim_time()))
        for child_id in node.children_id[tree_id]:
            if (child_id == node.leftest_id[tree_id] and
                node.leftest_id[tree_id] == node.neighbors[tree_id][0]) or \
                    (child_id == node.rightest_id[tree_id] and
                     node.rightest_id[tree_id] == node.neighbors[tree_id][1]):
                continue
            data = "both"
            # if child_id != self.node.neighbors[self._tree_id][0] and \
            #         child_id != self.node.neighbors[self._tree_id][1]:

            if child_id == node.leftest_id[tree_id]:
                data = "right"
            elif child_id == node.rightest_id[tree_id]:
                data = "left"
            if data == "both" and child_id == node.neighbors[tree_id][0]:
                data = "left"
            elif data == "both" and child_id == node.neighbors[tree_id][1]:
                data = "right"
            node.ports["cto_node" + str(child_id)].tx_output(
                ProtocolMessage(type=MessageType.START_ENTANGLING, tree_id=tree_id,
                                data=data, next_dst_id=-1, gen_time=ns.sim_time()))
            logging.debug(f"{ns.sim_time():.1f}: (TREE{tree_id}) sending ENTANGLING signal to node"
                          f" {str(child_id)} for {data} positions.")

    @staticmethod
    def tree_failure_v2(node: qPNode, tree_id: int, all_nodes: set[int], src_id: int, dst_id: int,
                        reset_swapping: bool = False):
        for node_id in all_nodes:
            if node_id == node.ID:
                continue
            if node_id == src_id:
                data = 'right'
            elif node_id == dst_id:
                data = 'left'
            else:
                data = 'both'
            node.ports[f"cto_node{node_id}"].tx_output(ProtocolMessage(MessageType.START_ENTANGLING, tree_id, data, -1,
                                                                       ns.sim_time()))
            node.ports[f"cto_node{node_id}"].tx_output(ProtocolMessage(MessageType.START_ENTANGLEMENT_GENERATING,
                                                                       tree_id, data, -1, ns.sim_time()))
            if reset_swapping and data == 'both':
                node.ports[f"cto_node{node_id}"].tx_output(ProtocolMessage(MessageType.RESET_SWAPPING, tree_id,
                                                                           ns.sim_time(), -1, ns.sim_time()))
        logging.info(f"{ns.sim_time():.1f}: (TREE{tree_id}) (Node{node.ID}) Restarting tree{tree_id} with nodes-"
                     f"{all_nodes}, source_id:{src_id}, and dest_id:{dst_id}")

    @staticmethod
    def fusion_tree_failure(node: qPNode, tree_id: int, all_nodes: set[int], end_nodes_key: str):
        # send instant message to its neighbors
        for neighbor in node.ghz_edges[f"{tree_id}_{end_nodes_key}"]:
            node.entangling[f"{tree_id}_{neighbor}"] = True
            node.ports["cto_neighbor" + str(neighbor)].tx_output(ProtocolMessage(
                MessageType.START_ENTANGLING, tree_id, end_nodes_key, -1, ns.sim_time()))
            node.ports["cto_neighbor" + str(neighbor)].tx_output(ProtocolMessage(
                MessageType.START_ENTANGLEMENT_GENERATING, tree_id, end_nodes_key, -1, ns.sim_time()))
        for child in all_nodes:
            if child != node.ID and child not in node.ghz_edges[f"{tree_id}_{end_nodes_key}"]:
                node.ports[f"cto_node{child}"].tx_output(ProtocolMessage(MessageType.START_ENTANGLING, tree_id,
                                                                         end_nodes_key, -1, ns.sim_time()))
                node.ports[f"cto_node{child}"].tx_output(ProtocolMessage(MessageType.START_ENTANGLEMENT_GENERATING,
                                                                         tree_id, end_nodes_key, -1, ns.sim_time()))

    @staticmethod
    def create_ghz_reference_state(n: int):
        ref_state = ns.qubits.create_qubits(2, no_state=True)
        ns.qubits.assign_qstate(ref_state, ns.b00)
        for _ in range(n - 2):
            new_state = ns.qubits.create_qubits(2, no_state=True)
            ns.qubits.assign_qstate(new_state, ns.b00)
            ns.qubits.operate(qubits=[ref_state[-1], new_state[0]], operator=ns.CNOT)
            m, _ = ns.qubits.measure(new_state[0], observable=ns.Z)
            if m:
                ns.qubits.operate(qubits=[new_state[1]], operator=ns.X)
            ref_state = ref_state + [new_state[1]]

        return ref_state[0].qstate.qubits, ref_state[0].qstate.qrepr

    @staticmethod
    def restart_swapping_trees(node: qPNode, tree_id: int, all_tree_nodes: set[int], src_id: int, dst_id: int):
        tools.tree_failure_v2(node, tree_id, all_tree_nodes, src_id, dst_id, True)
        if node.ID != src_id:
            node.ports[f"cto_node{src_id}"].tx_output(
                ProtocolMessage(MessageType.SRC_DST_RESET, tree_id, ns.sim_time(), dst_id, ns.sim_time()))
        if node.ID != dst_id:
            node.ports[f"cto_node{dst_id}"].tx_output(
                ProtocolMessage(MessageType.SRC_DST_RESET, tree_id, ns.sim_time(), src_id, ns.sim_time()))


class CharlieProtocol(NodeProtocol):
    """Charlie protocol that is responsible to do photonics BSM based on receiving from its left and right ports"""

    def __init__(self, node: Charlie, bsm_time_window: float = 10e-9):
        super(CharlieProtocol, self).__init__(node, name=node.name + "_charlie_protocol")
        self._bsm_time_window = bsm_time_window

    def run(self):
        qleft_port = self.node.ports["qin_left"]
        qright_port = self.node.ports["qin_right"]
        cright_port = self.node.ports["cout_right"]
        while True:
            expr = yield self.await_port_input(qleft_port) | \
                         self.await_port_input(qright_port)
            if expr.first_term.value:
                # left qubit comes
                for left_qubit in qleft_port.rx_input().items:
                    tree_id = int(left_qubit.name.split("_")[0])
                    left_time = ns.sim_time()
                    right_qubit, right_time = None, -float('inf')
                    if tree_id in self.node.qubit_data:
                        right_qubit, right_time = self.node.qubit_data[tree_id][1]
                    self.node.qubit_data[tree_id] = ((left_qubit, left_time),
                                                     (right_qubit, right_time))
            else:
                for right_qubit in qright_port.rx_input().items:
                    tree_id = int(right_qubit.name.split("_")[0])
                    right_time = ns.sim_time()
                    left_qubit, left_time = None, -float('inf')
                    if tree_id in self.node.qubit_data:
                        left_qubit, left_time = self.node.qubit_data[tree_id][0]
                    self.node.qubit_data[tree_id] = ((left_qubit, left_time),
                                                     (right_qubit, right_time))

            # time to decide if BSM is possible
            results = {}
            for tree_id in self.node.qubit_data.keys():
                left_qubit, left_time = self.node.qubit_data[tree_id][0]
                right_qubit, right_time = self.node.qubit_data[tree_id][1]
                if left_qubit is not None and right_qubit is not None and \
                        abs(left_time - right_time) < self._bsm_time_window:
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{str(tree_id)}) both link layer qubits "
                                  f"successfully received  at {self.node.name}")
                    qmemory = QuantumMemory("tmp" + str(tree_id), num_positions=2)
                    qmemory.put([left_qubit, right_qubit], positions=[0, 1])
                    results[tree_id] = bell_measurement(qmemory, positions=[0, 1], PHYSICAL=True)
            if len(results) > 0:
                yield self.await_timer(self.node.bsm_time * 1e9)
            for tree_id, res in results.items():
                if random() < self.node.bsm_success_rate:
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{str(tree_id)}) BSM was successfully done at "
                                  f"{self.node.name}")
                    cright_port.tx_output(ProtocolMessage(type=MessageType.LINK_CORRECTION, data=res,
                                                          tree_id=tree_id,
                                                          next_dst_id=self.node._left_node_id,
                                                          gen_time=ns.sim_time()))
                    # yield self.await_timer(random())
                self.node.qubit_data[tree_id] = ((None, -float('inf')), (None, -float('inf')))


class EntanglementGenerator(NodeProtocol):
    """A protocol that generates a pair of EPs, save one in node's qmemory and send the other to charlie to
    do optical bsm. rate is 1/s"""

    def __init__(self, node: qPNode, tree_id: int, rate: float('inf'), isLeft: bool, charlie_id: str,
                 link_layer_qmem_pos: int):
        super(EntanglementGenerator, self).__init__(node, name=node.name + ("_left" if isLeft else "_right") +
                                                               "_generator_for_tree" + str(tree_id))
        self._isLeft = isLeft
        self._rate = rate
        self._tree_id = tree_id
        self._charlie_id = charlie_id
        self._link_layer_qmem_pos = link_layer_qmem_pos

    def run(self):
        qout_port = self.node.ports["qout_" + ("left" if self._isLeft else "right") + self._charlie_id]
        while True:
            yield self.await_timer(1.0 / self._rate * 1e9)  # wait (nanoseconds)
            q1, q2 = ns.qubits.create_qubits(2, no_state=True)
            q1.name += f"{self.node.ID}-{self._link_layer_qmem_pos}"
            q2.name = str(self._tree_id) + ("_right" if self._isLeft else "_left")
            ns.qubits.assign_qstate([q1, q2], ns.b00)
            self.node.qmemory.put(q1, positions=self._link_layer_qmem_pos, replace=True)
            qout_port.tx_output(q2)
            logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) qubit was generated at {self.name}")
            # break


class MessagingProtocol(NodeProtocol):
    """a protocol that handles received messages from other nodes"""

    def __init__(self, node: qPNode):
        super().__init__(node, name=node.name + "_messaging_protocol")

    def add_signals(self, tree_id: int):
        self.add_signal(f"{MessageType.START_ENTANGLING.name}_{tree_id}")
        self.add_signal(f"{MessageType.SWAPPING_CORRECTION.name}_{tree_id}")
        self.add_signal(f"{MessageType.TELEPORTATION_CORRECTION.name}_{tree_id}")
        self.add_signal(f"{MessageType.LINK_CORRECTION.name}_{tree_id}")
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{tree_id}")
        self.add_signal(f"{MessageType.CORRECTION_ACK.name}_{tree_id}")

    def run(self):
        while True:
            expr = yield reduce(operator.or_, [self.await_port_input(self.node.ports[port_name])
                                               for port_name in self.node.ports if port_name.startswith("cfrom_")])
            for triggered_event in expr.triggered_events:
                for obj in triggered_event.source.input_queue:
                    received_port_name = obj[0].source.name
                    for message in obj[1].items:
                        # yield self.await_timer(0.1*random())
                        if message.type == MessageType.SWAPPING_CORRECTION:
                            # send signal and BSM results and leftest id (next_dst_id) to correction protocol
                            self.send_signal(signal_label=f"{MessageType.SWAPPING_CORRECTION.name}_{message.tree_id}",
                                             result=(received_port_name.split("cfrom_node")[1], message))
                            # (received_port_name.split("cfrom_node")[1]
                        elif message.type == MessageType.CORRECTION_ACK:
                            self.send_signal(signal_label=f"{MessageType.CORRECTION_ACK.name}_{message.tree_id}",
                                             result=(int(received_port_name.split("cfrom_node")[1]), message.tree_id))
                        elif message.type == MessageType.TELEPORTATION_CORRECTION:
                            self.send_signal(signal_label=f"{MessageType.TELEPORTATION_CORRECTION.name}_"
                                                          f"{message.tree_id}",
                                             result=[int(received_port_name.split("cfrom_node")[1]),
                                                     message.tree_id, message.data])
                        elif message.type == MessageType.ENTANGLEMENT_READY:
                            self.send_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{message.tree_id}",
                                             [message.tree_id,
                                              int(received_port_name.split("cfrom_neighbor")[1])])
                        elif message.type == MessageType.LINK_CORRECTION:
                            self.send_signal(f"{MessageType.LINK_CORRECTION.name}_{message.tree_id}", message)
                        elif message.type == MessageType.CORRECTION_FAILURE:
                            tools.tree_failure(self.node, message.tree_id)
                        elif message.type == MessageType.START_ENTANGLING:
                            is_neighbor = received_port_name.split("_")[1].startswith("neighbor")
                            if is_neighbor:
                                # signal comes from neighbors
                                left_neighbor = int(received_port_name.split("cfrom_neighbor")[1]) == \
                                                self.node.neighbors[message.tree_id][0]
                                if left_neighbor:
                                    # left neighbor
                                    self.node.entangling[message.tree_id] = (True,
                                                                             self.node.entangling[message.tree_id][1])
                                    logging.debug(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) set ENTANGLING to TRUE"
                                                  f" for LEFT position of node {str(self.node.ID)} because of left "
                                                  f"neighbor (node{received_port_name.split('cfrom_neighbor')[1]})")
                                else:
                                    self.node.entangling[message.tree_id] = (self.node.entangling[message.tree_id][0],
                                                                             True)
                                    logging.debug(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) set ENTANGLING to TRUE"
                                                  f" for RIGHT position of node {str(self.node.ID)} because of right"
                                                  f" neighbor (node{received_port_name.split('cfrom_neighbor')[1]})")
                            else:
                                # comes from parent
                                if self.node.qubit_mapping[message.tree_id][0][0] != -1 and \
                                        not self.node.entangling[message.tree_id][0] and \
                                        (message.data == "left" or message.data == "both"):
                                    # left qubit is part of the tree and was not set to entangling
                                    self.node.entangling[message.tree_id] = (True,
                                                                             self.node.entangling[message.tree_id][1])
                                    logging.debug(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) set ENTANGLING to TRUE"
                                                  f" for LEFT position of node {str(self.node.ID)} because of node"
                                                  f"{received_port_name.split('cfrom_node')[1]}")
                                    # send a message to left neighbor
                                    self.node.ports["cto_neighbor" +
                                                    str(self.node.neighbors[message.tree_id][0])].tx_output(message)
                                if self.node.qubit_mapping[message.tree_id][0][1] != -1 and \
                                        not self.node.entangling[message.tree_id][1] and \
                                        (message.data == "right" or message.data == "both"):
                                    # right qubit is part of the tree and was not set to entangling
                                    self.node.entangling[message.tree_id] = (self.node.entangling[message.tree_id][0],
                                                                             True)
                                    logging.debug(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) set ENTANGLING to TRUE"
                                                  f" for RIGHT position of node {str(self.node.ID)} because of node"
                                                  f"{received_port_name.split('cfrom_node')[1]}")
                                    # send a message to right neighbor
                                    self.node.ports["cto_neighbor" +
                                                    str(self.node.neighbors[message.tree_id][1])].tx_output(message)
                            logging.debug(
                                f"{ns.sim_time():.1f}: (TREE{message.tree_id}): node{self.node.ID} entangling:"
                                f" {self.node.entangling[message.tree_id]}")


class LinkCorrectionProtocol(NodeProtocol):
    """a protocol that waits for response from Charlie to do the BSM correction and entangle two ends and send a
    message to left end"""

    def __init__(self, node: qPNode, tree_id, messaging_protocol: MessagingProtocol, link_qmem_pos: int,
                 left_node_id: int):
        super(LinkCorrectionProtocol, self).__init__(node, name=node.name + "_link_correction_tree" + str(tree_id))
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._tree_id = tree_id
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{tree_id}")
        self._link_qmem_pos = link_qmem_pos
        self._left_node_id = left_node_id  # where we should send the results to

    def run(self):
        charlie_res = None
        while True:
            yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                    signal_label=f"{MessageType.LINK_CORRECTION.name}_{self._tree_id}")
            tmp_charlie_res = self.subprotocols["messaging_protocol"].get_signal_result(
                f"{MessageType.LINK_CORRECTION.name}_{self._tree_id}")
            if tmp_charlie_res.tree_id == self._tree_id and tmp_charlie_res.next_dst_id == self._left_node_id:
                charlie_res = tmp_charlie_res
            if charlie_res is not None:
                try:
                    if charlie_res.data[0]:
                        self.node.qmemory.operate(ns.Z, self._link_qmem_pos)
                    if charlie_res.data[1]:
                        self.node.qmemory.operate(ns.X, self._link_qmem_pos)
                    self.send_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}",
                                     [self._tree_id, self._left_node_id])
                    self.node.ports["cto_neighbor" + str(self._left_node_id)].tx_output(
                        ProtocolMessage(MessageType.ENTANGLEMENT_READY, self._tree_id, None, -1, ns.sim_time()))
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) {self.node.name} received results from "
                                  f"charlie and did "
                                  f"the CORRECTION. send an ACK to neighbor {self._left_node_id}")
                    logging.debug(f"(TREE{self._tree_id}) Entangling situation: {self.node.entangling[self._tree_id]}")
                except Exception as e:
                    logging.error(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Error happened at (link layer-waiting"
                                  f" for new one) node "
                                  f"{self.node.name}. \tError Msg: {e}")
                charlie_res = None


class EntanglingProtocol(NodeProtocol):
    """a protocol that accepts new qubits if the node allows. i.e. copy from link layer memories to BSM ones"""

    def __init__(self, node: qPNode, tree_id: int):
        super().__init__(node, name=node.name + "_entangling_protocol_tree" + str(tree_id))
        self._tree_id = tree_id

    def add_protocol(self, protocol, isLeft: bool):
        self.add_subprotocol(protocol, "left_link_layer" if isLeft else "right_link_layer")

    def run(self):
        left_qubit_pos, right_qubit_pos = self.node.qubit_mapping[self._tree_id][0]  # where BSM ops happen
        left_qubit_link_pos, right_qubit_link_pos = self.node.qubit_mapping[self._tree_id][1]  # where link qubits come

        if left_qubit_pos == -1:
            # src node, does not have left qubit memory attached to link layer
            while True:
                # yield self.await_port_input(self.node.ports["qin_right"])
                yield self.await_signal(sender=self.subprotocols["right_link_layer"],
                                        signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}")
                tree_id, _ = self.subprotocols["right_link_layer"].get_signal_result(
                    f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}")
                if tree_id == self._tree_id and self.node.entangling[self._tree_id][1]:
                    self.node.entangling[self._tree_id] = (False, False)  # reset to NOT ACCEPTING qubits
                    self.node.qmemory.pop(right_qubit_pos)
                    self.node.qmemory.mem_positions[right_qubit_pos].in_use = True
                    self.node.qmemory.put(self.node.qmemory.pop(right_qubit_link_pos), right_qubit_pos)
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{str(self._tree_id)}) node{self.node.ID} accepted right "
                                  f"qubit.")
        elif right_qubit_pos == -1:
            # dst node, does not have right qubit memory attached ro link layer
            while True:
                # yield self.await_port_input(self.node.ports["qin_left"])
                yield self.await_signal(sender=self.subprotocols["left_link_layer"],
                                        signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}")
                tree_id, _ = self.subprotocols["left_link_layer"].get_signal_result(
                    f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}")
                if tree_id == self._tree_id and self.node.entangling[self._tree_id][0]:
                    self.node.entangling[self._tree_id] = (False, False)  # reset to NOT ACCEPTING qubits
                    self.node.qmemory.pop(left_qubit_pos)
                    self.node.qmemory.mem_positions[left_qubit_pos].in_use = True
                    self.node.qmemory.put(self.node.qmemory.pop(left_qubit_link_pos), left_qubit_pos)
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted left qubit.")
        else:
            # intermediate node
            if self.subprotocols['left_link_layer'] is self.subprotocols['right_link_layer']:
                # cases where a node is the left node for left & right edges. Subprotocols are the same in this case
                while True:
                    yield self.await_signal(sender=self.subprotocols["left_link_layer"],
                                            signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}")
                    tree_id, other_node_id = self.subprotocols["left_link_layer"].get_signal_result(
                        f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}")
                    if tree_id == self._tree_id:
                        if self.node.neighbors[tree_id][0] == other_node_id and self.node.entangling[tree_id][0]:
                            # signal is for left qubit
                            self.node.entangling[tree_id] = (False, self.node.entangling[tree_id][1])
                            self.node.qmemory.pop(left_qubit_pos)
                            self.node.qmemory.mem_positions[left_qubit_pos].in_use = True
                            self.node.qmemory.put(self.node.qmemory.pop(left_qubit_link_pos), left_qubit_pos)
                            logging.debug(
                                f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted left "
                                f"qubit")
                        elif self.node.neighbors[tree_id][1] == other_node_id and self.node.entangling[tree_id][1]:
                            # signal is for right qubit
                            self.node.entangling[self._tree_id] = (self.node.entangling[self._tree_id][0], False)
                            self.node.qmemory.pop(right_qubit_pos)
                            self.node.qmemory.mem_positions[right_qubit_pos].in_use = True
                            self.node.qmemory.put(self.node.qmemory.pop(right_qubit_link_pos), right_qubit_pos)
                            logging.debug(
                                f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted right "
                                f"qubit.")
            else:
                while True:
                    # expr = yield self.await_port_input(self.node.ports["qin_left"]) | \
                    #              self.await_port_input(self.node.ports["qin_right"])
                    expr = yield self.await_signal(sender=self.subprotocols["left_link_layer"],
                                                   signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_"
                                                                f"{self._tree_id}") | \
                                 self.await_signal(sender=self.subprotocols["right_link_layer"],
                                                   signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_"
                                                                f"{self._tree_id}")

                    if expr.first_term.value and self.node.entangling[self._tree_id][0]:
                        # left qubit ready
                        tree_id, _ = self.subprotocols["left_link_layer"].get_signal_result(
                            f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}")
                        if tree_id == self._tree_id and self.node.entangling[self._tree_id][0]:
                            self.node.entangling[self._tree_id] = (False, self.node.entangling[self._tree_id][1])
                            self.node.qmemory.pop(left_qubit_pos)
                            self.node.qmemory.mem_positions[left_qubit_pos].in_use = True
                            self.node.qmemory.put(self.node.qmemory.pop(left_qubit_link_pos), left_qubit_pos)
                            logging.debug(
                                f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted left "
                                f"qubit.")
                    # else:
                    elif expr.second_term.value:
                        # right qubit ready
                        tree_id, _ = self.subprotocols["right_link_layer"].get_signal_result(
                            f"{MessageType.ENTANGLEMENT_READY.name}_{self._tree_id}")
                        if tree_id == self._tree_id and self.node.entangling[self._tree_id][1]:
                            self.node.entangling[self._tree_id] = (self.node.entangling[self._tree_id][0], False)
                            self.node.qmemory.pop(right_qubit_pos)
                            self.node.qmemory.mem_positions[right_qubit_pos].in_use = True
                            self.node.qmemory.put(self.node.qmemory.pop(right_qubit_link_pos), right_qubit_pos)
                            logging.debug(
                                f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted right "
                                f"qubit.")


class CorrectionProtocol(NodeProtocol):
    """A protocol class that does the correction when BSM results is ready"""

    def __init__(self, node: qPNode, messaging_protocol: MessagingProtocol, tree_id: int, main_root_id: int):
        super().__init__(node, node.name + "_correction_protocol_tree" + str(tree_id))
        self._tree_id = tree_id
        self.main_root_id = main_root_id
        self.add_subprotocol(messaging_protocol, "messaging_protocol")

    def run(self):
        entanglement_ready = False
        message = None
        entangle_qubit_pos = self.node.qubit_mapping[self._tree_id][0][0]  # qubit slot where correction happens on

        while True:
            evexp_entangle = self.await_mempos_in_use_toggle(self.node.qmemory, [entangle_qubit_pos])
            evexpr_meas_result = self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                   signal_label=f"{MessageType.SWAPPING_CORRECTION.name}_"
                                                                f"{self._tree_id}")
            expression = yield evexpr_meas_result | evexp_entangle
            if expression.first_term.value:
                # measurement is ready from parent
                tmp_message = self.subprotocols["messaging_protocol"].get_signal_result(
                    f"{MessageType.SWAPPING_CORRECTION.name}_{self._tree_id}")
                if tmp_message[1].tree_id == self._tree_id:
                    message = tmp_message
            else:
                # entangled qubit is ready
                entanglement_ready = True
            if message is not None and entanglement_ready:
                try:
                    if message[1].data[0]:
                        self.node.qmemory.operate(ns.Z, entangle_qubit_pos)
                    if message[1].data[1]:
                        self.node.qmemory.operate(ns.X, entangle_qubit_pos)
                    self.node.ports["cto_node" + str(message[1].next_dst_id)].tx_output(
                        ProtocolMessage(MessageType.CORRECTION_ACK, message[1].tree_id, None, -1, ns.sim_time()))
                    logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) {self.node.name} (CORRECTION) received"
                                 f" entangled qubit and corrections and sent ACK it to node{message[1].next_dst_id}!")

                    # reset for next round; send success signal to swapping protocol letting know left sub-tree is ready
                    if int(message[0]) == self.main_root_id:
                        self.send_signal(signal_label=Signals.SUCCESS, result=message[1].next_dst_id)
                        entanglement_ready = False
                except Exception as e:
                    logging.error(f"{ns.sim_time():.1f}: CORRECTION failed at node {self.node.name}. "
                                  f"Start accepting EPs from link layer for tree{self._tree_id} rooted at "
                                  f"(node{message[0]})\tError msg:{e}")
                    # start accepting for the left qubit pos
                    self.node.entangling[self._tree_id] = (True, self.node.entangling[self._tree_id][1])
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) set ENTANGLING to TRUE"
                                  f" for LEFT position of node {str(self.node.ID)} because of correction "
                                  f"failure.")
                    # send zero-latency msg to the left neighbor to start entangling
                    self.node.ports["cto_neighbor" + str(self.node.neighbors[self._tree_id][0])].tx_output(
                        ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, None, -1, message.gen_time))
                    # send a msg to root to restart the whole tree rooted at that root
                    self.node.ports["cto_node" + str(message[0])].tx_output(ProtocolMessage(
                        MessageType.CORRECTION_FAILURE, self._tree_id, None, -1, message.gen_time))
                message = None


class SwappingProtocol(NodeProtocol):
    """a protocol class that handles BSM operations and send results to rightest node"""

    def __init__(self, node: qPNode, tree_id: int, correction_protocol: CorrectionProtocol = None,
                 messaging_protocol: MessagingProtocol = None, swapping_time_window: float = float('inf')):
        super().__init__(node, name=node.name + "_swapping_protocol_tree" + str(tree_id))
        self._tree_id = tree_id
        if correction_protocol:
            self.add_subprotocol(correction_protocol, "correction_left")
        if messaging_protocol:
            self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._swapping_time_window = swapping_time_window

    def run(self):
        qubit_pos_left_ready = False  # left swapping qubit
        qubit_pos_right_ready = False  # right swapping qubit
        left_last_time, right_last_time = -float('inf'), -float('inf')

        while True:
            if self.node.left_height[self._tree_id] == 1:
                # qubit directly comes from charlie
                evexpr_left = self.await_mempos_in_use_toggle(self.node.qmemory,
                                                              [self.node.qubit_mapping[self._tree_id][0][0]])
            else:
                # qubit comes from lower level of the tree, should be after CORRECTION of the same node
                evexpr_left = self.await_signal(sender=self.subprotocols["correction_left"],
                                                signal_label=Signals.SUCCESS)

            if self.node.right_height[self._tree_id] == 1:
                evexpr_right = self.await_mempos_in_use_toggle(self.node.qmemory,
                                                               [self.node.qubit_mapping[self._tree_id][0][1]])
            else:
                # should wait for the rightest CORRECTION_ACK to receive
                evexpr_right = self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                 signal_label=f"{MessageType.CORRECTION_ACK.name}_{self._tree_id}")
            expression = yield evexpr_left | evexpr_right
            if expression.first_term.value:
                # first qubit is ready
                qubit_pos_left_ready = True
                left_last_time = ns.sim_time()
            else:
                #
                if self.node.right_height[self._tree_id] == 1:
                    qubit_pos_right_ready = True
                    right_last_time = ns.sim_time()
                else:
                    sender_id, tree_id = self.subprotocols["messaging_protocol"].get_signal_result(
                        f"{MessageType.CORRECTION_ACK.name}_{self._tree_id}")
                    if tree_id == self._tree_id and sender_id == self.node.rightest_id[tree_id]:
                        qubit_pos_right_ready = True
                        right_last_time = ns.sim_time()

            if qubit_pos_left_ready and qubit_pos_right_ready:
                if abs(right_last_time - left_last_time) < self._swapping_time_window:
                    # acceptable to do BSM, used for delft lp and sigcomm
                    yield self.await_timer(self.node.bsm_time * 1e9)
                    try:
                        m = bell_measurement(self.node.qmemory, [self.node.qubit_mapping[self._tree_id][0][0],
                                                                 self.node.qubit_mapping[self._tree_id][0][1]], False)

                        # if self.node.ID != 8:
                        if random() < self.node.bsm_success_rate:
                            # successful bsm
                            self.node.ports["cto_node" + str(self.node.rightest_id[self._tree_id])].tx_output(
                                ProtocolMessage(MessageType.SWAPPING_CORRECTION, self._tree_id, m,
                                                self.node.leftest_id[self._tree_id], ns.sim_time()))
                            logging.info(f"{ns.sim_time():.1f}:\t (TREE{self._tree_id}) {self.node.name}"
                                         " (SWAPPING) received entangled qubit, "
                                         f"measured qubits & sending corrections to node"
                                         + str(self.node.rightest_id[self._tree_id]))
                        else:
                            # unsuccessful bsm, should tell all the children to resume entangling again
                            tools.tree_failure(self.node, self._tree_id)
                    except Exception as e:
                        logging.error(f"{ns.sim_time():.1f}: (SWAPPING) Error happened at {self.node.name}.\n"
                                      f"Error message: {e}")
                        tools.tree_failure(self.node, self._tree_id)
                else:
                    # unacceptable timing
                    logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) swapping was UNSUCCESSFUL"
                                 f" at node{self.node.ID} because left & right did not arrive at the same time."
                                 f" Time difference: {abs(right_last_time - left_last_time)} ns.")
                    tools.tree_failure(self.node, self._tree_id)

                # reset for the next round
                qubit_pos_left_ready = False
                qubit_pos_right_ready = False
                left_last_time, left_last_time = -float('inf'), -float('inf')
                # break


class CorrectionAckProtocol(NodeProtocol):
    """A protocol class that change qubit position when a correction is finished
    THIS CLASS is OBSOLETE"""

    def __init__(self, node: qPNode, parent_id: int, rightest_id: int):
        super().__init__(node, node.name + "_correction_ack_protocol")
        self._parent_id, self._rightest_id = parent_id, rightest_id

    def run(self):
        port_correction_ack = self.node.ports["cfrom_node" + str(self._rightest_id)]
        while True:
            yield self.await_port_input(port_correction_ack)
            qubit_pos_from, qubit_pos_to = self.node.qubit_mapping[self._parent_id]
            self.node.qmemory.pop(qubit_pos_to)
            self.node.qmemory.mem_positions[qubit_pos_to].in_use = True
            self.node.qmemory.put(self.node.qmemory.pop(qubit_pos_from), qubit_pos_to)
            print(f"{self.node.name}" f" received CORRECTION_ACK from node{str(self._rightest_id)}"
                  f" and changed the qubit position.")
            break


class QubitGenerationProtocol(NodeProtocol):
    """A protocol that src is using to generate a qubit to later send to dst"""

    def run(self):
        qubit, = ns.qubits.create_qubits(1)
        mem_pos = self.node.avail_mem_pos()
        self.node.qmemory.put(qubit, mem_pos)
        self.node.qmemory.operate(ns.H, mem_pos)
        self.node.qmemory.operate(ns.S, mem_pos)
        self.send_signal(signal_label=Signals.SUCCESS, result=mem_pos)


class TeleportationProtocol(NodeProtocol):
    """Teleportation protocol that src uses when all the SwappingProtocol of the tree succeeds"""

    def __init__(self, node: qPNode, qubit_protocol: QubitGenerationProtocol, dst_id, is_path_one_hop, tree_id,
                 messaging_protocol: MessagingProtocol, tree_nodes_id: set, IS_TELEPORTING: bool):
        super().__init__(node, name=node.name + "_teleportation_protocol_tree" + str(tree_id))
        # self.add_subprotocol(qubit_protocol, "qprotocol")
        if not is_path_one_hop:
            self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._dst_id = dst_id
        self._is_path_one_hop = is_path_one_hop
        self._tree_id = tree_id
        self._teleporting_qubit_pos = node.avail_mem_pos()
        self._tree_nodes_id = tree_nodes_id
        self._IS_TELEPORTING = IS_TELEPORTING
        self._tree_success_count = 0

    def create_qubit(self):
        qubit, = ns.qubits.create_qubits(1)
        self.node.qmemory.put(qubit, self._teleporting_qubit_pos)
        self.node.qmemory.operate(ns.H, self._teleporting_qubit_pos)
        self.node.qmemory.operate(ns.S, self._teleporting_qubit_pos)

    def run(self):
        entanglement_ready = False  # when the tree construction is ready
        while True:
            if self._is_path_one_hop:
                evexpr_entanglement = self.await_mempos_in_use_toggle(self.node.qmemory,
                                                                      [self.node.qubit_mapping[self._tree_id][0][1]])
            else:
                evexpr_entanglement = self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                        signal_label=f"{MessageType.CORRECTION_ACK.name}_"
                                                                     f"{self._tree_id}")
            yield evexpr_entanglement
            if self._is_path_one_hop:
                entanglement_ready = True
            else:
                sender_id, tree_id = self.subprotocols["messaging_protocol"].get_signal_result(
                    f"{MessageType.CORRECTION_ACK.name}_{self._tree_id}")
                if tree_id == self._tree_id and sender_id == self._dst_id:
                    entanglement_ready = True
            if entanglement_ready:
                if self._IS_TELEPORTING:
                    # if we send qubit information to BOB
                    self.create_qubit()
                    yield self.await_timer(self.node.bsm_time * 1e9)
                    try:
                        m = bell_measurement(self.node.qmemory, [self._teleporting_qubit_pos,
                                                                 self.node.qubit_mapping[self._tree_id][0][1]], False)
                        if random() < self.node.bsm_success_rate:
                            # successful
                            self.node.ports["cto_node" + str(self._dst_id)].tx_output(
                                ProtocolMessage(MessageType.TELEPORTATION_CORRECTION, self._tree_id, m, -1,
                                                ns.sim_time()))
                            logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Alice (node {str(self.node.ID)})"
                                         f" received entangled qubit, entanglement ready. "
                                         f"measured qubits & sending corrections to Bob(node{str(self._dst_id)})")
                        else:
                            # unsuccessful bsm, should let the whole tree to start entangling
                            logging.info(
                                f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Unsuccessful TELEPORTATION at alice"
                                f" (node{str(self.node.ID)}). Tell all the nodes to start entangling again.")
                            self.restart_tree()
                    except Exception as e:
                        logging.error(f"{ns.sim_time():.1f}: (TELEPORTATION) Error happened at Alice "
                                      f"({self.node.name}).\tError message: {e}")
                        self.restart_tree()
                else:
                    # we stop here, when the tree succeeds; we don't send qubit information to Bob's side
                    logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Alice (node {str(self.node.ID)})"
                                 f" received entangled qubit, entanglement ready. ")
                    self.send_signal(signal_label=Signals.SUCCESS)
                    self._tree_success_count += 1
                    # self.await_timer(1)
                    self.restart_tree()
                entanglement_ready = False

    def restart_tree(self):
        self.node.entangling[self._tree_id] = (False, True)
        self.node.ports["cto_neighbor" + str(self.node.neighbors[self._tree_id][1])].tx_output(
            ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, None, -1, ns.sim_time()))
        if not self._is_path_one_hop:
            for node_id in self._tree_nodes_id:
                if node_id == self.node.ID:
                    # no need to send message to itself and its right neighbor
                    continue
                data = "both"
                if node_id == self.node.neighbors[self._tree_id][1]:
                    data = "right"
                if node_id == self._dst_id:
                    data = "left"
                self.node.ports["cto_node" + str(node_id)].tx_output(
                    ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, data, -1, ns.sim_time()))

    def start(self):
        super().start()

    @property
    def tree_success_count(self) -> int:
        return self._tree_success_count


class TeleportationCorrectionProtocol(NodeProtocol):
    """A protocol class that does the correction when BSM results is ready"""

    def __init__(self, node: qPNode, src_id: int, tree_id: int, is_path_one_hop,
                 messaging_protocol: MessagingProtocol,
                 correction_protocol: CorrectionProtocol,
                 tree_nodes_id: set):
        super().__init__(node, node.name + "_correction_protocol_for_source_node_" + str(src_id) + "_tree" +
                         str(tree_id))
        self._src_id = src_id
        self._tree_id = tree_id
        self._is_path_one_hop = is_path_one_hop
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._tree_nodes_id = tree_nodes_id
        self._success_count = 0
        if not is_path_one_hop:
            self.add_subprotocol(correction_protocol, "correction_protocol")

    @property
    def successful_count(self):
        return self._success_count

    def run(self):
        # port_meas = self.node.ports["cfrom_node" + str(self._src_id)]
        meas_results = None
        entanglement_ready = False

        while True:

            evexpr_meas_result = self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                   signal_label=f"{MessageType.TELEPORTATION_CORRECTION.name}_"
                                                                f"{self._tree_id}")
            if self._is_path_one_hop:
                evexpr_entanglement = self.await_mempos_in_use_toggle(self.node.qmemory,
                                                                      [self.node.qubit_mapping[self._tree_id][0][0]])
            else:
                evexpr_entanglement = self.await_signal(sender=self.subprotocols["correction_protocol"],
                                                        signal_label=Signals.SUCCESS)
            expression = yield evexpr_meas_result | evexpr_entanglement
            if expression.first_term.value:
                sender_id, tree_id, tmp_res = self.subprotocols["messaging_protocol"].get_signal_result(
                    f"{MessageType.TELEPORTATION_CORRECTION.name}_{self._tree_id}")
                if tree_id == self._tree_id and sender_id == self._src_id:
                    meas_results = tmp_res
            else:
                if self._is_path_one_hop:
                    entanglement_ready = True
                else:
                    dst_id = self.subprotocols["correction_protocol"].get_signal_result(Signals.SUCCESS)
                    if dst_id == self._src_id:
                        entanglement_ready = True

            if entanglement_ready and meas_results is not None:
                try:
                    if meas_results[0]:
                        self.node.qmemory.operate(ns.Z, self.node.qubit_mapping[self._tree_id][0][0])
                    if meas_results[1]:
                        self.node.qmemory.operate(ns.X, self.node.qubit_mapping[self._tree_id][0][0])
                    fidelity = ns.qubits.fidelity(
                        self.node.qmemory.peek(self.node.qubit_mapping[self._tree_id][0][0])[0],
                        ns.y0, squared=True)  # TODO we should decide how we handle fidelity
                    # reset for next round
                    logging.info(
                        f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Bob (node {self.node.ID}) received entangled "
                        f"qubit and corrections from node{self._src_id}! Fidelity = {fidelity:.3f}")
                    self._success_count += 1
                except Exception as e:
                    logging.error(f"{ns.sim_time():.1f}: CORRECTION failed at node {self.node.name}. "
                                  f"Start accepting EPs from link layer for tree{self._tree_id}.\tError msg: {e}")
                self.restart_tree()
                entanglement_ready = False
                meas_results = None

    def restart_tree(self):
        # tree completes, should tell all the nodes to start entangling for next round
        self.node.entangling[self._tree_id] = (True, False)
        self.node.ports["cto_neighbor" + str(self.node.neighbors[self._tree_id][0])].tx_output(
            ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, None, -1, ns.sim_time()))
        if not self._is_path_one_hop:
            for node_id in self._tree_nodes_id:
                if node_id == self.node.ID:
                    # no need to send message to itself
                    continue
                data = "both"
                if node_id == node_id == self.node.neighbors[self._tree_id][0]:
                    data = "left"
                if node_id == self._src_id:
                    data = "right"
                self.node.ports["cto_node" + str(node_id)].tx_output(
                    ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, data, -1, ns.sim_time()))
        # break


class FidelityCalculatorProtocol(Protocol):
    def __init__(self, tree_id: int, src: qPNode, dst: qPNode, is_path_one_hop: bool,
                 correction_protocol: CorrectionProtocol, teleportation_protocol: TeleportationProtocol):
        self._tree_id = tree_id
        self._src, self._dst = src, dst
        self._is_path_one_hop = is_path_one_hop
        if not is_path_one_hop:
            self.add_subprotocol(correction_protocol, "correction_protocol")
        else:
            self.add_subprotocol(teleportation_protocol, "teleportation_protocol")

        def calc_fidelity(evexpr):
            self.await_timer(1)
            qubit_a, = self._src.qmemory.peek(self._src.qubit_mapping[self._tree_id][0][1])
            qubit_b, = self._dst.qmemory.peek(self._dst.qubit_mapping[self._tree_id][0][0])
            try:
                fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ns.b00, squared=True)
                logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) ALice and Bob are entangled!. "
                             f"Fidelity: {fidelity}")
                return {"fidelity": fidelity}
            except Exception as e:
                logging.error(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Fidelity calculation failed!. "
                              f"Error msg: {e}")
                return

        self._dc = DataCollector(calc_fidelity, include_time_stamp=False, include_entity_name=False)

    def run(self):
        if not self._is_path_one_hop:
            self._dc.collect_on(self.await_signal(sender=self.subprotocols["correction_protocol"],
                                                  signal_label=Signals.SUCCESS))
        else:
            # self.await_timer(1)
            # self._dc.collect_on(self.await_mempos_in_use_toggle(self._dst.qmemory,
            #                                                       [self._dst.qubit_mapping[self._tree_id][0][0]]))
            self._dc.collect_on(self.await_signal(sender=self.subprotocols["teleportation_protocol"],
                                                  signal_label=Signals.SUCCESS))

    @property
    def average_fidelity(self):
        "return count and avg"
        try:
            return self._dc.dataframe.count()[0], self._dc.dataframe.agg("mean")[0]
        except IndexError as e:
            return 0, 0

#  ************************************ V2 Protocols ************************************


class MessagingProtocolV2(NodeProtocol):
    """a protocol that handles received messages from other nodes"""

    def __init__(self, node: qPNode):
        super().__init__(node, name=node.name + "_messaging_protocol")

    def add_signals(self, tree_id: int):
        self.add_signal(f"{MessageType.SWAPPING_CORRECTION.name}_{tree_id}")
        self.add_signal(f"{MessageType.TELEPORTATION_CORRECTION.name}_{tree_id}")
        self.add_signal(f"{MessageType.CORRECTION_ACK.name}_{tree_id}")

    def add_link_signals(self, tree_id: int, neighbor_id: int):
        # adding signals for one link
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{tree_id}_{neighbor_id}")
        self.add_signal(f"{MessageType.LINK_CORRECTION.name}_{tree_id}_{neighbor_id}")
        self.add_signal(f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_{tree_id}_{neighbor_id}")
        self.add_signal(f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_{tree_id}_{neighbor_id}")

    def add_links_signals(self, tree_id: int):
        # adding signals for all the edges in a tree
        for neighbor in self.node.neighbors[tree_id]:
            self.add_link_signals(tree_id, neighbor_id=neighbor)

    def add_ghz_signals(self, tree_id: int, end_nodes_key: str):
        self.add_signal(f"{MessageType.FUSION_CORRECTION.name}_{tree_id}_{end_nodes_key}")

    def add_ghz_fusion_correction_ack_signals(self, tree_id: int, end_nodes_key: str):
        # used for starting a fusion. Node waits for its sub-trees to finish, fusion ack from all the nodes in GHZ
        end_nodes = end_nodes_key.split('-')
        for node in end_nodes:
            self.add_signal(f"{MessageType.FUSION_CORRECTION_ACK.name}_{tree_id}_{end_nodes_key}_{node}")

    def add_correction_ack_signals(self, tree_id: int, rightest_node_id: int):
        """This signal is used to trigger a swapping where """
        if not f"{MessageType.CORRECTION_ACK.name}_{tree_id}_{rightest_node_id}" in self.signals:
            self.add_signal(f"{MessageType.CORRECTION_ACK.name}_{tree_id}_{rightest_node_id}")

    def run(self):
        while True:
            expr = yield reduce(operator.or_, [self.await_port_input(self.node.ports[port_name])
                                               for port_name in self.node.ports if port_name.startswith("cfrom_")])
            for triggered_event in expr.triggered_events:
                received_port_name = triggered_event.source.name
                messages = self.node.ports[received_port_name].rx_input()
                for message in messages.items:
                    if message.type == MessageType.SWAPPING_CORRECTION:
                        # send signal and BSM results and leftest id (next_dst_id) to correction protocol
                        self.send_signal(signal_label=f"{MessageType.SWAPPING_CORRECTION.name}_{message.tree_id}",
                                         result=(received_port_name.split("cfrom_node")[1], message))
                        # (received_port_name.split("cfrom_node")[1]
                    elif message.type == MessageType.CORRECTION_ACK:
                        self.send_signal(signal_label=f"{MessageType.CORRECTION_ACK.name}_{message.tree_id}_"
                                                      f"{received_port_name.split('cfrom_node')[1]}")
                    elif message.type == MessageType.TELEPORTATION_CORRECTION:
                        self.send_signal(signal_label=f"{MessageType.TELEPORTATION_CORRECTION.name}_"
                                                      f"{message.tree_id}",
                                         result=[int(received_port_name.split("cfrom_node")[1]),
                                                 message.tree_id, message.data])
                    elif message.type == MessageType.ENTANGLEMENT_READY:
                        self.send_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{message.tree_id}_"
                                         f"{received_port_name.split('cfrom_neighbor')[1]}")
                    elif message.type == MessageType.LINK_CORRECTION:
                        self.send_signal(f"{MessageType.LINK_CORRECTION.name}_{message.tree_id}_"
                                         f"{message.next_dst_id}", message.data)
                    elif message.type == MessageType.CORRECTION_FAILURE:
                        tools.tree_failure_v2(self.node, message.tree_id, self.node.children_id[message.tree_id],
                                              self.node.leftest_id[message.tree_id],
                                              self.node.rightest_id[message.tree_id])
                    elif message.type == MessageType.START_ENTANGLING:
                        if len(message.data.split("-")) > 1:
                            # designed for GHZ, data is end_nodes key
                            for neighbor in self.node.ghz_edges[f"{message.tree_id}_{message.data}"]:
                                if not self.node.entangling[f"{message.tree_id}_{neighbor}"]:
                                    self.node.entangling[f"{message.tree_id}_{neighbor}"] = True
                                    self.node.ports[f"cto_neighbor{neighbor}"].tx_output(ProtocolMessage(
                                        message.type, message.tree_id, message.data, -1, message.gen_time))
                        else:
                            # simple swapping
                            def start_entangling(is_left: bool):
                                if message.tree_id in self.node.last_not_accepting_ep_signal and\
                                        message.gen_time < self.node.last_not_accepting_ep_signal[message.tree_id]:
                                    return
                                self.node.last_accepting_ep_signal[message.tree_id] = message.gen_time
                                if is_left and not self.node.entangling[
                                    f"{message.tree_id}_{self.node.neighbors[message.tree_id][0]}"]:
                                    self.node.entangling[f"{message.tree_id}_" \
                                                         f"{self.node.neighbors[message.tree_id][0]}"] = True
                                    self.node.ports[f"cto_neighbor{self.node.neighbors[message.tree_id][0]}"].tx_output(
                                        ProtocolMessage(message.type, message.tree_id, 'right', -1, message.gen_time))
                                    logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} has "
                                                 f"resumed accepting EP for edge to "
                                                 f"node{self.node.neighbors[message.tree_id][0]}.")
                                elif not is_left and not self.node.entangling[
                                    f"{message.tree_id}_{self.node.neighbors[message.tree_id][1]}"]:
                                    self.node.entangling[f"{message.tree_id}_" \
                                                         f"{self.node.neighbors[message.tree_id][1]}"] = True
                                    self.node.ports[f"cto_neighbor{self.node.neighbors[message.tree_id][1]}"].tx_output(
                                        ProtocolMessage(message.type, message.tree_id, 'left', -1, message.gen_time))
                                    logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} has "
                                                 f"resumed accepting EP for edge to "
                                                 f"node{self.node.neighbors[message.tree_id][1]}.")

                            if message.data == 'left':
                                start_entangling(True)
                            elif message.data == 'right':
                                start_entangling(False)
                            elif message.data == 'both':
                                start_entangling(True)
                                start_entangling(False)
                    elif message.type == MessageType.STOP_ENTANGLING:
                        def stop_entangling(is_left: bool):
                            if message.tree_id in self.node.last_accepting_ep_signal and \
                                    message.gen_time < self.node.last_accepting_ep_signal[message.tree_id]:
                                return
                            self.node.last_not_accepting_ep_signal[message.tree_id] = message.gen_time
                            if is_left and \
                                    self.node.entangling[f"{message.tree_id}_{self.node.neighbors[message.tree_id][0]}"]:
                                self.node.entangling[
                                    f"{message.tree_id}_{self.node.neighbors[message.tree_id][0]}"] = False
                                self.node.ports[f"cto_neighbor{self.node.neighbors[message.tree_id][0]}"].tx_output(
                                    ProtocolMessage(message.type, message.tree_id, 'right', -1, message.gen_time))
                                logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} has "
                                             f"stopped accepting EP for edge to "
                                             f"node{self.node.neighbors[message.tree_id][0]}.")
                            elif not is_left and \
                                    self.node.entangling[f"{message.tree_id}_{self.node.neighbors[message.tree_id][1]}"]:
                                self.node.entangling[
                                    f"{message.tree_id}_{self.node.neighbors[message.tree_id][1]}"] = False
                                self.node.ports[f"cto_neighbor{self.node.neighbors[message.tree_id][1]}"].tx_output(
                                    ProtocolMessage(message.type, message.tree_id, 'left', -1, message.gen_time))
                                logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} has "
                                             f"stopped accepting EP for edge to "
                                             f"node{self.node.neighbors[message.tree_id][1]}.")
                        if message.data == 'left':
                            stop_entangling(True)
                        elif message.data == 'right':
                            stop_entangling(False)
                        elif message.data == 'both':
                            stop_entangling(True)
                            stop_entangling(False)
                    elif message.type == MessageType.FUSION_CORRECTION:
                        self.send_signal(f"{MessageType.FUSION_CORRECTION.name}_{message.tree_id}"
                                         f"_{message.data[0]}", [received_port_name.split("cfrom_node")[1],
                                                                 message.data[1]])
                    elif message.type == MessageType.START_ENTANGLEMENT_GENERATING:
                        def start_ep_generation(is_left: bool):
                            if message.tree_id in self.node.last_stopping_signal and \
                                    message.gen_time < self.node.last_stopping_signal[message.tree_id]:
                                return
                            self.node.last_starting_signal[message.tree_id] = message.gen_time
                            if is_left and \
                                    not self.node.running[f"{message.tree_id}_{self.node.neighbors[message.tree_id][0]}"]:
                                self.send_signal(f"{message.type.name}_{message.tree_id}_"
                                                 f"{self.node.neighbors[message.tree_id][0]}")
                                self.node.ports[f"cto_neighbor{self.node.neighbors[message.tree_id][0]}"].tx_output(
                                    ProtocolMessage(message.type, message.tree_id, 'right', -1, message.gen_time))
                                logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} has "
                                             f"resumed generating EP for edge to "
                                             f"node{self.node.neighbors[message.tree_id][0]}.")
                            elif not is_left and \
                                    not self.node.running[f"{message.tree_id}_{self.node.neighbors[message.tree_id][1]}"]:
                                self.send_signal(f"{message.type.name}_{message.tree_id}_"
                                                 f"{self.node.neighbors[message.tree_id][1]}")
                                self.node.ports[f"cto_neighbor{self.node.neighbors[message.tree_id][1]}"].tx_output(
                                    ProtocolMessage(message.type, message.tree_id, 'left', -1, message.gen_time))
                                logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} has "
                                             f"resumed generating EP for edge to "
                                             f"node{self.node.neighbors[message.tree_id][1]}.")
                        if len(message.data.split("-")) > 1:
                            # designed for GHZ, data is end_nodes key
                            for neighbor in self.node.ghz_edges[f"{message.tree_id}_{message.data}"]:
                                if not self.node.running[f"{message.tree_id}_{neighbor}"]:
                                    self.send_signal(f"{message.type.name}_{message.tree_id}_{neighbor}")
                                    self.node.ports[f"cto_neighbor{neighbor}"].tx_output(ProtocolMessage(
                                        message.type, message.tree_id, message.data, -1, message.gen_time))
                        elif message.data == 'left':
                            start_ep_generation(True)
                        elif message.data == 'right':
                            start_ep_generation(False)
                        elif message.data == 'both':
                            start_ep_generation(True)
                            start_ep_generation(False)
                    elif message.type == MessageType.STOP_ENTANGLEMENT_GENERATING:
                        def stop_ep_generation(is_left: bool):
                            if message.tree_id in self.node.last_starting_signal and \
                                    message.gen_time < self.node.last_starting_signal[message.tree_id]:
                                return
                            self.node.last_stopping_signal[message.tree_id] = message.gen_time
                            if is_left and \
                                    self.node.running[f"{message.tree_id}_{self.node.neighbors[message.tree_id][0]}"]:
                                self.send_signal(f"{message.type.name}_{message.tree_id}_"
                                                 f"{self.node.neighbors[message.tree_id][0]}")
                                self.node.ports[f"cto_neighbor{self.node.neighbors[message.tree_id][0]}"].tx_output(
                                    ProtocolMessage(message.type, message.tree_id, 'right', -1, message.gen_time))
                                logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} has "
                                             f"stopped generating EP for edge to "
                                             f"node{self.node.neighbors[message.tree_id][0]}.")
                            elif not is_left and \
                                    self.node.running[f"{message.tree_id}_{self.node.neighbors[message.tree_id][1]}"]:
                                self.send_signal(f"{message.type.name}_{message.tree_id}_"
                                                 f"{self.node.neighbors[message.tree_id][1]}")
                                self.node.ports[f"cto_neighbor{self.node.neighbors[message.tree_id][1]}"].tx_output(
                                    ProtocolMessage(message.type, message.tree_id, 'left', -1, message.gen_time))
                                logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} has "
                                             f"stopped generating EP for edge to "
                                             f"node{self.node.neighbors[message.tree_id][1]}.")
                        if message.data is None and \
                                self.node.running[f"{message.tree_id}_{received_port_name.split('cfrom_neighbor')[1]}"]:
                            # initiated by link-level
                            self.send_signal(f"{message.type.name}_{message.tree_id}_"
                                             f"{received_port_name.split('cfrom_neighbor')[1]}")
                            self.node.ports[f"cto_neighbor{received_port_name.split('cfrom_neighbor')[1]}"].tx_output(
                                ProtocolMessage(message.type, message.tree_id, None, -1, message.gen_time)
                            )
                        elif message.data == 'left':
                            stop_ep_generation(True)
                        elif message.data == 'right':
                            stop_ep_generation(False)
                        elif message.data == 'both':
                            stop_ep_generation(True)
                            stop_ep_generation(False)
                    elif message.type == MessageType.FUSION_CORRECTION_FAILURE:
                        if isinstance(self.node.all_nodes[f"{message.tree_id}_{message.data}"], set):
                            # general fusion
                            logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} "
                                         f"FUSION_CORRECTION_FAILED at node "
                                         f"{received_port_name.split('cfrom_node')[1]} for"
                                         f" GHZ {message.data}. RESTARTING the tree.")
                            tools.fusion_tree_failure(self.node, message.tree_id,
                                                      self.node.all_nodes[f"{message.tree_id}_{message.data}"],
                                                      end_nodes_key=message.data)
                        else:
                            # fro
                            for tree_id, all_tree_nodes, src_id, dst_id in \
                                    self.node.all_nodes[f"{message.tree_id}_{message.data}"]:
                                tools.tree_failure_v2(self.node, tree_id, all_tree_nodes, src_id, dst_id)
                    elif message.type == MessageType.FUSION_CORRECTION_ACK:
                        logging.info(f"{ns.sim_time():.1f}: (TREE{message.tree_id}) Node{self.node.ID} "
                                     f"receives FUSION_CORRECTION_ACK for GHZ-{message.data} from "
                                     f"node{received_port_name.split('cfrom_node')[1]}. Send a signal to "
                                     f"FUSION Protocol.")
                        self.send_signal(f"{message.type.name}_{message.tree_id}_{message.data}_"
                                         f"{received_port_name.split('cfrom_node')[1]}")
                    elif message.type == MessageType.SRC_DST_RESET:
                        if True:  # self.node.src_dst_ep_active[f"#{message.next_dst_id}"]:
                            self.node.src_dst_ep_active[f"#{message.next_dst_id}"] = False
                            self.node.acceptable_time[f"#{message.next_dst_id}"] = message.data
                            self.node.ports[f"cto_node{message.next_dst_id}"].tx_output(
                                ProtocolMessage(MessageType.SRC_DST_RESET, message.tree_id, message.data, self.node.ID,
                                                message.gen_time))
                    elif message.type == MessageType.RESET_SWAPPING:
                        self.node.acceptable_time[message.tree_id] = message.data


class CharlieManager(NodeProtocol):
    """A new class that handles multiple EntanglementGenerator for an edge. Since the photonics BSM, done by
    CharlieProtocolV2, involves time advances and a qubit can only be taken once from a port, having this class helps
    to handle all the link layer."""

    def __init__(self, node: Charlie):
        super(CharlieManager, self).__init__(node, name=f"{node.name}_charlie_manager")

    def add_signals(self, tree_id: int):
        self.add_signal(f"{MessageType.PHOTON_ARRIVED.name}_{tree_id}_left")  # for signaling left qubit arrival
        self.add_signal(f"{MessageType.PHOTON_ARRIVED.name}_{tree_id}_right")  # for signaling right qubit arrival

    def run(self):
        qleft_port = self.node.ports["qin_left"]
        qright_port = self.node.ports["qin_right"]
        while True:
            expr = yield self.await_port_input(qleft_port) | \
                         self.await_port_input(qright_port)
            if expr.first_term.value:
                # left qubit comes
                for left_qubit in qleft_port.rx_input().items:
                    tree_id = int(left_qubit.name.split("_")[0])
                    self.send_signal(signal_label=f"{MessageType.PHOTON_ARRIVED.name}_{tree_id}_left",
                                     result=left_qubit)
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{tree_id}) LEFT qubit was arrived at {self.name}")
            else:
                # right qubit comes
                for right_qubit in qright_port.rx_input().items:
                    tree_id = int(right_qubit.name.split("_")[0])
                    self.send_signal(signal_label=f"{MessageType.PHOTON_ARRIVED.name}_{tree_id}_right",
                                     result=right_qubit)
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{tree_id}) RIGHT qubit was arrived at {self.name}")


class CharlieProtocolV2(NodeProtocol):
    """Charlie protocol that is responsible to do photonics BSM based on receiving from its left and right ports.
    In this version, Charlie protocols are separated s.t. there is one protcol per link per tree. Since, there is
    time advancement happening in this protocol, handling multiple trees lead to miss some successful photon arrival."""

    def __init__(self, node: Charlie, tree_id: int, charlie_manager_protocol: CharlieManager,
                 bsm_time_window: float = 0):
        super(CharlieProtocolV2, self).__init__(node, name=f"{node.name}_tree{tree_id}_charlie_protocol")
        self._bsm_time_window = bsm_time_window
        self._tree_id = tree_id
        self.add_subprotocol(charlie_manager_protocol, "charlie_manager_protocol")

    def run(self):
        cright_port = self.node.ports["cout_right"]
        left_qubit, right_qubit = None, None
        left_last_arrival, right_last_arrival = -float('inf'), -float('inf')

        while True:
            expr = yield self.await_signal(sender=self.subprotocols["charlie_manager_protocol"],
                                           signal_label=f"{MessageType.PHOTON_ARRIVED.name}_{self._tree_id}_left") | \
                         self.await_signal(sender=self.subprotocols["charlie_manager_protocol"],
                                           signal_label=f"{MessageType.PHOTON_ARRIVED.name}_{self._tree_id}_right")
            if expr.first_term.value:
                # left qubit comes
                left_qubit = self.subprotocols["charlie_manager_protocol"].get_signal_result(
                    f"{MessageType.PHOTON_ARRIVED.name}_{self._tree_id}_left")
                left_last_arrival = ns.sim_time()
            else:
                # right qubit comes
                right_qubit = self.subprotocols["charlie_manager_protocol"].get_signal_result(
                    f"{MessageType.PHOTON_ARRIVED.name}_{self._tree_id}_right")
                right_last_arrival = ns.sim_time()
            if left_qubit is not None and right_qubit is not None \
                    and abs(left_last_arrival - right_last_arrival) <= self._bsm_time_window:
                logging.debug(f"{ns.sim_time():.1f}: (TREE{str(self._tree_id)}) both link layer qubits "
                              f"successfully received  at {self.node.name}")
                yield self.await_timer(self.node.bsm_time * 1e9)

                qmemory = QuantumMemory("tmp", num_positions=2)
                qmemory.put([left_qubit, right_qubit], positions=[0, 1])
                try:
                    results = bell_measurement(qmemory, positions=[0, 1], PHYSICAL=True)
                    if random() < self.node.bsm_success_rate:
                        logging.info(f"{ns.sim_time():.1f}: (TREE{str(self._tree_id)}) BSM was successfully done at "
                                     f"{self.node.name} and result sent to {self.node.right_node_id}")

                        cright_port.tx_output(ProtocolMessage(type=MessageType.LINK_CORRECTION, data=results,
                                                              tree_id=self._tree_id,
                                                              next_dst_id=self.node.left_node_id,
                                                              gen_time=ns.sim_time()))
                except Exception as e:
                    logging.error(f"error:{e}. left_qubit_name:{left_qubit}, right_qubit_name: {right_qubit}")

                left_qubit, right_qubit = None, None
                left_last_arrival, right_last_arrival = -float('inf'), -float('inf')


class EntanglementGeneratorV2(NodeProtocol):
    """A protocol that generates a pair of EPs, save one in node's qmemory and send the other to charlie to
    do optical bsm. rate is 1/s. This class has an optimization where node's EP generation stops if there is
    successful photonic BSM at Charlie node. It only resume generating by a request from MessagingProtocol."""

    def __init__(self, node: qPNode, tree_id: int, rate: float('inf'), isLeft: bool, charlie_id: str,
                 link_layer_qmem_pos: int, other_node_id: int, messaging_protocol: MessagingProtocol):
        super(EntanglementGeneratorV2, self).__init__(node, name=f"{node.name}_{'left' if isLeft else 'right'}"
                                                                 f"_generator_for_tree{tree_id}_edge{other_node_id}")
        self._qout_port = self.node.ports[f"qout_{'left' if isLeft else 'right'}{charlie_id}"]
        self._isLeft = isLeft
        self._rate = rate
        self._tree_id = tree_id
        self._charlie_id = charlie_id
        self._link_layer_qmem_pos = link_layer_qmem_pos
        self._other_node_id = other_node_id
        self.node.running[f"{self._tree_id}_{self._other_node_id}"] = True
        self.add_subprotocol(messaging_protocol, "messaging_protocol")

    def run(self):

        while True:
            expr = yield self.await_timer(1.0 / self._rate * 1e9) | \
                         self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                           signal_label=f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_"
                                                        f"{self._tree_id}_{self._other_node_id}")
            if expr.first_term.value:
                self.send_qubit()
            else:
                self.node.running[f"{self._tree_id}_{self._other_node_id}"] = False
                logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Node{self.node.ID} "
                              f"(Charlie {self._charlie_id}) has STOPPED generating link"
                              f" EP for edge {self._other_node_id}.")
                yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                        signal_label=f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_"
                                                     f"{self._tree_id}_{self._other_node_id}")
                self.node.running[f"{self._tree_id}_{self._other_node_id}"] = True
                logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Node{self.node.ID} "
                              f"(Charlie {self._charlie_id}) has RESUMED generating link"
                              f" EP for edge {self._other_node_id}.")

    def send_qubit(self):
        q1, q2 = ns.qubits.create_qubits(2, no_state=True)
        q1.name += f"{self._tree_id}_{self.node.ID}_{self._link_layer_qmem_pos}_{'left' if self._isLeft else 'right'}"
        q2.name = f"{self._tree_id}_{self.node.ID}_{self._link_layer_qmem_pos}_{'left' if self._isLeft else 'right'}"
        ns.qubits.assign_qstate([q1, q2], ns.b00)
        self.node.qmemory.put(q1, positions=self._link_layer_qmem_pos, replace=True)
        self._qout_port.tx_output(q2)
        logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) "
                      f"qubit was generated at {self.name}")

    @property
    def running(self):
        return self._running


class LinkCorrectionProtocolV2(NodeProtocol):
    """a protocol that waits for response from Charlie to do the BSM correction and entangle two ends and send a
    message to left end"""

    def __init__(self, node: qPNode, messaging_protocol: MessagingProtocol):
        super(LinkCorrectionProtocolV2, self).__init__(node, name=node.name + "_link_correction")
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._list_of_received_signals = []

    def add_tree_edge_signal(self, tree_id: int, left_node_id: int):
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{tree_id}_{left_node_id}")
        self._list_of_received_signals.append(f"{MessageType.LINK_CORRECTION.name}_{tree_id}_{left_node_id}")

    def run(self):
        if len(self._list_of_received_signals) == 0:
            # this node has no link correction involved
            self.subprotocols.__delitem__("messaging_protocol")
            return

        while True:
            expr = yield reduce(operator.or_, [self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                                 signal_label=label)
                                               for label in self._list_of_received_signals])
            for triggered_event in expr.triggered_events:
                signal = triggered_event.source.get_signal_by_event(triggered_event)
                tree_id, left_node_id = signal.label.split("_")[-2:]
                link_qubit_key = f"{tree_id}_{left_node_id}"
                link_qubit_position = self.node.qubit_mapping[link_qubit_key][1]
                try:
                    if signal.result[0]:
                        self.node.qmemory.operate(ns.Z, link_qubit_position)
                    if signal.result[1]:
                        self.node.qmemory.operate(ns.X, link_qubit_position)
                    self.send_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{tree_id}_{left_node_id}")
                    self.node.ports[f"cto_neighbor{left_node_id}"].tx_output(
                        ProtocolMessage(MessageType.ENTANGLEMENT_READY, tree_id, None, -1, ns.sim_time()))
                    logging.info(f"{ns.sim_time():.1f}: (TREE{tree_id}) {self.node.name} received results from "
                                 f"charlie and did "
                                 f"the CORRECTION. send an ACK to neighbor {left_node_id}")
                except Exception as e:
                    logging.error(f"{ns.sim_time():.1f}: (TREE{tree_id}) Error happened at (link layer-waiting"
                                  f" for new one) node "
                                  f"{self.node.name}, edge (node{left_node_id}). \tError Msg: {e}")


class EntanglingProtocolV2(NodeProtocol):
    """a protocol that accepts new qubits if the node allows. i.e. copy from link layer memories to BSM ones
    The difference between this class and EntanglingProtocol is handling more than two incident edges. In
    EntanglingProtocol, where it is used in swapping trees, one node can have two edges attached in a tree.
    Another improvement is, we only need one such protocol per node instead of creating one for each tree."""

    def __init__(self, node: qPNode, messaging_protocol: MessagingProtocolV2,
                 link_correction_protocol: LinkCorrectionProtocolV2):
        super(EntanglingProtocolV2, self).__init__(node, name=node.name + "_entangling_protocol_tree")
        self.add_subprotocol(messaging_protocol, "messaging_protocol")  # messaging requires once to be added
        self.add_subprotocol(link_correction_protocol, "link_correction_protocol")
        self._protocols_list = []

    def add_protocol(self, tree_id: int, other_id: int, isLeft: bool):
        if not isLeft:
            self._protocols_list.append((tree_id, other_id, "link_correction_protocol"))
        else:
            self._protocols_list.append((tree_id, other_id, "messaging_protocol"))
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{tree_id}_{other_id}")
        # used to stop link EP generating after accepting one. messaging_protocol is responsible to start them again

    def run(self):
        while True:
            expr = yield reduce(operator.or_,
                                [self.await_signal(sender=self.subprotocols[protocol_name],
                                                   signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_"
                                                                f"{tree_id}_{other_id}")
                                 for tree_id, other_id, protocol_name in self._protocols_list])
            for triggered_event in expr.triggered_events:
                signal = triggered_event.source.get_signal_by_event(triggered_event)
                tree_id, other_id = signal.label.split("_")[-2:]
                network_qubit_pos, link_qubit_pos = self.node.qubit_mapping[f"{tree_id}_{other_id}"]  # where BSM ops happen
                if self.node.entangling[f"{tree_id}_{other_id}"]:
                    self.node.entangling[f"{tree_id}_{other_id}"] = False
                    self.node.qmemory.pop(network_qubit_pos)
                    self.node.qmemory.mem_positions[network_qubit_pos].in_use = True
                    self.node.qmemory.put(self.node.qmemory.pop(link_qubit_pos), network_qubit_pos)
                    self.node.qubit_creation_time[network_qubit_pos] = ns.sim_time()
                    # to avoid cyclic dependency, each node, on an edge, send a message to the other side to stop entanglement
                    # generating
                    self.node.ports[f"cto_neighbor{other_id}"].tx_output(
                        ProtocolMessage(MessageType.STOP_ENTANGLEMENT_GENERATING, tree_id, None, -1, ns.sim_time()))
                    logging.info(f"{ns.sim_time():.1f}: (TREE{tree_id}) node{self.node.ID} "
                                 f" accepted qubit for edge {other_id}.")
                    self.send_signal(signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_{tree_id}_{other_id}")


class CorrectionProtocolV2(NodeProtocol):
    """New protocol for doing BSM corrections. Only one protocol per node exists."""

    def __init__(self, node: qPNode, messaging_protocol: MessagingProtocolV2):
        super(CorrectionProtocolV2, self).__init__(node, node.name + "_bsm_correction_protocol")
        self._trees = set()  # list of trees this node may get signal from
        self._main_root_id = {}  # a dictionary for holding the root id that this node is the rightmost node for
        self.add_subprotocol(messaging_protocol, "messaging_protocol")

    def add_trees(self, tree_id: int):
        if f"{MessageType.CORRECTION_ACK.name}_{tree_id}" not in self.signals:
            self.add_signal(f"{MessageType.CORRECTION_ACK.name}_{tree_id}")
        self._trees.add(tree_id)

    def add_main_root_id(self, tree_id: int, main_root_id: int):
        self._main_root_id[tree_id] = main_root_id

    def run(self):
        if len(self._trees) == 0:
            # this node has no link correction involved
            self.subprotocols.__delitem__("messaging_protocol")
            return
        while True:
            expression = yield reduce(operator.or_, [
                self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                  signal_label=f"{MessageType.SWAPPING_CORRECTION.name}_{tree_id}")
                for tree_id in self._trees])
            for triggered_event in expression.triggered_events:
                signal = triggered_event.source.get_signal_by_event(triggered_event)
                tree_id = signal.result[1].tree_id
                qubit_pos = self.node.qubit_mapping[f"{tree_id}_{self.node.neighbors[tree_id][0]}"][0]
                try:
                    if signal.result[1].data[0]:
                        self.node.qmemory.operate(ns.Z, qubit_pos)
                    if signal.result[1].data[1]:
                        self.node.qmemory.operate(ns.X, qubit_pos)
                    logging.info(f"{ns.sim_time():.1f}: (TREE{tree_id}) {self.node.name} (CORRECTION) received"
                                 f" entangled qubit and corrections and sent ACK to "
                                 f"node{signal.result[1].next_dst_id}!")
                    self.node.ports["cto_node" + str(signal.result[1].next_dst_id)].tx_output(
                        ProtocolMessage(MessageType.CORRECTION_ACK, tree_id, None, -1, ns.sim_time()))

                    # send a signal to SwappingProtocol if this is the last corrections
                    if int(signal.result[0]) == self._main_root_id[tree_id]:
                        self.send_signal(signal_label=f"{MessageType.CORRECTION_ACK.name}_{tree_id}",
                                         result=signal.result[1].next_dst_id)
                        logging.info(f"{ns.sim_time():.1f}: (TREE{tree_id}) {self.node.name} (CORRECTION) sent a signal"
                                     f" as this was its last correction for a tree rooted at "
                                     f"node{self._main_root_id[tree_id]}.")
                except Exception as e:
                    logging.error(f"{ns.sim_time():.1f}: (TREE{tree_id}) CORRECTION failed at node {self.node.name}. "
                                  f"Start accepting EPs from link layer for tree{tree_id} rooted at "
                                  f"(node{signal.result[0]})\tError msg:{e}")
                    # send a msg to root to restart the whole tree rooted at that root
                    self.node.ports["cto_node" + str(signal.result[0])].tx_output(ProtocolMessage(
                        MessageType.CORRECTION_FAILURE, tree_id, None, -1, ns.sim_time()))


class SwappingProtocolV2(NodeProtocol):
    """a protocol class that handles BSM operations and send results to rightest node"""

    def __init__(self, node: qPNode, tree_id: int, correction_protocol: CorrectionProtocolV2,
                 messaging_protocol: MessagingProtocolV2,
                 swapping_time_window: float = float('inf')):
        super().__init__(node, name=node.name + "_swapping_protocol_tree" + str(tree_id))
        self._tree_id = tree_id
        self.add_subprotocol(correction_protocol, "correction_left")
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._swapping_time_window = swapping_time_window

    def run(self):
        left_qubit_ready, right_qubit_ready = False, False  # left swapping qubit and right swapping qubit
        left_qubit_pos = self.node.qubit_mapping[f"{self._tree_id}_{self.node.neighbors[self._tree_id][0]}"][0]
        right_qubit_pos = self.node.qubit_mapping[f"{self._tree_id}_{self.node.neighbors[self._tree_id][1]}"][0]
        left_last_time, right_last_time = -float('inf'), -float('inf')
        if self.node.left_height[self._tree_id] == 1:
            # qubit directly comes from charlie
            evexpr_left = self.await_mempos_in_use_toggle(self.node.qmemory, [left_qubit_pos])
        else:
            # qubit comes from lower level of the tree, should be after CORRECTION of the same node
            evexpr_left = self.await_signal(sender=self.subprotocols["correction_left"],
                                            signal_label=f"{MessageType.CORRECTION_ACK.name}_{self._tree_id}")

        if self.node.right_height[self._tree_id] == 1:
            evexpr_right = self.await_mempos_in_use_toggle(self.node.qmemory, [right_qubit_pos])
        else:
            # should wait for the rightest CORRECTION_ACK to receive
            evexpr_right = self.await_signal(
                sender=self.subprotocols["messaging_protocol"],
                signal_label=f"{MessageType.CORRECTION_ACK.name}_{self._tree_id}_{self.node.rightest_id[self._tree_id]}")

        while True:
            expression = yield evexpr_left | evexpr_right
            if expression.first_term.value:
                # first qubit is ready
                left_qubit_ready = True
                left_last_time = ns.sim_time()
            else:
                # second qubit is ready
                right_qubit_ready = True
                right_last_time = ns.sim_time()

            if left_qubit_ready and right_qubit_ready:
                if abs(right_last_time - left_last_time) < self._swapping_time_window and\
                        not (self._tree_id in self.node.acceptable_time and
                             (left_last_time < self.node.acceptable_time[self._tree_id] or
                              right_last_time < self.node.acceptable_time[self._tree_id])):
                    # acceptable to do BSM, used for delft lp and sigcomm
                    yield self.await_timer(self.node.bsm_time * 1e9)
                    try:
                        m = bell_measurement(self.node.qmemory, [left_qubit_pos, right_qubit_pos], False)

                        # if self.node.ID != 8:
                        if random() < self.node.bsm_success_rate:
                            # successful bsm
                            self.node.ports["cto_node" + str(self.node.rightest_id[self._tree_id])].tx_output(
                                ProtocolMessage(MessageType.SWAPPING_CORRECTION, self._tree_id, m,
                                                self.node.leftest_id[self._tree_id], ns.sim_time()))
                            logging.info(f"{ns.sim_time():.1f}:\t (TREE{self._tree_id}) {self.node.name}"
                                         " (SWAPPING) received entangled qubit, "
                                         f"measured qubits & sending corrections to node"
                                         + str(self.node.rightest_id[self._tree_id]))
                        else:
                            # unsuccessful bsm, should tell all the children to resume entangling again
                            logging.info(f"{ns.sim_time():.1f}:\t (TREE{self._tree_id}) {self.node.name}"
                                         f" (SWAPPING) was UNSUCCESSFUL.")
                            tools.tree_failure_v2(self.node, self._tree_id, self.node.children_id[self._tree_id],
                                                  self.node.leftest_id[self._tree_id],
                                                  self.node.rightest_id[self._tree_id])
                    except Exception as e:
                        logging.error(f"{ns.sim_time():.1f}: (SWAPPING) Error happened at {self.node.name}.\n"
                                      f"Error message: {e}")
                        tools.tree_failure_v2(self.node, self._tree_id, self.node.children_id[self._tree_id],
                                              self.node.leftest_id[self._tree_id], self.node.rightest_id[self._tree_id])
                else:
                    # unacceptable timing
                    logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) swapping was UNSUCCESSFUL"
                                 f" at node{self.node.ID} because left & right did not arrive at the proper time."
                                 f" Time difference: {abs(right_last_time - left_last_time)} ns.")
                    tools.tree_failure_v2(self.node, self._tree_id, self.node.children_id[self._tree_id],
                                          self.node.leftest_id[self._tree_id], self.node.rightest_id[self._tree_id])

                # reset for the next round
                left_qubit_ready, right_qubit_ready = False, False
                left_last_time, left_last_time = -float('inf'), -float('inf')


class SwappingTreeControllerProtocol(NodeProtocol):
    """A class that will be used to handle multiple trees for a given (s, d) pair and is responsible for
    stopping/starting all the trees."""

    def __init__(self, node: qPNode, is_source: bool, other_end_id: int,
                 messaging_protocol: MessagingProtocolV2,
                 entangling_protocol: EntanglingProtocolV2,
                 correction_protocol: CorrectionProtocolV2):
        super().__init__(node, name=node.name + f"_swapping_tree_protocols_{'source' if is_source else 'destination'}"
                                                f"_other_end_node{other_end_id}")
        self._is_source = is_source
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self.add_subprotocol(entangling_protocol, "entangling_protocol")
        self.add_subprotocol(correction_protocol, "correction_protocol")
        self._listening_signals = []
        self._tree_nodes = {}
        self._other_end_id = other_end_id

    def add_tree(self, tree_id: int, is_one_hop: bool, all_nodes: set[int]):
        self._tree_nodes[tree_id] = all_nodes
        if is_one_hop:
            self._listening_signals.append((f"{MessageType.ENTANGLEMENT_READY.name}_{tree_id}_{self._other_end_id}",
                                            "entangling_protocol"))
        else:
            if self._is_source:
                self._listening_signals.append((f"{MessageType.CORRECTION_ACK.name}_{tree_id}_{self._other_end_id}",
                                                "messaging_protocol"))
            else:
                self._listening_signals.append((f"{MessageType.CORRECTION_ACK.name}_{tree_id}", "correction_protocol"))

    def run(self):
        src_dst_key = f"#{self._other_end_id}"
        target_src_dst_mem_pos = self.node.qubit_mapping[src_dst_key]
        while True:
            expression = yield reduce(operator.or_, [self.await_signal(sender=self.subprotocols[sender],
                                                                       signal_label=signal)
                                                     for signal, sender in self._listening_signals])
            for triggered_event in expression.triggered_events:
                if not self.node.src_dst_ep_active[src_dst_key]:
                    signal_name = triggered_event.type.name
                    gen_tree_id = signal_name.split("_")[-2]
                    if gen_tree_id == 'ACK':
                        gen_tree_id = signal_name.split("_")[-1]
                    if self._is_source:  # source node is responsible to turn off all tree ep generators
                        gen_ep_mem_pos = self.node.qubit_mapping[f"{gen_tree_id}_" \
                                                                 f"{self.node.neighbors[int(gen_tree_id)][1]}"][0]
                        if self.node.acceptable_time[src_dst_key] >= self.node.qubit_creation_time[gen_ep_mem_pos]:
                            tools.tree_failure_v2(self.node, int(gen_tree_id), self._tree_nodes[int(gen_tree_id)],
                                                  self.node.ID, self._other_end_id)
                            continue
                        for tree_id, all_tree_nodes in self._tree_nodes.items():
                            if tree_id == int(gen_tree_id):
                                continue
                            neighbor_id = self.node.neighbors[tree_id][1]
                            self.node.entangling[f"{tree_id}_{neighbor_id}"] = False
                            self.node.ports[f"cto_neighbor{neighbor_id}"].tx_output(
                                ProtocolMessage(MessageType.STOP_ENTANGLEMENT_GENERATING, tree_id, 'left', -1,
                                                ns.sim_time()))
                            self.node.ports[f"cto_neighbor{neighbor_id}"].tx_output(
                                ProtocolMessage(MessageType.STOP_ENTANGLING, tree_id, 'left', -1, ns.sim_time()))
                            for node_id in all_tree_nodes:
                                if node_id == self.node.ID or (node_id == self._other_end_id and
                                                               node_id == neighbor_id):
                                    continue
                                elif node_id == neighbor_id:
                                    data = 'left'
                                elif node_id == self._other_end_id:
                                    data = 'left'
                                else:
                                    data = 'both'
                                self.node.ports[f"cto_node{neighbor_id}"].tx_output(
                                    ProtocolMessage(MessageType.STOP_ENTANGLEMENT_GENERATING, tree_id, data, -1,
                                                    ns.sim_time()))
                                self.node.ports[f"cto_node{neighbor_id}"].tx_output(
                                    ProtocolMessage(MessageType.STOP_ENTANGLING, tree_id, data, -1, ns.sim_time()))
                        logging.info(
                            f"{ns.sim_time():.1f}; (S,D:{self.node.ID},{self._other_end_id}) One of its trees, "
                            f"tree{gen_tree_id}, "
                            f"succeeded. Source, node{self.node.ID}, received the ep in qubit{gen_ep_mem_pos}, put "
                            f"it in qubit{target_src_dst_mem_pos}. "
                            f"Stop all other trees"
                            f"{[tree_id for tree_id in self._tree_nodes.keys() if tree_id != int(gen_tree_id)]}.")
                    else:
                        gen_ep_mem_pos = self.node.qubit_mapping[f"{gen_tree_id}_" \
                                                                 f"{self.node.neighbors[int(gen_tree_id)][0]}"][0]
                        if self.node.acceptable_time[src_dst_key] >= self.node.qubit_creation_time[gen_ep_mem_pos]:
                            continue
                        logging.info(
                            f"{ns.sim_time():.1f}; (S,D:{self.node.ID},{self._other_end_id}) One of its trees,"
                            f" tree{gen_tree_id}, "
                            f"succeeded. Destination, node{self.node.ID}, received the ep in qubit{gen_ep_mem_pos}, "
                            f"put it in qubit{target_src_dst_mem_pos}.")
                    self.node.src_dst_ep_active[src_dst_key] = True
                    self.node.qmemory.pop(target_src_dst_mem_pos)
                    self.node.qmemory.mem_positions[target_src_dst_mem_pos].in_use = True
                    self.node.qmemory.put(self.node.qmemory.pop(gen_ep_mem_pos), target_src_dst_mem_pos)
                    self.send_signal(signal_label=Signals.SUCCESS)


#  ************************************ V2 Protocols ************************************
#  ************************************ GHZ Base Protocols **********************

class CentralGHZTeleportation(NodeProtocol):
    """A class to teleport a locally created GHZ, inside this node, to a set of terminals.
    This class takes some SwappingTreeControllerProtocol, each (s,d) can be one or multiple swapping trees.
     and their successful signals to initiate the teleportation"""
    def __init__(self, node: qPNode, tree_id: int,
                 swapping_trees_controllers: Dict[int, SwappingTreeControllerProtocol],
                 is_terminal: bool, trees_and_nodes: List):
        """@is_terminal: defines if the central node itself is a terminal or not"""
        super().__init__(node, name=node.name + f"central_node_ghz_creator_protocol")
        for dst_id, swapping_trees_controller in swapping_trees_controllers.items():
            self.add_subprotocol(swapping_trees_controller, f'swapping_tree_controller_{dst_id}')
        self._is_terminal = is_terminal
        self._trees_and_nodes = trees_and_nodes
        self._swapping_trees_controllers = swapping_trees_controllers
        self._tree_id = tree_id

    def run(self):
        received_signal = set()
        while True:
            expression = yield reduce(operator.or_, [self.await_signal(sender=subprotocol, signal_label=Signals.SUCCESS)
                                                     for subprotocol in self.subprotocols.values()])
            for triggered_event in expression.triggered_events:
                signal_name = triggered_event.source.name
                received_signal.add(signal_name)
            if len(received_signal) == len(self._swapping_trees_controllers):
                received_signal = set()
                # all the (s,d) succeeded, time to teleport all qubits. ALl (s, d) needs to restart if one fail
                logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) (GHZ) central node{self.node.ID} received "
                             f"all (s,d) EPs. It now performs teleportation on locally-created GHZ state."
                             f" Central node {'IS' if self._is_terminal else 'IS NOT'} part of the GHZ.")
                local_ghz_qubits, _ = tools.create_ghz_reference_state(len(self._swapping_trees_controllers) +
                                                                       (1 if self._is_terminal else 0))
                qubit_idx = 0
                results = {}
                for dst_id in self._swapping_trees_controllers.keys():
                    qmemory = QuantumMemory("tmp", num_positions=2)
                    qmemory.put([local_ghz_qubits[qubit_idx],
                                 self.node.qmemory.peek([self.node.qubit_mapping[f"#{dst_id}"]])[0]], positions=[0, 1])
                    m = bell_measurement(qmemory, [0, 1], False)
                    if random() < self.node.bsm_success_rate:
                        results[dst_id] = m
                        qubit_idx += 1
                    else:
                        # teleportation failed
                        break
                # advancing time with the number of teleportation
                yield self.await_timer(self.node.bsm_time * 1e9 * max(len(results) + 1,
                                                                      len(self._swapping_trees_controllers)))
                if len(results) == len(self._swapping_trees_controllers):
                    # all teleportation succeeded, time to send the results to destinations
                    logging.info(
                        f"{ns.sim_time():.1f}; (TREE{self._tree_id}) (GHZ) central node{self.node.ID} has performed "
                        f"all teleportation successfully. It now sends the results to the terminals "
                        f"({list(self._swapping_trees_controllers.keys())}) for Corrections")
                    for dst_id, result in results.items():
                        self.node.ports[f"cto_node{dst_id}"].tx_output(
                            ProtocolMessage(MessageType.SWAPPING_CORRECTION, self._tree_id, result, self.node.ID,
                                            ns.sim_time()))
                    if self._is_terminal:
                        # node itself is part of the GHZ, copy one of the local qubits into a memory
                        self.node.qmemory.put(local_ghz_qubits[-1], self.node.qubit_mapping[f"#{self._tree_id}"])
                    self.send_signal(Signals.SUCCESS)
                else:
                    # restart all the trees
                    logging.info(
                        f"{ns.sim_time():.1f}; (TREE{self._tree_id}) (GHZ) central node{self.node.ID} has failed "
                        f"to perform all teleportation successfully. It now sends ACK to restart all the swapping tree"
                        f" for all the (s,d). Terminals-{list(self._swapping_trees_controllers.keys())}")
                    for tree_id, tree_nodes, src_id, dst_id in self._trees_and_nodes:
                        tools.restart_swapping_trees(self.node, tree_id, tree_nodes, src_id, dst_id)


class FusionCorrectionProtocol(NodeProtocol):
    """A protocol that handles Fusion correction operations and send a signal in case to start a new Fusion.
    No message will be sent to any node"""

    # TODO: not tracking link layer (qubit) entanglement for now. if needed, we should keep track of them using await_me

    def __init__(self, node: qPNode, messaging_protocol: MessagingProtocolV2, is_fro: bool = False):
        """@messaging_protocol is needed to receive Fusion correction results. Node may be subject to multiple
        Fusion responses."""
        super(FusionCorrectionProtocol, self).__init__(node, name=f"{node.name}_fusion_correction_protocol")
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._list_of_received_signals = []
        self._is_fro = is_fro

    def add_signals(self, tree_id: int, end_nodes_key: str):
        self.add_signal(f"{MessageType.FUSION_CORRECTION_ACK.name}_{tree_id}_{end_nodes_key}")
        self._list_of_received_signals.append(f"{MessageType.FUSION_CORRECTION.name}_{tree_id}_{end_nodes_key}")

    def run(self):
        if len(self._list_of_received_signals) == 0:
            # no fusion correction involved
            self.subprotocols.__delitem__("messaging_protocol")
            return
        while True:
            expr = yield reduce(operator.or_, [self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                                 signal_label=label)
                                               for label in self._list_of_received_signals])
            for triggered_event in expr.triggered_events:
                signal = triggered_event.source.get_signal_by_event(triggered_event)
                tree_id, end_nodes = signal.label.split("_")[-2:]
                qubit_key = f"{tree_id}_{end_nodes}"

                try:
                    if signal.result[1][0]:
                        self.node.qmemory.operate(ns.X, self.node.qubit_mapping[qubit_key])
                    if signal.result[1][1]:
                        self.node.qmemory.operate(ns.Z, self.node.qubit_mapping[qubit_key])
                    if qPNode.parents[qubit_key][1] is None or qPNode.parents[qubit_key][1] == self.node.ID:
                        self.send_signal(f"{MessageType.FUSION_CORRECTION_ACK.name}_{qubit_key}")
                        logging.info(f"{ns.sim_time():.1f}: (TREE{tree_id}) ({self.node.name}) (FUSION CORRECTION)"
                                     f" applied received corrections for GHZ {end_nodes} and send "
                                     f"FUSION_CORRECTION_ACK SIGNAL as next FUSION is rooted at the same node.")
                    else:
                        self.node.ports[f"cto_node{qPNode.parents[qubit_key][1]}"].tx_output(
                            ProtocolMessage(MessageType.FUSION_CORRECTION_ACK, tree_id, end_nodes, -1,
                                            ns.sim_time()))
                        logging.info(f"{ns.sim_time():.1f}: (TREE{tree_id}) ({self.node.name}) (FUSION CORRECTION)"
                                     f" applied received corrections for GHZ {end_nodes} and send "
                                     f"FUSION_CORRECTION_ACK MESSAGE to node{qPNode.parents[qubit_key][1]} for next "
                                     f"FUSION.")
                except Exception as e:
                    if not self._is_fro:
                        logging.error(f"{ns.sim_time():.1f}: (TREE{tree_id}) ({self.node.name}) (FUSION CORRECTION) "
                                      f"Start accepting EPs from link layer for GHZ {end_nodes} rooted at "
                                      f"(node{signal.result[0]})\tError msg:{e}")
                        # start accepting all link layer for this GHZ
                        for other in self.node.ghz_edges[qubit_key]:
                            self.node.entangling[f"{tree_id}_{other}"] = True
                            logging.debug(f"{ns.sim_time():.1f}: (TREE{tree_id}) ({self.node.name})  set ENTANGLING to TRUE"
                                          f" for qubit assigned for the edge connected to node {other} because of "
                                          f"correction failure.")
                            # send zero-latency msg to the left neighbor to start entangling
                            self.node.ports[f"cto_neighbor{other}"].tx_output(
                                ProtocolMessage(MessageType.START_ENTANGLING, tree_id, end_nodes, -1, ns.sim_time()))
                    # send a msg to root to restart the whole tree rooted at that root
                    self.node.ports["cto_node" + str(signal.result[0])].tx_output(ProtocolMessage(
                        MessageType.FUSION_CORRECTION_FAILURE, tree_id, end_nodes, -1, ns.sim_time()))


class FusionProtocol(NodeProtocol):
    """a protocol class that handles FUSION operations on two qubits (to merge two GHZ states), measuring one of them
    and send the measurement result to all the nodes in the others state"""

    def __init__(self, node: qPNode, tree_id: int, end_nodes_left: set[int], end_nodes_right: set[int],
                 all_nodes_left: set[int], all_nodes_right: set[int],
                 messaging_protocol: MessagingProtocolV2,
                 fusion_correction_protocol: FusionCorrectionProtocol,
                 retain: bool = False, fro: bool = False, trees_and_nodes: List = None,
                 fusion_time_window: float = float('inf')):
        if len(end_nodes_left.intersection(end_nodes_right)) != 1:
            # ERROR, two legs of the GHZ state has another common node
            raise ValueError(f"GHZ state at node{node.ID}, left GHZ state as {list(end_nodes_left)}, and right GHZ "
                             f"state as {list(end_nodes_right)} has another common node.")

        end_nodes = end_nodes_left.union(end_nodes_right)
        if not retain:
            end_nodes.remove(node.ID)

        self._end_nodes_key = "-".join([str(end_id) for end_id in sorted(list(end_nodes))])
        self._tree_id = tree_id
        self._key = f"{self._tree_id}_{self._end_nodes_key}"
        super().__init__(node, name=f"{node.name}_fusion_protocol_tree{tree_id}_ghz_{self._end_nodes_key}")
        self._end_nodes_left = end_nodes_left
        self._end_nodes_right = end_nodes_right
        self._fro = fro
        if not fro:
            self.node.all_nodes[self._key] = all_nodes_left.union(all_nodes_right)
        else:
            self.node.all_nodes[self._key] = trees_and_nodes

        self.add_subprotocol(fusion_correction_protocol, 'fusion_correction_protocol')
        self.add_subprotocol(messaging_protocol, 'messaging_protocol')
        self._fusion_time_window = fusion_time_window
        self._retain = retain

    def run(self):
        qubit_pos_left_ready = False  # left fusion qubit
        qubit_pos_right_ready = False  # right fusion qubit
        left_last_time, right_last_time = -float('inf'), -float('inf')
        end_nodes_key_left = '-'.join([str(end_id) for end_id in sorted(list(self._end_nodes_left))])
        end_nodes_key_right = '-'.join([str(end_id) for end_id in sorted(list(self._end_nodes_right))])

        if self.node.left_height[self._key] == 1:
            other_node = self._end_nodes_left.difference({self.node.ID}).pop()
            if not self._fro:
                left_qubit_pos = self.node.qubit_mapping[f"{self._tree_id}_{other_node}"][0]
            else:
                left_qubit_pos = self.node.qubit_mapping[f"#{other_node}"]
        else:
            key = f"{self._tree_id}_{end_nodes_key_left}"
            left_qubit_pos = self.node.qubit_mapping[key]

        if self.node.right_height[self._key] == 1:
            other_node = self._end_nodes_right.difference({self.node.ID}).pop()
            if not self._fro:
                right_qubit_pos = self.node.qubit_mapping[f"{self._tree_id}_{other_node}"][0]
            else:
                right_qubit_pos = self.node.qubit_mapping[f"#{other_node}"]
        else:
            key = f"{self._tree_id}_{end_nodes_key_right}"
            right_qubit_pos = self.node.qubit_mapping[key]

        left_correction_signals = []
        left_correction_received = {}
        if self.node.left_height[self._key] != 1:
            # qubit comes from lower level of the tree, should be after ALL CORRECTIONS
            # All corrections from lower GHZ endnodes, except the root.
            # Correction from node itself comes from FusionCorrectionProtocol, otherwise from MessagingProtocol
            for node in self._end_nodes_left:
                if node != qPNode.parents[f"{self._tree_id}_{end_nodes_key_left}"][0]:
                    signal_name = f'{MessageType.FUSION_CORRECTION_ACK.name}_{self._tree_id}_{end_nodes_key_left}'
                    if node == self.node.ID:
                        left_correction_signals.append(('fusion_correction_protocol', signal_name))
                    else:
                        signal_name += f'_{node}'
                        left_correction_signals.append(('messaging_protocol', signal_name))
                    left_correction_received['PROTOCOL_' + signal_name] = False
            evexpr_left = reduce(operator.or_, [self.await_signal(sender=self.subprotocols[protocol],
                                                                  signal_label=signal) for protocol, signal in
                                                left_correction_signals])
        else:
            evexpr_left = self.await_mempos_in_use_toggle(self.node.qmemory, [left_qubit_pos])

        right_correction_signals = []
        right_correction_received = {}
        if self.node.right_height[self._key] != 1:
            # qubit comes from lower level of the tree, should be after ALL CORRECTIONS
            # All corrections from lower GHZ endnodes, except the root.
            # Correction from node itself comes from FusionCorrectionProtocol, otherwise from MessagingProtocol
            for node in self._end_nodes_right:
                if node != qPNode.parents[f"{self._tree_id}_{end_nodes_key_right}"][0]:
                    signal_name = f'{MessageType.FUSION_CORRECTION_ACK.name}_{self._tree_id}_{end_nodes_key_right}'
                    if node == self.node.ID:
                        right_correction_signals.append(('fusion_correction_protocol', signal_name))
                    else:
                        signal_name += f'_{node}'
                        right_correction_signals.append(('messaging_protocol', signal_name))
                    right_correction_received['PROTOCOL_' + signal_name] = False
            evexpr_right = reduce(operator.or_, [self.await_signal(sender=self.subprotocols[protocol],
                                                                   signal_label=signal) for protocol, signal in
                                                 right_correction_signals])
        else:
            evexpr_right = self.await_mempos_in_use_toggle(self.node.qmemory, [right_qubit_pos])

        while True:
            expression = yield evexpr_left | evexpr_right
            if expression.first_term.value:
                # first qubit is ready
                if len(left_correction_signals) == 0:
                    qubit_pos_left_ready = True
                else:
                    for triggered_event in expression.first_term.triggered_events:
                        source_name = triggered_event.type.name
                        left_correction_received[source_name] = True
                        if all(list(left_correction_received.values())):
                            qubit_pos_left_ready = True
                if left_last_time == -float('inf'):
                    left_last_time = ns.sim_time()
            else:
                if len(right_correction_signals) == 0:
                    qubit_pos_right_ready = True
                else:
                    for triggered_event in expression.second_term.triggered_events:
                        source_name = triggered_event.type.name
                        right_correction_received[source_name] = True
                        if all(list(right_correction_received.values())):
                            qubit_pos_right_ready = True
                if right_last_time == -float('inf'):
                    right_last_time = ns.sim_time()

            if qubit_pos_left_ready and qubit_pos_right_ready:
                if abs(right_last_time - left_last_time) < self._fusion_time_window:
                    # acceptable to do BSM, used for delft lp and sigcomm
                    yield self.await_timer(self.node.fusion_time * 1e9)
                    try:
                        m = fusion(qmemory=self.node.qmemory,
                                   positions=[left_qubit_pos, right_qubit_pos],
                                   RETAIN=self._retain, PHYSICAL=False)

                        if random() < self.node.fusion_success_rate:
                            # successful FUSION
                            # RESULTS always will be sent to end_nodes_left to do correction
                            # An ACK will be sent to end_nodes_right to notify
                            logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) ({self.node.name})"
                                         f" (FUSION) received entangled qubits for GHZ-{self._end_nodes_key}, "
                                         f"fuse qubits & sending ack to left end nodes"
                                         f"({end_nodes_key_left}), and sending corrections to right end nodes"
                                         f" ({end_nodes_key_right})")
                            for node in self._end_nodes_left:
                                if node != self.node.ID:
                                    self.node.ports[f"cto_node{node}"].tx_output(
                                        ProtocolMessage(MessageType.FUSION_CORRECTION, self._tree_id,
                                                        [self._end_nodes_key, (0, 0)], None, ns.sim_time()))
                            node_counter = 0
                            for node in self._end_nodes_right:
                                if node != self.node.ID:
                                    self.node.ports[f"cto_node{node}"].tx_output(
                                        ProtocolMessage(MessageType.FUSION_CORRECTION, self._tree_id,
                                                        [self._end_nodes_key,
                                                         (m if (node_counter == 0 and not self._retain) else (m[0], 0))],
                                                        None, ns.sim_time()))
                                    node_counter += 1
                            self.send_signal(signal_label=Signals.SUCCESS)
                        else:
                            # unsuccessful bsm, should tell all the children to resume entangling again
                            if not self._fro:
                                logging.info(
                                    f"{ns.sim_time():.1f}; (TREE{self._tree_id}) ({self.node.name})  FUSION was"
                                    f" UNSUCCESSFUL for GHZ-{self._end_nodes_key}, RESTARTING the tree"
                                    f" for all nodes {self.node.all_nodes[self._key]}.")
                                tools.fusion_tree_failure(self.node, self._tree_id, self.node.all_nodes[self._key],
                                                          self._end_nodes_key)
                            else:
                                logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) ({self.node.name}) "
                                             f"FUSION(FRO) was UNSUCCESSFUL for GHZ-{self._end_nodes_key}. "
                                             f"Restarting all the swapping trees.")
                                for tree_id, all_tree_nodes, src_id, dst_id in self.node.all_nodes[self._key]:
                                    tools.restart_swapping_trees(self.node, tree_id, all_tree_nodes, src_id, dst_id)

                    except Exception as e:
                        if not self._fro:
                            logging.error(
                                f"{ns.sim_time():.1f}: (TREE{self._tree_id}) ({self.node.name}) (FUSION), error"
                                f" happened for GHZ-{self._end_nodes_key}. RESTARTING the tree"
                                f" for all nodes {self.node.all_nodes[self._key]}\n"
                                f" Error message: {e}")
                            tools.fusion_tree_failure(self.node, self._tree_id, self.node.all_nodes[self._key],
                                                      self._end_nodes_key)
                        else:
                            logging.error(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) ({self.node.name}) "
                                          f"FUSION(FRO) error happened for GHZ-{self._end_nodes_key}. "
                                          f"Restarting all the swapping trees.\n"
                                          f"Error message: {e}")
                            for tree_id, all_tree_nodes, src_id, dst_id in self.node.all_nodes[self._key]:
                                tools.restart_swapping_trees(self.node, tree_id, all_tree_nodes, src_id, dst_id)
                else:
                    if not self._fro:
                        logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) ({self.node.name}) FUSION was"
                                     f" UNSUCCESSFUL for GHZ-{self._end_nodes_key} because left & right did not"
                                     f" arrive at the same time."
                                     f" Time difference: {abs(right_last_time - left_last_time)} ns. RESTARTING the tree"
                                     f" for all nodes {self.node.all_nodes[self._key]}")
                        tools.fusion_tree_failure(self.node, self._tree_id, self.node.all_nodes[self._key],
                                                  self._end_nodes_key)
                    else:
                        logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) ({self.node.name}) "
                                     f"FUSION(FRO) FUSION was UNSUCCESSFUL for GHZ-{self._end_nodes_key} because left &"
                                     f"right did not arrive in proper time."
                                     f" Time difference: {abs(right_last_time - left_last_time)} ns. RESTARTING the tree"
                                     f" Restarting all the swapping trees.")
                        for tree_id, all_tree_nodes, src_id, dst_id in self.node.all_nodes[self._key]:
                            tools.restart_swapping_trees(self.node, tree_id, all_tree_nodes, src_id, dst_id)

                # reset for the next round
                qubit_pos_left_ready = False
                qubit_pos_right_ready = False
                left_last_time, left_last_time = -float('inf'), -float('inf')
                left_correction_received = {key: False for key in left_correction_received.keys()}
                right_correction_received = {key: False for key in right_correction_received.keys()}


class CentralGHZController(Protocol):
    """A class that is responsible to measure fidelity and restarting central-node GHZ generation and all the
    swapping trees."""

    def __init__(self, central_node: qPNode, tree_id: int, central_ghz_teleportation: CentralGHZTeleportation,
                 teleportation_correction_protocols: List[CorrectionProtocolV2],
                 end_nodes: set[int], trees_and_nodes: List[List], end_qpNodes: List[qPNode]):
        super(CentralGHZController, self).__init__(name=f"central_controller_tree{tree_id}_ghz_{end_nodes}")
        self._central_node = central_node
        self._tree_id = tree_id
        self._end_nodes = end_nodes
        self._end_qpNodes = end_qpNodes
        self._trees_and_nodes = trees_and_nodes
        self._teleportation_correction_protocols = teleportation_correction_protocols
        self._teleportation_and_correction_done = {central_ghz_teleportation.name: False}
        self.add_subprotocol(central_ghz_teleportation, "central_teleportation")
        for idx, swapping_correction_protocol in enumerate(teleportation_correction_protocols):
            self.add_subprotocol(swapping_correction_protocol, f"teleportation_correction-{idx}")
            self._teleportation_and_correction_done[swapping_correction_protocol.name] = False
        self._success_count = 0
        self._fidelity = []

    def run(self):
        while True:
            expr = yield reduce(operator.or_,
                                [self.await_signal(sender=self.subprotocols["central_teleportation"],
                                                   signal_label=Signals.SUCCESS)] +
                                [self.await_signal(sender=self.subprotocols[f"teleportation_correction-{idx}"],
                                                   signal_label=f"{MessageType.CORRECTION_ACK.name}_{self._tree_id}")
                                 for idx in range(len(self._teleportation_correction_protocols))])
            for triggered_event in expr.triggered_events:
                source_name = triggered_event.source.name
                self._teleportation_and_correction_done[source_name] = True
            if all(list(self._teleportation_and_correction_done.values())):
                # fusion and its corrections are done, time to measure fidelity and restart swapping trees
                logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) GHZ was SUCCESSFULLY generated and"
                             f" all ACK are receive for GHZ{self._end_nodes}. "
                             f"Signals will be sent to all nodes to restart creating new link EPs to create another"
                             f" GHZ state.")
                self._success_count += 1
                self._teleportation_and_correction_done = {key: False for key in
                                                           self._teleportation_and_correction_done.keys()}
                # calculating the fidelity
                qubits = [qpnode.qmemory.peek(qpnode.qubit_mapping[f"#{self._central_node.ID}"])[0]
                          for qpnode in self._end_qpNodes]
                if self._central_node.ID in self._end_nodes:
                    # central node it self has a qubit in GHZ
                    qubits += [self._central_node.qmemory.peek(self._central_node.qubit_mapping[f"#{self._tree_id}"])[0]]
                try:
                    _fidelity = ns.qubits.fidelity(qubits,
                                                   tools.create_ghz_reference_state(len(self._end_nodes))[1],
                                                   squared=True)
                    self._fidelity.append(_fidelity)
                    logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) (CENTRAL MODE) GHZ-{self._end_nodes} was "
                                 f"created!. Qubits' qstate {qubits[0].qstate}"
                                 f"Fidelity: {self._fidelity[-1]}")
                except Exception as e:
                    logging.error(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) (CENTRAL MODE) GHZ-{self._end_nodes} "
                                  f"Fidelity calculation failed!. "
                                  f"Error msg: {e}")
                for tree_id, tree_nodes, src_id, dst_id in self._trees_and_nodes:
                    tools.restart_swapping_trees(self._central_node, tree_id, tree_nodes, src_id, dst_id)

    @property
    def successful_counts(self):
        return self._success_count

    @property
    def fidelity(self):
        return self._fidelity


class FusionTreeController(Protocol):
    """A class that counts number of successful GHZ generation and signals to restart creating another one.
    Also, it can be used for fidelity measurements"""

    def __init__(self, root_node: qPNode, tree_id: int, fusion_protocol: FusionProtocol,
                 fusion_correction_protocols: List[FusionCorrectionProtocol],
                 end_nodes: set[int], all_nodes: set[int], end_qPNodes: List[qPNode],
                 fro: bool = False, trees_and_nodes: List[List] = None):
        super(FusionTreeController, self).__init__(name=f"fusion_controller_tree{tree_id}_ghz_{end_nodes}")
        self._root_node = root_node
        self._success_count = 0
        self._tree_id = tree_id
        self._end_nodes = end_nodes
        self._fusion_protocol = fusion_protocol
        self._fusion_correction_protocols = fusion_correction_protocols
        self._all_nodes = all_nodes
        self.add_subprotocol(fusion_protocol, "fusion_protocol")
        self._fusion_correction_done = {fusion_protocol.name: False}
        for idx, fusion_correction_protocol in enumerate(fusion_correction_protocols):
            self.add_subprotocol(fusion_correction_protocol, f"fusion_correction_protocol-{idx}")
            self._fusion_correction_done[fusion_correction_protocol.name] = False
        self._end_qpNodes = end_qPNodes
        self._fidelity = []
        self._fro = fro
        self._trees_and_nodes = trees_and_nodes

    def run(self):
        end_nodes_key = "-".join([str(num) for num in sorted(list(self._end_nodes))])

        while True:
            expr = yield reduce(operator.or_,
                                [self.await_signal(sender=self.subprotocols["fusion_protocol"],
                                                   signal_label=Signals.SUCCESS)] +
                                [self.await_signal(sender=self.subprotocols[f"fusion_correction_protocol-{idx}"],
                                                   signal_label=f"{MessageType.FUSION_CORRECTION_ACK.name}"
                                                                f"_{self._tree_id}_{end_nodes_key}")
                                 for idx in range(len(self._fusion_correction_protocols))])
            for triggered_event in expr.triggered_events:
                source_name = triggered_event.source.name
                self._fusion_correction_done[source_name] = True

                if all(list(self._fusion_correction_done.values())):
                    # fusion and its corrections are done, time to restart
                    logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) FUSION was SUCCESSFULLY generated and"
                                 f" all ACK are receive for GHZ{self._end_nodes}. "
                                 f"Signals will be sent to all nodes to restart creating new link EPs to create another"
                                 f" GHZ state.")
                    self._success_count += 1
                    self._fusion_correction_done = {key: False for key in self._fusion_correction_done.keys()}
                    # calculating the fidelity
                    qubits = [qpnode.qmemory.peek(qpnode.qubit_mapping[f"{self._tree_id}_{end_nodes_key}"])[0]
                              for qpnode in self._end_qpNodes]
                    try:
                        _fidelity = ns.qubits.fidelity(qubits,
                                                       tools.create_ghz_reference_state(len(self._end_nodes))[1],
                                                       squared=True)

                        self._fidelity.append(_fidelity)
                        logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) GHZ-{end_nodes_key} was created!. "
                                     f"Qubits' qstate {qubits[0].qstate}"
                                     f"Fidelity: {self._fidelity[-1]}")
                    except Exception as e:
                        logging.error(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) GHZ-{end_nodes_key} Fidelity "
                                      f"calculation failed!. "
                                      f"Error msg: {e}")
                    if not self._fro:
                        tools.fusion_tree_failure(self._root_node, self._tree_id, self._all_nodes, end_nodes_key)
                    else:
                        for tree_id, tree_nodes, src_id, dst_id in self._trees_and_nodes:
                            tools.tree_failure_v2(self._root_node, tree_id, tree_nodes, src_id, dst_id)
                            if self._root_node.ID != src_id:
                                self._root_node.ports[f"cto_node{src_id}"].tx_output(
                                    ProtocolMessage(MessageType.SRC_DST_RESET, tree_id, ns.sim_time(), dst_id,
                                                    ns.sim_time()))
                            if self._root_node.ID != dst_id:
                                self._root_node.ports[f"cto_node{dst_id}"].tx_output(
                                    ProtocolMessage(MessageType.SRC_DST_RESET, tree_id, ns.sim_time(), src_id,
                                                    ns.sim_time()))

    @property
    def successful_counts(self):
        return self._success_count

    @property
    def fidelity(self):
        return self._fidelity
