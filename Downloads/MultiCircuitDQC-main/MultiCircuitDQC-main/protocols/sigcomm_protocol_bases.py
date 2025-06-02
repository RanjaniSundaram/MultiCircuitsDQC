import operator
import sys

import netsquid.components.qmemory

sys.path.append("..")

from netsquid.protocols import NodeProtocol, Signals, Protocol
from protocols.sigcomm_component_bases import qPNode, Charlie
from commons.tools import bell_measurement
import netsquid as ns
from netsquid.nodes import Node
from functools import reduce
from collections import namedtuple
from enum import Enum
from random import random
from netsquid.components import QuantumMemory
import logging
import datetime
import os
from netsquid.util.datacollector import DataCollector
import pandas as pd
import numpy as np

ProtocolMessage = namedtuple("ProtocolMessage", "type, tree_id, data, next_dst_id")

DEBUG_MODE = 0  # 0: no printing, 1: high level details, 2: deep details

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


class tools:
    @staticmethod
    def tree_failure(node: qPNode, tree_id: int):
        # TODO should send signal for only failed qubit positions
        logging.info(f"{ns.sim_time():.1f}; (TREE{tree_id}) swapping was UNSUCCESSFUL"
                     f" at node{node.ID}.")
        node.entangling[tree_id] = (True, True)  # node itself should start entangling
        # send instant message to its neighbors
        for neighbor in node.neighbors[tree_id]:
            node.ports["cto_neighbor" + str(neighbor)].tx_output(ProtocolMessage(
                MessageType.START_ENTANGLING, tree_id, None, -1))
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
                                data=data, next_dst_id=-1))
            logging.debug(f"{ns.sim_time():.1f}: (TREE{tree_id}) sending ENTANGLING signal to node"
                          f" {str(child_id)} for {data} positions.")
            if DEBUG_MODE > 1:
                print(f"{ns.sim_time():.1f}: (TREE{tree_id}) sending ENTANGLING signal to node"
                      f" {str(child_id)} for {data} positions.")


class CharlieProtocol(NodeProtocol):  # DONE
    """Charlie protocol that is responsible to do photonics BSM based on receiving from its left and right ports"""

    def __init__(self, node: Charlie, bsm_time_window: float = 0):
        super(CharlieProtocol, self).__init__(node, name=node.name + "_charlie_protocol")
        self._bsm_time_window = bsm_time_window

    def run(self):
        cright_port = self.node.ports["cout_right"]

        while True:
            expr = yield reduce(operator.or_, [self.await_port_input(self.node.ports[port_name])
                                               for port_name in self.node.ports if port_name.startswith("qin_") or
                                               port_name.startswith("cin_")])
            for triggered_event in expr.triggered_events:
                for obj in triggered_event.source.input_queue:
                    received_port_name = obj[0].source.name
                    if received_port_name.startswith("cin_"):
                        # time to decide if BSM is possible for prev events
                        tree_id = int(received_port_name.split("cin_tree")[1])
                        results = {}
                        succ_num = 0  # successful number BSM ops
                        for qubit_key in self.node.qubit_data[tree_id].keys():
                            left_qubit, left_time = self.node.qubit_data[tree_id][qubit_key][0]
                            right_qubit, right_time = self.node.qubit_data[tree_id][qubit_key][1]
                            if left_qubit is not None and right_qubit is not None and \
                                    abs(left_time - right_time) <= self._bsm_time_window:
                                logging.debug(
                                    f"{ns.sim_time():.1f}: (TREE{str(tree_id)}) both link layer qubits ({qubit_key})"
                                    f"successfully received  at {self.node.name}")
                                qmemory = QuantumMemory("tmp" + str(tree_id), num_positions=2)
                                qmemory.put([left_qubit, right_qubit], positions=[0, 1])
                                try:
                                    results[qubit_key] = bell_measurement(qmemory, positions=[0, 1], PHYSICAL=True)
                                    succ_num += 1
                                except Exception as e:
                                    logging.debug(f"{ns.sim_time():.2f} Error happened at node:{self.node.name}"
                                                  f"Error Msg:{e}. left_name: {left_qubit.name}, "
                                                  f"right_name: {right_qubit.name}")
                        if succ_num > 0:
                            pass
                            # yield self.await_timer(0 * self.node.bsm_time * 1e9)  # zero time for now
                        sending_data = []  # sending successful BSM ops to right node for a tree
                        for qubit_key, res in results.items():
                            left_qubit_pos, right_qubit_pos = int(qubit_key.split("-")[0]), int(qubit_key.split("-")[1])
                            if random() <= self.node.bsm_success_rate:
                                logging.debug(
                                    f"{ns.sim_time():.1f}: (TREE{str(tree_id)}) BSM was successfully done at "
                                    f"{self.node.name}")
                                sending_data.append((left_qubit_pos, right_qubit_pos, res))
                            self.node.qubit_data[tree_id][qubit_key] = ((None, -float('inf')), (None, -float('inf')))
                        if len(sending_data) > 0:
                            cright_port.tx_output(
                                ProtocolMessage(type=MessageType.LINK_CORRECTION, data=sending_data,
                                                tree_id=tree_id,
                                                next_dst_id=self.node._left_node_id))
                    else:
                        isLeft = received_port_name.startswith("qin_left")
                        for qubit in obj[1].items:
                            tree_id, arriving_qubit_pos = int(qubit.name.split('_')[0]), int(qubit.name.split('_')[1])
                            if isLeft:
                                other_qubit_pos = self.node._right_qubit_pos["{t}-{l}".format(t=tree_id,
                                                                                              l=arriving_qubit_pos)]
                                qubit_key = "{l}-{r}".format(l=arriving_qubit_pos, r=other_qubit_pos)
                                right_qubit, right_time = None, -float('inf')
                                left_time = ns.sim_time()
                                if tree_id in self.node.qubit_data and qubit_key in self.node.qubit_data[tree_id]:
                                    right_qubit, right_time = self.node.qubit_data[tree_id][qubit_key][1]
                                self.node.qubit_data[tree_id][qubit_key] = (
                                    (qubit, left_time), (right_qubit, right_time))
                            else:
                                other_qubit_pos = self.node._left_qubit_pos["{t}-{r}".format(t=tree_id,
                                                                                             r=arriving_qubit_pos)]
                                qubit_key = "{l}-{r}".format(l=other_qubit_pos, r=arriving_qubit_pos)
                                left_qubit, left_time = None, -float('inf')
                                right_time = ns.sim_time()
                                if tree_id in self.node.qubit_data and qubit_key in self.node.qubit_data[tree_id]:
                                    left_qubit, left_time = self.node.qubit_data[tree_id][qubit_key][0]
                                self.node.qubit_data[tree_id][qubit_key] = (
                                    (left_qubit, left_time), (qubit, right_time))


class CharlieInvoker(NodeProtocol):
    """A protocol that is used to invoke charlie to do BSM ops. This has been designed since charlie process
    events one by one"""

    def __init__(self, node: Node, rate: float, tree_id):
        super(CharlieInvoker, self).__init__(node, name=node.name + "_Invoker_" + str(tree_id))
        self._rate = rate
        self._tree_id = tree_id

    def run(self):
        cout_port = self.node.ports["cout_tree" + str(self._tree_id)]
        yield self.await_timer(1)  # to be a littler bit later than qubits
        while True:
            yield self.await_timer(1.0 / self._rate * 1e9)
            cout_port.tx_output(self._tree_id)


class EntanglementGenerator(NodeProtocol):  # DONE
    """A protocol that generates a pair of EPs, save one in node's qmemory and send the other to charlie to
    do optical bsm. rate is 1/s"""

    def __init__(self, node: qPNode, tree_id: int, rate: float('inf'), isLeft: bool, charlie_id: str,
                 link_layer_qmem_pos: int, gen_success_rate: float):
        super(EntanglementGenerator, self).__init__(node, name=node.name + ("_left" if isLeft else "_right") +
                                                               "_generator_for_tree" + str(tree_id) + "_qubit_" +
                                                               str(link_layer_qmem_pos))
        self._isLeft = isLeft
        self._rate = rate
        self._tree_id = tree_id
        self._charlie_id = charlie_id
        self._link_layer_qmem_pos = link_layer_qmem_pos
        self._gen_success_rate = gen_success_rate

    def run(self):
        qout_port = self.node.ports["qout_" + ("left" if self._isLeft else "right") +
                                    str(self.node.qubit_to_port[self._tree_id][self._link_layer_qmem_pos]) +
                                    "_" + self._charlie_id]
        while True:
            yield self.await_timer(1.0 / self._rate * 1e9)  # wait (nanoseconds)
            if random() <= self._gen_success_rate:
                q1, q2 = ns.qubits.create_qubits(2, no_state=True)
                q1.name += f"{self.node.ID}-{self._tree_id}-{self._link_layer_qmem_pos}"
                q2.name = str(self._tree_id) + "_" + str(self._link_layer_qmem_pos) + (
                    "_right" if self._isLeft else "_left")
                ns.qubits.assign_qstate([q1, q2], ns.b00)
                self.node.qmemory.put(q1, positions=self._link_layer_qmem_pos, replace=True)
                qout_port.tx_output(q2)
                logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) qubit was generated at {self.name}")
            # break


class MessagingProtocol(NodeProtocol):
    """a protocol that handles received messages from other nodes"""

    def __init__(self, node: qPNode):
        super().__init__(node, name=node.name + "_messaging_protocol")
        # add appropriate signals
        # self.add_signal(MessageType.START_ENTANGLING)
        # self.add_signal(MessageType.SWAPPING_CORRECTION)
        # self.add_signal(MessageType.TELEPORTATION_CORRECTION)
        # self.add_signal(MessageType.LINK_CORRECTION)
        # self.add_signal(MessageType.ENTANGLEMENT_READY)
        # self.add_signal(MessageType.CORRECTION_ACK)

    def add_signals(self, tree_id: int):
        self.add_signal(MessageType.START_ENTANGLING.name + str(tree_id))
        self.add_signal(MessageType.SWAPPING_CORRECTION.name + str(tree_id))
        self.add_signal(MessageType.TELEPORTATION_CORRECTION.name + str(tree_id))
        # self.add_signal(MessageType.LINK_CORRECTION.name + str(tree_id))
        self.add_signal(MessageType.ENTANGLEMENT_READY.name + str(tree_id) + "_left")
        self.add_signal(MessageType.ENTANGLEMENT_READY.name + str(tree_id) + "_right")
        self.add_signal(MessageType.CORRECTION_ACK.name + str(tree_id))

    def add_link_correction_signal(self, tree_id: int, left_node_id: int):
        self.add_signal(MessageType.LINK_CORRECTION.name + str(tree_id) + "_" + str(left_node_id))

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
                            self.send_signal(signal_label=MessageType.SWAPPING_CORRECTION.name + str(message.tree_id),
                                             result=(received_port_name.split("cfrom_node")[1], message))
                            # (received_port_name.split("cfrom_node")[1]
                        elif message.type == MessageType.CORRECTION_ACK:
                            self.send_signal(signal_label=MessageType.CORRECTION_ACK.name + str(message.tree_id),
                                             result=(int(received_port_name.split("cfrom_node")[1]), message.data))
                        elif message.type == MessageType.TELEPORTATION_CORRECTION:
                            self.send_signal(MessageType.TELEPORTATION_CORRECTION.name + str(message.tree_id),
                                             [int(received_port_name.split("cfrom_node")[1]), message.data])
                        elif message.type == MessageType.ENTANGLEMENT_READY:
                            left_or_right = "_left" if int(received_port_name.split("cfrom_neighbor")[1]) == \
                                                       self.node.neighbors[message.tree_id][0] else "_right"
                            self.send_signal(MessageType.ENTANGLEMENT_READY.name + str(message.tree_id) + left_or_right, message.data)
                        elif message.type == MessageType.LINK_CORRECTION:
                            self.send_signal(MessageType.LINK_CORRECTION.name + str(message.tree_id) + "_" +
                                             str(message.next_dst_id), message)
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


class LinkCorrectionProtocol(NodeProtocol):  # DONE
    """a protocol that waits for response from Charlie to do the BSM correction and entangle two ends and send a
    message to left end"""

    def __init__(self, node: qPNode, tree_id, messaging_protocol: MessagingProtocol, left_node_id: int):
        super(LinkCorrectionProtocol, self).__init__(node, name=node.name + "_link_correction_tree" + str(tree_id))
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._tree_id = tree_id
        self._is_left_leftnet = left_node_id == self.node.neighbors[self._tree_id][0]
        self.add_signal(MessageType.ENTANGLEMENT_READY.name + str(tree_id) + ("_left" if self._is_left_leftnet else
                                                                              "_right"))
        self._left_node_id = left_node_id  # where we should send the results to

    def run(self):
        charlie_res = None
        while True:
            yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                    signal_label=MessageType.LINK_CORRECTION.name + str(self._tree_id) + "_" +
                                                 str(self._left_node_id))
            tmp_charlie_res = self.subprotocols["messaging_protocol"].get_signal_result(
                MessageType.LINK_CORRECTION.name + str(self._tree_id) + "_" + str(self._left_node_id))
            if tmp_charlie_res.next_dst_id == self._left_node_id:
                charlie_res = tmp_charlie_res
            link_left_qubit_pos_lst = []  # sending list of left qubit pos to the left node where correction was successful
            link_right_qubit_pos_lst = []  # sending a list (signal) of right qubit pos to entangling protocol node where
            # correction was successful
            if charlie_res is not None:
                for link_left_qubit_pos, link_right_qubit_pos, res in charlie_res.data:
                    try:
                        if res[0]:
                            self.node.qmemory.operate(ns.Z, link_right_qubit_pos)
                        if res[1]:
                            self.node.qmemory.operate(ns.X, link_right_qubit_pos)
                        link_left_qubit_pos_lst.append(link_left_qubit_pos)
                        link_right_qubit_pos_lst.append(link_right_qubit_pos)
                        logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) {self.node.name}"
                                      f" received results from "
                                      f"charlie for qubits ({link_left_qubit_pos}-{link_right_qubit_pos}) and did "
                                      f"the CORRECTION. send an ACK to neighbor {self._left_node_id}")
                        # logging.debug(f"(TREE{self._tree_id}) Entangling situation: "
                        #               f"{self.node.entangling[self._tree_id][self.node.link_to_network_pos[self._tree_id][link_left_qubit_pos]]}")
                    except Exception as e:
                        logging.error(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Error happened at"
                                      f" (link layer-waiting"
                                      f" for new one) node for qubits ({link_left_qubit_pos}-{link_right_qubit_pos})"
                                      f"{self.node.name}. \tError Msg: {e}")
                        continue
                self.send_signal(MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) +
                                 ("_left" if self._is_left_leftnet else "_right"),
                                 link_right_qubit_pos_lst)
                self.node.ports["cto_neighbor" + str(self._left_node_id)].tx_output(
                    ProtocolMessage(MessageType.ENTANGLEMENT_READY, self._tree_id, link_left_qubit_pos_lst, -1))
                charlie_res = None


class EntanglingProtocol(NodeProtocol):
    """a protocol that accepts new qubits if the node allows. i.e. copy from link layer memories to BSM ones"""

    def __init__(self, node: qPNode, tree_id: int):
        super().__init__(node, name=node.name + "_entangling_protocol_tree" + str(tree_id))
        self._tree_id = tree_id

    def add_protocol(self, protocol, isLeft: bool):
        self.add_subprotocol(protocol, "left_link_layer" if isLeft else "right_link_layer")

    def run(self):
        if self.node.neighbors[self._tree_id][0] != -1:
            self.add_signal(MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left")
            init_left_paired, init_inverse_left_paired = self.node.left_paired_with[self._tree_id], \
                                                         self.node.inverse_left_paired_with[self._tree_id]
        if self.node.neighbors[self._tree_id][1] != -1:
            self.add_signal(MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right")
            init_right_paired, init_inverse_right_paired = self.node.right_paired_with[self._tree_id], \
                                                           self.node.inverse_right_paired_with[self._tree_id]

        if self.node.neighbors[self._tree_id][0] == -1:
            # src node, does not have left qubit memory attached to link layer
            while True:
                # yield self.await_port_input(self.node.ports["qin_right"])
                yield self.await_signal(sender=self.subprotocols["right_link_layer"],
                                        signal_label=MessageType.ENTANGLEMENT_READY.name + str(
                                            self._tree_id) + "_right")
                self.node.right_paired_with[self._tree_id] = init_right_paired
                self.node.inverse_right_paired_with[self._tree_id] = init_inverse_right_paired
                net_right_qubit_pos_lst = []
                link_right_qubit_pos_lst = self.subprotocols["right_link_layer"].get_signal_result(
                    MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right")
                for link_right_qubit_pos in link_right_qubit_pos_lst:
                    net_right_qubit_pos = self.node.link_to_network_pos[self._tree_id][link_right_qubit_pos]
                    if self.node.entangling[self._tree_id][net_right_qubit_pos]:
                        # self.node.entangling[self._tree_id][net_right_qubit_pos] = False  # reset to NOT ACCEPTING qubits
                        # self.node.qmemory.pop(net_right_qubit_pos)
                        # self.node.qmemory.mem_positions[net_right_qubit_pos].in_use = True
                        self.node.qmemory.put(self.node.qmemory.pop(link_right_qubit_pos), net_right_qubit_pos)
                        logging.debug(f"{ns.sim_time():.1f}: (TREE{str(self._tree_id)}) node{self.node.ID} accepted"
                                      f" right qubit. qubits:({net_right_qubit_pos}-{link_right_qubit_pos})")
                        net_right_qubit_pos_lst.append(net_right_qubit_pos)
                self.send_signal(signal_label=MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right",
                                 result=net_right_qubit_pos_lst)
        elif self.node.neighbors[self._tree_id][1] == -1:
            # dst node, does not have right qubit memory attached ro link layer
            while True:
                # yield self.await_port_input(self.node.ports["qin_left"])
                yield self.await_signal(sender=self.subprotocols["left_link_layer"],
                                        signal_label=MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left")
                self.node.left_paired_with[self._tree_id] = init_left_paired
                self.node.inverse_left_paired_with[self._tree_id] = init_inverse_left_paired
                net_left_qubit_pos_lst = []
                link_left_qubit_pos_lst = self.subprotocols["left_link_layer"].get_signal_result(
                    MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left")
                for link_left_qubit_pos in link_left_qubit_pos_lst:
                    net_left_qubit_pos = self.node.link_to_network_pos[self._tree_id][link_left_qubit_pos]
                    if self.node.entangling[self._tree_id][net_left_qubit_pos]:
                        # self.node.entangling[self._tree_id][net_left_qubit_pos] = False  # reset to NOT ACCEPTING qubits
                        # self.node.qmemory.pop(left_qubit_pos)
                        # self.node.qmemory.mem_positions[left_qubit_pos].in_use = True
                        self.node.qmemory.put(self.node.qmemory.pop(link_left_qubit_pos), net_left_qubit_pos)
                        net_left_qubit_pos_lst.append(net_left_qubit_pos)
                        logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} "
                                      f"accepted left qubit. qubits:({net_left_qubit_pos}-{link_left_qubit_pos})")
                self.send_signal(signal_label=MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left",
                                 result=net_left_qubit_pos_lst)
        else:
            # intermediate node
            # if self.subprotocols['left_link_layer'] is self.subprotocols['right_link_layer']:
            #     # cases where a node is the left node for left & right edges. Subprotocols (messaging) are the same in this case
            #     while True:
            #         expr = yield self.await_signal(sender=self.subprotocols["left_link_layer"],
            #                                        signal_label=MessageType.ENTANGLEMENT_READY.name + str(self._tree_id)
            #                                              + "_left") | \
            #                      self.await_signal(sender=self.subprotocols["left_link_layer"],
            #                                       signal_label=MessageType.ENTANGLEMENT_READY.name +
            #                                                    str(self._tree_id) + "_right")
            #         self.node.right_paired_with[self._tree_id] = init_right_paired
            #         self.node.inverse_right_paired_with[self._tree_id] = init_inverse_right_paired
            #         self.node.left_paired_with[self._tree_id] = init_left_paired
            #         self.node.inverse_left_paired_with[self._tree_id] = init_inverse_left_paired
            #
            #         if expr.first_term.value:
            #             # signal is for left qubit
            #             link_left_qubit_pos_lst = self.subprotocols["left_link_layer"].get_signal_result(
            #                 MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left")
            #             net_left_qubit_pos_lst = []
            #             for link_left_qubit_pos in link_left_qubit_pos_lst:
            #                 net_left_qubit_pos = self.node.link_to_network_pos[self._tree_id][link_left_qubit_pos]
            #                 if self.node.entangling[self._tree_id][net_left_qubit_pos]:
            #                     # self.node.entangling[self._tree_id][net_left_qubit_pos] = False
            #                     # self.node.qmemory.pop(left_qubit_pos)
            #                     # self.node.qmemory.mem_positions[left_qubit_pos].in_use = True
            #                     self.node.qmemory.put(self.node.qmemory.pop(link_left_qubit_pos), net_left_qubit_pos)
            #                     net_left_qubit_pos_lst.append(net_left_qubit_pos)
            #                     logging.debug(
            #                         f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted left "
            #                         f"qubit.  qubits:({net_left_qubit_pos}-{link_left_qubit_pos})")
            #             self.send_signal(
            #                 signal_label=MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left",
            #                 result=net_left_qubit_pos_lst)
            #         elif expr.second_term.value:
            #             # signal is for right qubit
            #             link_right_qubit_pos_lst = self.subprotocols["left_link_layer"].get_signal_result(
            #                 MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right")
            #             net_right_qubit_pos_lst = []
            #             for link_right_qubit_pos in link_right_qubit_pos_lst:
            #                 net_right_qubit_pos = self.node.link_to_network_pos[self._tree_id][link_right_qubit_pos]
            #                 if self.node.entangling[self._tree_id][net_right_qubit_pos]:
            #                     # self.node.entangling[self._tree_id][net_right_qubit_pos] = False
            #                     # self.node.qmemory.pop(right_qubit_pos)
            #                     # self.node.qmemory.mem_positions[right_qubit_pos].in_use = True
            #                     self.node.qmemory.put(self.node.qmemory.pop(link_right_qubit_pos), net_right_qubit_pos)
            #                     net_right_qubit_pos_lst.append(net_right_qubit_pos)
            #                     logging.debug(
            #                         f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted right "
            #                         f"qubit.  qubits:({net_right_qubit_pos}-{link_right_qubit_pos})")
            #             self.send_signal(
            #                 signal_label=MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right",
            #                 result=net_right_qubit_pos_lst)
            # else:
            while True:
                # expr = yield self.await_port_input(self.node.ports["qin_left"]) | \
                #              self.await_port_input(self.node.ports["qin_right"])
                expr = yield self.await_signal(sender=self.subprotocols["left_link_layer"],
                                               signal_label=MessageType.ENTANGLEMENT_READY.name +
                                                            str(self._tree_id) + "_left") | \
                             self.await_signal(sender=self.subprotocols["right_link_layer"],
                                               signal_label=MessageType.ENTANGLEMENT_READY.name +
                                                            str(self._tree_id) + "_right")
                self.node.right_paired_with[self._tree_id] = init_right_paired
                self.node.inverse_right_paired_with[self._tree_id] = init_inverse_right_paired
                self.node.left_paired_with[self._tree_id] = init_left_paired
                self.node.inverse_left_paired_with[self._tree_id] = init_inverse_left_paired

                if expr.first_term.value:
                    # left qubit ready
                    link_left_qubit_pos_lst = self.subprotocols["left_link_layer"].get_signal_result(
                        MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left")
                    net_left_qubit_pos_lst = []
                    for link_left_qubit_pos in link_left_qubit_pos_lst:
                        net_left_qubit_pos = self.node.link_to_network_pos[self._tree_id][link_left_qubit_pos]
                        if self.node.entangling[self._tree_id][net_left_qubit_pos]:
                            # self.node.entangling[self._tree_id][net_left_qubit_pos] = False
                            # self.node.qmemory.pop(left_qubit_pos)
                            # self.node.qmemory.mem_positions[left_qubit_pos].in_use = True
                            self.node.qmemory.put(self.node.qmemory.pop(link_left_qubit_pos), net_left_qubit_pos)
                            net_left_qubit_pos_lst.append(net_left_qubit_pos)
                            logging.debug(
                                f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted left "
                                f"qubit. qubits ({net_left_qubit_pos}-{link_left_qubit_pos})")
                    self.send_signal(
                        signal_label=MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left",
                        result=net_left_qubit_pos_lst)
                # else:
                elif expr.second_term.value:
                    # right qubit ready
                    link_right_qubit_pos_lst = self.subprotocols["right_link_layer"].get_signal_result(
                        MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right")
                    net_right_qubit_pos_lst = []
                    for link_right_qubit_pos in link_right_qubit_pos_lst:
                        net_right_qubit_pos = self.node.link_to_network_pos[self._tree_id][link_right_qubit_pos]
                        if self.node.entangling[self._tree_id][net_right_qubit_pos]:
                            # self.node.entangling[self._tree_id][net_right_qubit_pos] = False
                            # self.node.qmemory.pop(net_right_qubit_pos)
                            # self.node.qmemory.mem_positions[net_right_qubit_pos].in_use = True
                            self.node.qmemory.put(self.node.qmemory.pop(link_right_qubit_pos), net_right_qubit_pos)
                            net_right_qubit_pos_lst.append(net_right_qubit_pos)
                            logging.debug(
                                f"{ns.sim_time():.1f}: (TREE{self._tree_id}) node{self.node.ID} accepted right "
                                f"qubit.")
                    self.send_signal(
                        signal_label=MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right",
                        result=net_right_qubit_pos_lst)


class CorrectionProtocol(NodeProtocol):  # DONE
    """A protocol class that does the correction when BSM results is ready"""

    def __init__(self, node: qPNode, messaging_protocol: MessagingProtocol, tree_id: int, main_root_id: int):
        super().__init__(node, node.name + "_correction_protocol_tree" + str(tree_id))
        self._tree_id = tree_id
        self.main_root_id = main_root_id
        self.add_subprotocol(messaging_protocol, "messaging_protocol")

    def run(self):
        entanglement_ready = True
        message = None
        # entangle_qubit_pos = self.node.qubit_mapping[self._tree_id][0][0]  # qubit slot where correction happens on

        while True:
            # evexp_entangle = self.await_mempos_in_use_toggle(self.node.qmemory, [entangle_qubit_pos])
            evexpr_meas_result = self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                   signal_label=MessageType.SWAPPING_CORRECTION.name +
                                                                str(self._tree_id))
            yield evexpr_meas_result  # | evexp_entangle
            message = self.subprotocols["messaging_protocol"].get_signal_result(
                MessageType.SWAPPING_CORRECTION.name + str(self._tree_id))
            # if expression.first_term.value:
            #     # measurement is ready from parent
            #     message = self.subprotocols["messaging_protocol"].get_signal_result(
            #         MessageType.SWAPPING_CORRECTION.name + str(self._tree_id))
            # else:
            #     # entangled qubit is ready
            #     entanglement_ready = True
            if message is not None and entanglement_ready:
                left_qubit_pos_lst, right_qubit_pos_lst = [], []
                for left_qubit_pos, new_left_pos, swapping_right_qubit_pos, res in message[1].data:
                    right_qubit_pos = self.node.inverse_left_paired_with[self._tree_id][swapping_right_qubit_pos]
                    self.node.left_paired_with[self._tree_id][right_qubit_pos] = new_left_pos
                    self.node.inverse_left_paired_with[self._tree_id][new_left_pos] = right_qubit_pos
                    try:
                        if res[0]:
                            self.node.qmemory.operate(ns.Z, right_qubit_pos)
                        if res[1]:
                            self.node.qmemory.operate(ns.X, right_qubit_pos)
                        left_qubit_pos_lst.append((new_left_pos, right_qubit_pos))
                        right_qubit_pos_lst.append(right_qubit_pos)
                    except Exception as e:
                        logging.error(f"{ns.sim_time():.1f}: CORRECTION failed at node {self.node.name} (for qubit"
                                      f" {right_qubit_pos}). "
                                      f"Start accepting EPs from link layer for tree{self._tree_id} rooted at "
                                      f"(node{message[0]})\tError msg:{e}")
                        continue
                logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) {self.node.name} (CORRECTION) received"
                             f" entangled qubit and corrections (for qubits-{right_qubit_pos_lst}) and sent ACK it "
                             f"to node{message[1].next_dst_id}! (for qubits-{left_qubit_pos_lst}")
                self.node.ports["cto_node" + str(message[1].next_dst_id)].tx_output(
                    ProtocolMessage(MessageType.CORRECTION_ACK, message[1].tree_id, left_qubit_pos_lst, -1))

                # reset for next round; send success signal to swapping protocol letting know left sub-tree is ready
                if int(message[0]) == self.main_root_id:
                    self.send_signal(signal_label=Signals.SUCCESS, result=(message[1].next_dst_id, left_qubit_pos_lst,
                                                                           right_qubit_pos_lst))
                    # entanglement_ready = False

                    # start accepting for the left qubit pos
                    # self.node.entangling[self._tree_id] = (True, self.node.entangling[self._tree_id][1])
                    # logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) set ENTANGLING to TRUE"
                    #               f" for LEFT position of node {str(self.node.ID)} because of correction "
                    #               f"failure.")
                    # send zero-latency msg to the left neighbor to start entangling
                    # self.node.ports["cto_neighbor" + str(self.node.neighbors[self._tree_id][0])].tx_output(
                    #     ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, None, -1))
                    # send a msg to root to restart the whole tree rooted at that root
                    # self.node.ports["cto_node" + str(message[0])].tx_output(ProtocolMessage(
                    #     MessageType.CORRECTION_FAILURE, self._tree_id, None, -1))
                message = None


class SwappingProtocol(NodeProtocol):
    """a protocol class that handles BSM operations and send results to rightest node"""

    def __init__(self, node: qPNode, tree_id: int, correction_protocol: CorrectionProtocol = None,
                 messaging_protocol: MessagingProtocol = None, entangling_protocol: EntanglingProtocol = None,
                 swapping_time_window: float = float('inf')):
        super().__init__(node, name=node.name + "_swapping_protocol_tree" + str(tree_id))
        self._tree_id = tree_id
        if correction_protocol:
            self.add_subprotocol(correction_protocol, "correction_left")
        if messaging_protocol:
            self.add_subprotocol(messaging_protocol, "messaging_protocol")
        if entangling_protocol:
            self.add_subprotocol(entangling_protocol, "entangling_protocol")
        self._swapping_time_window = swapping_time_window

    def run(self):
        qubit_pos_left_lst = None  # left swapping qubit
        qubit_pos_right_lst = None  # right swapping qubit
        left_last_time, right_last_time = -float('inf'), -float('inf')

        while True:
            if self.node.left_height[self._tree_id] == 1:
                # qubit directly comes from charlie
                evexpr_left = self.await_signal(sender=self.subprotocols["entangling_protocol"],
                                                signal_label=MessageType.ENTANGLEMENT_READY.name +
                                                             str(self._tree_id) + "_left")
            else:
                # qubit comes from lower level of the tree, should be after CORRECTION of the same node
                evexpr_left = self.await_signal(sender=self.subprotocols["correction_left"],
                                                signal_label=Signals.SUCCESS)

            if self.node.right_height[self._tree_id] == 1:
                evexpr_right = self.await_signal(sender=self.subprotocols["entangling_protocol"],
                                                 signal_label=MessageType.ENTANGLEMENT_READY.name +
                                                              str(self._tree_id) + "_right")
            else:
                # should wait for the rightest CORRECTION_ACK to receive
                evexpr_right = self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                 signal_label=MessageType.CORRECTION_ACK.name + str(self._tree_id))
            expression = yield evexpr_left | evexpr_right
            if expression.first_term.value:
                # left qubit is ready
                if self.node.left_height[self._tree_id] == 1:
                    qubit_pos_left_lst = self.subprotocols["entangling_protocol"].get_signal_result(
                        MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left")
                else:
                    leftest_id, _, qubit_pos_left_lst_tmp = self.subprotocols["correction_left"].get_signal_result(
                        Signals.SUCCESS)
                    if leftest_id == self.node.leftest_id[self._tree_id]:
                        qubit_pos_left_lst = qubit_pos_left_lst_tmp
                left_last_time = ns.sim_time()
            else:
                #
                if self.node.right_height[self._tree_id] == 1:
                    qubit_pos_right_lst = self.subprotocols["entangling_protocol"].get_signal_result(
                        MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right")
                    right_last_time = ns.sim_time()
                else:
                    sender_id, qubit_pos_right_lst_tmp = self.subprotocols["messaging_protocol"].get_signal_result(
                        MessageType.CORRECTION_ACK.name + str(self._tree_id))
                    if sender_id == self.node.rightest_id[self._tree_id]:
                        qubit_pos_right_lst = []
                        for right_qubit_pos, paired_qubit_pos in qubit_pos_right_lst_tmp:
                            qubit_pos_right_lst.append(right_qubit_pos)
                            self.node.right_paired_with[self._tree_id][right_qubit_pos] = paired_qubit_pos
                            self.node.inverse_right_paired_with[self._tree_id][paired_qubit_pos] = right_qubit_pos
                        right_last_time = ns.sim_time()

            if qubit_pos_left_lst and qubit_pos_right_lst:
                if abs(right_last_time - left_last_time) <= self._swapping_time_window:
                    # acceptable to do BSM, used for delft lp and sigcomm
                    res = []  # (left_qubit, right_qubit, m)
                    # yield self.await_timer(0 * self.node.bsm_time * 1e9)  # zero time for now
                    for left_qubit_pos, right_qubit_pos in zip(qubit_pos_left_lst, qubit_pos_right_lst):
                        try:
                            m = bell_measurement(self.node.qmemory, [left_qubit_pos, right_qubit_pos], False)
                            # if self.node.ID != 8:
                            if random() <= self.node.bsm_success_rate:
                                # successful bsm
                                res.append((left_qubit_pos, self.node.left_paired_with[self._tree_id][left_qubit_pos],
                                            right_qubit_pos, m))
                            else:
                                # unsuccessful bsm, should tell all the children to resume entangling again
                                # tools.tree_failure(self.node, self._tree_id)
                                pass
                        except Exception as e:
                            logging.error(f"{ns.sim_time():.1f}: (SWAPPING) Error happened at {self.node.name} (for "
                                          f"qubits ({left_qubit_pos}-{right_qubit_pos})).\n"
                                          f"Error message: {e}")
                            continue
                            # tools.tree_failure(self.node, self._tree_id)
                    if len(res) > 0:
                        logging.info(f"{ns.sim_time():.1f}:\t (TREE{self._tree_id}) {self.node.name}"
                                     " (SWAPPING) received entangled qubit, "
                                     f"measured qubits (for qubits {[r[0] for r in res]}-{[r[1] for r in res]})"
                                     f" & sending corrections to node"
                                     + str(self.node.rightest_id[self._tree_id]))
                        self.node.ports["cto_node" + str(self.node.rightest_id[self._tree_id])].tx_output(
                            ProtocolMessage(MessageType.SWAPPING_CORRECTION, self._tree_id, res,
                                            self.node.leftest_id[self._tree_id]))
                    qubit_pos_left_lst, qubit_pos_right_lst = None, None
                else:
                    # unacceptable timing
                    logging.info(f"{ns.sim_time():.1f}; (TREE{self._tree_id}) swapping was UNSUCCESSFUL"
                                 f" at node{self.node.ID} because left & right did not arrive at the same time."
                                 f" Time difference: {abs(right_last_time - left_last_time)} ns.")
                    # tools.tree_failure(self.node, self._tree_id)

                # reset for the next round
                # qubit_pos_left_lst = None
                # qubit_pos_right_lst = None
                # left_last_time, left_last_time = -float('inf'), -float('inf')
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
                 messaging_protocol: MessagingProtocol, tree_nodes_id: set, IS_TELEPORTING: bool,
                 entangling_protocol: EntanglingProtocol):
        super().__init__(node, name=node.name + "_teleportation_protocol_tree" + str(tree_id))
        # self.add_subprotocol(qubit_protocol, "qprotocol")
        if not is_path_one_hop:
            self.add_subprotocol(messaging_protocol, "messaging_protocol")
        else:
            self.add_subprotocol(entangling_protocol, "entangling_protocol")
        self._dst_id = dst_id
        self._is_path_one_hop = is_path_one_hop
        self._tree_id = tree_id
        if IS_TELEPORTING:
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
        right_qubit_pos_lst = None  # when the tree construction is ready
        while True:
            if self._is_path_one_hop:
                evexpr_entanglement = self.await_signal(sender=self.subprotocols["entangling_protocol"],
                                                        signal_label=MessageType.ENTANGLEMENT_READY.name +
                                                                     str(self._tree_id) + "_right")
            else:
                evexpr_entanglement = self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                        signal_label=MessageType.CORRECTION_ACK.name +
                                                                     str(self._tree_id))
            yield evexpr_entanglement
            if self._is_path_one_hop:
                right_qubit_pos_lst = self.subprotocols["entangling_protocol"].get_signal_result(
                    MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_right")
            else:
                sender_id, right_qubit_pos_lst_tmp = self.subprotocols["messaging_protocol"].get_signal_result(
                    MessageType.CORRECTION_ACK.name + str(self._tree_id))
                if sender_id == self._dst_id:
                    right_qubit_pos_lst = right_qubit_pos_lst_tmp
            if right_qubit_pos_lst:
                if self._IS_TELEPORTING:
                    # TODO this part has not completed
                    self.create_qubit()
                    yield self.await_timer(self.node.bsm_time * 1e9)
                    try:
                        m = bell_measurement(self.node.qmemory, [self._teleporting_qubit_pos,
                                                                 self.node.qubit_mapping[self._tree_id][0][1]], False)
                        if random() < self.node.bsm_success_rate:
                            # successful
                            self.node.ports["cto_node" + str(self._dst_id)].tx_output(
                                ProtocolMessage(MessageType.TELEPORTATION_CORRECTION, self._tree_id, m, -1))
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
                    self._tree_success_count += len(right_qubit_pos_lst)
                    logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) EPs ({len(right_qubit_pos_lst)}) created"
                                 f" between Alice and Bob.")
                    # self.restart_tree()
                right_qubit_pos_lst = None

    def restart_tree(self):
        # TODO this part should be changed if we want BSM and classical time counts
        self.node.entangling[self._tree_id] = (False, True)
        self.node.ports["cto_neighbor" + str(self.node.neighbors[self._tree_id][1])].tx_output(
            ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, None, -1))
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
                    ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, data, -1))

    def start(self):
        super().start()

    @property
    def tree_success_count(self) -> int:
        return self._tree_success_count


class TeleportationCorrectionProtocol(NodeProtocol):
    # TODO this class has not been completed for sigcomm as we don't want to send information from alice to bo
    """A protocol class that does the correction when BSM results is ready"""

    def __init__(self, node: qPNode, src_id: int, tree_id: int, is_path_one_hop,
                 messaging_protocol: MessagingProtocol,
                 correction_protocol: CorrectionProtocol, entangling_protocol: EntanglingProtocol,
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
        else:
            self.add_subprotocol(entangling_protocol, "entangling_protocol")

    @property
    def successful_count(self):
        return self._success_count

    def run(self):
        # port_meas = self.node.ports["cfrom_node" + str(self._src_id)]
        meas_results = None
        entanglement_ready = False

        while True:

            evexpr_meas_result = self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                   signal_label=MessageType.TELEPORTATION_CORRECTION.name +
                                                                str(self._tree_id))
            if self._is_path_one_hop:
                evexpr_entanglement = self.await_signal(sender=self.subprotocols["entangling_protocol"],
                                                        signal_label=MessageType.ENTANGLEMENT_READY.name +
                                                                     str(self._tree_id) + "_left")
            else:
                evexpr_entanglement = self.await_signal(sender=self.subprotocols["correction_protocol"],
                                                        signal_label=Signals.SUCCESS)
            expression = yield evexpr_meas_result | evexpr_entanglement
            if expression.first_term.value:
                sender_id, tmp_res = self.subprotocols["messaging_protocol"].get_signal_result(
                    MessageType.TELEPORTATION_CORRECTION.name + str(self._tree_id))
                if sender_id == self._src_id:
                    meas_results = tmp_res
            else:
                if self._is_path_one_hop:
                    left_qubit_pos_lst = self.subprotocols["entangling_protocol"].get_signal_result(
                        MessageType.ENTANGLEMENT_READY.name + str(self._tree_id) + "_left")
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
                    if DEBUG_MODE > 0:
                        print(
                            f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Bob (node {self.node.ID}) received entangled qubit"
                            f" and corrections from node{self._src_id}! "
                            f"Fidelity = {fidelity:.3f}")
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
            ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, None, -1))
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
                    ProtocolMessage(MessageType.START_ENTANGLING, self._tree_id, data, -1))
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
            # yield self.await_timer(1)
            _, src_qubit_pos, dst_qubit_pos = self.subprotocols["correction_protocol"].get_signal_result(Signals.SUCCESS)
            fidelity_values = []
            for src_pos, dst_pos in zip(src_qubit_pos, dst_qubit_pos):
                try:
                    src_qubit = self._src.qmemory.peek([src_pos[0]])[0]
                    dst_qubit = self._dst.qmemory.peek([dst_pos])[0]
                    fidelity = ns.qubits.fidelity([src_qubit, dst_qubit], ns.b00, squared=True)
                    fidelity_values.append(fidelity)
                    logging.info(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) ALice ({src_pos[0]}) and Bob{dst_pos} "
                                 f"are entangled!. Fidelity: {fidelity}")
                except Exception as e:
                    logging.debug(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Fidelity calculation failed!. "
                                  f"positions: {src_pos[0]}-{dst_pos}"
                                  f"Error msg: {e}")
                    continue
            return {"fidelity": sum(fidelity_values)/len(fidelity_values)}

        self._dc = DataCollector(calc_fidelity, include_time_stamp=False, include_entity_name=False)

    def run(self):
        if not self._is_path_one_hop:
            self._dc.collect_on(self.await_signal(sender=self.subprotocols["correction_protocol"],
                                                  signal_label=Signals.SUCCESS))
        else:
            # TODO for one hop is not finished
            # self.await_timer(1)
            # self._dc.collect_on(self.await_mempos_in_use_toggle(self._dst.qmemory,
            #                                                       [self._dst.qubit_mapping[self._tree_id][0][0]]))
            self._dc.collect_on(self.await_signal(sender=self.subprotocols["teleportation_protocol"],
                                                  signal_label=Signals.SUCCESS))

    @property
    def average_fidelity(self):
        try:
            return self._dc.dataframe.count()[0], self._dc.dataframe.agg("mean")[0]
        except IndexError as e:
            return 0, 0
