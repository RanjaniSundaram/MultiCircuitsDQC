import operator
import sys

sys.path.append("..")

from netsquid.protocols import NodeProtocol, Signals, Protocol
from protocols.predistribution.component_bases import qPNode, Charlie
from commons.tools import bell_measurement
import netsquid as ns
from netsquid.nodes import Node
from functools import reduce
from collections import namedtuple
from enum import Enum
from random import random, choice
from netsquid.components import QuantumMemory
import logging
import datetime
import os
from netsquid.util.datacollector import DataCollector
from typing import List
from collections import defaultdict

ProtocolMessage = namedtuple("ProtocolMessage", "type, sd_id, tree_id, data, next_dst_id")
# sd_id = 0 for superlinks

DEBUG_MODE = 0  # 0: no printing, 1: high level details, 2: deep details

if not os.path.exists("protocol_logs"):
    os.makedirs("protocol_logs")
logging.basicConfig(filename="protocol_logs/log_pre_" + datetime.datetime.now().strftime('_%Y%m_%d%H_%M') + ".log",
                    level=logging.INFO)


class MessageType(Enum):
    SWAPPING_CORRECTION = 1
    START_ENTANGLING = 2
    CORRECTION_ACK = 3
    TELEPORTATION_CORRECTION = 4
    LINK_CORRECTION = 5
    ENTANGLEMENT_READY = 6  # when link layer entanglement is ready
    CORRECTION_FAILURE = 7
    START_ENTANGLEMENT_GENERATING = 8  # a signal to request link layer to START entanglement generating
    STOP_ENTANGLEMENT_GENERATING = 9  # a signal to request link layer to STOP entanglement generating
    SL_SWAPPING_FAILURE_1 = 10  # a signal to tell the other end of a BSM failure at the first SL to move its pointer
    SL_SWAPPING_FAILURE_2 = 11  # a signal to tell the other end of a BSM failure at the second SL to move its pointer
    SL_NOT_PRESENT = 12
    RESTART_SL = 13


class tools:
    @staticmethod
    def tree_failure(node: qPNode, key: str):
        sd_id, tree_id = int(key.split("_")[0]), int(key.split("_")[1])
        logging.info(f"{ns.sim_time():.1f}; (SD-{sd_id})(TREE{tree_id}) restart tree located"
                     f" at node{node.ID}.")
        node.entangling[key] = (True, True)  # node itself should start entangling
        # send instant message to its neighbors
        for neighbor in node.neighbors[key]:
            node.ports["cto_neighbor" + str(neighbor)].tx_output(ProtocolMessage(
                MessageType.START_ENTANGLING, sd_id, tree_id, None, -1))
        for child_id in node.children_id[key]:
            if (child_id == node.leftest_id[key] and
                node.leftest_id[key] == node.neighbors[key][0]) or \
                    (child_id == node.rightest_id[key] and
                     node.rightest_id[key] == node.neighbors[key][1]):
                continue
            data = "both"
            # if child_id != self.node.neighbors[self._tree_id][0] and \
            #         child_id != self.node.neighbors[self._tree_id][1]:

            if child_id == node.leftest_id[key]:
                data = "right"
            elif child_id == node.rightest_id[key]:
                data = "left"
            if data == "both" and child_id == node.neighbors[key][0]:
                data = "left"
            elif data == "both" and child_id == node.neighbors[key][1]:
                data = "right"
            node.ports["cto_node" + str(child_id)].tx_output(
                ProtocolMessage(type=MessageType.START_ENTANGLING, sd_id=sd_id, tree_id=tree_id,
                                data=data, next_dst_id=-1))
            logging.debug(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) sending ENTANGLING signal to node"
                          f" {str(child_id)} for {data} positions.")


class MessagingProtocol(NodeProtocol):
    """a protocol that handles received messages from other nodes"""

    def __init__(self, node: qPNode):
        super().__init__(node, name=node.name + "_messaging_protocol")

    def add_signals(self, sd_id: int, tree_id: int):
        self.add_signal(f"{MessageType.START_ENTANGLING.name}_{sd_id}_{tree_id}")
        self.add_signal(f"{MessageType.SWAPPING_CORRECTION.name}_{sd_id}_{tree_id}")
        self.add_signal(f"{MessageType.TELEPORTATION_CORRECTION.name}_{sd_id}_{tree_id}")
        self.add_signal(f"{MessageType.CORRECTION_ACK.name}_{sd_id}_{tree_id}")
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}_left")
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}_right")
        if sd_id == 0:
            self.add_signal(f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_{sd_id}_{tree_id}")
            self.add_signal(f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_{sd_id}_{tree_id}")

    def add_correction_ack_signal(self, sd_id: int, tree_id: int):
        # when an SL is located at the left end of a path, it's necessary to add this signal to left node of the SL's endpoint
        self.add_signal(f"{MessageType.CORRECTION_ACK.name}_{sd_id}_{tree_id}")

    def add_swapping_correction_signal(self, sd_id: int, tree_id: int):
        # when an SL is located at the end right of a path, it's necessary to add this signal to right node of the SL's endpoint
        self.add_signal(f"{MessageType.SWAPPING_CORRECTION.name}_{sd_id}_{tree_id}")

    def add_sl_end_signals(self, sd_id: int, tree_id: int):
        self.add_signal(f"{MessageType.SL_SWAPPING_FAILURE_1.name}_{sd_id}_{tree_id}")
        self.add_signal(f"{MessageType.SL_SWAPPING_FAILURE_2.name}_{sd_id}_{tree_id}")
        self.add_signal(f"{MessageType.RESTART_SL.name}_{sd_id}_{tree_id}")
        # self.add_signal(f"{MessageType.SL_NOT_PRESENT.name}_{sd_id}_{tree_id}")

    def add_non_sl_signals(self, sd_id: int):
        self.add_signal(f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_{sd_id}")  # TODO add only once
        self.add_signal(f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_{sd_id}")  # TODO add only once

    def add_link_correction_signal(self, sd_id: int, tree_id: int, left_node_id: int):
        self.add_signal(f"{MessageType.LINK_CORRECTION.name}_{sd_id}_{tree_id}_{left_node_id}")

    def run(self):
        last_sd_started = -1
        last_sd_orig_entangling_status = []
        while True:
            expr = yield reduce(operator.or_, [self.await_port_input(self.node.ports[port_name])
                                               for port_name in self.node.ports if port_name.startswith("cfrom_")])
            for triggered_event in expr.triggered_events:
                for obj in triggered_event.source.input_queue:
                    received_port_name = obj[0].source.name
                    for message in obj[1].items:
                        if message.type == MessageType.SWAPPING_CORRECTION:
                            # send signal and BSM results and leftest id (next_dst_id) to correction protocol
                            self.send_signal(signal_label=f"{MessageType.SWAPPING_CORRECTION.name}_{message.sd_id}_"
                                                          f"{message.tree_id}",
                                             result=(received_port_name.split("cfrom_node")[1], message))
                            # (received_port_name.split("cfrom_node")[1]
                        elif message.type == MessageType.CORRECTION_ACK:
                            self.send_signal(signal_label=f"{MessageType.CORRECTION_ACK.name}_{message.sd_id}_"
                                                          f"{message.tree_id}",
                                             result=int(received_port_name.split("cfrom_node")[1]))
                        elif message.type == MessageType.TELEPORTATION_CORRECTION:
                            self.send_signal(signal_label=f"{MessageType.TELEPORTATION_CORRECTION.name}_{message.sd_id}"
                                                          f"_{message.tree_id}",
                                             result=[int(received_port_name.split("cfrom_node")[1]), message.data])
                        elif message.type == MessageType.ENTANGLEMENT_READY:
                            sender_id = int(received_port_name.split("cfrom_neighbor")[1])
                            if sender_id == self.node.neighbors[f"{message.sd_id}_{message.tree_id}"][0]:
                                direction = "left"
                            elif sender_id == self.node.neighbors[f"{message.sd_id}_{message.tree_id}"][1]:
                                direction = "right"
                            else:
                                logging.error(f"left_node: {sender_id} is not a valid neighbor of {self.node.name}")
                                continue
                            self.send_signal(signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_{message.sd_id}_"
                                                          f"{message.tree_id}_{direction}")
                        elif message.type == MessageType.LINK_CORRECTION:
                            self.send_signal(f"{MessageType.LINK_CORRECTION.name}_{message.sd_id}_{message.tree_id}_"
                                             f"{message.next_dst_id}", message.data)
                        elif message.type == MessageType.CORRECTION_FAILURE:
                            tools.tree_failure(self.node, f"{message.sd_id}_{message.tree_id}")
                        elif message.type == MessageType.START_ENTANGLING:
                            is_neighbor = received_port_name.split("_")[1].startswith("neighbor")
                            key = f"{message.sd_id}_{message.tree_id}"
                            if is_neighbor:
                                # signal comes from neighbors
                                left_neighbor = int(received_port_name.split("cfrom_neighbor")[1]) == \
                                                self.node.neighbors[key][0]
                                if left_neighbor:
                                    # left neighbor
                                    self.node.entangling[key] = (True, self.node.entangling[key][1])
                                    logging.debug(f"{ns.sim_time():.1f}: (SD-{message.sd_id})(TREE{message.tree_id}) "
                                                  f"set ENTANGLING to TRUE"
                                                  f" for LEFT position of node {str(self.node.ID)} because of left "
                                                  f"neighbor (node{received_port_name.split('cfrom_neighbor')[1]})")
                                else:
                                    self.node.entangling[key] = (self.node.entangling[key][0], True)
                                    logging.debug(f"{ns.sim_time():.1f}: (SD-{message.sd_id})(TREE{message.tree_id})"
                                                  f" set ENTANGLING to TRUE"
                                                  f" for RIGHT position of node {str(self.node.ID)} because of right"
                                                  f" neighbor (node{received_port_name.split('cfrom_neighbor')[1]})")
                            else:
                                # comes from parent
                                if self.node.qubit_mapping[key][0][0] != -1 and not self.node.entangling[key][0] and \
                                        (message.data == "left" or message.data == "both"):
                                    # left qubit is part of the tree and was not set to entangling
                                    self.node.entangling[key] = (True, self.node.entangling[key][1])
                                    logging.debug(f"{ns.sim_time():.1f}: (SD-{message.sd_id})(TREE{message.tree_id})"
                                                  f" set ENTANGLING to TRUE"
                                                  f" for LEFT position of node {str(self.node.ID)} because of node"
                                                  f"{received_port_name.split('cfrom_node')[1]}")
                                    # send a message to left neighbor
                                    self.node.ports[f"cto_neighbor{self.node.neighbors[key][0]}"].tx_output(message)
                                if self.node.qubit_mapping[key][0][1] != -1 and not self.node.entangling[key][1] and \
                                        (message.data == "right" or message.data == "both"):
                                    # right qubit is part of the tree and was not set to entangling
                                    self.node.entangling[key] = (self.node.entangling[key][0], True)
                                    logging.debug(f"{ns.sim_time():.1f}: (SD-{message.sd_id})(TREE{message.tree_id}) "
                                                  f"set ENTANGLING to TRUE"
                                                  f" for RIGHT position of node {str(self.node.ID)} because of node"
                                                  f"{received_port_name.split('cfrom_node')[1]}")
                                    # send a message to right neighbor
                                    self.node.ports[f"cto_neighbor{self.node.neighbors[key][1]}"].tx_output(message)
                            logging.debug(
                                f"{ns.sim_time():.1f}: (SD-{message.sd_id})(TREE{message.tree_id}): node{self.node.ID} "
                                f"entangling: {self.node.entangling[key]}")
                        elif message.type == MessageType.STOP_ENTANGLEMENT_GENERATING:
                            if message.sd_id != 0:
                                self.send_signal(f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_{message.sd_id}")
                                if message.sd_id == last_sd_started:
                                    for i in range(len(last_sd_orig_entangling_status)):
                                        self.node.entangling[last_sd_orig_entangling_status[i][0]] = \
                                            last_sd_orig_entangling_status[i][1]
                                        self.node.accepting_eps[last_sd_orig_entangling_status[i][0]] = False
                                else:
                                    logging.error("STOP generation signal sent without starting it before")
                            else:  # for an SL
                                key = f"{message.sd_id}_{message.tree_id}"
                                if key in self.node.is_left_sl:  # reached to right end, send a message back to left end
                                    self.node.ports[f"cto_sl_node{self.node.other_sl_id[key]}"].tx_output(message)
                                self.send_signal(f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_{key}")
                                self.node.accepting_eps[key] = False
                        elif message.type == MessageType.START_ENTANGLEMENT_GENERATING:
                            if message.sd_id != 0:
                                self.send_signal(f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_{message.sd_id}")
                                last_sd_started = message.sd_id
                                last_sd_orig_entangling_status = []
                                for key, ent_status in self.node.entangling.items():
                                    if int(key.split("_")[0]) == message.sd_id:
                                        last_sd_orig_entangling_status.append((key, ent_status))
                                        self.node.accepting_eps[key] = True
                            else:
                                key = f"{message.sd_id}_{message.tree_id}"
                                if key in self.node.is_left_sl:  # reached to right end, send a message back to left end
                                    self.node.ports[f"cto_sl_node{self.node.other_sl_id[key]}"].tx_output(message)
                                self.send_signal(f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_{key}")
                                self.node.accepting_eps[key] = True
                        elif message.type == MessageType.SL_SWAPPING_FAILURE_1:
                            self.send_signal(f"{MessageType.SL_SWAPPING_FAILURE_1.name}_{message.sd_id}_"
                                             f"{message.tree_id}")
                        elif message.type == MessageType.SL_SWAPPING_FAILURE_2:
                            self.send_signal(f"{MessageType.SL_SWAPPING_FAILURE_2.name}_{message.sd_id}_"
                                             f"{message.tree_id}", message.data)
                        elif message.type == MessageType.RESTART_SL:
                            self.send_signal(f"{MessageType.RESTART_SL.name}_{message.sd_id}_{message.tree_id}")
                        # elif message.type == MessageType.SL_NOT_PRESENT:
                        #     self.send_signal(f"{MessageType.SL_NOT_PRESENT.name}_{message.sd_id}_"
                        #                      f"{message.tree_id}", message.data)
                        else:
                            logging.error(f"{ns.sim_time():.1f}: (SD-{message.sd_id}(TREE{message.tree_id}) Unknown "
                                          f"message {message.type} was received at node{self.node.ID}.")
                    self.node.ports[received_port_name].rx_input()


class EntanglementGenerator(NodeProtocol):
    """A protocol that generates a pair of EPs, save one in node's qmemory and send the other to charlie to
    do optical bsm. rate is 1/s"""

    def __init__(self, node: qPNode, rate: float('inf'), isLeft: bool, charlie_id: str,
                 link_layer_qmem_pos: int, sd_id: int, tree_id: int, IS_SL: bool = False,
                 messaging_protocol: MessagingProtocol = None):
        super(EntanglementGenerator, self).__init__(node, name=node.name + ("_left" if isLeft else "_right") +
                                                               "_generator_for" + f"_sd{sd_id}_tree{tree_id}")
        self._isLeft = isLeft
        self._IS_SL = IS_SL
        self._rate = rate
        self._sd_id = sd_id
        self._tree_id = tree_id
        self._charlie_id = charlie_id
        self._link_layer_qmem_pos = link_layer_qmem_pos
        # if not IS_SL:
        #     self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self.add_subprotocol(messaging_protocol, "messaging_protocol")

    def run2(self):
        while True:
            if not self._IS_SL:
                yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                        signal_label=f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_{self._sd_id}")
            while True:
                # wait (nanoseconds) or wait for stop signal
                if self._IS_SL:
                    yield self.await_timer(1.0 / self._rate * 1e9)
                    self.send_qubit()
                else:
                    expr = yield self.await_timer(1.0 / self._rate * 1e9) | \
                                 self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                   signal_label=f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_"
                                                                f"{self._sd_id}")
                    if expr.first_term.value:
                        self.send_qubit()
                    else:
                        break
                # break

    def run(self):
        if self._IS_SL:
            while True:
                expr = yield self.await_timer(1.0 / self._rate * 1e9) | \
                             self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                               signal_label=f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_"
                                                            f"{self._sd_id}_{self._tree_id}")
                if expr.first_term.value:
                    self.send_qubit()
                else:
                    yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                            signal_label=f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_"
                                                         f"{self._sd_id}_{self._tree_id}")
        else:
            while True:
                yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                        signal_label=f"{MessageType.START_ENTANGLEMENT_GENERATING.name}_{self._sd_id}")
                while True:
                    expr = yield self.await_timer(1.0 / self._rate * 1e9) | \
                                 self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                   signal_label=f"{MessageType.STOP_ENTANGLEMENT_GENERATING.name}_"
                                                                f"{self._sd_id}")
                    if expr.first_term.value:
                        self.send_qubit()
                    else:
                        break

    def send_qubit(self):
        qout_port = self.node.ports["qout_" + ("left" if self._isLeft else "right") + self._charlie_id]
        q1, q2 = ns.qubits.create_qubits(2, no_state=True)
        q1.name += f"{self._sd_id}_{self._tree_id}_{self.node.ID}_{self._link_layer_qmem_pos}"
        q2.name = f"{self._sd_id}_{self._tree_id}" + ("_right" if self._isLeft else "_left")
        ns.qubits.assign_qstate([q1, q2], ns.b00)
        self.node.qmemory.put(q1, positions=self._link_layer_qmem_pos, replace=True)
        qout_port.tx_output(q2)
        logging.debug(f"{ns.sim_time():.1f}: (SD-{self._sd_id})(TREE{self._tree_id}) "
                      f"qubit was generated at {self.name}")


class CharlieProtocol(NodeProtocol):
    """Charlie protocol that is responsible to do photonics BSM based on receiving from its left and right ports"""

    def __init__(self, node: Charlie, bsm_time_window: float = 10):
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
                    sd_id, tree_id = int(left_qubit.name.split("_")[0]), int(left_qubit.name.split("_")[1])
                    key = f"{sd_id}_{tree_id}"
                    left_time = ns.sim_time()
                    right_qubit, right_time = None, -float('inf')
                    if key in self.node.qubit_data:
                        right_qubit, right_time = self.node.qubit_data[key][1]
                    self.node.qubit_data[key] = ((left_qubit, left_time), (right_qubit, right_time))
            else:
                for right_qubit in qright_port.rx_input().items:
                    sd_id, tree_id = int(right_qubit.name.split("_")[0]), int(right_qubit.name.split("_")[1])
                    key = f"{sd_id}_{tree_id}"
                    right_time = ns.sim_time()
                    left_qubit, left_time = None, -float('inf')
                    if key in self.node.qubit_data:
                        left_qubit, left_time = self.node.qubit_data[key][0]
                    self.node.qubit_data[key] = ((left_qubit, left_time), (right_qubit, right_time))

            # time to decide if BSM is possible
            results = {}
            for key in self.node.qubit_data.keys():  # one sd_id will be requested each time
                left_qubit, left_time = self.node.qubit_data[key][0]
                right_qubit, right_time = self.node.qubit_data[key][1]
                if left_qubit is not None and right_qubit is not None and \
                        abs(left_time - right_time) < self._bsm_time_window:
                    logging.debug(f"{ns.sim_time():.1f}: ({key}) both link layer qubits "
                                  f"successfully received  at {self.node.name}")
                    qmemory = QuantumMemory("tmp" + key, num_positions=2)
                    qmemory.put([left_qubit, right_qubit], positions=[0, 1])
                    results[key] = bell_measurement(qmemory, positions=[0, 1], PHYSICAL=False)
            if len(results) > 0:
                yield self.await_timer(self.node.bsm_time * 1e9)
            for key, res in results.items():
                if random() < self.node.bsm_success_rate:
                    logging.debug(f"{ns.sim_time():.1f}: ({key}) BSM was successfully done at "
                                  f"{self.node.name}")
                    sd_id, tree_id = key.split("_")
                    cright_port.tx_output(ProtocolMessage(type=MessageType.LINK_CORRECTION, data=res,
                                                          tree_id=int(tree_id), sd_id=int(sd_id),
                                                          next_dst_id=self.node._left_node_id))
                self.node.qubit_data[key] = ((None, -float('inf')), (None, -float('inf')))


class LinkCorrectionProtocol(NodeProtocol):
    """a protocol that waits for response from Charlie to do the BSM correction and entangle two ends and send a
    message to left end"""

    def __init__(self, node: qPNode, messaging_protocol: MessagingProtocol):
        super(LinkCorrectionProtocol, self).__init__(node, name=node.name + "_link_correction_protocol")
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self._labels = set()
        self._link_mem_positions = {}
        # self._left_node_ids = {}  # left_node_id, key: sd_id-tree_id

    def add_signal_label(self, sd_id: int, tree_id: int, link_qmem_pos: int, left_node_id: int):
        key = f"{sd_id}_{tree_id}"
        self._link_mem_positions[f"{key}_{left_node_id}"] = link_qmem_pos
        # self._left_node_ids[f"{key}"] = left_node_id
        if f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}_left" not in self.signals:
            self.add_signal(
                f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}_left")  # for notifying qubit is ready
        if f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}_right" not in self.signals:
            self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}_right")
        self._labels.add(f"{MessageType.LINK_CORRECTION.name}_{sd_id}_{tree_id}_{left_node_id}")
        # to get notified EP generation from MessagingProtocol

    def run(self):
        while True:
            expr = yield reduce(operator.or_, [self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                                                 signal_label=label) for label in self._labels])
            triggered_res = expr.triggered_events[0].source.get_signal_by_event(expr.triggered_events[0])
            triggered_label, charlie_res = triggered_res.label, triggered_res.result
            sd_id, tree_id, left_node_id = triggered_label.split("_")[2:]
            key = f"{sd_id}_{tree_id}"
            link_qmem_pos = self._link_mem_positions[f"{key}_{left_node_id}"]
            try:
                if charlie_res[0]:
                    self.node.qmemory.operate(ns.Z, link_qmem_pos)
                if charlie_res[1]:
                    self.node.qmemory.operate(ns.X, link_qmem_pos)
                if self.node.neighbors[key][0] == int(left_node_id):  # node is located in the left along the path
                    direction = "left"
                elif self.node.neighbors[key][1] == int(left_node_id):  # node is located in right along the path
                    direction = "right"
                else:
                    logging.error(f"left_node: {left_node_id} is not a valid neighbor of {self.node.name}")
                    continue
                self.send_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}_{direction}")
                self.node.ports["cto_neighbor" + str(left_node_id)].tx_output(
                    ProtocolMessage(MessageType.ENTANGLEMENT_READY, int(sd_id), int(tree_id), None, -1))
                logging.debug(f"{ns.sim_time():.1f}: (SD-{sd_id}(TREE{tree_id}) {self.node.name} "
                              f"received results from charlie and did "
                              f"the CORRECTION. send an ACK to neighbor {left_node_id}")
                logging.debug(f"(SD-{sd_id})(TREE{tree_id}) Entangling situation: "
                              f"{self.node.entangling[key]}")
            except Exception as e:
                logging.error(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) "
                              f"Error happened at (link layer-waiting"
                              f" for new one) node "
                              f"{self.node.name}. \tError Msg: {e}")


class EntanglingProtocol(NodeProtocol):
    """a protocol that accepts new qubits if the node allows. i.e. copy from link layer memories to BSM ones"""

    def __init__(self, node: qPNode, messaging: MessagingProtocol, link_correction: LinkCorrectionProtocol = None):
        super().__init__(node, name=f"{node.name}_entangling_protocol")
        if link_correction is not None:
            self.add_subprotocol(link_correction, "link_correction")
        self.add_subprotocol(messaging, "messaging")
        self._protocol_labels = set()

    def add_signal_label(self, sd_id: int, tree_id: int, isLeft: bool, isMessaging: bool):
        self._protocol_labels.add(("messaging" if isMessaging else "link_correction",
                                   f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_"
                                   f"{tree_id}" + ("_left" if isLeft else "_right")))
        # (protocol: either messaging_protocol or link_correction_protocol,
        #  label)
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}" + ("_left" if isLeft else "_right"))

    def run(self):
        while True:
            expr = yield reduce(operator.or_, [self.await_signal(sender=self.subprotocols[x[0]], signal_label=x[1])
                                               for x in self._protocol_labels])
            triggered_label = expr.triggered_events[0].source.get_signal_by_event(expr.triggered_events[0]).label
            sd_id, tree_id, direction = triggered_label.split("_")[2:]
            direction_idx = 0 if direction == "left" else 1
            key = f"{sd_id}_{tree_id}"
            if self.node.qubit_mapping[key][0][direction_idx] == -1:
                # check if the qubit mapping and the triggered event do not match
                logging.error(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) "
                              f"link layer notification received for {direction} where it has not been assigned for"
                              f"this path")
                continue
            if self.node.accepting_eps[key] and self.node.entangling[key][
                direction_idx]:  # if EP is allowed for this node and path
                qubit_net_pos, qubit_link_pos = self.node.qubit_mapping[key][0][direction_idx], \
                                                self.node.qubit_mapping[key][1][direction_idx]
                other_qmem_status = self.node.entangling[key][0 if direction_idx else 1]
                self.node.entangling[key] = (other_qmem_status, False) if direction_idx else (False, other_qmem_status)
                # self.node.qmemory.pop(qubit_net_pos)
                # self.node.qmemory.mem_positions[qubit_net_pos].in_use = True
                self.send_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}_{direction}")
                self.node.qmemory.put(self.node.qmemory.pop(qubit_link_pos), qubit_net_pos)
                logging.debug(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) "
                              f"node{self.node.ID} accepted {direction} qubit.")


class CorrectionProtocol(NodeProtocol):
    """A protocol class that does the correction when BSM results is ready"""

    def __init__(self, node: qPNode, messaging_protocol: MessagingProtocol, entangling_protocol: EntanglingProtocol,
                 NUMBER_OF_SL_MEMORIES: int):
        super().__init__(node, node.name + "_correction_protocol")
        self._main_root_id = {}
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self.add_subprotocol(entangling_protocol, "entangling_protocol")
        self._protocol_labels = {}
        self._is_node_dst_right_sl = {}
        self._NUMBER_OF_SL_MEMORIES = NUMBER_OF_SL_MEMORIES

    def add_signal_label(self, sd_id: int, tree_id: int, IS_NON_SL: bool = True):
        key = f"{sd_id}_{tree_id}"
        self._protocol_labels[f"{MessageType.SWAPPING_CORRECTION.name}_{key}"] = "messaging_protocol"
        if IS_NON_SL:
            self._protocol_labels[f"{MessageType.ENTANGLEMENT_READY.name}_{key}_left"] = "entangling_protocol"
        # self._protocol_labels.append(("messaging_protocol", f"{MessageType.SWAPPING_CORRECTION.name}_{key}"))
        # self._protocol_labels.append(("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_{key}_left"))
        if f"{MessageType.CORRECTION_ACK.name}_{key}" not in self.signals:
            self.add_signal(f"{MessageType.CORRECTION_ACK.name}_{key}")

    def add_main_root_id(self, sd_id: int, tree_id: int, root_id):
        self._main_root_id[f"{sd_id}_{tree_id}"] = root_id

    def node_is_dst(self, sd_id: int, tree_id: int, sl_tree_id: int, sl_sd_id: int = 0):
        # used to move SL's saved pos pointer
        self._is_node_dst_right_sl[f"{sd_id}_{tree_id}"] = f"{sl_sd_id}_{sl_tree_id}"

    def run(self):
        messages = {}
        entanglement_ready = set()

        if len(self._protocol_labels) > 0:
            while True:
                expr = yield reduce(operator.or_, [self.await_signal(sender=self.subprotocols[protocol],
                                                                     signal_label=label)
                                                   for label, protocol in self._protocol_labels.items()])
                triggered_event = expr.triggered_events[0].source.get_signal_by_event(expr.triggered_events[0])
                triggered_label, triggered_res = triggered_event.label, triggered_event.result
                event_type = triggered_label.split("_")[0]
                sd_id, tree_id = triggered_label.split("_")[2:4]
                key = f"{sd_id}_{tree_id}"
                if event_type == "SWAPPING":
                    messages[key] = triggered_res
                elif event_type == "ENTANGLEMENT":  # ENTANGLING
                    # entangled qubit is ready
                    entanglement_ready.add(key)
                else:
                    logging.error(
                        f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) {self.node.name} Error happened at "
                        f"CorrectionProtocol.")
                    continue
                if key in messages and (key in entanglement_ready or key in self.node.left_sl_id):
                    message = messages[key]
                    if key in self.node.left_sl_id:
                        sl_key = self.node.left_sl_id[key]  # left qubit is the rightmost position of an SL
                        entangle_qubit_pos = next(iter(self.node.sl_saved_pos[sl_key]))
                    else:
                        entangle_qubit_pos = self.node.qubit_mapping[key][0][
                            0]  # qubit slot where correction happens on
                    try:
                        if message[1].data[0]:
                            self.node.qmemory.operate(ns.Z, entangle_qubit_pos)
                        if message[1].data[1]:
                            self.node.qmemory.operate(ns.X, entangle_qubit_pos)
                        self.node.ports["cto_node" + str(message[1].next_dst_id)].tx_output(
                            ProtocolMessage(MessageType.CORRECTION_ACK, sd_id, tree_id, None, -1))
                        logging.info(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) {self.node.name} "
                                     f"(CORRECTION) received"
                                     f" entangled qubit and corrections and sent ACK it to node{message[1].next_dst_id}!")
                        # reset for next round; send success signal to swapping protocol letting know left sub-tree is ready
                        if int(message[0]) == self._main_root_id[key]:
                            self.send_signal(f"{MessageType.CORRECTION_ACK.name}_{key}", message[1].next_dst_id)
                            if key in entanglement_ready:
                                entanglement_ready.remove(key)  # resetting for next round
                            # path reaches to its end and its right side of an SL, should restart SL if need it
                            if key in self._is_node_dst_right_sl:
                                sl_key = self._is_node_dst_right_sl[key]
                                self.node.ports[f"cto_node{self.node.other_sl_id[sl_key]}"].tx_output(
                                    ProtocolMessage(MessageType.RESTART_SL, int(sl_key.split("_")[0]),
                                                    int(sl_key.split("_")[1]), None, -1))
                                if len(self.node.sl_saved_pos[
                                           self._is_node_dst_right_sl[key]]) == self._NUMBER_OF_SL_MEMORIES and \
                                        sl_key in self.node.is_right_sl:
                                    SwappingProtocol.sl_generating(self.node, sl_key, True)
                                    SwappingProtocol.restart_tree(self.node, sl_key)  # leftmost node of an SL
                                logging.info(
                                    f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) (SL-{sl_key}) Bob "
                                    f"is an end of an SL. Should drop the proper EP "
                                    f"(pos: {next(iter(self.node.sl_saved_pos[sl_key]))})")
                                self.node.sl_saved_pos[sl_key].pop(next(iter(self.node.sl_saved_pos[sl_key])))
                    except Exception as e:
                        logging.error(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) CORRECTION failed at "
                                      f"node {self.node.name}. Start accepting EPs from link layer rooted at "
                                      f"(node{message[0]})\tError msg:{e}")
                        # start accepting for the left qubit pos
                        self.node.entangling[key] = (True, self.node.entangling[key][1])
                        logging.debug(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{self._tree_id}) set ENTANGLING to TRUE"
                                      f" for LEFT position of node {str(self.node.ID)} because of correction "
                                      f"failure.")
                        # send zero-latency msg to the left neighbor to start entangling
                        self.node.ports["cto_neighbor" + str(self.node.neighbors[key][0])].tx_output(
                            ProtocolMessage(MessageType.START_ENTANGLING, sd_id, tree_id, None, -1))
                        # send a msg to root to restart the whole tree rooted at that root
                        self.node.ports["cto_node" + str(message[0])].tx_output(ProtocolMessage(
                            MessageType.CORRECTION_FAILURE, sd_id, tree_id, None, -1))
                    messages.pop(key)
        else:
            yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                    signal_label=Signals.SUCCESS)  # dummy signal to avoid ending other protocols


class SwappingProtocol(NodeProtocol):
    """a protocol class that handles BSM operations and send results to rightest node"""

    def __init__(self, node: qPNode, entangling_protocol: EntanglingProtocol, correction_protocol: CorrectionProtocol,
                 messaging_protocol: MessagingProtocol, NUMBER_OF_SL_MEMORIES: int,
                 swapping_time_window: float = float('inf')):
        super().__init__(node, name=node.name + "_swapping_protocol")
        self.add_subprotocol(correction_protocol, "correction_left")
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self.add_subprotocol(entangling_protocol, "entangling_protocol")
        self._swapping_time_window = swapping_time_window
        self._protocol_labels = set()
        self._left_right_tree_root = {}  # if node is part of an SL, we should have the attached tree root to tell its failure
        self._is_first_bsm = set()  # if node is an end of sl, this define the ordering of last bsm operations
        self._NUMBER_OF_SL_MEMORIES = NUMBER_OF_SL_MEMORIES

    def add_protocol_label(self, sd_id: int, tree_id: int):
        key = f"{sd_id}_{tree_id}"
        # left swapping event
        if self.node.left_height[key] == 1:  # left qubit signal comes directly from Entangling (no BSM in its left)
            self._protocol_labels.add(("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_{key}_left"))
        else:  # some BSM in its left to be first done
            self._protocol_labels.add(("correction_left", f"{MessageType.CORRECTION_ACK.name}_{key}"))
        # right swapping event
        if self.node.right_height[key] == 1:  # right qubit signal comes directly from Entangling (no BSM in its right)
            self._protocol_labels.add(("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_{key}_right"))
        else:
            self._protocol_labels.add(("messaging_protocol", f"{MessageType.CORRECTION_ACK.name}_{key}"))

    def update_sl_information(self, sd_id: int, tree_id: int, other_node_id: int,
                              IS_LEFT_SL: bool, is_path_one_hop: bool = False, sl_nodes_id: set = ()):
        # to update information about an SL
        key = f"{sd_id}_{tree_id}"
        self.node.other_sl_id[key] = other_node_id
        self._protocol_labels.add(("messaging_protocol", f"{MessageType.SL_SWAPPING_FAILURE_1.name}_{key}"))
        self._protocol_labels.add(("messaging_protocol", f"{MessageType.SL_SWAPPING_FAILURE_2.name}_{key}"))
        self._protocol_labels.add(("messaging_protocol", f"{MessageType.RESTART_SL.name}_{key}"))
        # self._protocol_labels.add(("messaging_protocol", f"{MessageType.SL_NOT_PRESENT.name}_{key}"))
        if IS_LEFT_SL:
            self.node.is_left_sl.add(key)
            if is_path_one_hop:
                self._protocol_labels.add(("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_{key}"
                                                                  f"_left"))
            else:  # some BSM in its left to be first done
                self._protocol_labels.add(("correction_left", f"{MessageType.CORRECTION_ACK.name}_{key}"))
        else:
            self.node.is_right_sl.add(key)
            if is_path_one_hop:  # right qubit signal comes directly from Entangling (no BSM in its right)
                self._protocol_labels.add(
                    ("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_{key}_right"))
            else:
                self.node.sl_nodes_id[key] = sl_nodes_id
                self._protocol_labels.add(("messaging_protocol", f"{MessageType.CORRECTION_ACK.name}_{key}"))

    def link_path_to_sl(self, path_sd_id: int, path_tree_id: int, sl_tree_id: int, IS_SL_LEFT: bool, sl_sd_id: int = 0,
                        IS_FIRST_BSM: bool = True, path_root_id: int = -1):
        path_key, sl_key = f"{path_sd_id}_{path_tree_id}", f"{sl_sd_id}_{sl_tree_id}"
        if IS_SL_LEFT:
            self.node.left_sl_id[path_key] = sl_key
        else:
            self.node.right_sl_id[path_key] = sl_key
        if IS_FIRST_BSM:
            self._is_first_bsm.add(path_key)
        if path_root_id != -1:
            self._left_right_tree_root[path_key] = path_root_id

        if not IS_SL_LEFT:  # left side of SL
            if self.node.left_height[path_key] == 1:
                # left qubit signal comes directly from Entangling (no BSM in its left)
                self._protocol_labels.add(("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_"
                                                                  f"{path_key}_left"))
            else:  # some BSM in its left to be first done
                self._protocol_labels.add(("correction_left", f"{MessageType.CORRECTION_ACK.name}_{path_key}"))
            if not IS_FIRST_BSM:
                # right swapping event
                # if self.node.right_height[path_key] == 1:
                #     # right qubit signal comes directly from Entangling (no BSM in its right)
                #     self._protocol_labels.append(
                #         ("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_{path_key}_right"))
                # else:
                self._protocol_labels.add(("messaging_protocol", f"{MessageType.CORRECTION_ACK.name}_"
                                                                 f"{path_key}"))
        else:  # right side of SL
            if self.node.right_height[path_key] == 1:
                # left qubit signal comes directly from Entangling (no BSM in its left)
                self._protocol_labels.add(("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_"
                                                                  f"{path_key}_right"))
            else:  # some BSM in its left to be first done
                self._protocol_labels.add(("messaging_protocol", f"{MessageType.CORRECTION_ACK.name}_{path_key}"))
            if not IS_FIRST_BSM:
                self._protocol_labels.add(("correction_left", f"{MessageType.CORRECTION_ACK.name}_"
                                                              f"{path_key}"))

    def run(self):
        qubit_left_ready, qubit_right_ready = set(), set()  # which sd-tree pairs are ready if there are multiple
        left_last_time, right_last_time = {}, {}
        prv_bsm_count = 0

        if len(self._protocol_labels) > 0:
            while True:
                expr = yield reduce(operator.or_, [self.await_signal(sender=self.subprotocols[x[0]], signal_label=x[1])
                                                   for x in self._protocol_labels] +
                                    [self.await_timer(self.node.bsm_time * 1e9) if prv_bsm_count > 0 else None])
                for triggered_idx, tt_event in enumerate(expr.triggered_events):
                    if tt_event.type.name.lower() == "sleep_event":
                        prv_bsm_count -= 1
                        continue  # previous bsm timer, do nothing
                    triggered_event = expr.triggered_events[triggered_idx].source.get_signal_by_event(
                        expr.triggered_events[triggered_idx])
                    triggered_label, triggered_res = triggered_event.label, triggered_event.result
                    if triggered_label.startswith(f"{MessageType.SL_SWAPPING_FAILURE_1.name}") or \
                            triggered_label.startswith(f"{MessageType.SL_SWAPPING_FAILURE_2.name}") or \
                            triggered_label.startswith(f"{MessageType.RESTART_SL.name}"):
                        # failed SL, drop used qubits and update left or right tree if necessary
                        sl_sd_id, sl_tree_id = triggered_label.split("_")[-2:]
                        sl_key = f"{sl_sd_id}_{sl_tree_id}"
                        if sl_key in self.node.is_right_sl and \
                                len(self.node.sl_saved_pos[sl_key]) == self._NUMBER_OF_SL_MEMORIES:
                            self.sl_generating(self.node, sl_key, True)
                            self.restart_tree(self.node, sl_key)  # left of an SL and space for new EP
                        logging.info(f"{ns.sim_time():.1f}:\t (SD-{sl_sd_id})(TREE{sl_tree_id}) (SL-{sl_key}) "
                                     f"{self.node.name} dropped an "
                                     f"EP (pos: {next(iter(self.node.sl_saved_pos[sl_key]))}) "
                                     f"because of an ACK from node{self.node.other_sl_id[sl_key]}.")
                        self.node.sl_saved_pos[sl_key].pop(next(iter(self.node.sl_saved_pos[sl_key])))

                        if triggered_label.startswith(f"{MessageType.SL_SWAPPING_FAILURE_2.name}"):
                            sd_id, tree_id = triggered_res
                            key = f"{sd_id}_{tree_id}"
                            neighbor = 0 if key in self.node.right_sl_id else 1
                            if key in self._left_right_tree_root:
                                logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                                             f"(SL-{sl_key}) {self.node.name} send signal to the tree"
                                             f" rooted at node"
                                             f"{self._left_right_tree_root[key]} to restart the tree.")
                                self.node.ports["cto_node" + str(self._left_right_tree_root[key])]. \
                                    tx_output(ProtocolMessage(MessageType.CORRECTION_FAILURE, sd_id,
                                                              tree_id, None, None))
                            elif key in self.node.neighbors and self.node.neighbors[key][
                                neighbor] != -1:  # left side of SL is an edge
                                if neighbor == 0:
                                    self.node.entangling[key] = (True, self.node.entangling[key][1])
                                else:
                                    self.node.entangling[key] = (self.node.entangling[key][0], True)
                                self.node.ports[f"cto_neighbor" \
                                                f"{self.node.neighbors[key][neighbor]}"].tx_output(
                                    ProtocolMessage(MessageType.START_ENTANGLING, sd_id, tree_id, None, -1)
                                )
                                logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                                             f"(SL-{sl_key}) {self.node.name} The other side was just an edge. "
                                             f"Fix Entangling status and send a signal to "
                                             f"neighbor{self.node.neighbors[key][neighbor]}")
                    else:
                        event_type = triggered_label.split("_")[0]
                        source = expr.triggered_events[0].source.name.split("_")[1]
                        sd_id, tree_id = triggered_label.split("_")[2:4]
                        is_left = True if ((event_type == "ENTANGLEMENT" and triggered_label.split("_")[-1] == "left") or
                                           (event_type == "CORRECTION" and source == "correction")) else False
                        # for left qubit (toward right of a path)
                        key = f"{sd_id}_{tree_id}"
                        if key in self.node.is_right_sl or key in self.node.is_left_sl:  # an end of an SL, save the EP and move on
                            if event_type == "ENTANGLEMENT" or triggered_res == self.node.other_sl_id[key]:
                                # right (left) qubit for leftmost (rightmost) node of an SL is ready
                                saving_pos = -1
                                for pos in range(self.node.qubit_mapping[key][0][0 if is_left else 1] + 1,
                                                 self.node.qubit_mapping[key][0][
                                                     0 if is_left else 1] + self._NUMBER_OF_SL_MEMORIES
                                                 + 1):
                                    if pos not in self.node.sl_saved_pos[key]:
                                        saving_pos = pos
                                        self.node.sl_saved_pos[key][saving_pos] = None
                                        break
                                if saving_pos == -1:
                                    saving_pos = next(iter(self.node.sl_saved_pos[key]))
                                    self.node.sl_saved_pos[key].pop(saving_pos)
                                    self.node.sl_saved_pos[key][saving_pos] = None
                                self.node.qmemory.put(self.node.qmemory.pop(
                                    self.node.qubit_mapping[key][0][0 if is_left else 1]), saving_pos)
                                logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) Node{self.node.ID} "
                                             f"accepted new EP and saved at pos={saving_pos}. " +
                                             ("LEFT" if not is_left else "RIGHT") + " side of an SL.")
                                if not is_left:
                                    if len(self.node.sl_saved_pos[key]) < self._NUMBER_OF_SL_MEMORIES:
                                        self.restart_tree(self.node, key)  # leftmost node received EP, restart the SL
                                    else:
                                        self.sl_generating(self.node, key, False)
                        else:  # not for an end node of an SL
                            if (is_left and key in self.node.right_sl_id) or \
                                    (not is_left and key in self.node.left_sl_id and
                                     (event_type == "ENTANGLEMENT" or triggered_res == self.node.rightest_id[key])):
                                if key in self._is_first_bsm:
                                    # node's left come and it should be entangled with right SL (the first SL BSM), assume SL always ready
                                    if is_left:
                                        left_qubit_pos = self.node.qubit_mapping[key][0][0]
                                        sl_key = self.node.right_sl_id[key]
                                        if len(self.node.sl_saved_pos[sl_key]) == 0:  # there is no SL EP ready
                                            self._failed_sl_swapping(sd_id, tree_id, sl_key, IS_FIRST=True,
                                                                     IS_SL_LEFT=not is_left, SL_NOT_PRESENT=True)
                                            continue
                                        right_qubit_pos = next(iter(self.node.sl_saved_pos[sl_key]))  # from right sl
                                    else:  # right
                                        right_qubit_pos = self.node.qubit_mapping[key][0][1]
                                        sl_key = self.node.left_sl_id[key]
                                        if len(self.node.sl_saved_pos[sl_key]) == 0:  # there is no SL EP ready
                                            self._failed_sl_swapping(sd_id, tree_id, sl_key, IS_FIRST=True,
                                                                     IS_SL_LEFT=not is_left, SL_NOT_PRESENT=True)
                                            continue
                                        left_qubit_pos = next(iter(self.node.sl_saved_pos[sl_key]))
                                    prv_bsm_count += 1
                                    bsm_success, bsm_res = self._bsm([left_qubit_pos, right_qubit_pos], False)

                                    if bsm_success:  # send to other end of the SL
                                        if is_left:
                                            self.node.ports["cto_node" + str(self.node.other_sl_id[sl_key])].tx_output(
                                                ProtocolMessage(MessageType.SWAPPING_CORRECTION, sd_id, tree_id, bsm_res,
                                                                self.node.leftest_id[key]))
                                        else:
                                            self.node.ports[f"cto_node{self.node.rightest_id[key]}"].tx_output(
                                                ProtocolMessage(MessageType.SWAPPING_CORRECTION, sd_id, tree_id, bsm_res,
                                                                self.node.other_sl_id[sl_key]))
                                        logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                                                     f"(SL-{sl_key}) {self.node.name} (SWAPPING) received entangled qubit, "
                                                     f"measured qubits (pos: {left_qubit_pos}, {right_qubit_pos})"
                                                     f" & sending corrections to node"
                                                     + (str(self.node.other_sl_id[sl_key]) if is_left
                                                        else str(self.node.rightest_id[key])))
                                    else:
                                        self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_SUCCESS=False, IS_FIRST=True,
                                                                  IS_SL_LEFT=not is_left,
                                                                  pos=[left_qubit_pos, right_qubit_pos])
                                else:
                                    if is_left:
                                        if key in qubit_right_ready:
                                            # second SL BSM when the left side gets ready later
                                            left_qubit_pos = self.node.qubit_mapping[key][0][0]
                                            sl_key = self.node.right_sl_id[key]
                                            if len(self.node.sl_saved_pos[sl_key]) == 0:  # there is no SL EP ready
                                                self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_FIRST=False,
                                                                          IS_SL_LEFT=False, SL_NOT_PRESENT=True,
                                                                          IS_SUCCESS=False)
                                                continue
                                            right_qubit_pos = next(iter(self.node.sl_saved_pos[sl_key]))
                                            prv_bsm_count += 1
                                            bsm_success, bsm_res = self._bsm([left_qubit_pos, right_qubit_pos], False)
                                            if bsm_success:  # send to other end of the SL
                                                self.node.ports["cto_node" + str(self.node.rightest_id[key])].tx_output(
                                                    ProtocolMessage(MessageType.SWAPPING_CORRECTION, sd_id, tree_id,
                                                                    bsm_res,
                                                                    self.node.leftest_id[key]))
                                                self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_SUCCESS=True,
                                                                          IS_FIRST=False, IS_SL_LEFT=False,
                                                                          pos=[left_qubit_pos, right_qubit_pos])
                                            else:
                                                self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_SUCCESS=False,
                                                                          IS_FIRST=False, IS_SL_LEFT=False,
                                                                          pos=[left_qubit_pos, right_qubit_pos])
                                            qubit_right_ready.remove(key)
                                        else:
                                            qubit_left_ready.add(key)
                                    else:
                                        if key in qubit_left_ready:
                                            # second SL BSM when the right side gets ready later
                                            right_qubit_pos = self.node.qubit_mapping[key][0][1]
                                            sl_key = self.node.left_sl_id[key]
                                            if len(self.node.sl_saved_pos[sl_key]) == 0:  # there is no SL EP ready
                                                self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_FIRST=False,
                                                                          IS_SUCCESS=False,
                                                                          IS_SL_LEFT=True, SL_NOT_PRESENT=True)
                                                continue
                                            left_qubit_pos = next(iter(self.node.sl_saved_pos[sl_key]))
                                            prv_bsm_count += 1
                                            bsm_success, bsm_res = self._bsm([left_qubit_pos, right_qubit_pos], False)
                                            if bsm_success:  # send to other end of the SL
                                                self.node.ports["cto_node" + str(self.node.rightest_id[key])].tx_output(
                                                    ProtocolMessage(MessageType.SWAPPING_CORRECTION, sd_id, tree_id,
                                                                    bsm_res,
                                                                    self.node.leftest_id[key]))
                                                self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_SUCCESS=True,
                                                                          IS_FIRST=False, IS_SL_LEFT=True,
                                                                          pos=[left_qubit_pos, right_qubit_pos])
                                            else:
                                                self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_SUCCESS=False,
                                                                          IS_FIRST=False, IS_SL_LEFT=True,
                                                                          pos=[left_qubit_pos, right_qubit_pos])
                                            qubit_left_ready.remove(key)
                                        else:
                                            qubit_right_ready.add(key)
                            elif is_left and key in self.node.left_sl_id:
                                # second SL BSM if right is ready
                                if key in qubit_right_ready:
                                    sl_key = self.node.left_sl_id[key]
                                    if len(self.node.sl_saved_pos[sl_key]) == 0:  # there is no SL EP ready
                                        self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_FIRST=False, IS_SUCCESS=False,
                                                                  IS_SL_LEFT=True, SL_NOT_PRESENT=True)
                                        continue
                                    left_qubit_pos = next(iter(self.node.sl_saved_pos[sl_key]))
                                    right_qubit_pos = self.node.qubit_mapping[key][0][1]
                                    prv_bsm_count += 1
                                    bsm_success, bsm_res = self._bsm([left_qubit_pos, right_qubit_pos], False)
                                    if bsm_success:  # send to other end of the SL
                                        self.node.ports["cto_node" + str(self.node.rightest_id[key])].tx_output(
                                            ProtocolMessage(MessageType.SWAPPING_CORRECTION, sd_id, tree_id, bsm_res,
                                                            self.node.leftest_id[key]))
                                        self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_FIRST=False, IS_SL_LEFT=True,
                                                                  IS_SUCCESS=True, pos=[left_qubit_pos, right_qubit_pos])
                                    else:
                                        self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_FIRST=False, IS_SL_LEFT=True,
                                                                  IS_SUCCESS=False, pos=[left_qubit_pos, right_qubit_pos])
                                    qubit_right_ready.remove(key)
                                else:
                                    qubit_left_ready.add(key)
                            elif not is_left and key in self.node.right_sl_id and \
                                    (event_type == "ENTANGLEMENT" or triggered_res == self.node.rightest_id[key]):
                                if key in qubit_left_ready:
                                    # do BSM right: should be from SL
                                    sl_key = self.node.right_sl_id[key]
                                    if len(self.node.sl_saved_pos[sl_key]) == 0:  # there is no SL EP ready
                                        self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_FIRST=False, IS_SUCCESS=False,
                                                                  IS_SL_LEFT=False, SL_NOT_PRESENT=True)
                                        continue
                                    right_qubit_pos = next(iter(self.node.sl_saved_pos[sl_key]))
                                    left_qubit_pos = self.node.qubit_mapping[key][0][0]
                                    prv_bsm_count += 1
                                    bsm_success, bsm_res = self._bsm([left_qubit_pos, right_qubit_pos], False)
                                    if bsm_success:  # send to other end of the SL
                                        self.node.ports["cto_node" + str(self.node.rightest_id[key])].tx_output(
                                            ProtocolMessage(MessageType.SWAPPING_CORRECTION, sd_id, tree_id, bsm_res,
                                                            self.node.leftest_id[key]))
                                        self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_FIRST=False, IS_SL_LEFT=False,
                                                                  IS_SUCCESS=True, pos=[left_qubit_pos, right_qubit_pos])
                                    else:
                                        self._sl_swapping_manager(sd_id, tree_id, sl_key, IS_FIRST=False, IS_SL_LEFT=False,
                                                                  IS_SUCCESS=False, pos=[left_qubit_pos, right_qubit_pos])
                                    qubit_left_ready.remove(key)
                                else:
                                    qubit_right_ready.add(key)
                            else:  # normal BSM
                                if (is_left and key in qubit_right_ready) or \
                                        (not is_left and key in qubit_left_ready and
                                         (event_type == "ENTANGLEMENT" or triggered_res == self.node.rightest_id[key])):
                                    left_qubit_pos, right_qubit_pos = self.node.qubit_mapping[key][0]
                                    prv_bsm_count += 1
                                    bsm_success, bsm_res = self._bsm([left_qubit_pos, right_qubit_pos], False)
                                    if bsm_success:  # send to other end of the SL
                                        self.node.ports["cto_node" + str(self.node.rightest_id[key])].tx_output(
                                            ProtocolMessage(MessageType.SWAPPING_CORRECTION, sd_id, tree_id, bsm_res,
                                                            self.node.leftest_id[key]))
                                        logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                                                     f"{self.node.name} (SWAPPING) received entangled qubit, "
                                                     f"measured qubits & sending corrections to node"
                                                     + str(self.node.rightest_id[key]))
                                    else:
                                        logging.info(
                                            f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}){self.node.name}"
                                            " (SWAPPING) Unsuccessful swapping")
                                        tools.tree_failure(self.node, key)
                                    # reset for the next round
                                    if key in qubit_left_ready:
                                        qubit_left_ready.remove(key)
                                    if key in qubit_right_ready:
                                        qubit_right_ready.remove(key)
                                else:  # 1st qubit is ready
                                    if is_left:
                                        qubit_left_ready.add(key)
                                    elif event_type == "ENTANGLEMENT" or triggered_res == self.node.rightest_id[key]:
                                        qubit_right_ready.add(key)
        else:
            yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                    signal_label=Signals.SUCCESS)  # dummy signal to avoid ending other protocols

    @staticmethod
    def restart_tree(node: qPNode, key: str):  # only used for SL paths to let the pipelining goes on
        sd_id, tree_id = int(key.split("_")[0]), int(key.split("_")[1])
        node.entangling[key] = (False, True)
        node.ports["cto_neighbor" + str(node.neighbors[key][1])].tx_output(
            ProtocolMessage(MessageType.START_ENTANGLING, sd_id, tree_id, None, -1))
        if len(node.sl_nodes_id[key]) > 2:
            for node_id in node.sl_nodes_id[key]:
                if node_id == node.ID:
                    # no need to send message to itself and its right neighbor
                    continue
                data = "both"
                if node_id == node.neighbors[key][1]:
                    data = "right"
                if node_id == node.other_sl_id[key]:
                    data = "left"
                node.ports["cto_node" + str(node_id)].tx_output(
                    ProtocolMessage(MessageType.START_ENTANGLING, sd_id, tree_id, data, -1))

    @staticmethod
    def sl_generating(node: qPNode, key: str, is_starting: bool):
        sd_id, tree_id = int(key.split("_")[0]), int(key.split("_")[1])
        node.ports["cto_sl_node" + str(node.neighbors[key][1])].tx_output(
            ProtocolMessage(MessageType.START_ENTANGLEMENT_GENERATING if is_starting
                            else MessageType.STOP_ENTANGLEMENT_GENERATING, sd_id, tree_id, None, -1))
        if len(node.sl_nodes_id[key]) > 2:
            for node_id in node.sl_nodes_id[key]:
                if node_id == node.ID or node_id == node.neighbors[key][1]:
                    # no need to send message to itself and its right neighbor
                    continue
                node.ports["cto_sl_node" + str(node_id)].tx_output(
                    ProtocolMessage(MessageType.START_ENTANGLEMENT_GENERATING if is_starting
                                    else MessageType.STOP_ENTANGLEMENT_GENERATING, sd_id, tree_id, None, -1))

    def _bsm(self, pos: List[int], is_physical: False):
        if random() < self.node.bsm_success_rate:
            try:
                m = bell_measurement(self.node.qmemory, pos, is_physical)
                return True, m
            except Exception as e:
                logging.error(f"{ns.sim_time():.1f}: (SWAPPING) Error happened at {self.node.name}.\n"
                              f"Error message: {e}")
                return False, []
        return False, []

    def _failed_sl_swapping(self, sd_id: str, tree_id: str, sl_key: str, IS_FIRST: bool, IS_SL_LEFT: bool,
                            pos: List[int] = None, SL_NOT_PRESENT: bool = False):
        sl_id, sl_tree_id = sl_key.split("_")[:2]
        key = f"{sd_id}_{tree_id}"
        order = "first" if IS_FIRST else "second"
        if not SL_NOT_PRESENT:
            logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                         f"(SL-{sl_key}){self.node.name} SWAPPING failed at the {order}"
                         f" SL swapping (pos: {str(pos)})."
                         f" Drop the failed EP and send signal to the other node "
                         f"{self.node.other_sl_id[sl_key]}.")
            self.node.ports["cto_node" + str(self.node.other_sl_id[sl_key])].tx_output(
                ProtocolMessage(MessageType.SL_SWAPPING_FAILURE_1 if IS_FIRST else MessageType.SL_SWAPPING_FAILURE_2,
                                sl_id, sl_tree_id, [sd_id, tree_id], None))
            if sl_key in self.node.is_right_sl and \
                    len(self.node.sl_saved_pos[sl_key]) == self._NUMBER_OF_SL_MEMORIES:
                self.sl_generating(self.node, sl_key, True)
                self.restart_tree(self.node, sl_key)
        else:
            left_or_right = "left" if IS_SL_LEFT else "right"
            logging.error(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                          f"(SL-{sl_key}){self.node.name} the {order} SL ({left_or_right}) is not PRESENT at "
                          f"the {order} SL swapping. Restarting "
                          f"the tree and waiting for SL to come")
            # self.node.ports["cto_node" + str(self.node.other_sl_id[sl_key])].tx_output(
            #     ProtocolMessage(MessageType.SL_NOT_PRESENT, sl_id, sl_tree_id, [sd_id, tree_id], None))
        if key in self._left_right_tree_root:
            logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                         f"(SL-{sl_key}){self.node.name} send signal to the left tree"
                         f" rooted at node"
                         f"{self._left_right_tree_root[key]} to restart the tree.")
            self.node.ports["cto_node" + str(self._left_right_tree_root[key])]. \
                tx_output(ProtocolMessage(MessageType.CORRECTION_FAILURE, sd_id,
                                          tree_id, None, None))
        elif self.node.neighbors[key][1 if IS_SL_LEFT else 0] != -1:  # left side of SL is an edge
            self.node.entangling[key] = (True if not IS_SL_LEFT else self.node.entangling[key][0],
                                         True if IS_SL_LEFT else self.node.entangling[key][1])
            self.node.ports[f"cto_neighbor" \
                            f"{self.node.neighbors[key][1 if IS_SL_LEFT else 0]}"].tx_output(
                ProtocolMessage(MessageType.START_ENTANGLING, sd_id, tree_id, None, -1)
            )
            logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                         f"(SL-{sl_key}) The other side was just an edge. "
                         f"Fix Entangling status and send a signal to "
                         f"neighbor{self.node.neighbors[key][1 if IS_SL_LEFT else 0]}")

    def _sl_swapping_manager(self, sd_id: str, tree_id: str, sl_key: str, IS_SUCCESS: bool,
                             IS_FIRST: bool, IS_SL_LEFT: bool, pos: List[int] = None, SL_NOT_PRESENT: bool = False):
        sl_id, sl_tree_id = sl_key.split("_")[:2]
        key = f"{sd_id}_{tree_id}"
        order = "first" if IS_FIRST else "second"
        if IS_SUCCESS and not IS_FIRST:
            # second successful sl swapping
            self.node.ports["cto_node" + str(self.node.other_sl_id[sl_key])].tx_output(
                ProtocolMessage(MessageType.RESTART_SL, sl_id, sl_tree_id, [sd_id, tree_id], None))
            if len(self.node.sl_saved_pos[sl_key]) == self._NUMBER_OF_SL_MEMORIES and \
                    sl_key in self.node.is_right_sl:
                self.sl_generating(self.node, sl_key, True)
                self.restart_tree(self.node, sl_key)
            logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                         f"(SL-{sl_key}) {self.node.name} (SWAPPING) received entangled qubit, "
                         f"measured qubits (pos: {pos[0]}, {pos[1]}) "
                         f"& sending corrections to node"
                         + str(self.node.rightest_id[key]))
            logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                         f"(SL-{sl_key}) {self.node.name} dropped EP at pos: "
                         f"{next(iter(self.node.sl_saved_pos[sl_key]))}.")
            self.node.sl_saved_pos[sl_key].pop(next(iter(self.node.sl_saved_pos[sl_key])))
        elif not IS_SUCCESS or SL_NOT_PRESENT:
            if SL_NOT_PRESENT:
                left_or_right = "left" if IS_SL_LEFT else "right"
                logging.error(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                              f"(SL-{sl_key}){self.node.name} the {order} SL ({left_or_right}) is not PRESENT at "
                              f"the {order} SL swapping. Restarting "
                              f"the tree and waiting for SL to come")
            else:  # first or second sl failure
                logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                             f"(SL-{sl_key}){self.node.name} SWAPPING failed at the {order}"
                             f" SL swapping (pos: {str(pos)})."
                             f" Drop the failed EP and send signal to the other node "
                             f"{self.node.other_sl_id[sl_key]}.")
                self.node.ports["cto_node" + str(self.node.other_sl_id[sl_key])].tx_output(
                    ProtocolMessage(
                        MessageType.SL_SWAPPING_FAILURE_1 if IS_FIRST else MessageType.SL_SWAPPING_FAILURE_2,
                        sl_id, sl_tree_id, [sd_id, tree_id], None))
                if len(self.node.sl_saved_pos[sl_key]) == self._NUMBER_OF_SL_MEMORIES \
                        and sl_key in self.node.is_right_sl:
                    logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                                 f"(SL-{sl_key}) {self.node.name} Memory was full. Restart the tree")
                    self.sl_generating(self.node, sl_key, True)
                    self.restart_tree(self.node, sl_key)
                logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                             f"(SL-{sl_key}) {self.node.name} dropped EP at pos: "
                             f"{next(iter(self.node.sl_saved_pos[sl_key]))}.")
                self.node.sl_saved_pos[sl_key].pop(next(iter(self.node.sl_saved_pos[sl_key])))
            # restart non-sl attached to it
            if key in self._left_right_tree_root:
                logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                             f"(SL-{sl_key}) {self.node.name} send signal to the left tree"
                             f" rooted at node{self._left_right_tree_root[key]} to restart the tree.")
                self.node.ports["cto_node" + str(self._left_right_tree_root[key])]. \
                    tx_output(ProtocolMessage(MessageType.CORRECTION_FAILURE, sd_id, tree_id, None, None))
            elif self.node.neighbors[key][1 if IS_SL_LEFT else 0] != -1:  # left side of SL is an edge
                self.node.entangling[key] = (True if not IS_SL_LEFT else self.node.entangling[key][0],
                                             True if IS_SL_LEFT else self.node.entangling[key][1])
                self.node.ports[f"cto_neighbor{self.node.neighbors[key][1 if IS_SL_LEFT else 0]}"].tx_output(
                    ProtocolMessage(MessageType.START_ENTANGLING, sd_id, tree_id, None, -1))
                logging.info(f"{ns.sim_time():.1f}:\t (SD-{sd_id})(TREE{tree_id}) "
                             f"(SL-{sl_key}) {self.node.name} The other side was just an edge. "
                             f"Fix Entangling status and send a signal to "
                             f"neighbor{self.node.neighbors[key][1 if IS_SL_LEFT else 0]}")


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

    def __init__(self, node: qPNode,
                 messaging_protocol: MessagingProtocol, entangling_protocol: EntanglingProtocol,
                 NUMBER_OF_SL_MEMORIES: int):
        super().__init__(node, name=node.name + "_teleportation_protocol")
        # self.add_subprotocol(qubit_protocol, "qprotocol")
        self.add_subprotocol(messaging_protocol, "messaging_protocol")
        self.add_subprotocol(entangling_protocol, "entangling_protocol")
        self._dst_ids = {}
        self._is_paths_one_hop = {}
        # self._teleporting_qubit_pos = node.avail_mem_pos()
        # self._tree_nodes_id = {}
        self._is_teleporting = {}
        self._tree_success_count = {}
        self._protocol_labels = []
        self._is_node_src_left_sl = {}
        self._NUMBER_OF_SL_MEMORIES = NUMBER_OF_SL_MEMORIES

    def add_protocol_label(self, sd_id: int, tree_id: int, dst_id: int, is_path_one_hop: bool,
                           IS_TELEPORTING: bool = False):
        key = f"{sd_id}_{tree_id}"
        self._dst_ids[key] = dst_id
        self._is_paths_one_hop[key] = is_path_one_hop
        self._is_teleporting[key] = IS_TELEPORTING
        # self._tree_nodes_id[key] = tree_nodes_id
        self._tree_success_count[key] = 0
        if is_path_one_hop:
            self._protocol_labels.append(("entangling_protocol", f"{MessageType.ENTANGLEMENT_READY.name}_{key}_right"))
        else:
            self._protocol_labels.append(("messaging_protocol", f"{MessageType.CORRECTION_ACK.name}_{key}"))
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}")

    def node_is_src_left_sl(self, sd_id: int, tree_id: int, sl_tree_id: int, sl_sd_id: int = 0):
        self._is_node_src_left_sl[f"{sd_id}_{tree_id}"] = f"{sl_sd_id}_{sl_tree_id}"

    def create_qubit(self):
        qubit, = ns.qubits.create_qubits(1)
        self.node.qmemory.put(qubit, self._teleporting_qubit_pos)
        self.node.qmemory.operate(ns.H, self._teleporting_qubit_pos)
        self.node.qmemory.operate(ns.S, self._teleporting_qubit_pos)

    def run(self):
        if len(self._protocol_labels) > 0:
            entanglement_ready = set()  # when the tree construction is ready
            while True:
                expr = yield reduce(operator.or_, [self.await_signal(sender=self.subprotocols[x[0]], signal_label=x[1])
                                                   for x in self._protocol_labels])
                triggered_event = expr.triggered_events[0].source.get_signal_by_event(expr.triggered_events[0])
                triggered_label, triggered_res = triggered_event.label, triggered_event.result
                event_type = triggered_label.split("_")[0]
                sd_id, tree_id = triggered_label.split("_")[2:4]
                key = f"{sd_id}_{tree_id}"
                if event_type == "ENTANGLEMENT":
                    entanglement_ready.add(key)
                elif event_type == "CORRECTION":
                    if triggered_res == self._dst_ids[key]:
                        entanglement_ready.add(key)
                else:
                    logging.error(
                        f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{self._tree_id}) TELEPORTATION received unknown "
                        f"{triggered_label}")
                if key in entanglement_ready:
                    sd_id, tree_id = int(key.split("_")[0]), int(key.split("_")[1])
                    if self._is_teleporting[key]:
                        # if we send qubit information to BOB
                        self.create_qubit()
                        yield self.await_timer(self.node.bsm_time * 1e9)
                        try:
                            m = bell_measurement(self.node.qmemory, [self._teleporting_qubit_pos,
                                                                     self.node.qubit_mapping[self._tree_id][0][1]],
                                                 False)
                            if random() < self.node.bsm_success_rate:
                                # successful
                                self.node.ports["cto_node" + str(self._dst_id)].tx_output(
                                    ProtocolMessage(MessageType.TELEPORTATION_CORRECTION, self._tree_id, m, -1))
                                logging.info(
                                    f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Alice (node {str(self.node.ID)})"
                                    f" received entangled qubit, entanglement ready. "
                                    f"measured qubits & sending corrections to Bob(node{str(self._dst_id)})")
                                if DEBUG_MODE > 0:
                                    print(f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Alice (node {str(self.node.ID)})"
                                          f" received entangled qubit, entanglement ready. "
                                          f"measured qubits & sending corrections to Bob(node{str(self._dst_id)})")
                            else:
                                # unsuccessful bsm, should let the whole tree to start entangling
                                logging.info(
                                    f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Unsuccessful TELEPORTATION at alice"
                                    f" (node{str(self.node.ID)}). Tell all the nodes to start entangling again.")
                                if DEBUG_MODE > 0:
                                    print(
                                        f"{ns.sim_time():.1f}: (TREE{self._tree_id}) Unsuccessful TELEPORTATION at alice"
                                        f" (node{str(self.node.ID)}). Tell all the nodes to start entangling again.")
                                self.restart_tree()
                        except Exception as e:
                            logging.error(f"{ns.sim_time():.1f}: (TELEPORTATION) Error happened at Alice "
                                          f"({self.node.name}).\tError message: {e}")
                            self.restart_tree()
                    else:
                        key = f"{sd_id}_{tree_id}"
                        if key in self._is_node_src_left_sl:
                            sl_key = self._is_node_src_left_sl[key]  # left qubit is the rightmost position of an SL
                            # path reaches to its end, should restart SL if need it
                            self.node.ports[f"cto_node{self.node.other_sl_id[sl_key]}"].tx_output(
                                ProtocolMessage(MessageType.RESTART_SL, int(sl_key.split("_")[0]),
                                                int(sl_key.split("_")[1]), None, -1))
                            if len(self.node.sl_saved_pos[sl_key]) == self._NUMBER_OF_SL_MEMORIES \
                                    and sl_key in self.node.is_right_sl:
                                SwappingProtocol.sl_generating(self.node, sl_key, True)
                                SwappingProtocol.restart_tree(self.node, sl_key)  # leftmost node of an SL
                            logging.info(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) (SL-{sl_key}) Alice "
                                         f"is an end of an SL. Should drop the proper EP "
                                         f"(pos: {next(iter(self.node.sl_saved_pos[sl_key]))})")
                            self.node.sl_saved_pos[sl_key].pop(next(iter(self.node.sl_saved_pos[sl_key])))

                        # we stop here, when the tree succeeds; we don't send qubit information to Bob's side
                        logging.info(
                            f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{tree_id}) Alice (node {str(self.node.ID)})"
                            f" received entangled qubit, entanglement ready. ")
                        self.send_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}_{tree_id}")
                        # send signal to manager to stop
                        self.node.ports["cto_manager"].tx_output(ProtocolMessage(MessageType.ENTANGLEMENT_READY,
                                                                                 sd_id,
                                                                                 tree_id, None, -1))
                        self._tree_success_count[key] += 1
                    entanglement_ready.remove(key)
        else:
            yield self.await_signal(sender=self.subprotocols["messaging_protocol"],
                                    signal_label=Signals.SUCCESS)  # dummy signal to avoid ending other protocols

    def restart_tree(self):
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
                                                   signal_label=MessageType.TELEPORTATION_CORRECTION.name +
                                                                str(self._tree_id))
            if self._is_path_one_hop:
                evexpr_entanglement = self.await_mempos_in_use_toggle(self.node.qmemory,
                                                                      [self.node.qubit_mapping[self._tree_id][0][0]])
            else:
                evexpr_entanglement = self.await_signal(sender=self.subprotocols["correction_protocol"],
                                                        signal_label=Signals.SUCCESS)
            expression = yield evexpr_meas_result | evexpr_entanglement
            if expression.first_term.value:
                sender_id, tree_id, tmp_res = self.subprotocols["messaging_protocol"].get_signal_result(
                    MessageType.TELEPORTATION_CORRECTION.name + str(self._tree_id))
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
    def __init__(self, sd_id: int, tree_id: int, src: qPNode, dst: qPNode, is_path_one_hop: bool,
                 correction_protocol: CorrectionProtocol, teleportation_protocol: TeleportationProtocol):
        self._tree_id = tree_id
        self._sd_id = sd_id
        self._src, self._dst = src, dst
        self._is_path_one_hop = is_path_one_hop
        if not is_path_one_hop:
            self.add_subprotocol(correction_protocol, "correction_protocol")
        else:
            self.add_subprotocol(teleportation_protocol, "teleportation_protocol")

        def calc_fidelity(evexpr):
            self.await_timer(1)
            qubit_a, = self._src.qmemory.peek(self._src.qubit_mapping[f"{self._sd_id}_{self._tree_id}"][0][1])
            qubit_b, = self._dst.qmemory.peek(self._dst.qubit_mapping[f"{self._sd_id}_{self._tree_id}"][0][0])
            try:
                fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ns.b00, squared=True)
                logging.info(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{self._tree_id}) ALice and Bob are entangled!. "
                             f"Fidelity: {fidelity}")
                return {"fidelity": fidelity}
            except Exception as e:
                logging.debug(f"{ns.sim_time():.1f}: (SD-{sd_id})(TREE{self._tree_id}) Fidelity calculation failed!. "
                              f"Error msg: {e}")
                return

        self._dc = DataCollector(calc_fidelity, include_time_stamp=False, include_entity_name=False)

    def run(self):
        if not self._is_path_one_hop:
            self._dc.collect_on(self.await_signal(sender=self.subprotocols["correction_protocol"],
                                                  signal_label=f"{MessageType.CORRECTION_ACK.name}_{self._sd_id}_"
                                                               f"{self._tree_id}"))
        else:
            # self._dc.collect_on(self.await_mempos_in_use_toggle(self._dst.qmemory,
            #                                                       [self._dst.qubit_mapping[self._tree_id][0][0]]))
            self._dc.collect_on(self.await_signal(sender=self.subprotocols["teleportation_protocol"],
                                                  signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_{self._sd_id}_"
                                                               f"{self._tree_id}"))

    @property
    def average_fidelity(self):
        "return count and avg"
        try:
            return self._dc.dataframe.count()[0], self._dc.dataframe.agg("mean")[0]
        except IndexError as e:
            return 0, 0


class ManagerMessagingProtocol(NodeProtocol):
    """a protocol that handles received messages from other nodes"""

    def __init__(self, node: Node):
        super().__init__(node, name=node.name + "_manager_messaging_protocol")

    def add_signals(self, sd_id: int):
        self.add_signal(f"{MessageType.ENTANGLEMENT_READY.name}_{sd_id}")

    def run(self):
        while True:
            expr = yield reduce(operator.or_, [self.await_port_input(self.node.ports[port_name])
                                               for port_name in self.node.ports if port_name.startswith("cfrom_")])
            for triggered_event in expr.triggered_events:
                for obj in triggered_event.source.input_queue:
                    received_port_name = obj[0].source.name
                    for message in obj[1].items:
                        if message.type == MessageType.ENTANGLEMENT_READY:
                            self.send_signal(signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_{message.sd_id}")


class ManagerProtocol(NodeProtocol):
    """a protocol that manage EP generation order for multiple (s, d) pairs"""

    def __init__(self, node: Node, manager_messaging_protocol: ManagerMessagingProtocol, time_period: float,
                 max_sl_latency: float):
        """:param time_period (s) a period in which an (s,d) entanglement is supposed to finish, frequency demand"""
        super(ManagerProtocol, self).__init__(node, name="manager_protocol")
        self._time_period = time_period
        self._nodes_id = defaultdict(set)  # a dict in which non-sl ids for each pair is stored
        # self._sd_id_to_src_id = {}
        # self._srcs = set()
        self.add_subprotocol(manager_messaging_protocol, "manager_messaging")
        self._latencies, self._unsuccessful_tries = None, None
        self._has_whole_sl_path, self._max_sl_Latency = {}, max_sl_latency

    def add_sd_information(self, sd_id: int, nodes_id: set):
        # if src_id not in self._srcs:
        #     # self.add_subprotocol(src_tele_sub, f"src_tele_sub{src_id}")
        #     self._srcs.add(src_id)
        # if sd_id not in self._sd_id_to_src_id:
        #     self._sd_id_to_src_id[sd_id] = src_id
        self._nodes_id[sd_id] = self._nodes_id[sd_id].union(nodes_id)

    def update_sd_whole_sl(self, sd_id: int, sl_latency: float):
        if sd_id not in self._has_whole_sl_path:
            self._has_whole_sl_path[sd_id] = sl_latency
        self._has_whole_sl_path[sd_id] = min(sl_latency, self._has_whole_sl_path[sd_id])

    def run(self):
        prev_sd_id = -1
        id_list = list(self._nodes_id.keys())
        last_time = 0.0
        self._latencies = [[] for _ in range(len(self._nodes_id))]
        self._unsuccessful_tries = [0] * len(self._nodes_id)
        while True:
            if prev_sd_id != -1:
                xpr = yield self.await_timer(self._time_period * 1e9) | \
                            self.await_signal(sender=self.subprotocols[f"manager_messaging"],
                                              signal_label=f"{MessageType.ENTANGLEMENT_READY.name}_{prev_sd_id}")
                if xpr.first_term:
                    self._unsuccessful_tries[prev_sd_id - 1] += 1
                else:
                    self._latencies[prev_sd_id - 1].append(ns.sim_time() - last_time)
                logging.info(f"{ns.sim_time():.1f}: (SD-{prev_sd_id}) STOP signal was sent")
                for node_id in self._nodes_id[prev_sd_id]:
                    self.node.ports[f"cto_node{node_id}"].tx_output(
                        ProtocolMessage(MessageType.STOP_ENTANGLEMENT_GENERATING, prev_sd_id, -1, None, -1))
            if last_time + self._time_period * 1e9 - ns.sim_time() > 0:
                yield self.await_timer(last_time + self._time_period * 1e9 - ns.sim_time())
            last_time += self._time_period * 1e9
            new_sd_id = choice(id_list)
            while new_sd_id in self._has_whole_sl_path:
                if self._has_whole_sl_path[new_sd_id] < self._max_sl_Latency:
                    self._latencies[new_sd_id - 1].append(0)
                else:
                    self._latencies[new_sd_id - 1].append(self._has_whole_sl_path[new_sd_id] / len(id_list))
                yield self.await_timer(self._time_period * 1e9)
                last_time += self._time_period * 1e9
                new_sd_id = choice(id_list)
            logging.info(f"{ns.sim_time():.1f}: (SD-{new_sd_id}) START signal was sent")
            for node_id in self._nodes_id[new_sd_id]:
                self.node.ports[f"cto_node{node_id}"].tx_output(
                    ProtocolMessage(MessageType.START_ENTANGLEMENT_GENERATING, new_sd_id, -1, None, -1))
            prev_sd_id = new_sd_id

    @property
    def latencies(self):
        return self._latencies

    @property
    def failed_tries(self):
        return self._unsuccessful_tries
