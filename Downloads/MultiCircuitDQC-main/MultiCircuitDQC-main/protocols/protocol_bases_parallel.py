import sys

sys.path.append("..")

from netsquid.protocols import NodeProtocol, Signals
from protocols.component_bases_parallel import qPNode
from commons.tools import bell_measurement
import netsquid as ns


class SwappingProtocol(NodeProtocol):
    """a protocol class that handles BSM operations and send results to rightest node"""
    def __init__(self, node: qPNode, tree_id: int):
        super().__init__(node, name=node.name + "_swapping_protocol")

    def run(self):
        qubit_pos_left_ready = False  # left swapping qubit
        qubit_pos_right_ready = False  # right swapping qubit

        while True:
            if self.node.leftest_qubit_pos == self.node.init_left_qubit_pos:
                # qubit directly comes from charlie
                evexpr_pos_left = self.await_port_input(self.node.ports["qin_left"])
            else:
                # qubit comes from lower level of the tree
                evexpr_pos_left = self.await_mempos_in_use_toggle(self.node.qmemory, [self.node.leftest_qubit_pos])

            if self.node.rightest_qubit_pos == self.node.init_right_qubit_pos:
                evexpr_pos_right = self.await_port_input(self.node.ports["qin_right"])
            else:
                evexpr_pos_right = self.await_mempos_in_use_toggle(self.node.qmemory, [self.node.rightest_qubit_pos])
            expression = yield evexpr_pos_left | evexpr_pos_right
            if expression.first_term.value:
                # first qubit is ready
                qubit_pos_left_ready = True
            elif expression.second_term.value:
                #
                qubit_pos_right_ready = True

            if qubit_pos_left_ready and qubit_pos_right_ready:
                m = bell_measurement(self.node.qmemory, [self.node.leftest_qubit_pos, self.node.rightest_qubit_pos])
                self.node.ports["cto_node" + str(self.node.rightest_id)].tx_output(m)
                # self.send_signal(Signals.SUCCESS)
                print(f"{ns.sim_time():.1f}:\t" + self.node.name + " received entangled qubit, "
                      f"measured qubits & sending corrections to node" + str(self.node.rightest_id))

                # reset for the next round
                qubit_pos_left_ready = False
                qubit_pos_right_ready = False
                # break


class CorrectionProtocol(NodeProtocol):
    """A protocol class that does the correction when BSM results is ready"""

    def __init__(self, node: qPNode, parent_id: int, leftest_id: int):
        super().__init__(node,  node.name + "_correction_protocol_for_parent_node_" + str(parent_id))
        self._parent_id = parent_id
        self._leftest_id = leftest_id

    def run(self):
        port_meas = self.node.ports["cfrom_node" + str(self._parent_id)]
        entanglement_ready = False
        meas_results = None
        entangle_qubit_pos = self.node.qubit_mapping[self._parent_id][0]  # qubit slot where correction happens on
        result_qubit_pos = self.node.qubit_mapping[self._parent_id][1]  # qubit slot where result goes to

        while True:
            if self.node.init_left_qubit_pos == entangle_qubit_pos:
                # qubit comes directly from charlie
                evexp_entangle = self.await_port_input(self.node.ports["qin_left"])
            else:
                # qubit was moved to another position as part of previous swapping
                evexp_entangle = self.await_mempos_in_use_toggle(self.node.qmemory, [entangle_qubit_pos])
            evexpr_meas_result = self.await_port_input(port_meas)
            expression = yield evexpr_meas_result | evexp_entangle
            if expression.first_term.value:
                # measurement is ready from parent
                meas_results = port_meas.rx_input().items
            elif expression.second_term.value:
                # entangled qubit is ready
                entanglement_ready = True
            if meas_results is not None and entanglement_ready:
                if meas_results[0]:
                    self.node.qmemory.operate(ns.Z, entangle_qubit_pos)
                if meas_results[1]:
                    self.node.qmemory.operate(ns.X, entangle_qubit_pos)
                fidelity = ns.qubits.fidelity(self.node.qmemory.peek(entangle_qubit_pos)[0],
                                              ns.y0, squared=True)  # TODO we should decide how we handle fidelity
                # self.send_signal(Signals.SUCCESS, self._parent_id)
                self.node.ports["cto_node" + str(self._leftest_id)].tx_output(["CORRECTION_ACK", self._parent_id])
                print(f"{ns.sim_time():.1f}: {self.node.name} received entangled qubit and "  # TODO this print should be commented
                      f"corrections from node{self._parent_id} and sent it to node{self._leftest_id}! "
                      f"Fidelity = {fidelity:.3f}")
                self.node.qmemory.pop(result_qubit_pos)
                self.node.qmemory.mem_positions[result_qubit_pos].in_use = True
                self.node.qmemory.put(self.node.qmemory.pop(entangle_qubit_pos), result_qubit_pos)
                # put in the next level of memory for future entanglement

                # reset for next round
                entanglement_ready = False
                meas_results = None
                break


class CorrectionAckProtocol(NodeProtocol):
    """A protocol class that change qubit position when a correction is finished"""
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
    def __init__(self, node: qPNode, qubit_protocol: QubitGenerationProtocol, dst_id):
        super().__init__(node)
        self.add_subprotocol(qubit_protocol, "qprotocol")
        self._dst_id = dst_id

    def run(self):
        qubit_initialised = False  # when the target qubit is ready
        entanglement_ready = False  # when the tree construction is ready
        teleported_qubit_pos = -1
        while True:
            evexpr_qubit = self.await_signal(sender=self.subprotocols["qprotocol"],
                                             signal_label=Signals.SUCCESS)
            evexpr_entanglement = self.await_mempos_in_use_toggle(self.node.qmemory, [self.node.rightest_qubit_pos])
            expression = yield evexpr_qubit | evexpr_entanglement
            if expression.first_term.value:
                qubit_initialised = True
            elif expression.second_term.value:
                entanglement_ready = True
                teleported_qubit_pos = self.subprotocols["qprotocol"].get_signal_result(Signals.SUCCESS)
            if qubit_initialised and entanglement_ready:
                m = bell_measurement(self.node.qmemory, [teleported_qubit_pos, self.node.rightest_qubit_pos])
                # TODO leftest_qubit_pos should be retrieved from evexpr_qubit message
                self.node.ports["cto_node" + str(self._dst_id)].tx_output(m)
                self.send_signal(Signals.SUCCESS)
                print(f"{ns.sim_time():.1f}: Alice (node {str(self.node.ID)})"
                      f" received entangled qubit, entanglement ready"
                      f"measured qubits & sending corrections to Bob(node{str(self._dst_id)})")
                break

    def start(self):
        super().start()
        self.start_subprotocols()


class TeleportationCorrectionProtocol(NodeProtocol):
    """A protocol class that does the correction when BSM results is ready"""

    def __init__(self, node: qPNode, src_id: int):
        super().__init__(node,  node.name + "_correction_protocol_for_source_node_" + str(src_id))
        self._src_id = src_id

    def run(self):
        port_meas = self.node.ports["cfrom_node" + str(self._src_id)]
        entanglement_ready = False
        meas_results = None

        while True:
            if self.node.init_left_qubit_pos == self.node.leftest_qubit_pos:
                # qubit comes directly from charlie
                evexp_entangle = self.await_port_input(self.node.ports["qin_left"])
            else:
                # qubit was moved to another position as part of previous swapping
                evexp_entangle = self.await_mempos_in_use_toggle(self.node.qmemory, [self.node.leftest_qubit_pos])
            evexpr_meas_result = self.await_port_input(port_meas)
            expression = yield evexpr_meas_result | evexp_entangle
            if expression.first_term.value:
                # measurement is ready from parent
                meas_results = port_meas.rx_input().items
            elif expression.second_term.value:
                # entangled qubit is ready
                entanglement_ready = True
            if meas_results is not None and entanglement_ready:
                if meas_results[0]:
                    self.node.qmemory.operate(ns.Z, self.node.leftest_qubit_pos)
                if meas_results[1]:
                    self.node.qmemory.operate(ns.X, self.node.leftest_qubit_pos)
                fidelity = ns.qubits.fidelity(self.node.qmemory.peek(self.node.leftest_qubit_pos)[0],
                                              ns.y0, squared=True)  # TODO we should decide how we handle fidelity
                # reset for next round
                print(f"{ns.sim_time():.1f}: Bob (node {self.node.ID}) received entangled qubit and "  # TODO this print should be commented
                    f"corrections from node{self._src_id}! "
                    f"Fidelity = {fidelity:.3f}")
                entanglement_ready = False
                meas_results = None
                break
