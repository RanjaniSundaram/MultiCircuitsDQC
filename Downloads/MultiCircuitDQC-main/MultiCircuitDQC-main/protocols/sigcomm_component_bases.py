import sys
sys.path.append("..")

from netsquid.nodes import Node, Connection
from commons.qNode import qNode
from netsquid.components import QuantumMemory
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.qubits import StateSampler
import netsquid.qubits.ketstates as ks
from commons.Point import Point
from collections import defaultdict


class qPNode(Node):
    def __init__(self, node: qNode):
        super().__init__("Node" + str(node.id), ID=node.id,
                         qmemory=QuantumMemory(self.name + "Memory", num_positions=2 * node.memory))
        self.mem_pos = -1
        self._loc = node.loc
        self._bsm_success_rate = node.bsm_success_rate
        self._bsm_time = node.bsm_time
        self._gen_success_rate = node.gen_success_rate
        self.leftest_id, self.rightest_id = {}, {}  # (key=tree_id) leftest and rightest node if rooted at this node
        # self.qubit_mapping = {}
        # each tree has two tuples (left_qubit_pos, right_qubit_pos), (left_link_qubit_pos, right_qubit_pos)
        # (left_qubit_pos, right_qubit_pos) is for BSM and (left_link_qubit_pos, right_qubit_pos) is for link layer
        self.children_id = defaultdict(set)  # set of children ids used when BSM fails, direct connection to each
        self.left_height, self.right_height = {}, {}  # (key=tree_id) holds the height of left and right subtree
        self.entangling = {}  # a dictionary that indicates node should accept new qubits from link layer for left&right
        self.neighbors = {}  # a dictionary to hold left-right neighbors if node is in the tree
        self.qubit_to_port = {}  # a dictionary that converts link_layer_Qubit_pos to port. Assumed each qubit uses its
        # own specific port and channel
        self.link_to_network_pos = {}  # a dictionary that converts link_layer_Qubit_pos to network_layer_Qubit_pos
        self.left_paired_with, self.inverse_left_paired_with = {}, {}
        self.right_paired_with, self.inverse_right_paired_with = {}, {}

    def avail_mem_pos(self) -> int:
        self.mem_pos += 1
        return self.mem_pos

    @property
    def loc(self) -> Point:
        return self._loc

    @property
    def bsm_time(self) -> float:
        return self._bsm_time

    @property
    def bsm_success_rate(self) -> float:
        return self._bsm_success_rate

    @property
    def gen_success_rate(self) -> float:
        return self._gen_success_rate


class Charlie(Node):
    """Charlie Node"""
    def __init__(self, left_node_id: int, right_node_id: int, bsm_success_rate: float, bsm_time: float,
                 number_of_subchannels: int):
        super(Charlie, self).__init__(name="Charlie_" + str(left_node_id) + "_" + str(right_node_id))
        self._bsm_success_rate, self._optical_bsm_time = bsm_success_rate, bsm_time
        self.qubit_data = {}  # key=tree_id,
        # value: (new_dict) key: left_qubit_pos-right_qubit_pos, value: tuple that store qubit & time for both side,
        self._left_node_id, self._right_node_id = left_node_id, right_node_id
        self._left_qubit_pos = {}  # a dictionary to get the left qubit pos based on tree_id and right_qubit_pos
        self._right_qubit_pos = {}  # a dictionary to get the right qubit pos based on tree_id and left_qubit_pos
        self.add_ports(["cout_right"])  # only one classical is enough to send the results for W qubits
        self._number_of_channels = number_of_subchannels
        self._avail_channels = 0
        self.add_ports(["qin_left" + str(idx) for idx in range(number_of_subchannels)])
        self.add_ports(["qin_right" + str(idx) for idx in range(number_of_subchannels)])

    def add_request(self, tree_id: int, left_qubit_pos: int, right_qubit_pos: int):
        if tree_id not in self.qubit_data:
            self.qubit_data[tree_id] = {}
        self.qubit_data[tree_id]["{l}-{r}".format(l=left_qubit_pos, r=right_qubit_pos)] =\
            ((None, -float('inf')), (None, -float('inf')))
        self._left_qubit_pos["{t}-{r}".format(t=tree_id, r=right_qubit_pos)] = left_qubit_pos
        self._right_qubit_pos["{t}-{l}".format(t=tree_id, l=left_qubit_pos)] = right_qubit_pos
        # always send signal to the right

    def avail_channels(self) -> int:
        avail = -1
        if self._avail_channels < self._number_of_channels:
            avail = self._avail_channels
            self._avail_channels += 1
        return avail

    @property
    def bsm_success_rate(self) -> float:
        return self._bsm_success_rate

    @property
    def bsm_time(self) -> float:
        return self._optical_bsm_time


class ClassicalConnection(Connection):
    """A connection that transmits classical messages in one direction, from A to B.

    Parameters
    ----------
    length : float
        End to end length of the connection [km].
    name : str, optional
       Name of this connection.

    """

    def __init__(self, length, name="ClassicalConnection"):
        super().__init__(name=name)
        self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length,
                                               models={"delay_model": FibreDelayModel()}),
                              forward_input=[("A", "send")],
                              forward_output=[("B", "recv")])


class QuantumConnection(Connection):
    """A connection that transmits quantum messages in one direction, from A to B.

        Parameters
        ----------
        length : float
            End to end length of the connection [km].
        name : str, optional
           Name of this connection.

        """
    def __init__(self, length: float, channel_success_rate: float, name="QuantumConnection"):
        super(QuantumConnection, self).__init__(name=name)
        self.add_subcomponent(QuantumChannel("QChannel_A2B", length=length,
                                             models={"delay_model": FibreDelayModel(),
                                                     "quantum_loss_model": FibreLossModel(
                                                         p_loss_init=1 - channel_success_rate,
                                                         p_loss_length=0)}),
                              forward_input=[("A", "send")],
                              forward_output=[("B", "recv")])


class EntanglingConnection(Connection):
    """A connection that generates entanglement.

    Consists of a midpoint holding a quantum source that connects to
    outgoing quantum channels.

    Parameters
    ----------
    length : float
        End to end length of the connection [km].
    source_frequency : float
        Frequency with which midpoint entanglement source generates entanglement [Hz].
    name : str, optional
        Name of this connection.

    """

    def __init__(self, length, source_frequency, name="EntanglingConnection"):
        super().__init__(name=name)
        qsource = QSource(f"qsource_{name}", StateSampler([ks.b00], [1.0]), num_ports=2,
                          timing_model=FixedDelayModel(delay=1e9 / source_frequency),
                          status=SourceStatus.INTERNAL)
        self.add_subcomponent(qsource, name="qsource")
        qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2,
                                      models={"delay_model": FibreDelayModel()})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])
