import sys
sys.path.append("..")

from netsquid.nodes import Node, Connection
from commons.qNode import qNode
from netsquid.components import QuantumMemory
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel
from netsquid.qubits import StateSampler
import netsquid.qubits.ketstates as ks
from commons.Point import Point
from collections import defaultdict, OrderedDict


class qPNode(Node):
    def __init__(self, node: qNode):
        super().__init__("Node" + str(node.id), ID=node.id,
                         qmemory=QuantumMemory(self.name + "Memory", num_positions=20 * node.memory,
                                               memory_noise_models=[DepolarNoiseModel(0.245e-3)] * 20 * node.memory))
        self.mem_pos = {}  # free memory position available for each (s-d) tree.Multiple (s-d) pair won't be requested
        self._loc = node.loc
        self._bsm_success_rate = node.bsm_success_rate
        self._bsm_time = node.bsm_time
        self.leftest_id, self.rightest_id = {}, {}  # (key=tree_id) leftest and rightest node if rooted at this node
        self.qubit_mapping = {}
        # each tree has two tuples (left_qubit_pos, right_qubit_pos), (left_link_qubit_pos, right_qubit_pos)
        # (left_qubit_pos, right_qubit_pos) is for BSM and (left_link_qubit_pos, right_qubit_pos) is for link layer
        self.children_id = defaultdict(set)  # set of children ids used when BSM fails, direct connection to each
        self.left_height, self.right_height = {}, {}  # (key=tree_id) holds the height of left and right subtree
        self.entangling = {}  # a dictionary that indicates node should accept new qubits from link layer for left&right
        self.neighbors = {}  # a dictionary to hold left-right neighbors if node is in the tree
        self.sl_saved_pos = defaultdict(OrderedDict)  # holds generated EPs at SL ends
        self.right_sl_id, self.left_sl_id = {}, {}  # info about if left or right of a node is SL
        self.accepting_eps = {}  # this is used for cases when new eps are received after STOP_GENERATING signal
        self.sl_nodes_id = defaultdict(set)  # used to hold ids of an sl path; for restarting
        self.other_sl_id = {}  # if the node is an end node of an sl, record the other end node id
        self.is_left_sl, self.is_right_sl = set(), set()  # define if left or right qubit is an SL end

    # def avail_mem_poss(self, sd_id: int) -> int:
    #     if sd_id not in self.mem_pos:
    #         if 0 in self.mem_pos:
    #             self.mem_pos[sd_id] = self.mem_pos[0] + NUMBER_OF_SL_MEMORIES
    #         else:
    #             self.mem_pos[sd_id] = -1
    #     self.mem_pos[sd_id] += 1
    #     return self.mem_pos[sd_id]

    def avail_mem_pos(self, sd_id: int, num: int = 1):
        if sd_id in self.mem_pos:
            pos = self.mem_pos[sd_id]
            self.mem_pos[sd_id] += num
        else:
            if 0 in self.mem_pos:
                self.mem_pos[sd_id] = self.mem_pos[0] + num
                pos = self.mem_pos[0]
            else:
                pos = 0
                self.mem_pos[sd_id] = num
        return pos

    @property
    def loc(self) -> Point:
        return self._loc

    @property
    def bsm_time(self) -> float:
        return self._bsm_time

    @property
    def bsm_success_rate(self) -> float:
        return self._bsm_success_rate


class Charlie(Node):
    """Charlie Node"""
    def __init__(self, left_node_id: int, right_node_id: int, bsm_success_rate: float, bsm_time: float):
        super(Charlie, self).__init__(name="Charlie_" + str(left_node_id) + "_" + str(right_node_id))
        self._bsm_success_rate, self._optical_bsm_time = bsm_success_rate, bsm_time
        self.qubit_data = {}  # key=tree_id,value: a tuple that store qubit & time for both side
        self.add_ports(["qin_left", "qin_right", "cout_right"])  # always send signal to the right
        self._left_node_id, self._right_node_id = left_node_id, right_node_id

    def add_request(self, tree_id: int, sd_id: int):
        if sd_id not in self.qubit_data:
            self.qubit_data[sd_id] = {}
        self.qubit_data[sd_id][tree_id] = ((None, -float('inf')), (None, -float('inf')))

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
