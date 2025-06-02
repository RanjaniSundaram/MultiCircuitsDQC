import sys

sys.path.append("..")

from netsquid.nodes import Node, Connection
from commons.qNode import qNode, BSM_SUCCESS_RATE
from netsquid.components import QuantumMemory
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel
from netsquid.qubits import StateSampler
import netsquid.qubits.ketstates as ks
from commons.Point import Point
from collections import defaultdict
from netsquid.examples import repeater_chain


class qPNode(Node):
    parents = {}  # a dict that holds a tuple (parent, next_parent) with the key of GHZ, used in ghz

    def __init__(self, node: qNode):
        memory_const = 4
        super().__init__("Node" + str(node.id), ID=node.id,
                         qmemory=QuantumMemory(self.name + "Memory", num_positions=memory_const * node.memory,
                                               memory_noise_models=[DepolarNoiseModel(
                                                   0.05)] * memory_const * node.memory))  # 0.245e-3
        self.mem_pos = -1
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
        # For GHZ states
        self._fusion_success_rate = node.fusion_success_rate
        self._fusion_time = node.fusion_time
        self.ghz_edges = defaultdict(set)  # a set of edges involved for a ghz_state, key={tree_id}_{endnodes}
        self.running = {}  # a dict that defines if an entanglement generator for an edge is running. key:{tree}_{other}
        self.all_nodes = {}  # a dict that holds all the nodes in the tree rooted at this node, used in ghz
        self.src_dst_ep_active = {}  # a dict, used in fro, to not accepting new EPs from its trees when a ghz is going
        self.qubit_creation_time = [-float('inf')] * memory_const * node.memory  # time a qubit is delivered from link
        self.acceptable_time = {}  # a dict, used in single or a set of trees, for acceptable new qubits
        self.last_starting_signal = {}  # a dict that holds the last received starting time to resume generating a gen
        self.last_stopping_signal = {}  # a dict that holds the last received starting time to stop generating a gen
        self.last_accepting_ep_signal = {}
        self.last_not_accepting_ep_signal = {}


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
    def fusion_success_rate(self) -> float:
        return self._fusion_success_rate

    @property
    def fusion_time(self) -> float:
        return self._fusion_time


# ************************* FOR Multipartite (GHZ) *************************

class qPNodeGHZ(qPNode):
    # All dictionaries in the base class will be used with a key format of "tree_id_endnode1-endnode2_..." such that
    # list of end nodes is sorted.
    # For key generation, suffix 0 will be used for left subtree and suffix 1 for right subtrees.
    # For qubit_mapping, if any leg of fusion tree has a height of 1, i.e. directly operating on link layer, a tuple of
    # (network_qubit_pos, link_layer_qubit_pos) will be used; otherwise, "None" will be used for link_layer_qubit_pos
    def __init__(self, node: qNode):
        super(qPNodeGHZ, self).__init__(node)
        self._fusion_success_rate = node.fusion_success_rate
        self._fusion_time = node.fusion_time
        # list of all end nodes in the GHZ state (left and right). used for sending fusion results
        self.end_nodes = defaultdict(set)
        # list of all nodes involved in the GHZ states (left and right), end or intermediate nodes. Used in case of
        # fusion failure to accept new EP from link layer
        self.all_nodes = defaultdict(set)

    @property
    def fusion_success_rate(self):
        return self._fusion_success_rate

    @property
    def fusion_time(self):
        return self._fusion_time


# ************************* FOR Multipartite (GHZ) *************************


class Charlie(Node):
    """Charlie Node"""

    def __init__(self, left_node_id: int, right_node_id: int, bsm_success_rate: float, bsm_time: float):
        super(Charlie, self).__init__(name="Charlie_" + str(left_node_id) + "_" + str(right_node_id))
        self._bsm_success_rate, self._optical_bsm_time = bsm_success_rate, bsm_time
        self.qubit_data = {}  # key=tree_id,value: a tuple that store qubit & time for both side
        self.add_ports(["qin_left", "qin_right", "cout_right"])  # always send signal to the right
        self._left_node_id, self._right_node_id = left_node_id, right_node_id

    def add_request(self, tree_id: int):
        self.qubit_data[tree_id] = ((None, -float('inf')), (None, -float('inf')))

    @property
    def bsm_success_rate(self) -> float:
        return self._bsm_success_rate

    @property
    def bsm_time(self) -> float:
        return self._optical_bsm_time

    @property
    def left_node_id(self):
        return self._left_node_id

    @property
    def right_node_id(self):
        return self._right_node_id


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
