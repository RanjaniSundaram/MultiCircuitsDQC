import sys

sys.path.append("..")

from commons.qNode import qNode
from commons.qChannel import qChannel
from commons.tree_node import tree_node
import netsquid as ns
from protocols.protocol_bases_parallel import *
from netsquid.nodes import Network
from protocols.component_bases_parallel import qPNode
from protocols.component_bases import ClassicalConnection, EntanglingConnection
from typing import List


from netsquid.components import QuantumMemory
from commons.Point import Point
from netsquid.nodes import Node

PORT_NAMES_CLASSICAL = ["c" + inOrOut + "_" + direction for inOrOut in ["in", "out"]
                        for direction in ["left", "right", "parent"]]
PORT_NAMES_QUANTUM = ["q" + inOrOut + "_" + direction for inOrOut in ["in", "out"]
                      for direction in ["left", "right"]]


class TreeProtocol:

    def __init__(self, nodes: List[qNode], tree: tree_node, src: qNode, dst: qNode, duration: int):
        ns.set_qstate_formalism(ns.QFormalism.DM)
        ns.sim_reset()
        network = Network("network", [qPNode(node) for node in nodes])
        protocols = []
        TreeProtocol.setup_tree_connections(network=network, node=tree, protocols=protocols)
        # TODO add src & dst ports&connections&protocols
        srcQPNode = network.get_node("ProtocolNode" + str(src.id))
        dstQPNOde = network.get_node("ProtocolNode" + str(dst.id))
        # add ports
        srcQPNode.add_ports(["cto_node" + str(dstQPNOde.ID)])
        dstQPNOde.add_ports(["cfrom_node" + str(srcQPNode.ID)])
        # add a classical connection from src to dst
        src_dst_cchannel = ClassicalConnection(srcQPNode.loc.distance(dstQPNOde.loc),
                                                  srcQPNode.name + "->" + dstQPNOde.name + "|classical")
        network.add_connection(srcQPNode, dstQPNOde, connection=src_dst_cchannel,
                               port_name_node1="cto_node" + str(dstQPNOde.ID),
                               port_name_node2="cfrom_node" + str(srcQPNode.ID),
                               label="classical")
        # add protocols
        protocols.append(TeleportationProtocol(srcQPNode, QubitGenerationProtocol(srcQPNode), dstQPNOde.ID))
        protocols.append(TeleportationCorrectionProtocol(dstQPNOde, srcQPNode.ID))
        for protocol in protocols:
            protocol.start()
        stats = ns.sim_run(duration=duration)

    @staticmethod
    def setup_tree_connections(network: Network, protocols: List, node: tree_node,
                               parent: tree_node = None, tree_id: int = 0):
        if isinstance(node.data, qChannel):
            # get qNodes based on name
            qpnode1 = network.get_node("ProtocolNode" + str(node.data.this.id))
            qpnode2 = network.get_node("ProtocolNode" + str(node.data.other.id))

            # add ports for this quantum channel
            qpnode1.add_ports(PORT_NAMES_QUANTUM)
            qpnode2.add_ports(PORT_NAMES_QUANTUM)

            # assigning qubits
            qpnode1_qubit_pos = qpnode1.avail_mem_pos()
            qpnode2_qubit_pos = qpnode2.avail_mem_pos()

            # this (qpnode1) as left, other (qpnode2) as right
            qpnode1.rightest_qubit_pos = qpnode1_qubit_pos
            qpnode1.init_right_qubit_pos = qpnode1_qubit_pos
            qpnode2.leftest_qubit_pos = qpnode2_qubit_pos
            qpnode2.init_left_qubit_pos = qpnode2_qubit_pos

            # quantum entangling
            qconn = EntanglingConnection(node.data.this.loc.distance(node.data.other.loc), 0.01,
                                         qpnode1.name + "-" + qpnode2.name + "|Entangling")  # frequency should be based on demand
            port_ac, port_bc = network.add_connection(qpnode1, qpnode2, connection=qconn,
                                                      label="quantum",
                                                      port_name_node1="qin_right",
                                                      port_name_node2="qin_left")
            qpnode1.ports[port_ac].forward_input(qpnode1.qmemory.ports["qin" + str(qpnode1_qubit_pos)])
            qpnode2.ports[port_bc].forward_input(qpnode2.qmemory.ports["qin" + str(qpnode2_qubit_pos)])

            # # classical connections
            # add_classical_connections(network, qpnode1, qpnode2)
            return qpnode1, qpnode2
        nodeqp = network.get_node("ProtocolNode" + str(node.data.id))
        left_left, _ = TreeProtocol.setup_tree_connections(network=network, protocols=protocols,
                                                           node=node.left, parent=node)
        _, right_right = TreeProtocol.setup_tree_connections(network=network, protocols=protocols,
                                                             node=node.right, parent=node)
        # classical connections
        TreeProtocol.add_tree_classical_connection(network, nodeqp, left_left, right_right)
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

        # adding & starting protocol
        # subtree (left_left -- nodeqp -- right_right
        protocols.append(SwappingProtocol(nodeqp))  # swap at this root
        protocols.append(CorrectionProtocol(right_right, parent_id=nodeqp.ID, leftest_id=left_left.ID))
        protocols.append(CorrectionAckProtocol(left_left, parent_id=nodeqp.ID, rightest_id=right_right.ID))

        return left_left, right_right

    @staticmethod
    def add_tree_classical_connection(network: Network, node: qPNode, leftest: qPNode, rightest: qPNode):
        # add ports
        node.add_ports(["cto_node" + str(rightest.ID)])
        rightest.add_ports(["cfrom_node" + str(node.ID), "cto_node" + str(leftest.ID)])
        leftest.add_ports(["cfrom_node" + str(rightest.ID)])
        # node to rightest connections to send BSM res
        # if network.get_connection(node, rightest, "classical") is None and \
        #         not node.ports["cto_node" + str(rightest.ID)].is_connected and \
        #         not rightest.ports["cfrom_node" + str(node.ID)].is_connected:
        node_right_cchannel = ClassicalConnection(node.loc.distance(rightest.loc),
                                                  node.name + "->" + rightest.name + "|classical")
        network.add_connection(node, rightest, connection=node_right_cchannel,
                               port_name_node1="cto_node" + str(rightest.ID),
                               port_name_node2="cfrom_node" + str(node.ID),
                               label="classical")

        # rightest to leftest connections to send CORRECTION_ACK
        # if network.get_connection(rightest, leftest, "classical") is None and \
        #         not rightest.ports["cto_node" + str(leftest.ID)].is_connected and \
        #         not leftest.ports["cfrom_node" + str(rightest.ID)].is_connected:
        right_left_cchannel = ClassicalConnection(rightest.loc.distance(leftest.loc),
                                                  rightest.name + "->" + leftest.name + "|classical")
        network.add_connection(rightest, leftest, connection=right_left_cchannel,
                               port_name_node1="cto_node" + str(leftest.ID),
                               port_name_node2="cfrom_node" + str(rightest.ID),
                               label="classical")


def add_classical_connections(network:Network, node1: qPNode, node2: qPNode):
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
    nodes = [qNode(20, Point(0, 5000)), qNode(20, Point(0, 10000)), qNode(20, Point(0, 15000)),
             qNode(20, Point(0, 20000)), qNode(20, Point(0, 25000))]
    root = tree_node(nodes[2],
                     tree_node(nodes[1], tree_node(qChannel(nodes[0], nodes[1])),
                               tree_node(qChannel(nodes[1], nodes[2]))),
                     tree_node(nodes[3], tree_node(qChannel(nodes[2], nodes[3])),
                               tree_node(qChannel(nodes[3], nodes[4])))
                     )
    network_protocol = TreeProtocol(nodes, root, nodes[0], nodes[4], 1e12)

