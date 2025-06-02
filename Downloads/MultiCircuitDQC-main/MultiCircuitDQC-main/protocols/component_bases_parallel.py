import sys
sys.path.append("..")

from netsquid.nodes import Node
from commons.qNode import qNode
from netsquid.components import QuantumMemory
from commons.Point import Point


class qPNode(Node):
    def __init__(self, node: qNode):
        super().__init__("ProtocolNode" + str(node.id), ID=node.id,
                         qmemory=QuantumMemory(self.name + "Memory", num_positions=node.memory))
        # TODO memory should be 2nlog(n) for parallel
        self.mem_pos = -1
        self._loc = node.loc
        self._leftest_id, self._rightest_id = None, None
        self.qubit_mapping = {}
        self._leftest_qubit_pos, self._rightest_qubit_pos = -1, -1
        self.init_left_qubit_pos, self.init_right_qubit_pos = -1, -1

    def avail_mem_pos(self) -> int:
        self.mem_pos += 1
        return self.mem_pos

    @property
    def loc(self) -> Point:
        return self._loc

    @property
    def leftest_id(self) -> int:
        return self._leftest_id

    @leftest_id.setter
    def leftest_id(self, ID: int):
        self._leftest_id = ID

    @property
    def rightest_id(self) -> int:
        return self._rightest_id

    @rightest_id.setter
    def rightest_id(self, ID: int):
        self._rightest_id = ID

    @property
    def leftest_qubit_pos(self) -> int:
        return self._leftest_qubit_pos

    @leftest_qubit_pos.setter
    def leftest_qubit_pos(self, pos: int):
        self._leftest_qubit_pos = pos

    @property
    def rightest_qubit_pos(self) -> int:
        return self._rightest_qubit_pos

    @rightest_qubit_pos.setter
    def rightest_qubit_pos(self, pos: int):
        self._rightest_qubit_pos = pos