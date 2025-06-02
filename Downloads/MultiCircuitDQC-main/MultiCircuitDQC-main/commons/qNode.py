from .Point import Point
from .tools import time_slot_num, V_T, V_H, P_HT, TAU_H, TAU_T, TAU_P
from typing import List




GENERATION_TIME = TAU_P + max(TAU_H, TAU_T)  # time (seconds) it takes to generate an entangled pair of qubits
GEN_SUCCESS_RATE = P_HT * V_H * V_T  # probability of successfully generating an entangled pair  of qubits
BSM_SUCCESS_RATE = 1.0  # probability of successfully performing a BSM operations
BSM_TIME = 10e-6  # duration (seconds) of performing a BSM operation
FUSION_TIME = 10e-6  # duration (seconds) of performing GHZ fusion
FUSION_SUCCESS_RATE = 1.0  # probability of successfully performing GHZ FUSION operations


class qNode:
    """qNode is quantum node objects that can be a routing (repeater) and an end node (source and destination).
    It also capable of generating an entanglement pair of qubits, with a specific probability of success,
    between itself and its adjacent qNodes.
    EVery qNode has a memory capacity which is an integer and defines how many qubits it can save simultaneously."""
    _stat_id = 0

    def __init__(self, memory: int, loc: Point, gen_success_rate: float = GEN_SUCCESS_RATE,
                 generation_time: int = GENERATION_TIME, bsm_time: float = BSM_TIME,
                 bsm_success_rate: float = BSM_SUCCESS_RATE, node_id: int = -1, used_capacity: float = 0,
                 fusion_time: float = FUSION_TIME, fusion_success_rate: float = FUSION_SUCCESS_RATE):
        """
        :param memory: number of memories the qNode has
        :param gen_success_rate: successful rate (probability) for generating an entangled pair of qubits
        :param loc: location of the qNode
        :param generation_time: time (seconds) it takes to generate an entangled pair of qubits. 1 / generation_time
        is the highest capacity of node generation.
        :param bsm_time: duration (seconds) of performing a BSM operation
        :param bsm_success_rate: probability of successfully performing a BSM operation
        :param node_id: an internal id. you may never use it
        :param used_capacity this value (1/time) defines how much of node's capacity is already used
        """
        self._memory, self._gen_success_rate = memory, gen_success_rate
        self._loc = loc
        self._generation_time = generation_time
        self._bsm_time, self._bsm_success_rate = bsm_time, bsm_success_rate
        self._id = qNode._stat_id if node_id == -1 else node_id
        self._used_capacity = used_capacity
        if node_id == -1:
            qNode._stat_id += 1
        self._fusion_time = fusion_time
        self._fusion_success_rate = fusion_success_rate

    def success_generation_time(self) -> int:
        """
        :return: returns expected number of time slots it takes the qNode generate an entangled pair of qubits
        """
        remaining_generation_capacity_rate = max(1 / self.generation_time - self.used_capacity, 0)
        remaining_generation_time = 1 / remaining_generation_capacity_rate if remaining_generation_capacity_rate != 0 \
            else float('inf')
        return time_slot_num(remaining_generation_time / self._gen_success_rate)

    def clone(self, new_memory: int = -1, new_used_capacity: float = -1.0) -> 'qNode':
        """
        :param new_memory: new memory. Old one is used if not provided.
        :param new_used_capacity: new used_capacity. Old one is used if not provided
        :return: create an exact copy of the node
        """
        return qNode(memory=new_memory if new_memory != -1 else self.memory,
                     used_capacity=new_used_capacity if new_used_capacity != -1.0 else self.used_capacity,
                     loc=self.loc,
                     gen_success_rate=self.gen_success_rate, generation_time=self.generation_time,
                     bsm_success_rate=self.bsm_success_rate, bsm_time=self.bsm_time,
                     node_id=self.id,
                     fusion_success_rate=self.fusion_success_rate, fusion_time=self.fusion_time)

    def clone_with_new_id(self, id: int):
        """almost a clone, but the id is different
        :param id: new id
        """
        return qNode(memory=self.memory,
                     used_capacity=self.used_capacity,
                     loc=self.loc,
                     gen_success_rate=self.gen_success_rate, generation_time=self.generation_time,
                     bsm_success_rate=self.bsm_success_rate, bsm_time=self.bsm_time,
                     node_id=id)

    @property
    def used_capacity(self) -> float:
        return self._used_capacity

    @used_capacity.setter
    def used_capacity(self, new_value: float):
        # if new_value > self.max_capacity and int(new_value) != int(self.max_capacity):
        #     # raise ValueError("Used capacity is higher than maximum.")
        #     print("Used capacity is higher than maximum.")
        self._used_capacity = min(new_value, self.max_capacity)

    @property
    def remaining_capacity(self) -> float:
        """Return #of successful atom-photon / 1s"""
        if self.used_capacity > self.max_capacity:
            raise ValueError("Used capacity is higher than maximum.")
        if self.memory <= 0:  # no more memory to handle new paths
            return 0
        return self.max_capacity - self.used_capacity

    @property
    def loc(self) -> Point:
        return self._loc

    @property
    def bsm_success_rate(self) -> float:
        return self._bsm_success_rate

    @property
    def bsm_time(self) -> float:
        return self._bsm_time

    @property
    def generation_time(self) -> int:
        return self._generation_time

    @property
    def max_capacity(self) -> float:
        return 1.0 / self.generation_time

    @property
    def gen_success_rate(self) -> float:
        return self._gen_success_rate

    @property
    def id(self) -> int:
        return self._id

    @property
    def memory(self) -> int:
        return self._memory

    @memory.setter
    def memory(self, new_value):
        if new_value < 0:
            print(f"Warning! node memory violation in Node{self.id}")
            # raise ValueError("Memory of the node is negative")
        self._memory = new_value

    @property
    def fusion_time(self) -> float:
        return self._fusion_time

    @property
    def fusion_success_rate(self) -> float:
        return self._fusion_success_rate

    def __str__(self):
        return "qNOde:{id}, mem={mem}\n{loc}".format(id=self.id, mem=self.memory, loc=self.loc)
