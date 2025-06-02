import math
from netsquid.components import QuantumMemory
from typing import List
import netsquid as ns
from netsquid.components.instructions import INSTR_H, INSTR_CNOT
from netsquid.qubits.qubitapi import delay_dephase
from collections import namedtuple

TIME_SLOT = 1e-6  # smallest scale of time (seconds)
ATT_LENGTH = 22e3  # Attenuation length of the medium is used. For fiber: 22km
LIGHT_SPEED = 2e8  # light speed in fiber
TAU_P, TAU_H, TAU_T, TAU_D, TAU_O = 5.9e-6, 20e-6, 10e-6, 100e-6, 10e-6
# some physical parameters related to generating an entangled pair. all in seconds
V_O = 1.0  # optical BSM efficiency for entanglement generation
P_HT, V_H, V_T = 1.0, 0.8, 0.8  # some physical probabilities related to generating an entangled pair
T_CH = 10e-3  # decoherence time (seconds)

SL = namedtuple("SL", ("src", "dst", "tree", "cost"))


def get_classical_time(distance: float):
    """
    :param distance: distance where the signal travels in its medium (fiber)
    :return: number of time slots it takes to send a signal travel over the distance
    """
    return distance / LIGHT_SPEED


def bell_measurement(qmemory: QuantumMemory, positions: List[int], PHYSICAL: bool):
    """@qmemory: the memory that BSM is operating on
        @positions: BSM is operating on two memory positions such that positions[0] is teleported
    """
    if len(positions) != 2:
        raise ValueError("list's length of positions should be two.")
    if qmemory.num_positions < 2:
        raise ValueError("input qmemory does not have enough slots")

    if max(positions) > qmemory.num_positions - 1:
        raise ValueError("One of the positions is out of range")

    # if not PHYSICAL:
    if PHYSICAL:
        delay_dephase(qmemory.peek(positions[0])[0], 1000, 10000)
    qmemory.operate(ns.CNOT, positions)
    qmemory.operate(ns.H, positions[0])
    # else:
    #     INSTR_CNOT(qmemory, positions=positions)
    #     INSTR_H(qmemory, positions=positions)

    m, _ = qmemory.measure(positions, observable=ns.Z)

    return m


def fusion(qmemory: QuantumMemory, positions: List[int], RETAIN: bool = False, PHYSICAL: bool = False):
    """@qmemory: the memory that fusion is operating on
        @positions: fusion is operating on two memory positions such that positions[0] is maintained in the final
        GHZ state unless it's not supposed to (RETAIN = False)
        @RETAIN: If set, one of the qubit (the first one in the positions) will be maintained
        @PHYSICAL: If set, some noise will be introduced
    """
    if len(positions) != 2:
        raise ValueError("list's length of positions should be two.")
    if qmemory.num_positions < 2:
        raise ValueError("input qmemory does not have enough slots")

    if max(positions) > qmemory.num_positions - 1:
        raise ValueError("One of the positions is out of range")

    if PHYSICAL:
        delay_dephase(qmemory.peek(positions[0])[0], 1000, 10000)
    qmemory.operate(ns.CNOT, positions)
    m, _ = qmemory.measure(positions[1], observable=ns.Z)

    mm = [0]
    if not RETAIN:
        mm, _ = qmemory.measure(positions[0], observable=ns.X)

    return m[0], mm[0]



def time_slot_num(time: float):
    """
    convert time (seconds) to number of time-slots
    :param time: time
    :return: number of time slots based on TIME_SLOT
    """
    if time == float('inf'):
        return float('inf')
    return math.ceil(time / TIME_SLOT) if time > 0 else 0


def time_slot_to_time(time_slot_number: int):
    """
    convert number of time-slots to real time
    :param time_slot_number: number of time-slots
    :return: time (seconds)
    """
    return time_slot_number * TIME_SLOT


def get_min_max(u: int, v: int):
    '''sometimes, you want an order between u and v
    Args:
        u: an index of a node
        u: an index of a node
    '''
    return min(u, v), max(u, v)


class DisJointSets:
    def __init__(self, N):
        # Initially, all elements are single element subsets
        self._parents = [node for node in range(N)]
        self._ranks = [1 for _ in range(N)]

    def find(self, u):
        while u != self._parents[u]:
            # path compression technique
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]
        return u

    def connected(self, u, v):
        return self.find(u) == self.find(v)

    def union(self, u, v):
        # Union by rank optimization
        root_u, root_v = self.find(u), self.find(v)
        if root_u == root_v:
            return True
        if self._ranks[root_u] > self._ranks[root_v]:
            self._parents[root_v] = root_u
        elif self._ranks[root_v] > self._ranks[root_u]:
            self._parents[root_u] = root_v
        else:
            self._parents[root_u] = root_v
            self._ranks[root_v] += 1
        return False
