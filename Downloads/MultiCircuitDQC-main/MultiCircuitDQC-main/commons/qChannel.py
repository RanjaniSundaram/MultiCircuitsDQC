from .qNode import qNode
from .tools import ATT_LENGTH, TAU_T, TAU_P, TAU_H, TAU_D, get_classical_time, \
    time_slot_num, TAU_O, V_O, time_slot_to_time
import math


class qChannel:
    """
    qChannel simulate a quantum channel connecting two qNodes and take its probabilistic nature into measurements.
    To create an entangled pair of qubits, a third node in the middle is assumed.
    """

    def __init__(self, this: qNode, other: qNode, channels_num: int = 1, optical_bsm_rate: float = 0.5 * V_O,
                 optical_bsm_time: float = TAU_O):
        """
        :param this: an qNode end
        :param other: an qNOde end
        :param channels_num: number of parallel sub channels a qChannel can have.
        :param length: is used for cases when the real distance is ignored (used for sigcomm)
        """
        self._this, self._other = this, other
        self._len = this.loc.distance(other.loc)  # distance between this and other
        self._channel_success_rate = math.exp(-self._len / (2 * ATT_LENGTH))  # successful rate of delivering an
        # entangled pair successfully. It's related to inverse of exp of distance. Charlie is in the middle
        self._optical_bsm_rate = optical_bsm_rate
        self._optical_bsm_time = optical_bsm_time
        self._tau = optical_bsm_time + get_classical_time(self._len / 2)  # len / 2 as charlie is supposed to be in the middle
        # time (seconds) of sending an entangled pair along qChannel. This is inverse of maximum capacity for an edge
        self._t_s = TAU_P + max(TAU_H, self._tau)  # average time (seconds) to establish one round of the qChannel link
        self._current_flow = 0
        self._channels_num = channels_num

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def optical_bsm_time(self) -> float:
        return self._optical_bsm_time

    @property
    def length(self) -> float:
        return self._len

    def get_avr_ent_time(self, this_usage: int = 50, other_usage: int = 50) -> float:
        """General methode to calculate avr_ent_time based on given usages for left and right nodes"""
        capacity = self.get_residual_capacity(this_usage=this_usage, other_usage=other_usage)
        return float('inf') if capacity == 0 else 1.0 / capacity
    avr_ent_time = property(get_avr_ent_time)

    # @property
    # def avr_ent_time(self) -> float:
    #     """return average time (# time slots) to generate an entanglement between the two nodes.
    #     This value includes both successful and failure tries"""
    #     return float('inf') if self.residual_capacity == 0 else 1.0 / self.residual_capacity

    @property
    def avr_success_time(self) -> float:
        """"The time (# time slots) one (successful) entanglement try takes"""
        return self._t_s

    @property
    def optical_bsm_rate(self) -> float:
        return self._optical_bsm_rate

    @property
    def current_flow(self) -> float:
        return self._current_flow

    @current_flow.setter
    def current_flow(self, new_flow):
        # if new_flow > self.max_channel_capacity and int(new_flow) != int(self.max_channel_capacity):
        #     # raise ValueError("The new flow is higher than maximum possible for this edge.")
        #     print("The new flow is higher than maximum possible for this edge.")
        self._current_flow = min(new_flow, self.max_channel_capacity)

    def get_residual_capacity(self, this_usage: int = 50, other_usage: int = 50) -> float:
        # general residual capacity calculations
        max_flow_nodes = min(self.this.remaining_capacity * this_usage / 100, self.other.remaining_capacity *
                             other_usage / 100) * \
                         self.this.gen_success_rate * self.other.gen_success_rate * self.channel_success_rate ** 2 * \
                         self.optical_bsm_rate
        return min(max_flow_nodes, self.max_channel_capacity - self.current_flow)

    # use half of nodes' generation rate. min gives us maximum possible rate between two nodes
    residual_capacity = property(get_residual_capacity)

    # @property
    # def residual_capacity(self, this_usage: int = 50, other_usage: int = 50) -> float:
    #     # use half of nodes' generation rate. min gives us maximum possible rate between two nodes
    #     max_flow_nodes = min(self.this.remaining_capacity * this_usage / 100, self.other.remaining_capacity *
    #                          other_usage / 100) * \
    #                      self.this.gen_success_rate * self.other.gen_success_rate * self.channel_success_rate ** 2 * \
    #                      self.optical_bsm_rate
    #     return min(max_flow_nodes, self.max_channel_capacity - self.current_flow)

    @property
    def max_channel_capacity(self):
        return (self._channels_num / self._tau) * self.channel_success_rate ** 2 * self.optical_bsm_rate * \
               min(self.this.gen_success_rate, self.other.gen_success_rate)

    def max_channel_capacity_delftlp(self):
        '''after discussion with Himanshu, decide to use this for delft lp
        '''
        return self._channels_num / self._tau

    @property
    def this(self) -> qNode:
        return self._this

    @property
    def other(self) -> qNode:
        return self._other

    @property
    def channel_success_rate(self) -> float:
        return self._channel_success_rate

    @property
    def channels_num(self):
        return self._channels_num

    @channels_num.setter
    def channels_num(self, num: int):
        self._channels_num = num

    @staticmethod
    def update_flows_cancel(e1: 'qChannel', e2: 'qChannel'):
        """Update flows of a pair of edges between a pair of nodes"""
        if e1.this != e2.other or e1.other != e2.this:
            raise ValueError("e1 is not the opposite directed edge of e2.")
        if e1.current_flow > e2.current_flow:
            e1.current_flow = e1.current_flow - e2.current_flow
            e2.current_flow = 0
        else:
            e2.current_flow = e2.current_flow - e1.current_flow
            e1.current_flow = 0

    @staticmethod
    def update_residual_capacities_cancel(e1: 'qChannel', e2: 'qChannel'):
        """
        This function calculate residual capacities of the two edges e1 and e2.
        These edges must be between a unique pair of qNodes and they should not have flow at the same time
        :param e1: a qChannel edge
        :param e2: the counter-directed edge of e2
        """
        if e1.this != e2.other or e1.other != e2.this:
            raise ValueError("e1 is not the opposite directed edge of e2.")
        if e1.current_flow != 0 and e2.current_flow != 0:
            raise ValueError("Both edges have flow at the same time")
        # use half of nodes' generation rate. min gives us maximum possible rate between two nodes
        max_flow_nodes = min(e1.this.remaining_capacity / 2 * e1.this.gen_success_rate,
                             e1.other.remaining_capacity / 2 * e1.other.gen_success_rate) * e1.channel_success_rate
        forward, backward = (e1, e2) if e1.current_flow > 0 else (e2, e1)
        forward_flow = forward.current_flow
        forward.residual_capacity = min(max_flow_nodes, forward.max_channel_capacity - forward_flow)
        backward.residual_capacity = min(max_flow_nodes + forward_flow, backward.max_channel_capacity)

    def __str__(self):
        return "Node1:{node1}\nNode2:{node2}\nsuccess rate={ss_rate}" \
               "\navr_ent_time={ent_t}\nchannel_num={num_chan}".format(node1=self.this, node2=self.other,
                                                                       ss_rate=self.channel_success_rate,
                                                                       ent_t=self.avr_ent_time,
                                                                       num_chan=self.channels_num)

    @property
    def link_success_rate(self):
        '''a charlie in the middle. five events need to success all at once.
        '''
        return self.this.gen_success_rate * self.other.gen_success_rate * \
               self.channel_success_rate * self.channel_success_rate * self.optical_bsm_rate
