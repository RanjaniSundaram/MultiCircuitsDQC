'''
link layer (EGProtocol)
physical layer (MidpointHeraldingProtocol)
SIGCOMM 2019 paper, https://docs.netsquid.org/latest-release/learn_examples/learn.examples.simple_link.html
'''

import sys
sys.path.append('..')

import abc
import time
import numpy as np
import operator
import random
from functools import reduce
from netsquid.qubits.qformalism import QFormalism
from collections import namedtuple, deque
import netsquid.components.instructions as instr
import netsquid as ns
from netsquid.nodes import Network
from netsquid.components.qprocessor import QuantumProcessor, QuantumProgram
from netsquid.nodes.connections import Connection
from netsquid.protocols import NodeProtocol
from netsquid.protocols.protocol import Protocol, Signals
from netsquid.protocols.nodeprotocols import DataNodeProtocol
from netsquid.protocols.serviceprotocol import ServiceProtocol
from netsquid.components.component import Message
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qdetector import QuantumDetector
from netsquid.qubits.operators import Operator

from commons.qNode import GENERATION_TIME
GENERATION_TIME_NETSQUID = int(GENERATION_TIME / 1e-9)   # in nano-seconds

DEBUG = False  # if True, there will be some output from print()


class HeraldedConnection(Connection):
    '''a connection that takes in two qubits, and returns a message how they were measured at a detector
    Either no clicks, a signal click or double click, or an error when the qubits didn't arrive within the time window

    Args:
        name: str -- the name of this connection
        length_to_a: float -- the length in km between the detector and side A. We assume a speed of 200000 km/s
        length_to_b: float -- the length in km between the detector and side B.
        time_window: float -- the interval where qubits are still able to be measured correctly
    '''
    def __init__(self, name, length_to_a, length_to_b, time_window=0):
        super().__init__(name)
        delay_a = length_to_a / 200000 * 1e9
        delay_b = length_to_b / 200000 * 1e9
        channel_a = ClassicalChannel('ChannelA', delay=delay_a) # TODO replace
        channel_b = ClassicalChannel('ChannelB', delay=delay_b)
        qchannel_a = QuantumChannel('QChannelA', delay=delay_a)
        qchannel_b = QuantumChannel('QChannelB', delay=delay_b)
        # add all channels as subcomponents
        self.add_subcomponent(channel_a)
        self.add_subcomponent(channel_b)
        self.add_subcomponent(qchannel_a)
        self.add_subcomponent(qchannel_b)
        # add midpoint detector
        detector = BSMDetector('Midpoint', system_delay=time_window)
        self.add_subcomponent(detector)
        # connect the ports
        self.ports['A'].forward_input(qchannel_a.ports['send'])
        self.ports['B'].forward_input(qchannel_b.ports['send'])
        qchannel_a.ports['recv'].connect(detector.ports['qin0'])
        qchannel_b.ports['recv'].connect(detector.ports['qin1'])
        channel_a.ports['send'].connect(detector.ports['cout0'])
        channel_b.ports['send'].connect(detector.ports['cout1'])
        channel_a.ports['recv'].forward_output(self.ports['A'])  # bug: this is forward_output, not connect
        channel_b.ports['recv'].forward_output(self.ports['B'])


def create_meas_ops(visibility=1):
    """Sets the photon beamsplitter POVM measurements.

    We are assuming a lot here, see the Netsquid-Physlayer snippet for more info.

    Parameters
    ----------
    visibility : float, optional -- The visibility of the qubits in the detector.

    """
    mu = np.sqrt(visibility)
    s_plus = (np.sqrt(1 + mu) + np.sqrt(1 - mu)) / (2. * np.sqrt(2))
    s_min = (np.sqrt(1 + mu) - np.sqrt(1 - mu)) / (2. * np.sqrt(2))
    m0 = np.diag([1, 0, 0, 0])
    ma = np.array([[0, 0, 0, 0],
                   [0, s_plus, s_min, 0],
                   [0, s_min, s_plus, 0],
                   [0, 0, 0, np.sqrt(1 + mu * mu) / 2]],
                  dtype=np.complex)
    mb = np.array([[0, 0, 0, 0],
                   [0, s_plus, -1. * s_min, 0],
                   [0, -1. * s_min, s_plus, 0],
                   [0, 0, 0, np.sqrt(1 + mu * mu) / 2]],
                  dtype=np.complex)
    m1 = np.diag([0, 0, 0, np.sqrt(1 - mu * mu) / np.sqrt(2)])
    n0 = Operator("n0", m0)
    na = Operator("nA", ma)
    nb = Operator("nB", mb)
    n1 = Operator("n1", m1)
    return [n0, na, nb, n1]


class BSMDetector(QuantumDetector):
    '''a component that performs bell basis measurements

    Measure two incoming qubits in the Bell basis if they arrive within the specified measurement delay
    only informs the connections that send a qubit of the measurement result
    '''
    def __init__(self, name, system_delay=0., dead_time=0., models=None, output_meta=None, error_on_fail=False, properties=None):
        super().__init__(name, num_input_ports=2, num_output_ports=2, meas_operators=create_meas_ops(), system_delay=system_delay, 
                         dead_time=dead_time, models=models, output_meta=output_meta, error_on_fail=error_on_fail, properties=properties)
        self._sender_ids = []

    def preprocess_inputs(self):
        '''preprocess and capture the qubit metadata
        '''
        super().preprocess_inputs()
        for port_name, qubit_list in self._qubits_per_port.items():
            if len(qubit_list) > 0:
                self._sender_ids.append(port_name[3:])

    def measure(self):
        '''perform a measurement on the received qubits
        '''
        super().measure()   # will not raise error because of error_on_fail = False

    def inform(self, port_outcomes):
        '''inform the MHP of the measurement result
        send a result to the node that send a qubit. if the result is empty, change the result and header
        
        Args:
            port_outcomes: dict -- a dictionary with the port names as keysand the post-processed measurement outcomes as values
        '''
        for port_name, outcomes in port_outcomes.items():
            if len(outcomes) == 0:
                outcomes = ['TIMEOUT']
                header = 'error'
            else:
                header = 'photonoutcome'
            # extract the ids from the port names (cout...)
            if port_name[4:] in self._sender_ids:
                msg = Message(outcomes, header=header, **self._meta)
                self.ports[port_name].tx_output(msg)

    def finish(self):
        '''clear sender ids after the measurement has finished
        '''
        super().finish()
        self._sender_ids.clear()


class MidpointHeraldingProtocol(NodeProtocol):
    '''
    Args:
        node: `~netsquid.nodes.node.Node` -- the node this protocol runs on
        time_step: int    -- the period of the triggers in nanoseconds
        q_port_name: str  -- the name of port connected to the heraled connection
        request_name: str -- virtual generators, i.e. each MHP is associated with a request    
    Attributes:
        trigger_label: str -- the label of the trigger signal
        answer_label: str -- the label of the answer signal
    '''
    def __init__(self, node, time_step, q_port_name, request_name, emit_protocol):
        super().__init__(node=node)
        self.time_step = time_step
        self.node.qmemory.ports['qout'].forward_output(self.node.ports[q_port_name])
        # wait for an outcome on the input port
        self.port_name = q_port_name
        self.q_port_name = q_port_name
        self.request_name = request_name
        # each request will have a unique random number generator, alice and bob's same request MHP will have the same random_num_gen
        self.random_num_gen = random.Random(request_name)
        # remember if we already send a photon
        self.trigger_label = f'TRIGGER-{request_name}'
        self.answer_label = f'ANSWER-{request_name}'
        self.add_signal(self.trigger_label)
        self.add_signal(self.answer_label)
        self._do_task_label = f'do_task_label-{request_name}'
        self.add_signal(self._do_task_label)
        self.add_subprotocol(emit_protocol, name='emit')
        self.same_time_emit_label = 'SAME_TIME_EMIT'
        self.add_signal(self.same_time_emit_label)


    class EmitProgram(QuantumProgram):
        '''program to create a qubit and emit an entangled photon to the 'qout' port
        '''
        def __init__(self):
            super().__init__(num_qubits=2)

        def program(self):
            #emit from q2 using q1
            q1, q2 = self.get_qubit_indices(self.num_qubits)  # convenience method that returns the program's qubit indices
            self.apply(instr.INSTR_INIT, q1)                  # INSTR_INIT: initialize a qubit on a quantum memory
            self.apply(instr.INSTR_EMIT, [q1, q2])            # INSTR_EMIT: emit a qubit entangled with a qubit in memory
            yield self.run()

    def run(self):
        '''sends trigger periodically or starts an entanglemnt attempt
        the triggers are skipped during the entanglement attempt for simplicity
        Yields
        ------
        :class:`~pydynaa.core.EventExpression` -- await a timer signal or a response from the connection
        '''
        while True:
            time = (1 + ns.sim_time() // self.time_step) * self.time_step             # bug: self.time_step being float will sometimes arise a bug here
            rand = self.random_num_gen.randint(0, self.time_step)
            time += rand
            wait_timer = self.await_timer(end_time=time)
            wait_signal = self.await_signal(self, self._do_task_label)
            evexpr = yield wait_timer | wait_signal                                   # bug: AttributeError: 'thread_safe_generator' object has no attribute 'send'
            if evexpr.second_term.value:
                # start the entanglement attempt
                qpos = self.get_signal_result(self._do_task_label)                    # qpos is the memory position
                prog = self.EmitProgram()
                try:
                    self.node.qmemory.execute_program(prog, qubit_mapping=[qpos, 0])      # atom-photon entangled pair. qpos is the position for atom, and 0 is for photon (emited to BSM Detector)
                    self.send_signal(self.same_time_emit_label, result=False)
                except Exception as e:
                    outcome = 'multiple qubit emit exact same time (this fail)'     # multiple qubits are emitted at the same time, the "first" one to execute success
                    if DEBUG:
                        print(f'{int(ns.sim_time()):,} ns, {self.node.name:<5} FAIL, {self.request_name}, item = {outcome}')
                    self.send_signal(self.answer_label, result=(outcome, qpos))
                    self.send_signal(self.same_time_emit_label, result=True)
                    continue

                port = self.node.ports[self.q_port_name]
                yield self.await_port_input(port)
                message = port.rx_input()

                if self.subprotocols['emit'].same_time == True:
                    outcome = 'multiple qubit emit exact same time (this success, other fail)'
                    self.subprotocols['emit'].same_time = False  # reset the label
                    if DEBUG:
                        print(f'{int(ns.sim_time()):,} ns, {self.node.name:<5} SUCCESS, {self.request_name}, item = {outcome}')
                elif message and message.meta.get('header') == 'photonoutcome' and self.subprotocols['emit'].same_time is False:
                    outcome = message.items[0]
                    if DEBUG:
                        print(f'{int(ns.sim_time()):,} ns, {self.node.name:<5} SUCCESS, {self.request_name}, item = {outcome}')
                else:
                    outcome = 'multiple qubit emit in the time window'
                    if DEBUG:
                        print(f'{int(ns.sim_time()):,} ns, {self.node.name:<5} FAIL, {self.request_name}, item = {outcome}')
                self.send_signal(self.answer_label, result=(outcome, qpos))
            else:
                self.send_signal(self.trigger_label)

    def do_task(self, qpos):
        '''start the task
        
        Args:
            qpos: int -- the number indicating which qubit pair we are making
        '''
        self.send_signal(self._do_task_label, result=qpos)  # result or intermediate result results of the protocol that you want to broadcast with this signal


class EGService(ServiceProtocol, metaclass=abc.ABCMeta):
    '''abstract interface for an entanglement generation service
    Args:
        node: :class:`~netsquid.nodes.node.Node` -- the node this protocol runs on
        name: str -- the name of this protocol. default EGService
    
    Attributes:
        req_create: namedtuple -- a request to create entanglement with a remote node
        res_ok: namedtuple -- a response to indicate a create request has finished
    '''
    # Define the requests and responses as class attributes so they are identical for every EGProtocol instance
    req_create = namedtuple('LinkLayerCreate', ['request_name', 'rate', 'node1', 'node1_memory_pos', 'node2', 'node2_memory_pos'])
    # res_ok     = namedtuple('LinkLayerOk', ['request_name', 'create_id', 'rate', 'node1', 'node1_memory_pos', 'node2', 'node2_memory_pos'])
    res_ok     = namedtuple('LinkLayerOk', ['request_name', 'create_id', 'memory_pos'])

    def __init__(self, node, name, recv_msg_protocol):
        super().__init__(node=node, name=name)
        # register the request and response
        self.register_request(self.req_create, self.create)
        self.register_response(self.res_ok)
        # use a queue for requests
        self.queue = deque()
        self._new_req_signal = 'new request in queue'
        self.add_signal(self._new_req_signal)
        self._create_id = 0
        self.recv_msg_protocol = recv_msg_protocol
        self.node1 = False         # node1 is alice, node2 is bob

    def handle_request(self, request, identifier, start_time=None, **kwargs):
        '''schedule the request on the queue and signal to :meth:`~netsquid.examples.simple_link.EGProtocol.run`
        new items have been put into the queue

        Args:
            request -- the object representing the request
            identifier: str -- the identifier for this request
            start_time: float -- the time at which the request can be executed
            kwargs: dict -- additional arguments
        Returns:
            dict -- the dictionary with additional arguments
        '''
        if start_time is None:
            start_time = ns.sim_time()
        self.queue.append((start_time, (identifier, request, kwargs)))
        self.send_signal(self._new_req_signal)       # broadcasted to the world. Other protocols can wait for this signal
        return kwargs

    def run(self):
        '''wait for a new request signal, then run the requests one by one
        assumes request handlers are generators and not functions
        '''
        while True:
            wait_new_req_signal = self.await_signal(self, self._new_req_signal)
            wait_new_message = self.await_signal(self.recv_msg_protocol, Signals.READY)
            evexpr = yield wait_new_req_signal | wait_new_message
            if evexpr.first_term.value:
                while len(self.queue) > 0:
                    start_time, (handler_id, request, kwargs) = self.queue.popleft()
                    if self.name != request.request_name:
                        self.queue.append((start_time, (handler_id, request, kwargs))) # this EGService protocol is receiving a request signal from another EGService
                        break
                    if start_time > ns.sim_time():
                        yield self.await_timer(end_time=start_time)
                    func = self.request_handlers[handler_id]        # self.create registered at self.__init__
                    args = request._asdict()
                    gen = func(**{**args, **kwargs})                # all the parameters passed into self.create
                    yield from gen
            else:  # receives and handle a message from the ReceiveDataProtocol through signal
                event = evexpr.second_term.triggered_events[0]
                label, msg = self.recv_msg_protocol.get_signal_by_event(event, receiver=self)
                request, handler_id, start_time, kwargs = msg.items
                print(f'{self.node}-{self.name} received {request}')
                if self.name == request.request_name:         # one EGProtocol only handle one type of request
                    self.handle_request(request, handler_id, start_time, **kwargs)  # bob receives request msg from alice


    def _get_next_create_id(self):
        # return a unique create id
        self._create_id += 1
        return self._create_id

    @abc.abstractmethod
    def create(self, request_name, rate, create_id, **kwargs):
        '''implement the entanglement generation

        Args:
            request_name: int   -- string describing what this entanglement is for. used to communicate to higher layers
            rate: int       -- the entangled qubit pair generation rate to achieve
            create_id: int  -- the unique identifier provided by the service
            kwargs: dict    -- optional
        '''
        pass


class EGProtocol(EGService):
    '''implement an entanglement generatoin service

    upon a *LinkLayerCreate* request generates pairs of entangled qubits, of which one is locally stored and the other is stored at the remote node
    requests are fulfilled in FIFO order, and upon completion a response signal is sent.
    
    Args:
        node: :class:`~netsquid.nodes.node.Node` -- the node this protocol runs on
        c_port_name: str -- the name of the port which is connected to the remote node
        name: str -- the name of this protocol
    '''
    def __init__(self, node, c_port_name, name, recv_msg_protocol):
        super().__init__(node=node, name=name, recv_msg_protocol=recv_msg_protocol)
        # require a physical layer protocol
        self._mh_names = []                                # a link layer will handle several physical layer MHPs (update: give up this idea)
        # setup queue synchronization
        self.c_port = self.node.ports[c_port_name]

    def add_phys_layer(self, mh_protocol):
        '''add a physical layer protocol as a subprotocol

        Args:
            mh_protocol  -- the protocol which implements a physical layer
        '''
        self.add_subprotocol(mh_protocol, name=mh_protocol.request_name)
        self._mh_names.append(mh_protocol.request_name)

    def handle_request(self, request, identifier, start_time=None, **kwargs):
        '''synchronize the request with the other service and schedule it on the queue

        Args:
            request -- the object representing the request
            identifier: str -- identifier for this request
            start_time: float -- the time at which the request can be executed
            kwargs: dict -- additional arguments

        Return:
            dict -- for create request this is the unique create id
        '''
        if kwargs.get('create_id') is None:        # only for alice, not for bob
            self.node1 = True                      # it is node2, i.e. alice
            kwargs['create_id'] = self._get_next_create_id()
        else:
            self.node1 = False                     # it is node2, i.e. bob     
        if start_time is None:                     # only for alice, not for bob
            travel_time = 9000
            start_time= ns.sim_time() + travel_time
            # start_time = 10
            # make sure message don't combine by specifying a header
            msg = Message([request, identifier, start_time, kwargs], header=request)
            print(f'{self.node}-{self.name} SEND {request}')
            self.c_port.tx_output(msg)
        return super().handle_request(request, identifier, start_time, **kwargs)

    def run(self):
        '''make sure we have an subprotocol before running our program
        '''
        for mh_name in self._mh_names:
            if mh_name in self.subprotocols or isinstance(self.subprotocols[mh_name], Protocol):
                break
        else:
            raise ValueError('EGProtocol requires a physical layer protocol to be added')
        self.start_subprotocols()   # all MHP are running
        yield from super().run()

    def create(self, request_name, **kwargs):
        '''handler for create requests. create qubits together with a remote note
        
        Args:
            request_name: str -- the string used to tag this request for a specific purpose in a higher layer
            rate: int     -- the entangled qubit pair generation rate to achieve

        Yields:
            :class:`~pydynaa.core.EventExpression` -- the expressions required to execute the create request

        Returns:
            :obj:`~netsquid.examples.simple_link.EGProtocol.res_ok` -- the response object indicating we successfully made the requested qubits
        '''
        create_id  = kwargs['create_id']
        if self.node1:   # the alice node
            memory_pos = kwargs['node1_memory_pos']
        else:            # the bob node
            memory_pos = kwargs['node2_memory_pos']
        self._create_id = create_id
        sub_proto = self.subprotocols[request_name]               # choose the correct sub protocol (MHP)
        wait_trigger = self.await_signal(sub_proto, sub_proto.trigger_label)
        wait_answer  = self.await_signal(sub_proto, sub_proto.answer_label)

        # start the main loop
        while True:
            evexpr = yield wait_trigger | wait_answer
            for event in evexpr.triggered_events:
                qpos = self._handle_event(event, memory_position=memory_pos, request_name=request_name)
                if qpos is not None:
                    response = self.res_ok(request_name, create_id, qpos)  # entangled qubit is successfully created, the third element is the memory position
                    self.send_response(response)                           # send a response via a signal, name of signal is self.get_name(response)

    def _handle_event(self, event, memory_position, request_name):
        # communicate with the physical layer on trigger and answer signals
        sub_proto = self.subprotocols[request_name]
        label, value = sub_proto.get_signal_by_event(event, receiver=self)
        if label == sub_proto.trigger_label:
            if DEBUG:
                print(f'{int(ns.sim_time()):,} do task: {self.node} trigger phy {self.name}')
            sub_proto.do_task(qpos=memory_position)            # link layer telling physical layer to generate entangled pair of qubits
        elif label == sub_proto.answer_label:                  # link layer getting anwsers from the physical layer
            outcome, qpos = value
            # outcome of 1 is |01>+|10>, 2 is |01>-|10>
            # other outcomes are non-entangled states
            if outcome == 1 or outcome == 2:
                return qpos
        return None


class ReceiveDataProtocol(DataNodeProtocol):
    '''Receives messages from one port, and sends the message through signals to all linklayer EGProtocols
    '''
    def __init__(self, node, port_name, name):
        super().__init__(node=node, port_name=port_name, name=name)

    def process_data(self, message):
        '''Process the available data
           Don't need to process
        '''
        return True
    
    def post_process_data(self, message):
        return message


class SameTimeEmitProtocol(NodeProtocol):
    '''receive signals to see if there exist multiple qubit emit happen at the same time
    '''
    def __init__(self, node):
        super().__init__(node)
        self.name = node.name + '-same_time_emit'
        self.phy_protos = []
        self.same_time = False

    def add_phy_layer_protocol(self, phy_proto):
        '''
        Args:
            phy_proto -- this physical layer protocol is "registered" to this protocol
        '''
        self.phy_protos.append(phy_proto)

    def run(self):
        '''check if a MidpointHeralding Protocol is emitting two photons at the same time
        '''
        while True:
            expr = yield reduce(operator.or_, [self.await_signal(proto, proto.same_time_emit_label) for proto in self.phy_protos])
            for triggered_event in expr.triggered_events:
                proto_source = triggered_event.source
                res = proto_source.get_signal_result(proto_source.same_time_emit_label, self)
                self.same_time = self.same_time or res   # if there is one True, then it is True
                if DEBUG:
                    print(f'{int(ns.sim_time()):,} ns {proto_source}, {proto_source.request_name} same time = {res}')



class NetworkProtocol(Protocol):
    '''send requests to an EG service protocol
    will automatically stop the service and their sub-protocols when the request are finished

    Args:
        name: str -- name to identify this protocol
        request_link_dict: dict -- {request: {'alice': EGService, 'bob': EGService}}
                                -- 'alice' is the serivce that will get the requests from this network
                                -- 'bob' is the  service that will create the pairs together with the entangled_protocol
    '''
    def __init__(self, name, request_link_dict, msg_recv_proto_dict, emit_proto_dict):
        super().__init__(name=name)
        self.request_link_dict = request_link_dict
        for req, link in self.request_link_dict.items():
            alice_egp = link['alice']
            bob_egp = link['bob']
            self.add_subprotocol(alice_egp, 'alice:' + req.request_name)
            self.add_subprotocol(bob_egp,   'bob:' + req.request_name)
        self.msg_recv_proto_dict = msg_recv_proto_dict
        for name, msg_recv_protocol in msg_recv_proto_dict.items():
            self.add_subprotocol(msg_recv_protocol, name)
        self.emit_proto_dict = emit_proto_dict
        for name, emit_protocol in emit_proto_dict.items():
            self.add_subprotocol(emit_protocol, name)

    def run(self):
        '''start the protocols and put requests to Alice
        '''
        for _, msg_recv_proto in self.msg_recv_proto_dict.items():
            msg_recv_proto.start()

        for  _, emit_proto in self.emit_proto_dict.items():
            emit_proto.start()

        alice_protos = []
        for req, link in self.request_link_dict.items():
            alice_egp = link['alice']
            bob_egp   = link['bob']
            alice_egp.start()
            bob_egp.start()
            alice_egp.put(req)
            alice_protos.append(alice_egp)
        
        proto = alice_protos[0]
        ok_signal = proto.get_name(proto.res_ok)
        while True:
            expr = yield reduce(operator.or_, [self.await_signal(proto, ok_signal) for proto in alice_protos])

            for triggered_event in expr.triggered_events:
                proto_source = triggered_event.source
                res = proto_source.get_signal_result(ok_signal, self)
                qubits = proto_source.node.qmemory.peek(res.memory_pos)[0].qstate.qubits  #  namedtuple('LinkLayerOk', ['purpose_id', 'create_id', 'memory_pos'])
                global counter
                counter[res.request_name] += 1
                print(f'{int(ns.sim_time()):,} ns {qubits}, request = {res.request_name}, counter = {counter[res.request_name]}')

            if ns.sim_time() > 2_000_000_000:
                break


from collections import Counter
counter = Counter()


def create_example_network():
    '''create the example network, the topology
    Args:
        rate: int -- the entangled qubit pair generation rate to achieve
    Returns:
        :class:`~netsquid.nodes.network.Network` -- the example network for a simple link
    '''
    network = Network('simple_link_network')
    nodes = network.add_nodes(['alice', 'bob'])
    distance = 2    # km
    for node in nodes:
        node.add_subcomponent(QuantumProcessor(f'qmem_{node.name}', num_positions=11, fallback_to_nonphysical=True))
    conn = HeraldedConnection('heraled_connection', length_to_a = distance / 2, length_to_b=distance / 2, time_window=5)
    network.add_connection(nodes[0], nodes[1], connection=conn, label='quantum')
    network.add_connection(nodes[0], nodes[1], delay=distance / 200000 * 1e9, label='classical')
    return network


def setup_protocol(network, requests):
    '''configure the protocols

    Args:
        network: :class:`Network` -- the network to configure the protocols on. Should consist of two nodes Alice & Bob
        requests: namedtuple -- a list of requests    
    Return:
        :class: Protocol -- a protocol describing the complete simple link setup
    '''
    nodes   = network.nodes
    q_ports = network.get_connected_ports(*nodes, label='quantum')     # single link
    c_ports = network.get_connected_ports(*nodes, label='classical')
    alice_recv_msgp = ReceiveDataProtocol(nodes['alice'], port_name=c_ports[0], name='alice_recv_msg')
    bob_recv_msgp   = ReceiveDataProtocol(nodes['bob'],   port_name=c_ports[1], name='bob_recv_msg')
    alice_emit_protocol = SameTimeEmitProtocol(nodes['alice'])
    bob_emit_protocol   = SameTimeEmitProtocol(nodes['bob'])
    recv_msg_proto_dict = {alice_recv_msgp.name: alice_recv_msgp, bob_recv_msgp.name: bob_recv_msgp}
    emit_proto_dict = {alice_emit_protocol.name: alice_emit_protocol, bob_emit_protocol.name: bob_emit_protocol}
    request_link_dict = {}
    for req in requests:
        # each request will have it's own dedicated link layer and the accordingly physical layer
        alice_egp = EGProtocol(nodes['alice'], c_ports[0], name=req.request_name, recv_msg_protocol=alice_recv_msgp)
        bob_egp   = EGProtocol(nodes['bob'],   c_ports[1], name=req.request_name, recv_msg_protocol=bob_recv_msgp)
        time_step = int(1 / (req.rate / 0.74) * 1e9)         # BSM success rate is approximately 0.74~0.75 by experimental measurements, bug: being a float here will arise a bug, so let it be int
        # setup Alice's MHP
        alice_mhp = MidpointHeraldingProtocol(nodes['alice'], time_step, q_ports[0], request_name=req.request_name, emit_protocol=alice_emit_protocol)
        alice_egp.add_phys_layer(alice_mhp)
        alice_emit_protocol.add_phy_layer_protocol(alice_mhp)
        # setup Bob's   MHP
        bob_mhp = MidpointHeraldingProtocol(nodes['bob'], time_step, q_ports[1], request_name=req.request_name, emit_protocol=bob_emit_protocol)
        bob_egp.add_phys_layer(bob_mhp)
        bob_emit_protocol.add_phy_layer_protocol(bob_mhp)
        link = {'alice':alice_egp, 'bob':bob_egp}
        request_link_dict[req] = link
    return NetworkProtocol('SimpleLinkProtocol', request_link_dict, recv_msg_proto_dict, emit_proto_dict)


def run_siumulation():
    '''run the example simulation
    '''
    ns.sim_reset()
    ns.set_random_state(42)
    ns.set_qstate_formalism(QFormalism.DM)
    network = create_example_network()
    protocol = setup_protocol(network, requests)
    start = time.time()
    protocol.start()
    ns.sim_run()
    print(f'time = {time.time() - start:.3f} s')


if __name__ == '__main__':
    request_name_template = '{}-{}-{}-{}-{}'   # node1-qpos1-node2-qpos2-rate
    node1 = 'alice'
    node2 = 'bob'
    rate = 100
    node1_memory_pos = 1 # NOTE cannot use memory position 0, it is reserved for emiting the atom of photon-atom entanglement
    node2_memory_pos = 2
    req1_name = request_name_template.format(node1, node1_memory_pos, node2, node2_memory_pos, rate)
    req_1 = EGProtocol.req_create(request_name=req1_name, rate=rate, node1=node1, node1_memory_pos=node1_memory_pos,\
                                                                     node2=node2, node2_memory_pos=node2_memory_pos)
    # requests = [req_1]

    rate = 100
    node1_memory_pos = 2
    node2_memory_pos = 3
    req2_name = request_name_template.format(node1, node1_memory_pos, node2, node2_memory_pos, rate)
    req_2 = EGProtocol.req_create(request_name=req2_name, rate=rate, node1=node1, node1_memory_pos=node1_memory_pos,\
                                                                     node2=node2, node2_memory_pos=node2_memory_pos)
    requests = [req_1, req_2]

    # NOTE for two request of rate 100 and 300, same time emit won't (very rarely) happen, because of int() in time_step = int(...)

    run_siumulation()
