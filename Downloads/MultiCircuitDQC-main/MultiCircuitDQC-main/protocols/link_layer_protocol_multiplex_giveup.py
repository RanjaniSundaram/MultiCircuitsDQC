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
from netsquid.qubits.qformalism import QFormalism
from collections import namedtuple, deque
import netsquid.components.instructions as instr
import netsquid as ns
from netsquid.nodes import Network
from netsquid.components.qprocessor import QuantumProcessor, QuantumProgram
from netsquid.nodes.connections import Connection
from netsquid.protocols import NodeProtocol
from netsquid.protocols.protocol import Protocol
from netsquid.protocols.serviceprotocol import ServiceProtocol
from netsquid.components.component import Message
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.qdetector import QuantumDetector
from netsquid.qubits.operators import Operator

from protocols.generator_multiplex import multiplex
from commons.qNode import GENERATION_TIME
GENERATION_TIME_NETSQUID = int(GENERATION_TIME / 1e-9)   # in nano-seconds


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
    def __init__(self, name, system_delay=0., dead_time=0., models=None, output_meta=None, error_on_fail=True, properties=None):
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
    def __init__(self, node, time_step, q_port_name, request_name):
        super().__init__(node=node)
        self.time_step = time_step
        self.node.qmemory.ports['qout'].forward_output(self.node.ports[q_port_name])
        # wait for an outcome on the input port
        self.port_name = q_port_name
        self.q_port_name = q_port_name
        self.request_name = request_name
        # remember if we already send a photon
        self.trigger_label = 'TRIGGER'
        self.answer_label = 'ANSWER'
        self.add_signal(self.trigger_label)
        self.add_signal(self.answer_label)
        self._do_task_label = 'do_task_label'
        self.add_signal(self._do_task_label)


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
            wait_timer = self.await_timer(end_time=time)
            wait_signal = self.await_signal(self, self._do_task_label)
            evexpr = yield wait_timer | wait_signal                                   # bug: AttributeError: 'thread_safe_generator' object has no attribute 'send'
            if evexpr.second_term.value:
                # start the entanglement attempt
                qpos = self.get_signal_result(self._do_task_label)                    # qpos is the memory position
                prog = self.EmitProgram()
                self.node.qmemory.execute_program(prog, qubit_mapping=[qpos, 0])      # atom-photon entangled pair. qpos is the position for atom, and 0 is for photon (emited to BSM Detector)
                port = self.node.ports[self.q_port_name]
                yield self.await_port_input(port)
                message = port.rx_input()
                if message.meta.get('header') == 'photonoutcome':
                    outcome = message.items[0]
                    # print(f'{int(ns.sim_time()):,} ns, {self.node.name:<5} SUCCESS, item = {outcome}')
                else:
                    outcome = 'FAIL'
                    # print(f'{int(ns.sim_time()):,} ns, {self.node.name:<5} FAIL')
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
    req_create = namedtuple('LinkLayerCreate', ['request_name', 'rate', 'memory_pos'])
    res_ok     = namedtuple('LinkLayerOk', ['request_name', 'create_id', 'logical_qubit_id'])

    def __init__(self, node, name=None):
        super().__init__(node=node, name=name)
        # register the request and response
        self.register_request(self.req_create, self.create)
        self.register_response(self.res_ok)
        # use a queue for requests
        self.queue = deque()
        self._new_req_signal = 'new request in queue'
        self.add_signal(self._new_req_signal)
        self._create_id = 0
    
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
            yield self.await_signal(self, self._new_req_signal)
            gens = []
            while len(self.queue) > 0:
                start_time, (handler_id, request, kwargs) = self.queue.popleft()
                if start_time > ns.sim_time():
                    yield self.await_timer(end_time=start_time)
                func = self.request_handlers[handler_id]        # self.create registered at self.__init__
                args = request._asdict()
                gen = func(**{**args, **kwargs})                # all the parameters passed into self.create
                gens.append(gen)
                # yield from gen
                # threadsafe_gen = thread_safe_generator(gen)
                # yield from threadsafe_gen
            yield from multiplex(gens)


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
    def __init__(self, node, c_port_name, name=None):
        super().__init__(node=node, name=name)
        # require a physical layer protocol
        # self._mh_name = 'MH_Protocol'
        self._mh_names = []                                # a link layer will handle several physical layer MHPs
        # setup queue synchronization
        self.c_port = self.node.ports[c_port_name]
        self.c_port.bind_input_handler(self._handle_msg)   # request is put at Alice. alice
    
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
        if kwargs.get('create_id') is None:
            kwargs['create_id'] = self._get_next_create_id()
        if start_time is None:
            travel_time = 10000
            start_time= ns.sim_time() + travel_time
            # start_time = 10
            # make sure message don't combine by specifying a header
            self.c_port.tx_output(Message([request, identifier, start_time, kwargs], header=request))
        return super().handle_request(request, identifier, start_time, **kwargs)

    def _handle_msg(self, msg):
        '''handle incoming messages from the other service

        the service use these messages to ensure they start the same request at the same time

        Args:
            msg: Message -- a Message from another ServiceProtocol containing request and scheduling data
        '''
        request, handler_id, start_time, kwargs = msg.items
        self.handle_request(request, handler_id, start_time, **kwargs)  # bob receives request msg from alice

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
        memory_pos = kwargs['memory_pos']
        self._create_id = create_id
        sub_proto = self.subprotocols[request_name]               # choose the correct sub protocol (MHP)
        wait_trigger = self.await_signal(sub_proto, sub_proto.trigger_label)
        wait_answer  = self.await_signal(sub_proto, sub_proto.answer_label)

        # start the main loop
        while True:
            evexpr = yield wait_trigger | wait_answer              # bug: AttributeError: 'thread_safe_generator' object has no attribute 'send'
            # if evexpr is None:
            #     evexpr = wait_trigger | wait_answer
            for event in evexpr.triggered_events:                  # bug: 'NoneType' object has no attribute 'triggered_events' (using the multiplex)
                qpos = self._handle_event(event, memory_position=memory_pos, request_name=request_name)
                if qpos is not None:
                    response = self.res_ok(request_name, create_id, qpos)  # entangled qubit is successfully created, the third element is the memory position
                    self.send_response(response)                           # send a response via a signal, name of signal is self.get_name(response)

    def _handle_event(self, event, memory_position, request_name):
        # communicate with the physical layer on trigger and answer signals
        sub_proto = self.subprotocols[request_name]
        label, value = sub_proto.get_signal_by_event(event, receiver=self)
        if label == sub_proto.trigger_label:
            sub_proto.do_task(qpos=memory_position)            # link layer telling physical layer to generate entangled pair of qubits
        elif label == sub_proto.answer_label:                  # link layer getting anwsers from the physical layer
            outcome, qpos = value
            # outcome of 1 is |01>+|10>, 2 is |01>-|10>
            # other outcomes are non-entangled states
            if outcome == 1 or outcome == 2:
                return qpos
        return None


class NetworkProtocol(Protocol):
    '''send requests to an EG service protocol
    will automatically stop the service and their sub-protocols when the request are finished

    Args:
        name: str -- name to identify this protocol
        entangled_protocol: :class:'EGService'        -- the serivce that will get the requests from this network
        remote_entangle_protocol: :class:'EGService' -- the service that will create the pairs together with the entangled_protocol
    '''
    def __init__(self, name, entangled_protocol, remote_entangle_protocol):
        super().__init__(name=name)
        self.add_subprotocol(entangled_protocol, 'EGP_Alice')
        self.add_subprotocol(remote_entangle_protocol, 'EGP_Bob')
    
    def run(self):
        '''start the protocols and put requests to Alice
        '''
        proto = self.subprotocols['EGP_Alice'].start()   # bug: forgot to start ...
        self.subprotocols['EGP_Bob'].start()
        # threads = []
        # for req in requests:
        #     threads.append(threading.Thread(target=self.request_function, args=(proto, req)))
        #     # threads.append(threading.Thread(target=thread_function, args=(req,), daemon=True))
        # for t in threads:
        #     t.start()
        # for t in threads:
        #     t.join()

        # start with a single request
        proto.put(req_1)                    # EGProtocol.handle_request will handle this proto.put
        proto.put(req_2)
        yield from self.show_results(proto, req_1)

        # create_id = proto.put(req_2)['create_id']
        # req_3 = proto.req_create(purpose_id=4, number=1)
        # create_id = proto.put(req_3)['create_id']
        # print(f'Waiting for responses with create_id')
        # yield from self.show_results(proto, req_2)
        # yield from self.show_results(proto, req_3)
        print('Finished all network requests')
    
    def request_function(self, proto, request):
        '''when "yield from" statement is in a thread's target the thread cannot be executed.
           still investigating why ...
        '''
        print(f'{request} starts ...')
        proto.put(request)
        yield from self.show_results(proto, request)
        print(f'{request} ends ...')

    def show_results(self, proto, request):
        '''show the qubits which are entangled as a result of the request

        Args:
            proto: :class:`EGProtocol` -- the entanglement generation service
            request: :class:`EGService.req_create`
        Yields:
            :class:`pydynaa.core.EventExpression`
        '''
        ok_signal = proto.get_name(proto.res_ok)
        while True:
            yield self.await_signal(proto, ok_signal)          # ok_signal is the response from the link layer
            res = proto.get_signal_result(ok_signal, self)     # get the result of the last signal sent with this label
            qubits = proto.node.qmemory.peek(res.logical_qubit_id)[0].qstate.qubits  #  namedtuple('LinkLayerOk', ['purpose_id', 'create_id', 'logical_qubit_id'])
            global counter
            counter += 1
            print(f'{int(ns.sim_time()):,} ns {qubits}, request = {res.request_name}, counter = {counter}')

counter = 0

def thread_function(name):
    print("Thread %s: starting", name)
    for i in range(10):
        print(f'{name} - {i}')
        time.sleep(1)
    print("Thread %s: finishing", name)

def create_example_network():
    '''create the example network
    Args:
        rate: int -- the entangled qubit pair generation rate to achieve
    Returns:
        :class:`~netsquid.nodes.network.Network` -- the example network for a simple link
    '''
    network = Network('simple_link_network')
    nodes = network.add_nodes(['Alice', 'Bob'])
    distance = 2    # km
    for node in nodes:
        node.add_subcomponent(QuantumProcessor(f'qmem_{node.name}', num_positions=11, fallback_to_nonphysical=True))
    conn = HeraldedConnection('heraled_connection', length_to_a = distance / 2, length_to_b=distance / 2, time_window=20)
    network.add_connection(nodes[0], nodes[1], connection=conn, label='quantum')
    network.add_connection(nodes[0], nodes[1], delay=distance / 200000 * 1e9, label='classical')
    return network


def setup_protocol(network, requests):
    '''configure the protocols

    Args:
        network: :class:`Network` -- the network to configure the protocols on. Should consist of two nodes Alice & Bob
    
    Return:
        :class: Protocol -- a protocol describing the complete simple link setup
    '''
    nodes   = network.nodes
    q_ports = network.get_connected_ports(*nodes, label='quantum')     # single link
    c_ports = network.get_connected_ports(*nodes, label='classical')
    alice_egp = EGProtocol(nodes['Alice'], c_ports[0])
    bob_egp   = EGProtocol(nodes['Bob'],   c_ports[1])
    for req in requests:
        time_step = int(1 / (req.rate / 0.74) * 1e9)         # BSM success rate is approximately 0.74~0.75 by experimental measurements, bug: being a float here will arise a bug, so let it be int
        # setup Alice's MHP
        alice_mhp = MidpointHeraldingProtocol(nodes['Alice'], time_step, q_ports[0], request_name=req.request_name)
        alice_egp.add_phys_layer(alice_mhp)
        # setup Bob's   MHP
        bob_mhp = MidpointHeraldingProtocol(nodes['Bob'], time_step, q_ports[1], request_name=req.request_name)
        bob_egp.add_phys_layer(bob_mhp)
    return NetworkProtocol('SimpleLinkProtocol', alice_egp, bob_egp)


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
    memory_pos = 2
    req1_name = request_name_template.format(node1, memory_pos, node2, memory_pos, rate)
    req_1 = EGProtocol.req_create(request_name=req1_name, rate=rate, memory_pos=memory_pos)
    requests = [req_1]

    rate = 150
    memory_pos = 3
    req2_name = request_name_template.format(node1, memory_pos, node2, memory_pos, rate)
    req_2 = EGProtocol.req_create(request_name=req2_name, rate=rate, memory_pos=memory_pos)
    requests = [req_1, req_2]

    run_siumulation()
