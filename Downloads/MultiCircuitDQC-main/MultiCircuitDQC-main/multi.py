from MinQAP import *
import random as rn
from Circuit import Circuit
from NetworkGraph import NetworkGraph
from unittest.mock import MagicMock
from scipy.optimize import quadratic_assignment, linear_sum_assignment
from itertools import chain, combinations
locked_library = ['netsquid','netsquid.util','netsquid.examples','netsquid.util.datacollector',
'netsquid.components','netsquid.components.qmemory','netsquid.components.models',
'netsquid.components.models.delaymodels','netsquid.components.models.qerrormodels',
'netsquid.components.qsource','netsquid.components.qchannel','netsquid.components.cchannel',
'netsquid.components.instructions','netsquid.qubits','netsquid.qubits.ketstates',
'netsquid.qubits.qubitapi','netsquid.protocols','netsquid.nodes','protocols.predistribution.network_protocol',
'protocols.network_protocol','execnet','kahypar','ortools','ortools.linear_solver','OM'
]
for library in locked_library:
    sys.modules[library] = MagicMock()

dec=1.0
M=100
num=10

def ReadInput(folder):
    Circuits=[]
    path='MultiCircuits/'+folder
    for filename in os.listdir(path):
        if 'random' in filename or 'simple' in filename:
            continue
        Circuits.append(Circuit(path+'/'+filename))
    return Circuits
        
        

def MinQAPSmallCircuit(circ, network_graph):
    distance_matrix=nx.floyd_warshall_numpy(network_graph)
    numMem=network_graph.nodes()
    flow_matrix=np.zeros((len(numMem),len(numMem)))
    for entry in circ.MyData:
        #print( entry)
        flow_matrix[int(entry[0]), int(entry[1])]= -int(entry[2])
        flow_matrix[int(entry[1]), int(entry[0])]= -int(entry[2])
    res = quadratic_assignment(flow_matrix,distance_matrix,method='faq',  options={"maximize":True})
    assigned_memory={}
    #print("QAP done")
    #print('min value:',res['fun'])
    for qubit,location in enumerate(res['col_ind']): 
        assigned_memory[qubit]=location
        
    return assigned_memory, distance_matrix, res.fun

def MinQAP_qG(circ, network_graph):
    '''
    circ: class for a circuit
    network_graph: nx.Graph() with computer/node as unit
    '''
    distance_matrix=nx.floyd_warshall_numpy(network_graph)
    node_list=network_graph.nodes()
    flow_matrix=np.zeros((len(node_list),len(node_list)))
    for entry in circ.qubit_partition:
        flow_matrix[int(entry[0]), int(entry[1])]= int(entry[2])
        flow_matrix[int(entry[1]), int(entry[0])]= int(entry[2])
    res = quadratic_assignment(flow_matrix, distance_matrix, options={"maximize":False})
    assigned_memory={}
    for qubit,location in enumerate(res['col_ind']): 
        assigned_memory[qubit]=location
        
    return assigned_memory, distance_matrix, res.fun

def LatencyofCircuitonNetwork(circ, network):
    assigned_memory, distance_matrix, latency=MinQAP_qG(circ, network.network_graph)
    #latency=CallGreedy(assigned_memory,distance_matrix,circ, network)
    #print(latency)
    
    return latency
    
def CallGreedy(parts,dist_matrix,circ, network):
    EP_list=[]
    Bin_gates=[sorted(gate) for i,gate in enumerate(circ.gates) if len(gate)==2]
    #print("doing dag")
    DAG2,G_full=DAG_of_EPs(network.network_graph, network.qG,Bin_gates)
    #print("dag done")
    EP_gate_map=[]
    for i,(q1,q2) in enumerate(Bin_gates):
        if dist_matrix[parts[int(q1)],parts[int(q2)]]>=10.0:
            EP_list.append(tuple(sorted([network.node_map[parts[int(q1)]],network.node_map[parts[int(q2)]]])+[i]))
            EP_gate_map.append(i)
    #print("Number of gates ", len(Bin_gates))
    #print("number of EPs ", len(EP_list))
    #print("made EP  list")
    Ent_Counter=collections.Counter(EP_list)
    DAG,G=DAG_of_EPs(network.network_graph, network.qG,EP_list)
    layers=[[Bin_gates[i] for i in sorted(generation)] for generation in nx.topological_generations(G_full)]
    #print(layers)
    new_layers=[]
    for layer in layers:
        new_layer=[]
        for q1,q2 in layer:
            if dist_matrix[parts[int(q1)],parts[int(q2)]]>=10.0:
                new_layer.append(tuple(sorted((network.node_map[parts[int(q1)]],network.node_map[parts[int(q2)]]))))
        if new_layer:
            new_layers.append(new_layer)

    print("in Greedy")
    greedy_batches,greedy_val,greed_time=Greedy_Ent_Gen_Dec_par(network.qG, EP_list, dec, dist_matrix, parts,  DAG2,Bin_gates)

    return greedy_val
    
def CombineCircuits(Circuits, network, IgnoreLatency=False):
    Newqubits=[]
    Newgates=[]
    TotQubitNumber=0
    NewQubitMap={}
    for circ in Circuits:
        NewQubitMap[circ]=TotQubitNumber
        for qubit in circ.qubits:
            Newqubits.append(TotQubitNumber+int(qubit))
        for gate in circ.gates:
            if len(gate)==2:
                Newgates.append([str(TotQubitNumber+int(gate[0])),str(TotQubitNumber+int(gate[1]))])
            else:
                Newgates.append([str(TotQubitNumber+int(gate[0]))])
        TotQubitNumber+=len(circ.qubits)    
    Newqubits=[str(qubit) for qubit in Newqubits]
    BigCircuit=Circuit(None,Newqubits,Newgates)
    if IgnoreLatency:
        return BigCircuit, NewQubitMap
    if TotQubitNumber>M:
        return BigCircuit, 1
    BigCircuit.latency=LatencyofCircuitonNetwork(BigCircuit, network)
    return BigCircuit, 0
        

def SetCover1(Circuits, network):
    batches=[]
    currentBatch=[]
    Total_latency=0
    remainCircuits=Circuits
    best_circ=np.argmin([circ.latency for circ in Circuits])
    currentBatch.append(Circuits[best_circ])
    currentBigCircuit=Circuits[best_circ]
    del remainCircuits[best_circ]
    while remainCircuits:
        best_latency=np.inf
        for i,circ in enumerate(remainCircuits):
            NewBigCircuit, bad=CombineCircuits([currentBigCircuit,circ], network)
            if bad==1:
                continue
            if best_latency>NewBigCircuit.latency:
                best_latency=NewBigCircuit.latency
                best_circ=i
        if best_latency==np.inf:
            batches.append(currentBatch)
            currentBatch=[]
            best_circ=np.argmin([circ.latency for circ in remainCircuits])
            currentBatch.append(remainCircuits[best_circ])
            currentBigCircuit=remainCircuits[best_circ]
            del remainCircuits[best_circ]
            continue

        NewBigCircuit,_=CombineCircuits([currentBigCircuit,remainCircuits[best_circ]],network)
        if NewBigCircuit.latency <= currentBigCircuit.latency + remainCircuits[best_circ].latency:
            currentBatch.append(remainCircuits[best_circ])
            currentBigCircuit=NewBigCircuit
            del remainCircuits[best_circ]
        else:
            batches.append(currentBatch)
            Total_latency+=currentBigCircuit.latency
            currentBatch=[remainCircuits[best_circ]]
            currentBigCircuit=remainCircuits[best_circ]
            del remainCircuits[best_circ]
    batches.append(currentBatch)
    Total_latency+=currentBigCircuit.latency
    return batches, Total_latency

def SetCover2subroutine(Circuits_set,  network):
    HugeCircuit=[]
    Circuits=list(Circuits_set)
    TotNumQubits=sum([len(circ.qubits) for circ in Circuits])
    buffer=min([len(circ.qubits) for circ in Circuits])
    print(TotNumQubits)
    HugeNetwork=network.AddDummyNodes(TotNumQubits-M+buffer)
    print(len(HugeNetwork.nodes()))
    NewQubitMap={}
    for circ in Circuits:
        if not HugeCircuit:
            HugeCircuit=circ
            NewQubitMap[circ.filepath]=0
            continue
        NewQubitMap[circ.filepath]=len(HugeCircuit.qubits)
        HugeCircuit,_=CombineCircuits([HugeCircuit,circ], network, True)
    parts, _, cost=MinQAPSmallCircuit(HugeCircuit, HugeNetwork)
    print("cost: ", cost)
    print(parts)
    currentBatchCircuits=set()
    currentBigCircuit=0
    for circ in Circuits:
        print(circ.filepath, " ", NewQubitMap[circ.filepath])
        flag=0
        for qubit in circ.qubits:
            if parts[NewQubitMap[circ.filepath]+int(qubit)]>=M:
                flag+=1
                print(circ.filepath, " broken at ", qubit)
                break
        if flag<=3:
            if currentBigCircuit:
                NewBigCircuit,bad= CombineCircuits([currentBigCircuit,circ], network)
                if bad:
                    return currentBatchCircuits, currentBigCircuit
                else:
                    currentBigCircuit=NewBigCircuit
                    currentBatchCircuits.add(circ)
            else:
                currentBigCircuit=circ
                currentBatchCircuits.add(circ)
    return currentBatchCircuits, currentBigCircuit

def SetCover2(Circuits, network):
    
    Circuits_set=set(Circuits)
    circuit_batches=[]
    total_latency=0
    while Circuits_set:
        new_batch, batch_circuit=SetCover2subroutine(Circuits_set,  network)
        circuit_batches.append(new_batch)
        total_latency+=batch_circuit.latency
        Circuits_set-=new_batch
        print([circ.filepath for circ in new_batch])
    
    return circuit_batches, total_latency

def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            if len(seq[0].qubits)+sum([len(it.qubits) for it in item])<=M:
                yield [seq[0]]+item
                yield item


def gen_subsets(items, target):
    items = sorted(items, key=lambda item: len(item.qubits))
    #print([item.filepath for item in items])
    def helper(i, t, acc):
        #print("helper call ", (i, t, [ac.filepath for ac in acc]))
        if i >= len(items): # no more items
            yield acc
        elif len(items[i].qubits) > t: # all remaining items are too big
            yield acc
        else:
            #yield acc
            yield from helper(i+1, t - len(items[i].qubits), (*acc, items[i])) # include item[i]
            yield from helper(i+1, t, acc)                  # do not include item[i]
    yield from helper(0, target, ())


def ApproximateGreedy(Circuits, network):
    Powerset=list(gen_subsets(Circuits, M))
    PowersetCircuits=[]
    for subset in Powerset:
        BigCircuit,_=CombineCircuits(subset, network)
        PowersetCircuits.append(BigCircuit)
        #print([(item.filepath, len(item.qubits)) for item in subset])
    ToBeCovered=set(Circuits)
    batches=[]
    Total_latency=0
    while ToBeCovered:
        best_value=np.inf
        best_circuit=None
        best_subset=None
        for i, BigCircuit in enumerate(PowersetCircuits):
            inter= len(ToBeCovered&set(Powerset[i]))
            if inter==0:
                continue
            valuation=BigCircuit.latency/inter
            if best_value>valuation:
                best_value=valuation
                best_circuit=BigCircuit
                best_subset=set(Powerset[i])
        print([(item.filepath, len(item.qubits)) for item in best_subset])
        ToBeCovered-=best_subset
        batches.append(list(best_subset))
        Total_latency+=best_circuit.latency
    
    return batches, Total_latency


if __name__ == "__main__":
    i=0
    #qG=Create_qG(10)
    network= NetworkGraph(M,num)
    
    Circuits=ReadInput('dir1')
    for circ in Circuits:
        print(circ.filepath)
        circ.latency=LatencyofCircuitonNetwork(circ, network)
    '''
    batches1, Total_latency1=SetCover1(Circuits, network)
    
    for batch in batches1:
        print([circ.filepath for circ in batch])
    print("Total latency ", Total_latency1)
    
    batches2, Total_latency2=SetCover2(Circuits, network)
    
    for batch in batches2:
        print([circ.filepath for circ in batch])
    print("Total latency ", Total_latency2)
    '''
    batches3, Total_latency3=ApproximateGreedy(Circuits, network)
    for batch in batches3:
        print([circ.filepath for circ in batch])
    print("Total latency ", Total_latency3)
    '''
    for filename in os.listdir('circuitsVaryQubits'):
        if 'random' in filename or 'simple' in filename:
            continue
        #num=int(re.findall('[0-9]+', filename)[-1])
        #print("power ",power)
        num=10
        print(num)
        qNode._stat_id=0
        if filename not in os.listdir('Testresults'):
            start_time=time.time()
            CallGreedy(filename,i,qG,num, 1)
            qNode._stat_id=0
            print(i, filename)
        i+=1
    '''
    
    #PrepareInput(5)
