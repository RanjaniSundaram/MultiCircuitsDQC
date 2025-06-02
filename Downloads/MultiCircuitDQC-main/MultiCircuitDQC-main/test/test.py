import sys
from unittest.mock import MagicMock
sys.modules['netsquid'] = MagicMock()
from MinQAP import *
import random as rn
from Circuit import Circuit
from NetworkGraph import NetworkGraph


dec=1.0
M=100
num=10

def ReadInput(folder):
	Circuits=[]
	path='MultiCircuits/'+folder
	for filename in os.listdir(path):
		Circuits.append(Circuit(path+'/'+filename))
	return Circuits
		
		

def MinQAPSmallCircuit(circ, network_graph):
	distance_matrix=nx.floyd_warshall_numpy(network_graph)
	numMem=network_graph.nodes()
	flow_matrix=np.zeros((len(numMem),len(numMem)))
	for entry in circ.MyData:
		flow_matrix[int(entry[0]), int(entry[1])]= int(entry[2])
		flow_matrix[int(entry[1]), int(entry[0])]= int(entry[2])
	res = quadratic_assignment(distance_matrix, flow_matrix, options={"maximize":False})
	assigned_memory={}
	#print("QAP done")
	#print('min value:',res['fun'])
	for qubit,location in enumerate(res['col_ind']): 
		assigned_memory[qubit]=location
		
	return assigned_memory, distance_matrix, res.fun


def LatencyofCircuitonNetwork(circ, network):
	assigned_memory, distance_matrix, latency=MinQAPSmallCircuit(circ, network.main_network_graph)
	latency=CallGreedy(assigned_memory,distance_matrix,circ, network)
	print(latency)
	
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
	
def CombineCircuits(Circuits, network):
	Newqubits=[]
	Newgates=[]
	TotQubitNumber=0
	for circ in Circuits:
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
	if TotQubitNumber>M:
		return BigCircuit, 1
	BigCircuit.latency=LatencyofCircuitonNetwork(BigCircuit, network)
	return BigCircuit, 0
		

def SetCover1(Circuits, network):
	batches=[]
	currentBatch=[]
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

		NewBigCircuit=CombineCircuits([currentBigCircuit,remainCircuits[best_circ]],network)
		if NewBigCircuit.latency <= currentBigCircuit.latency + remainCircuits[best_circ].latency:
			currentBatch.append(remainCircuits[best_circ])
			currentBigCircuit=NewBigCircuit
			del remainCircuits[best_circ]
		else:
			batches.append(currentBatch)
			currentBatch=[remainCircuits[best_circ]]
			currentBigCircuit=remainCircuits[best_circ]
			del remainCircuits[best_circ]
	batches.append(currentBatch)
	return batches

def SetCover2(Circuits, network):
	HugeCircuit=[]
	for circ in Circuits:
		HugeCircuit,_=CombineCircuits([HugeCircuit,circ], network)
	HugeNetwork=NetworkGraph(M, HugeCircuit.qubits)


i=0
#qG=Create_qG(10)
network= NetworkGraph(M,num)

Circuits=ReadInput('dir1')
for circ in Circuits:
	print(circ.filepath)
	circ.latency=LatencyofCircuitonNetwork(circ, network)

batches=SetCover1(Circuits, network)

for batch in batches:
	print([circ.filepath for circ in batch])
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
