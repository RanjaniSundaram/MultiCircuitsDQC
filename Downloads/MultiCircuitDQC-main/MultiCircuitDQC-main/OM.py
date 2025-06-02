#import Balancer_Cut
import sys
import csv
#from qiskit.circuit.random import random_circuit
#from qiskit.tools.visualization import circuit_drawer
from vertex_cov import min_vertex_cover
from pprint import pprint
import collections
import execnet
import numpy as np
import regex as re
from collections import defaultdict 
from networkx.algorithms import bipartite
import networkx as nx
from collections import defaultdict
from pprint import pprint
import os
import random
import kahypar as kahypar
import copy
import time

from scipy.sparse import csr_matrix
import bisect
import pandas as pd

def find_ge(a, low, high):
	i = bisect.bisect_left(a, low)
	g = bisect.bisect_right(a, high)
	if i <= len(a) and g <= len(a):
		return a[i:g]
	raise ValueError
def KaHyPar_partition(qubits,MyData, k, epsilon):
	if k==1:
		return [qubits]
	num_nodes = len(qubits)
	num_edges = len(MyData)
	edgelist=[]
	for data in MyData:
		edgelist.append(int(data[0]))
		edgelist.append(int(data[1]))
	hyperedge_indices = [2*i for i in range(num_edges+1)]
	node_weights = [1]*num_nodes
	edge_weights = [int(data[2]) for data in MyData]
	hypergraph = kahypar.Hypergraph(num_nodes, num_edges, hyperedge_indices, edgelist, k, edge_weights, node_weights)
	context = kahypar.Context()
	context.loadINIconfiguration("./kahypar_config.ini")
	context.setK(k)
	context.setEpsilon(epsilon)
	context.suppressOutput(True)
	kahypar.partition(hypergraph, context)
	n=hypergraph.numBlocks()
	parts=[[x for x in range(num_nodes) if hypergraph.blockID(x)==y] for y in range(n)]
	print(parts)
	return parts
	

def call_python_version(Version, Module, Function, ArgumentList):
	gw      = execnet.makegateway("popen//python=python%s" % Version)
	channel = gw.remote_exec("""
		from %s import %s as the_function
		channel.send(the_function(*channel.receive()))
	""" % (Module, Function))
	channel.send(ArgumentList)
	return channel.receive()

def correct(parts,k):
	c_parts=parts[:]
	#print(len(parts),k)
	if len(parts)>k:
		n=str(parts).count(",")+1
		minimum1=min(c_parts,key=len)
		c_parts.remove(minimum1)
		minimum2=min(c_parts,key=len)
		c_parts.remove(minimum2)
		c_parts.append(minimum1+minimum2)

	return c_parts

def vertex_partition(sourceFile, destinationFile, k, epsilon):
	call_python_version("2", "Balancer_Cut", "main",  [sourceFile, destinationFile, k, epsilon]) 
	#parts=Balancer_Cut.main(sourceFile, destinationFile, k, epsilon)
	parts=[]
	with open("bestPartition.txt") as f:
		parts=[[int(x) for x in y.strip().strip('[').strip(']').split(',')] for y in f.readlines()]
	parts2=correct(parts,k)
	return parts2
			

def get_circuit(filename):
	file1 = open(filename, 'r') 
	Lines = file1.readlines()
	qubits=[]
	gates=[]
	qubits=re.findall('[0-9]+', Lines[0])
	qubits=[int(s) for s in qubits]
	for line in Lines[1:len(Lines)]:
		if "QGate" in line:
			gates.append(re.findall('[0-9]+', line))
		if "QInit" in line:
			val= re.findall("\\(([0-9]+)\\)", line)[0]
			if int(val) not in qubits:
				qubits.append(int(val))
		#inputs=[]
		#for i in line:
		#   if i.isdigit() ==True:
		#       inputs.append(i)
		#gates.append(inputs)
	#print("hi")
	return qubits, gates
	#print(gates)_
#print
		



def Circuit_to_Graph(qubits, gates):
	edges=[tuple(sorted(elem)) for elem in gates if len(elem)>1]
	cost=0
	weights2={}
	seen=[]
	for edge in list(dict.fromkeys(edges)):
		q1=edge[0]
		q2=edge[1]
		res_gates=[gate for gate in gates if (((q1 in gate) and (q2 in gate)) or ((q1 in gate) and (len(gate)==1)) or ((q2 in gate) and (len(gate)==1)) )]
		ans,ans2,weights2[edge], dummy, dummy2=OE([[int(q1)], [int(q2)]], res_gates, [int(q1), int(q2)])
		seen.append(q1)
		seen.append(q2)
		
	weights=collections.Counter(edges)
	myData=[list(edge)+[str(weights2[edge])] for edge in edges]
	for qubit in qubits:
		if seen==[]:
			seen.append(str(qubit))
		if qubit not in seen:
			myData.append([str(qubit), seen[0], "0"])
	Data=[]
	for entry in myData:
		Data.append([" ".join(entry)])
	graph=open('graph2.csv','w')
	with graph:
		writer = csv.writer(graph)
		writer.writerows(Data)
	return myData
		
def OE(parts, gates, qubits):
	numparts=len(parts)
	timestamps=[(i+1,e) for i, e in enumerate(gates) if len(e)<2]
	partition={}
	#print(parts)
	#print(qubits)
	for qubit in qubits:
		flag=0
		for i in range(0, numparts):
			if qubit in parts[i]:
				flag=1
				partition[qubit]=i
		if flag==0:
			parts[0].append(qubit)
			partition[qubit]=0
	vertices= []
	#print(partition)
	vertices2=[]
	edges=defaultdict(list)     
	for qubit in qubits:
		for i,gate in enumerate(gates):
			if str(qubit) in gate: 
				if len(gate)<2:
					break
				elif partition[int(gate[0])] != partition[int(gate[1])]:
					if str(qubit)==gate[0]:
						vertices.append(tuple([str(gate[0]), str(partition[int(gate[1])]), '0']))
						vertices2.append(tuple(gate+['0']))
						edges[tuple([str(gate[0]), str(partition[int(gate[1])]), '0'])].append(i+1)
					else: 
						vertices.append(tuple([str(gate[1]), str(partition[int(gate[0])]), '0']))
						vertices2.append(tuple([gate[1],gate[0], '0']))
						edges[tuple([str(gate[1]), str(partition[int(gate[0])]), '0'])].append(i+1)
					
		
	index=0
	for qubit in qubits:
		index=0
		for i,gate in enumerate(gates):
			if str(qubit) in gate and len(gate)<2:
				index=[j for (j,e) in timestamps if gate==e]
			if str(qubit) in gate and len(gate)>1 and index!=0 and (partition[int(gate[0])]!= partition[int(gate[1])]):
				pos=max([position for position in index if position <i+1])
				if str(qubit)==gate[0]:
						vertices.append(tuple([str(gate[0]), str(partition[int(gate[1])]), str(pos)]))
						vertices2.append(tuple(gate+[str(pos)]))
						edges[tuple([str(gate[0]), str(partition[int(gate[1])]), str(pos)])].append(i+1)
				else: 
						vertices.append(tuple([str(gate[1]), str(partition[int(gate[0])]), str(pos)]))
						vertices2.append(tuple([gate[1],gate[0], str(pos)]))
						edges[tuple([str(gate[1]), str(partition[int(gate[0])]), str(pos)])].append(i+1)
	
	vertices=list(dict.fromkeys(vertices))
	vertices2=list(dict.fromkeys(vertices2))
	n=len(vertices)
	answer=[]
	answer2=[]
	#A = np.zeros(shape=(n,n))
	#for i,gate1 in enumerate(vertices):
	#   for j,gate2 in enumerate(vertices):
	#       if set(edges[gate1]) & (set(edges[gate2])) and i!=j:
	#           A[i][j]+=len(set(edges[gate1]).intersection(set(edges[gate2])))
	
	graph = defaultdict(list)
	for i,gate1 in enumerate(vertices):
		for j,gate2 in enumerate(vertices):
			if set(edges[gate1]) & set(edges[gate2]) and i!=j:
				graph[i].append(j)
	U,V=bfs([],graph, 0)
	U2=defaultdict(list)
	V2=defaultdict(list)
	for key,value in graph.items():
		if key in U:
			U2[key]=value
		else:
			V2[key]=value
	result= min_vertex_cover(U2,V2)
	for i,vertex in enumerate(vertices):
		if i in result.keys():
			answer.append(vertex)
	cost=len(answer)
	Memory_matrix=csr_matrix((numparts, len(gates) ), dtype=np.int8)
	MigrationtoCZ=defaultdict(set)
	uncovered_CZ=set([i+1 for i,gate in enumerate(gates) if len(gate)==2 and partition[int(gate[0])]!=partition[int(gate[1])]])
	for vertex in answer:
		MigrationtoCZ[vertex]=MigrationtoCZ[vertex]|(set(edges[vertex])&uncovered_CZ)
		uncovered_CZ=uncovered_CZ-set(edges[vertex])
		
	for key in answer:
			latest_CZ1=max(MigrationtoCZ[key])
		
			alive_time=find_ge([elem[0]-1 for elem in timestamps] , int(key[2]), latest_CZ1)
			#print(alive_time)
			#print("interval ", int(best_migration[2]),latest_CZ1, cover[best_migration])
			for timestamp in alive_time:
				#print(Memory_matrix[int(best_migration[1]),timestamp])
				Memory_matrix[int(key[1]),timestamp]+=1
	print(Memory_matrix.max())
	memory=Memory_matrix.max()
	#print(answer)
	return answer,answer2,cost, edges, memory

def bfs(visited, graph, node):
	visited={node:1}
	fulllist=list(graph.keys())
	queue=[node]
	color={}
	U=[]
	V=[]
	U.append(node)
	color[node]="red"
	while queue or fulllist:
		if queue!=[]:
			s = queue.pop(0)
		else:
			s = fulllist.pop(0)
			if s not in visited:
				visited[s]=1
				color[s]="red"
				U.append(s)
			else:
				continue
		for neighbour in graph[s]:
			if neighbour not in visited:
				visited[neighbour]=1
				queue.append(neighbour)
				if color[s]=="red":
					V.append(neighbour)
					color[neighbour]="blue"
				else:
					U.append(neighbour)
					color[neighbour]="red"
	'''U1=[]
	V1=[]
	if len(visited)<len(graph):
		for s,v in graph.items():
			if s not in visited:
				U1,V1=bfs(visited,graph,s)
				break'''
	return U,V
		

def printing(qubits, gates, v_part, answer, cost, filename,num, epsilon, max_ebits):
	tot_time=time.time() - start_time
	minutes, seconds = divmod(tot_time, 60)
	f=open('Results/'+filename, "w")
	f.write("Original Circuit: ")
	f.write('\n')
	f.write("qubits: "+ str(qubits))
	f.write('\n')
	f.write("gates: "+ str(len(gates)))
	f.write('\n')
	f.write("num of CZ: " + str(len([gate for gate in gates if len(gate)>1])))
	f.write('\n')
	f.write("partitions: "+ str(v_part))
	f.write('\n')
	f.write("number of partitions: "+ str(num))
	#print("answer: ", answer)
	f.write('\n')
	n=len(qubits)
	f.write('size: '+ str(int((1+epsilon)*(n/num))+1))
	f.write('\n')
	f.write("cost: "+str(cost))
	f.write('\n')
	f.write('maximum number of ebits in partition: '+ str(max_ebits))
	f.write('\n')
	f.write("time: "+ str(minutes)+ " minutes "+ str(seconds)+ " seconds")
	
	
def verify(qubits, gates, answer, parts2, edges):
	partition={}
	ebits_per_part=defaultdict(set)
	parts=copy.deepcopy(parts2)
	max_ebits=0
	numparts=len(parts)
	for qubit in qubits:
		for i in range(0, numparts):
			if qubit in parts[i]:
				partition[qubit]=i
	init_migrations=[(q,P,t) for (q,P,t) in answer if int(t)==0]
	max_ebits_per_part={}
	for q,P,t in init_migrations:
		qubit=int(q)
		parts[int(P)].append(qubit)
		ebits_per_part[int(P)]=ebits_per_part[int(P)]|{qubit}

	ebit_count=max([len(ebits_per_part[p]) for p in ebits_per_part.keys()]+[0])
	for i in range(numparts):
		max_ebits_per_part[i]=len(ebits_per_part[i])
	if ebit_count>max_ebits:
		max_ebits=ebit_count
	flag=0
	
	whenToDestroy= defaultdict(list)
	for i,gate in enumerate(gates):
		#print(i+1)
		if len(gate)==1:
			qubit=int(gate[0])
			for j in range(0, len(parts)-1):
				if j!= partition[qubit]:
					if qubit in parts[j]:
						parts[j].remove(qubit)
						ebits_per_part[j]=ebits_per_part[j]-{qubit}
		if (i+1) in [int(t) for (q,P,t) in answer]:
			migrations=[(q,P,t) for (q,P,t) in answer if int(t)==i+1]
			for q,P,t in migrations:
				qubit=int(q)
				parts[int(P)].append(qubit)
				ebits_per_part[int(P)]=ebits_per_part[int(P)]|{qubit}
				destroy=max(edges[(q,P,t)])
				whenToDestroy[destroy].append((q,P,t))
				#print((q,P,t), " destroy: ", destroy)
			ebit_count=max([len(ebits_per_part[p]) for p in ebits_per_part.keys()])
			for i in range(numparts):
				if len(ebits_per_part[i])>max_ebits_per_part[i]:
					max_ebits_per_part[i]=len(ebits_per_part[i])
			if ebit_count>max_ebits:
				max_ebits=ebit_count
				#print(ebits_per_part)
			#print(i+1)
			#print(ebits_per_part)

		if len(gate)>1:
			check=0
			qubit1=int(gate[0])
			qubit2=int(gate[1])
				
			for p,part in enumerate(parts):
				if int(gate[0]) in part and int(gate[1]) in part :
					#print(gate, "satisfied")
					check=1
			
			if check==0:
				print(i+1, gate, "not satisfied")
				print(parts)
				flag=1
			if (i+1) in whenToDestroy.keys():
				for vertex in whenToDestroy[i+1]:
					#print(vertex)
					parts[int(vertex[1])].remove(int(vertex[0]))
					ebits_per_part[int(vertex[1])]=ebits_per_part[int(vertex[1])]-{int(vertex[0])}
	print(max_ebits_per_part)
	#print(ebits_per_part)
	if flag==1:
		print("something is wrong")
	return max_ebits
'''
def main_func(filename):
	qubits, gates= get_circuit('circuits3/'+filename)
	print(filename)
	MyData=Circuit_to_Graph(qubits, gates)
	print("graph constructed")
	if(len(sys.argv)==1):
		num=random.randint(int(len(qubits)/30)+1, int(len(qubits)/3)+1)
		print(num)
		epsilon=random.uniform(0.1, 0.6)
		print(epsilon)
		v_part=KaHyPar_partition(qubits,MyData, num, epsilon)
		#v_part=vertex_partition("graph2.csv","bestPartition.txt",4, 0.1)     #if sourceFile=="" a networkx graph is generated otherwise the file is read from graph
	print("OE")
	
	answer,answer2, cost=OE(v_part, gates, qubits)
	printing(qubits, gates, v_part, answer, cost, filename, num, epsilon)
	verify(qubits, gates, answer, v_part)               
'''
def getPart(i):
	fact=int(i/17)*2
	return max(2,fact)


def main_func_OM(filename,i):
	start_time=time.time()
	qubits, gates= get_circuit('circuitsVaryQubits/'+filename)
	print(filename)
	MyData=Circuit_to_Graph(qubits, gates)
	print("graph constructed")
	print(qubits)
	num=int(i)
	print(num)
	#epsilon=get_epsilon(filename, num, qubits)
	epsilon=0.2
	print(epsilon)
	v_part=KaHyPar_partition(qubits,MyData, num, epsilon)
	#v_part=vertex_partition("graph2.csv","bestPartition.txt",4, 0.1)     #if sourceFile=="" a networkx graph is generated otherwise the file is read from graph
	print("OE")
	#old_stdout = sys.stdout
	#sys.stdout = open(os.devnull, "w")
	answer,answer2, cost, edges,max_memory=OE(v_part, gates, qubits)
	#sys.stdout = old_stdout
	#print(gates)
	print(answer)
	#max_ebits=verify(qubits, gates, answer, v_part, edges)
	#printing(qubits, gates, v_part, answer, cost, filename, num, epsilon,max_memory)
	return answer
i=5

start_time=time.time()

"""
for filename in os.listdir('circuitsVaryPart0.5'):
	if 'random' in filename:
		continue
	if filename not in os.listdir('OMVaryPart0.5'):
		start_time=time.time()
		main_func(filename,i)
	i+=1
"""

#main_func_OM("example.txt",i)
