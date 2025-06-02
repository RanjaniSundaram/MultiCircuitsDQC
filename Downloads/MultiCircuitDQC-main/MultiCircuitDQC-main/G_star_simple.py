#import Balancer_Cut
import sys
import csv
from qiskit.circuit.random import random_circuit
from qiskit.tools.visualization import circuit_drawer
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
import copy
import random
import itertools
import kahypar as kahypar
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
	print(len(parts),k)
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
		

def refined_weights(qubit1, qubit2, qubits, gates):
	dummy_qubit=max(qubit1, qubit2)+1
	new_gates=[]
	print("in refined weights")
	for gate in gates:
		if str(qubit1) in gate and str(qubit2) in gate:
			#print("actual CZ")
			new_gates.append([str(0), str(1)])
		elif len(gate)==1:
			if str(qubit1) in gate:
				new_gates.append(['0'])
			elif str(qubit2) in gate:
				#print("actual unary")
				new_gates.append(['1'])
			elif new_gates!=[] and new_gates[-1]!=[str(2)]:
				new_gates.append([str(2)])
		else:
			if str(qubit1) in gate:
				#print(gate)
				new_gates.append([str(0),str(2)])
			if str(qubit2) in gate:
				new_gates.append([str(1),str(2)])
	print(new_gates)
	return new_gates



def Circuit_to_Graph(qubits, gates):
	edges=[tuple(sorted(elem)) for elem in gates if len(elem)>1]
	cost=0
	weights2={}
	seen=[]
	for edge in list(dict.fromkeys(edges)):
		q1=edge[0]
		q2=edge[1]
		res_gates=[gate for gate in gates if (((q1 in gate) and (q2 in gate)) or ((q1 in gate) and (len(gate)==1)) or ((q2 in gate) and (len(gate)==1)) )]
		#ans,ans2,weights2[edge],dummy=OE([[0], [1], [2]], refined_weights(int(q1), int(q2), qubits, gates), [0, 1, 2])
		seen.append(q1)
		seen.append(q2)
		
	weights=collections.Counter(edges)
	myData=[list(edge)+[str(weights[edge])] for edge in edges]
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

def check_cover(vertex1, vertex2, gates,edges, inv_edges, Back_Pointers):
	boolean=False
	answer=list(dict.fromkeys(edges[vertex1]+edges[vertex2]))
	if vertex1[1] != vertex2[1]:
		return answer
	else:
		for i in gates:
			v1=inv_edges[i][0]
			v2=inv_edges[i][1]
			if (i+1 > int(vertex1[2])) and (i+1 > int(vertex2[2])):
				if v1[0] == vertex1[0] and v2[0] == vertex2[0] and v1[2]== vertex1[2] and v2[2]== vertex2[2]:
					answer.append(i)
					Back_Pointers[i].append((vertex1,vertex2))
					boolean=True
				elif v1[0] == vertex2[0] and v2[0] == vertex1[0] and v1[2]== vertex2[2] and v2[2]== vertex1[2]:
					answer.append(i)
					Back_Pointers[i].append((vertex2,vertex1))
					boolean=True
		return set(answer), boolean


				
def OE(parts, gates, qubits):
	max_memory=40
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
	complete_migrations=[]
	edges=defaultdict(list)  
	inv_edges=defaultdict(list)   
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
						inv_edges[i+1].append(tuple([str(gate[0]), str(partition[int(gate[1])]), '0']))
					else: 
						vertices.append(tuple([str(gate[1]), str(partition[int(gate[0])]), '0']))
						vertices2.append(tuple([gate[1],gate[0], '0']))
						edges[tuple([str(gate[1]), str(partition[int(gate[0])]), '0'])].append(i+1)
						inv_edges[i+1].append(tuple([str(gate[1]), str(partition[int(gate[0])]), '0']))
					
		
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
						inv_edges[i+1].append(tuple([str(gate[0]), str(partition[int(gate[1])]), str(pos)]))
				else: 
						vertices.append(tuple([str(gate[1]), str(partition[int(gate[0])]), str(pos)]))
						vertices2.append(tuple([gate[1],gate[0], str(pos)]))
						edges[tuple([str(gate[1]), str(partition[int(gate[0])]), str(pos)])].append(i+1)
						inv_edges[i+1].append(tuple([str(gate[1]), str(partition[int(gate[0])]), str(pos)]))
	
	vertices=list(dict.fromkeys(vertices))
	vertices2=list(dict.fromkeys(vertices2))
	n=len(vertices)
	answer=[]
	answer2=[]
	uncovered_CZ=set([i+1 for i,gate in enumerate(gates) if len(gate)==2 and partition[int(gate[0])]!=partition[int(gate[1])]]) 
	
	for vertex in vertices:
		for P,part in enumerate(parts):
			if P!= partition[int(vertex[0])]:
				complete_migrations.append(tuple([vertex[0], str(P), vertex[2]]))
	complete_migrations=list(dict.fromkeys(complete_migrations))
	tupled_vertices=[]
	length={}
	cover=defaultdict(set) 
	inv_migrations=defaultdict(list)
	j=0
	numqubits=len(qubits)
	gate_matrix=[[0 for _ in range(numqubits)] for _ in range(numqubits)]
	gate_adjacency=[[] for i in range(numqubits)]
	for i in uncovered_CZ:
		gate=gates[i-1]
		gate_matrix[int(gate[0])][int(gate[1])]=1
		gate_matrix[int(gate[1])][int(gate[0])]=1
		gate_adjacency[int(gate[0])].append(i)
		gate_adjacency[int(gate[1])].append(i)
		
	print(len(complete_migrations))
	for migration in complete_migrations:
		if migration not in vertices:
			edges[migration]=[]
		cover[migration]=set(edges[migration])
		length[migration]=1
		inv_migrations[migration[1]].append(migration)
	temp_inv=copy.deepcopy(inv_migrations)
	Back_Pointers=copy.deepcopy(inv_edges)
	for i in range(len(complete_migrations)):
		vertex1=complete_migrations[i]
		for vertex2 in temp_inv[vertex1[1]]:
			if gate_matrix[int(vertex1[0])][int(vertex2[0])]==1 and not (partition[int(vertex1[0])]==partition[int(vertex2[0])]):
				v1_gates=set(gate_adjacency[int(vertex1[0])])
				v2_gates=set(gate_adjacency[int(vertex2[0])])
				intersection=v1_gates & v2_gates
				cover[(vertex1,vertex2)], boolean=check_cover(vertex1,vertex2,intersection,edges, inv_edges, Back_Pointers)
				if boolean:
					tupled_vertices.append((vertex1,vertex2))
					inv_migrations[vertex1[1]].append((vertex1,vertex2))
					length[(vertex1,vertex2)]=2
				else:
					cover[(vertex1,vertex2)]=set()
				j+=1
		#print(j)
	#print(tupled_vertices)
	#print(Back_Pointers)
	cover_migrations=complete_migrations+tupled_vertices
	print(len(tupled_vertices), len(vertices), len(inv_edges))
	#print(cover_migrations)
	
	covered=set()
	step_size=0
	#print(len(uncovered_CZ))
	Memory_matrix=csr_matrix((numparts, len(gates) ), dtype=np.int8)

	while uncovered_CZ:
		values=[(len(cover[elem])/length[elem] ) for elem in cover_migrations]
		best_migration= cover_migrations[np.argmax(values)]
		
		if len(cover[best_migration])==1:
			alternate=inv_edges[list(cover[best_migration])[0]]
			latest_CZ1=max(cover[best_migration])
			alive_time=find_ge([elem[0]-1 for elem in timestamps] , int(best_migration[2]), latest_CZ1)
			best_ebit=0
			for timestamp in alive_time:
				if Memory_matrix[int(best_migration[1]),timestamp]>best_ebit:
					best_ebit=Memory_matrix[int(best_migration[1]),timestamp]
			for migration in alternate:
				latest_CZ1=max(cover[best_migration])
				alive_time=find_ge([elem[0]-1 for elem in timestamps] , int(migration[2]), latest_CZ1)
				current_ebit=0
				for timestamp in alive_time:
					if Memory_matrix[int(migration[1]),timestamp]>current_ebit:
						current_ebit=Memory_matrix[int(migration[1]),timestamp]
				if current_ebit<best_ebit:
					best_ebit=current_ebit
					best_migration=migration
				
				
		if len(best_migration)==2:
			
			answer.append(best_migration[0])
			answer.append(best_migration[1])
			cover_migrations.remove(best_migration)
			if best_migration[0] in cover_migrations:
				cover_migrations.remove(best_migration[0])
			if best_migration[1] in cover_migrations:
				cover_migrations.remove(best_migration[1])
			latest_CZ=max(cover[best_migration])
			alive_time1=find_ge([elem[0]-1 for elem in timestamps] , int(best_migration[0][2]), latest_CZ)
			#print(alive_time1)
			for timestamp in alive_time1:
				Memory_matrix[int(best_migration[0][1]),timestamp]+=1
			alive_time2=find_ge([elem[0]-1 for elem in timestamps] , int(best_migration[1][2]), latest_CZ)
			#print(alive_time2)
			for timestamp in alive_time2:
				Memory_matrix[int(best_migration[1][1]),timestamp]+=1
		else:
			answer.append(best_migration)
			cover_migrations.remove(best_migration)
			latest_CZ1=max(cover[best_migration])
		
			alive_time=find_ge([elem[0]-1 for elem in timestamps] , int(best_migration[2]), latest_CZ1)
			#print(alive_time)
			#print("interval ", int(best_migration[2]),latest_CZ1, cover[best_migration])
			for timestamp in alive_time:
				#print(Memory_matrix[int(best_migration[1]),timestamp])
				Memory_matrix[int(best_migration[1]),timestamp]+=1
		covered=covered|cover[best_migration]
		uncovered_CZ=uncovered_CZ-cover[best_migration]
		print(len(uncovered_CZ), [[(gates[index-1][0], partition[int(gates[index-1][0])]), (gates[index-1][1], partition[int(gates[index-1][1])])] for index in cover[best_migration]])
		print(best_migration)
		for migration in tupled_vertices+complete_migrations:
			cover[migration]=cover[migration]-covered
		latest_migration_partition=answer[-1][1]
		for migration in set(inv_migrations[latest_migration_partition])&set(cover_migrations):
			if length[migration]==1:
				cover[migration]=cover[migration].union(cover[(migration,answer[-1])])-covered
				#print(cover[(migration,answer[-1])])
				if len(answer)>2:
					cover[migration]=cover[migration].union(cover[(migration,answer[-2])])-covered

			elif length[migration]==2:
				cover[migration]=cover[migration]|cover[(migration[0],answer[-1])]-covered
				if len(answer)>=2:
					cover[migration]=cover[migration]|cover[(migration[0],answer[-2])]|cover[(migration[1],answer[-2])]-covered
				cover[migration]=cover[migration]|cover[(migration[1],answer[-1])]-covered
		#print("here")
	answer=list(dict.fromkeys(answer))
	#print(covered)

	cost=len(answer)
	#print(answer)
	pd.DataFrame(Memory_matrix.toarray()).to_csv('memory_matrix115.csv')
	print(Memory_matrix.max())
	memory=Memory_matrix.max()
	return answer,answer2,cost, memory

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
	
	

def verify(qubits, gates, answer, parts2):
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
	for q,P,t in init_migrations:
		qubit=int(q)
		parts[int(P)].append(qubit)
		ebits_per_part[int(P)]=ebits_per_part[int(P)]|{qubit}
	ebit_count=max([len(ebits_per_part[p]) for p in ebits_per_part.keys()])
	if ebit_count>max_ebits:
		max_ebits=ebit_count
	flag=0
	for i,gate in enumerate(gates):
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
			ebit_count=max([len(ebits_per_part[p]) for p in ebits_per_part.keys()])
			if ebit_count>max_ebits:
				max_ebits=ebit_count
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


def main_func_Gstar_simple(filename,i):
	start_time=time.time()
	qubits, gates= get_circuit('Inputs/'+filename)
	print(filename)
	MyData=Circuit_to_Graph(qubits, gates)
	print("graph constructed")
	if(len(sys.argv)==1):
		num=int(i)
		print(num)
		#epsilon=get_epsilon(filename, num, qubits)
		epsilon=0.2
		print(epsilon)
		v_part=KaHyPar_partition(qubits,MyData, num, epsilon)
		#v_part=vertex_partition("graph2.csv","bestPartition.txt",4, 0.1)     #if sourceFile=="" a networkx graph is generated otherwise the file is read from graph
	print("OE")
	old_stdout = sys.stdout
	sys.stdout = open(os.devnull, "w")
	answer,answer2, cost, max_memory=OE(v_part, gates, qubits)
	sys.stdout = old_stdout
	print("answer: ",answer)
	max_ebits=verify(qubits, gates, answer, v_part)
	printing(qubits, gates, v_part, answer, cost, filename, num, epsilon,max_memory)
i=0
start_time=time.time()

"""
for filename in os.listdir('circuitsVaryPart0.5'):
	if 'random' in filename:
		continue
	if filename not in os.listdir('Algo2VaryPart0.5'):
		start_time=time.time()
		main_func(filename,i)
	i+=1
"""
start_time=time.time()
#main_func_Gstar_simple("example.txt",i)
