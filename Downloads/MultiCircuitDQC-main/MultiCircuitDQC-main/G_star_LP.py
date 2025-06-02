#import Balancer_Cut
import sys
import csv
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
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
from pulp import PULP_CBC_CMD
import time
from scipy.optimize import linprog
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
		
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value



def Circuit_to_Graph(qubits, gates):
	edges=[tuple(sorted(elem)) for elem in gates if len(elem)>1]
	cost=0
	weights2={}
	seen=[]
	for edge in list(dict.fromkeys(edges)):
		q1=edge[0]
		q2=edge[1]
		res_gates=[gate for gate in gates if (((q1 in gate) and (q2 in gate)) or ((q1 in gate) and (len(gate)==1)) or ((q2 in gate) and (len(gate)==1)) )]
		#ans,ans2,weights2[edge]=OE([[int(q1)], [int(q2)]], res_gates, [int(q1), int(q2)])
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

def check_cover(vertex1, vertex2, gates,edges, inv_edges):
	boolean=False
	answer2=set(dict.fromkeys(edges[vertex1]+edges[vertex2]))
	answer=[]
	if vertex1[1] != vertex2[1]:
		return answer
	else:
		for i in gates:
			v1=inv_edges[i][0]
			v2=inv_edges[i][1]
			if (i+1 > int(vertex1[2])) and (i+1 > int(vertex2[2])):
				if v1[0] == vertex1[0] and v2[0] == vertex2[0] and v1[2]== vertex1[2] and v2[2]== vertex2[2]:
					answer.append(i)
					#Back_Pointers[i].add((vertex1,vertex2))
					boolean=True
				elif v1[0] == vertex2[0] and v2[0] == vertex1[0] and v1[2]== vertex2[2] and v2[2]== vertex1[2]:
					answer.append(i)
					#Back_Pointers[i].add((vertex2,vertex1))
					boolean=True
		if not set(answer)-answer2:
			boolean=False
		return set(answer)-answer2, boolean

def best_candidate(deletion_loss_copy1):
	best_value=0
	candidate=[]
	not_covered=set()

	deletion_loss_copy=copy.deepcopy(deletion_loss_copy1)
	while deletion_loss_copy:
		min_weight_migration=min(deletion_loss_copy, key=lambda k: len(deletion_loss_copy[k]))
		cov=deletion_loss_copy[min_weight_migration]
		not_covered=not_covered|cov
		del deletion_loss_copy[min_weight_migration]

		for key in deletion_loss_copy.keys():
			deletion_loss_copy[key]=deletion_loss_copy[key]-cov
		if not deletion_loss_copy:
			break
		value=sum([len(v) for v in deletion_loss_copy.values()])/len(deletion_loss_copy)

		if value>best_value:
			best_value=value
			candidate=list(deletion_loss_copy.keys())
			
	return candidate


def LP(cover_migrations, mapped_migrations,cover):
	bounds=[]
	choose_one=[]
	variables={}
	threshold_candidates=[]
	model = LpProblem(name="DensestSubgraph", sense=LpMaximize)
	finished=set()

	for vertex in set(cover_migrations):
		if len(vertex)==2 and vertex[0] !=vertex[1]:
			variables[mapped_migrations[vertex]]=LpVariable(name="x"+str(mapped_migrations[vertex[0]])+"_"+str(mapped_migrations[vertex[1]]), lowBound=0 )
			if vertex[0] not in finished:
				variables[mapped_migrations[vertex[0]]]=LpVariable(name="y"+str(mapped_migrations[vertex[0]]), lowBound=0 )
				finished=finished|{vertex[0]}
			if vertex[1] not in finished:
				variables[mapped_migrations[vertex[1]]]=LpVariable(name="y"+str(mapped_migrations[vertex[1]]), lowBound=0 )
				finished=finished|{vertex[1]}
			model += (variables[mapped_migrations[vertex]]-variables[mapped_migrations[vertex[0]]]<=0, "")
			model += (variables[mapped_migrations[vertex]]-variables[mapped_migrations[vertex[1]]]<=0, "")
			finished=finished|{vertex[0],vertex[1]}
		elif vertex not in finished:
			variables[mapped_migrations[vertex]]=LpVariable(name="y"+str(mapped_migrations[vertex]), lowBound=0 )
			finished=finished|{vertex}
		if len(vertex)!=2:
			threshold_candidates.append(mapped_migrations[vertex])


	model+=(sum([variables[mapped_migrations[vertex]] for vertex in finished])<=1)

	model += lpSum([len(cover[vertex])*variables[mapped_migrations[vertex]] for vertex in cover_migrations ])
	#print(model)
	
	status = model.solve(solver=PULP_CBC_CMD(msg=False))
	print("LP solved")
	#print(model.objective.value())
	"""
	for var in model.variables():
		if 'y' in var.name:
			threshold_candidates.append(var.value())
	"""
	
	
	if model.objective.value()==0 or model.objective.value()==None:
		return []
	ignore=1/10*(len(threshold_candidates))
	max_obj=0
	best_candidate=[]
	best_threshold=0
	threshold_candidates=list(set([round(variables[threshold].value(),4) for threshold in threshold_candidates if variables[threshold].value()>1e-10]))
	threshold_candidates.sort()
	important_vertices=cover_migrations
	print(threshold_candidates)
	for threshold in threshold_candidates:
		temp_important=[]
		candidate=[]
		objective=0
		for vertex in important_vertices:
			if round(variables[mapped_migrations[vertex]].value(),4)>=threshold:
				temp_important.append(vertex)
				if len(vertex)!=2:
					candidate.append(vertex)
				objective+=len(cover[vertex])
		important_vertices=temp_important
		objective=objective/len(candidate)
		if objective>max_obj:
			max_obj=objective
			best_candidate=candidate
			
	"""
	for threshold in threshold_candidates:
		important_vertices=[]
		for vertex in cover_migrations:
			if round(variables[mapped_migrations[vertex]].value(),4)>=threshold:
				important_vertices.append(vertex)
		important_vertices=[vertex for vertex in cover_migrations if round(variables[mapped_migrations[vertex]].value(),4)>=threshold]
		candidate=[vertex for vertex in important_vertices if (len(vertex)!=2)]
		objective=sum([len(cover[vertex]) for vertex in important_vertices])/len(candidate)
		if objective>max_obj:
			max_obj=objective
			best_candidate=candidate
	"""
	return best_candidate
				
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
	inv_migrations=defaultdict(set)
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
	ind=0
	mapped_migrations={}
	for migration in complete_migrations:
		if migration not in vertices:
			edges[migration]=[]
		cover[migration]=set(edges[migration])
		length[migration]=1
		inv_migrations[migration[1]].add(migration)
		mapped_migrations[migration]=ind
		ind+=1
	temp_inv=copy.deepcopy(inv_migrations)
	deletion_loss=AutoVivification()
	#Back_Pointers=copy.deepcopy(inv_migrations)
	for i in range(len(complete_migrations)):
		vertex1=complete_migrations[i]
		deletion_loss[int(vertex1[1])][vertex1]=set().union(cover[vertex1])
		for vertex2 in temp_inv[vertex1[1]]:
			if gate_matrix[int(vertex1[0])][int(vertex2[0])]==1 and not (partition[int(vertex1[0])]==partition[int(vertex2[0])]):
				v1_gates=set(gate_adjacency[int(vertex1[0])])
				v2_gates=set(gate_adjacency[int(vertex2[0])])
				intersection=v1_gates & v2_gates
				cover[(vertex1,vertex2)], boolean=check_cover(vertex1,vertex2,intersection,edges, inv_edges)
				deletion_loss[int(vertex1[1])][vertex1]=deletion_loss[int(vertex1[1])][vertex1]|cover[(vertex1,vertex2)]
				if boolean:
					tupled_vertices.append((vertex1,vertex2))
					inv_migrations[vertex1[1]].add((vertex1,vertex2))
					length[(vertex1,vertex2)]=2
					mapped_migrations[(vertex1,vertex2)]=ind
					ind+=1
				else:
					cover[(vertex1,vertex2)]=set()
				j+=1
		#print(j)
	#print(tupled_vertices)
	#print(Back_Pointers)
	cover_migrations=tupled_vertices+complete_migrations
	print(len(tupled_vertices), len(complete_migrations), len(inv_edges))
	length_cover_mig=len(cover_migrations)

	inv_migrations_copy=inv_migrations
	covered=set()
	deletion_loss_copy=deletion_loss

	answer=defaultdict(set) 
	answer2=set()
	cover_copy=copy.deepcopy(cover)
	print(len(uncovered_CZ))
	
	print(parts)
	Memory_matrix=csr_matrix((numparts, len(gates) ), dtype=np.int8)
	MigrationtoCZ=defaultdict(set)
	while uncovered_CZ:
		candidates=set()
		best_value=0
		buffer_list=[]
		for i in range(numparts):
			covered_temp=defaultdict(set) 
			for key in deletion_loss_copy[i]:
				deletion_loss_copy[i][key]=deletion_loss_copy[i][key]-covered
			print("before LP")
			answer[i]=set(LP(inv_migrations_copy[str(i)],mapped_migrations,cover_copy))
			print("after LP")
			"""
			deletion_loss_copy1=copy.deepcopy(deletion_loss_copy[i])
			for key in answer[i]:
				del deletion_loss_copy1[key]"""		
			not_covered=set()
			for key,value in deletion_loss_copy[i].items():
				if key not in answer[i]:
					not_covered=not_covered|value

			#not_covered=set().union(*deletion_loss_copy1.values())
			for key in answer[i]:			
				covered_temp[i]=covered_temp[i]|(deletion_loss_copy[i][key]-not_covered)
			if not answer[i]:
				break
			value=len(covered_temp[i])/len(answer[i])
			if value>best_value:
				index=i
				best_value=value
				candidates=answer[i]
			
		for i in range(numparts):
			if not answer[i]:
				break
			value=len(covered_temp[i])/len(answer[i])
			if value==best_value and i!=index:
				if not covered_temp[i]&covered_temp[index]:
					buffer_list.append(i)
					candidates=candidates|answer[i]

		for i in buffer_list:
			for entry in inv_migrations_copy[str(i)]:
				 if entry in candidates:
					 inv_migrations_copy[str(i)]= inv_migrations_copy[str(i)]-{entry}
				 elif entry[0] in candidates:
					 cover_copy[entry[1]]=cover_copy[entry[1]]|cover_copy[entry]
					 inv_migrations_copy[str(i)]= inv_migrations_copy[str(i)]-{entry}
				 elif entry[1] in candidates:
					 cover_copy[entry[0]]=cover_copy[entry[0]]|cover_copy[entry]
					 inv_migrations_copy[str(i)]= inv_migrations_copy[str(i)]-{entry}

		answer2=answer2|candidates
		for vertex1 in candidates:
			for vertex2 in answer2&set(temp_inv[vertex1[1]]):
				covered=covered|(cover[vertex1]|cover[(vertex1,vertex2)])
				MigrationtoCZ[vertex1]=MigrationtoCZ[vertex1]|cover_copy[vertex1]|cover_copy[(vertex1,vertex2)]
				MigrationtoCZ[vertex2]=MigrationtoCZ[vertex2]|cover_copy[vertex2]|cover_copy[(vertex1,vertex2)]
				

		for key in cover_copy.keys():
			cover_copy[key]=cover_copy[key]-covered
		
		uncovered_CZ=uncovered_CZ-covered
		print(len(uncovered_CZ))

	answer=answer2
	
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
	cost=len(answer)
	#print(answer)
	return answer,answer2,cost, memory


		

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
	max_ebits=0
	parts=copy.deepcopy(parts2)
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
def get_epsilon(filename, num, qubits):
	f=open('outputs3/'+filename,'r')
	Lines=f.readlines()
	result = [e for e in re.split("[^0-9]",Lines[6]) if e != '']
	size=max(map(int, result))-1
	epsilon=size/len(qubits)*num -1

	return epsilon

def main_func_Gstar_LP(filename,i):
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

"""
for filename in os.listdir('circuitsVaryQubits0.51'):
	if 'random' in filename:
		continue
	if filename not in os.listdir('GstarVaryQubits0.5'):
		start_time=time.time()
		main_func(filename,i)
	i+=1

"""

start_time=time.time()
#main_func_Gstar_LP("example.txt",i)
#main_func("qft10.txt",i)
