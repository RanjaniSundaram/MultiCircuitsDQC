import sys
import csv
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
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import bisect
from heapq import heapify, heappush, heappop 
import pandas as pd
import itertools
import heapq
from itertools import combinations, groupby
from scipy.optimize import quadratic_assignment
import matplotlib.pyplot as plt
import math
import time
import signal
from contextlib import contextmanager
from networkx.classes import graph
from commons.qChannel import qChannel
from commons.qNode import qNode, BSM_SUCCESS_RATE
from commons.qGraph import qGraph
from commons.Point import Point
from random import randint
from routing_algorithms.max_flow import max_flow
from routing_algorithms.dp_shortest_path import dp_shortest_path as dp
from routing_algorithms.sigcomm import Sigcomm_W
from routing_algorithms.dp_alternate import DP_Alt
from networkx.generators.geometric import waxman_graph
from networkx import Graph
from routing_algorithms.delft_lp import modified_qGraph, LP
import datetime
from routing_algorithms.naive_shortest_path import hop_shortest_path
from routing_algorithms.optimal_linear_programming import optimal_linear_programming
from protocols.predistribution.network_protocol import TreeProtocol as PredistProtocol
from protocols.network_protocol import TreeProtocol
from protocols.sigcomm_network_protocol import TreeProtocol as SigcommTreeProtocol
from commons.tree_node import tree_node
from plot_results import Plot
import os
from predistributed_algorithms.greedy_predist import greedy_predist
from predistributed_algorithms.multi_pair_greedy_predist import multi_pair_greedy_predist
from OM import OE
from itertools import zip_longest

def get_circuit(filename):  ## Reading input circuit as text files (Quipper format), outputs qubits and gates in order of execution
	file1 = open(filename, 'r') 
	Lines = file1.readlines()
	qubits=[]
	gates=[]
	qubits=re.findall('[0-9]+', Lines[0])
	MapQtoG=defaultdict(list)
	for line in Lines[1:len(Lines)]:
		if "QGate" in line:
			gate=re.findall('[0-9]+', line)
			if len(gate)==2:
				gates.append(sorted(gate,key=int))
				MapQtoG[gate[0]].append(sorted(gate,key=int)+[len(gates)])
				MapQtoG[gate[1]].append(sorted(gate,key=int)+[len(gates)])
			
			elif MapQtoG[gate[0]]==[] or gate != MapQtoG[gate[0]][-1][:2]:
				gates.append(gate)
				MapQtoG[gate[0]].append(gate+[len(gates)])
				
		if "QInit" in line:
			val= re.findall("\\(([0-9]+)\\)", line)[0]
			if val not in qubits:
				qubits.append(int(val))
				MapQtoG[val]=[]
	return qubits, gates, MapQtoG


def get_circuit2(filename):  ## Reading input circuit as text files (Quipper format), outputs qubits and gates in order of execution
	file1 = open(filename, 'r') 
	Lines = file1.readlines()
	qubits=[]
	gates=[]
	qubits=[str(i) for i in range(int(re.findall('[0-9]+', filename)[-1]))]
	print(re.findall('[0-9]+', filename))
	MapQtoG=defaultdict(list)
	for val in qubits:
		MapQtoG[val]=[]
	for line in Lines[1:len(Lines)]:
		if 'measure' not in line:
			if "cx" in line:
				gate=re.findall('(?<![\d.])[0-9]+(?![\d.])', line)
				#print(gate)
				if not gate:
					continue
				if len(gate)==2:
					gates.append(sorted(gate,key=int))
					MapQtoG[gate[0]].append(sorted(gate,key=int)+[len(gates)])
					MapQtoG[gate[1]].append(sorted(gate,key=int)+[len(gates)])
			elif "rz" in line or "x" in line:
				#print(line)
				gate=re.findall('(?<![\d.])[0-9]+(?![\d.])', line)
				#print(gate)
				if not gate:
					continue
				
				elif len(gate)==1:
					gates.append(gate)
					MapQtoG[gate[0]].append(gate+[len(gates)])
	print("here")
	return qubits, gates, MapQtoG

def Circuit_to_Graph(qubits, gates):  ## Converts a given circuit into a weighted graph
	edges=[tuple(sorted(elem)) for elem in gates if len(elem)>1]
	cost=0
	weights2={}
	seen=[]
	for edge in list(dict.fromkeys(edges)):
		q1=edge[0]
		q2=edge[1]
		#res_gates=[gate for gate in gates if (((q1 in gate) and (q2 in gate)) or ((q1 in gate) and (len(gate)==1)) or ((q2 in gate) and (len(gate)==1)) )]
		#ans,ans2,weights2[edge],dummy=OE([['0'], ['1'], ['2']], refined_weights(q1, q2, qubits, gates), [0, 1, 2])
		seen.append(q1)
		seen.append(q2)
		
	weights=collections.Counter(edges)
	myData=[list(edge)+[str(weights[edge])] for edge in edges]
	for qubit in qubits:
		if seen==[]:
			seen.append(qubit)
		if qubit not in seen:
			myData.append([qubit, seen[0], "0"])

	return myData
	
def MinQAP(qubits, MyData, network_graph):
	#network_matrix=nx.adjacency_matrix(network_graph)
	#print("network matrix")
	#print(network_matrix)
	distance_matrix=nx.floyd_warshall_numpy(network_graph)
	#print(network_graph.edges(data=True))
	flow_matrix=np.zeros((len(qubits),len(qubits)))
	for entry in MyData:
		flow_matrix[int(entry[0]), int(entry[1])]= int(entry[2])
		flow_matrix[int(entry[1]), int(entry[0])]= int(entry[2])
	options={"maximize":False}
	print(distance_matrix)
	print(distance_matrix.shape,flow_matrix.shape)
	res = quadratic_assignment( distance_matrix, flow_matrix, options=options)
	#print('res ', res['col_ind'], "val ", res.fun)
	parts={}
	print("QAP done")
	for i,qubit in enumerate(res['col_ind']):
		parts[qubit]=i
	#print("gates ", len(MyData))
	return parts, distance_matrix


def MinQAP3(qubits, MyData, distance_matrix):

	flow_matrix=np.zeros((len(qubits),len(qubits)))
	for entry in MyData:
		flow_matrix[int(entry[0]), int(entry[1])]= int(entry[2])
		flow_matrix[int(entry[1]), int(entry[0])]= int(entry[2])
	options={"maximize":False}
	#print(distance_matrix)
	#print(distance_matrix.shape,flow_matrix.shape)
	res = quadratic_assignment( distance_matrix, flow_matrix, options=options)
	#print('res ', res['col_ind'], "val ", res.fun)
	parts={}
	#print("QAP done")
	for i,qubit in enumerate(res['col_ind']):
		parts[qubit]=i
	#print("gates ", len(MyData))
	
	
	return parts


def connected_graph(graph: qGraph, min_channel_num, max_channel_num):
	"""Check if a graph is connected. IF not, make it a connected graph."""

	def dfs(src: int, adj_list, visited: list, components):
		components[-1].add(src)
		visited[src] = True
		for dst in adj_list[src]:
			if not visited[dst]:
				dfs(dst, adj_list, visited, components)

	id_to_pos = {node.id: idx for idx, node in enumerate(graph.V)}
	adj_list = [[] for _ in range(len(graph.V))]
	for e in graph.E:
		adj_list[id_to_pos[e.this.id]].append(id_to_pos[e.other.id])
		adj_list[id_to_pos[e.other.id]].append(id_to_pos[e.this.id])
	visited = [False] * len(graph.V)
	components = []

	for i in range(len(graph.V)):
		if not visited[i]:
			components.append(set())
			dfs(i, adj_list, visited, components)

	while len(components) > 1:
		min_dist = float('inf')
		selected_components = []
		selected_nodes = []
		for i in range(len(components) - 1):
			for j in range(i + 1, len(components)):
				for node_i_idx in components[i]:
					node_i = graph.V[node_i_idx]
					for node_j_idx in components[j]:
						node_j = graph.V[node_j_idx]
						if node_i.loc.distance(node_j.loc) < min_dist:
							min_dist = node_i.loc.distance(node_j.loc)
							selected_nodes = [node_i, node_j]
							selected_components = [i, j]
		graph.add_edge(qChannel(this=selected_nodes[0], other=selected_nodes[1],
								channels_num=randint(min_channel_num, max_channel_num)))
		merged_component = components[selected_components[0]].union(components[selected_components[1]])
		components = components[:selected_components[0]] + components[selected_components[0] + 1:selected_components[1]] \
					 + components[selected_components[1] + 1:]
		components.append(merged_component)
def create_random_graph(area_side: int, cell_size: int, number_nodes: int, min_memory_size: int, max_memory_size: int,
						min_node_degree: int, max_node_degree: int,
						min_channel_num: int, max_channel_num: int) -> qGraph:
	"""
	:param area_side: side size of a square area
	:param cell_size: size of each cell (square)
	:param number_nodes: number of nodes (qNodes) in the network
	:param min_memory_size: minimum number of memories for each node
	:param max_memory_size: maximum number of memories for each node
	:param min_node_degree: minimum number of edges for each node
	:param max_node_degree: maximum number of edges per each node
	:param min_channel_num: minimum number of channels for each edge
	:param max_channel_num: maximum number of channels for each edge
	:return:
	"""
	points = [Point(randint(0, area_side - 1) * cell_size, randint(0, area_side - 1) * cell_size)
			  for _ in range(number_nodes)]
	V = [qNode(randint(min_memory_size, max_memory_size), point) for point in points]
	E = []
	num_edges = [0] * len(V)  # number of edges for each node
	edges_set = set()  # a set to avoid parallel edges with key "u-v"
	for i, v in enumerate(V):
		num_v_edges = randint(min_node_degree, max_node_degree) - num_edges[i]  # number of edges to be created
		distance_to_v = [(v.loc.distance(V[k].loc), k) for k in range(i)] + \
						[(v.loc.distance(V[k].loc), k) for k in range(i + 1, len(V))]
		distance_to_v.sort(key=lambda x: x[0])  # sort based on distances
		neighbor_idx, edge_num = 0, 0
		while edge_num < num_v_edges and neighbor_idx < len(distance_to_v):
			k_id = distance_to_v[neighbor_idx][1]
			if "{u}-{v}".format(u=i, v=k_id) in edges_set:
				neighbor_idx += 1
				continue
			E.append(qChannel(v, V[k_id], randint(min_channel_num, max_channel_num)))  # v -> k
			num_edges[k_id] += 1
			edge_num += 1
			neighbor_idx += 1
			edges_set.add("{u}-{v}".format(u=i, v=k_id))
			edges_set.add("{u}-{v}".format(u=k_id, v=i))
		num_edges[i] += num_v_edges
	return qGraph(V=V, E=E)

def create_random_waxman(area_side: int, cell_size: int, number_nodes: int, min_memory_size: int, max_memory_size: int,
						 min_channel_num: int, max_channel_num: int, atomic_bsm_success_rate: float,
						 edge_density: float) -> qGraph:
	dist = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
	alpha_min, alpha_max, beta_min, beta_max = 0, 1, 0, 1
	while alpha_max - alpha_min > 0.02 and beta_max - beta_min > 0.02:
		alpha = (alpha_min + alpha_max) / 2
		beta = (beta_min + beta_max) / 2
		graphx = waxman_graph(number_nodes, domain=(0, 0, area_side * cell_size, area_side * cell_size), metric=dist,
							  beta=beta, alpha=alpha, L=10e3)
		if (edge_density - 0.002) * number_nodes ** 2 <= len(graphx.edges()) <= (
				edge_density + 0.002) * number_nodes ** 2:
			break
		elif len(graphx.edges()) > (edge_density - 0.002) * number_nodes ** 2:
			alpha_max = alpha
			beta_min = beta
		else:
			alpha_min = alpha
			beta_max = beta
	if atomic_bsm_success_rate is not None:
		V = [qNode(randint(min_memory_size, max_memory_size),
				   Point(int(graphx.nodes[i]["pos"][0]), int(graphx.nodes[i]["pos"][1])),
				   bsm_success_rate=atomic_bsm_success_rate) for i in range(number_nodes)]
	else:
		V = [qNode(randint(min_memory_size, max_memory_size),
				   Point(int(graphx.nodes[i]["pos"][0]), int(graphx.nodes[i]["pos"][1]))) for i in range(number_nodes)]
	E = []
	# if optical_bsm_rate is not None:
	#     for u, v in graphx.edges():
	#         E.append(qChannel(V[u], V[v], randint(min_channel_num, max_channel_num),
	#                           optical_bsm_rate=optical_bsm_rate))
	# else:
	for u, v in graphx.edges():
		E.append(qChannel(V[u], V[v], randint(min_channel_num, max_channel_num),
						  optical_bsm_rate=V[0].bsm_success_rate / 2))
	return qGraph(V=V, E=E)

def DAG_of_EPs(nG,qG, EP_list):
	n=len(EP_list)
	#APSP=dict(nx.all_pairs_shortest_path(nG))
	G=nx.DiGraph()
	G.add_nodes_from(range(n))
	for i in range(n):
		q1,q2=EP_list[i][0],EP_list[i][1]
		for j,p in enumerate(EP_list[i+1:]):
			p1,p2=p[0],p[1]
			if p1==q1 or p1==q2 or p2==q1 or p2==q2:
				G.add_edge(i,j+i+1)

	
	G=nx.transitive_reduction(G)
	#print(list(nx.topological_sort(G)))
	generations=[sorted(generation) for generation in nx.topological_generations(G)]
	"""
	for g, generation in enumerate(generations[:-1]):
		for i in generation:
			for j in generations[g+1]:
				q1,q2=EP_list[i]
				p1,p2=EP_list[j]
				if bool(set(APSP[q1][q2])&set(APSP[p1][p2])):
					gar,l1=Ent_Gen(qG, [(q1,q2)])
					gar,l2=Ent_Gen(qG, [p])
					if l1<=l2:
						if j in nx.ancestors(G,i):
							continue
						G.add_edge(i,j)
					else:
						if i in nx.ancestors(G,j):
							#print("here?")
							continue
						G.add_edge(j,i)
	"""
	#print([[EP_list[i] for i in sorted(generation)] for generation in nx.topological_generations(G)])
	return flatten([[EP_list[i] for i in sorted(generation)] for generation in nx.topological_generations(G)]),G

def Greedy_DP_Ent_Gen(qg, EP_list, Dec_time, dist_matrix, parts,DAG2, Bin_gates):
	n=len(EP_list)
	start_time=datetime.datetime.now()
	G=nx.DiGraph()
	G.add_nodes_from(range(n))
	for i in range(n):
		q1,q2,gar=EP_list[i]
		for j,p in enumerate(EP_list[i+1:]):
			p1,p2,gar=p
			if p1==q1 or p1==q2 or p2==q1 or p2==q2:
				G.add_edge(i,j+i+1)	
	G=nx.transitive_reduction(G)
	#print(EP_list)
	layers=[[EP_list[i] for i in sorted(generation)] for generation in nx.topological_generations(G)]
	#print(layers)
	Layers_of_batches=[]
	for layer in layers:
		#print("layer ")
		greedy_sol,greedy_time, garb= Greedy_Ent_Gen_Dec_par( qg, layer, Dec_time, dist_matrix, parts, DAG2, Bin_gates, True)
		Layers_of_batches.append(greedy_sol)
	new_layers={}
	for i, layer in enumerate(Layers_of_batches[:-1]):
		flag=0
		best_left=0
		best_right=0
		for batch1 in layer:
			for batch2 in Layers_of_batches[i+1]:
				gar,time=Ent_Gen(qG,[(EP_list[i][0],EP_list[i][1]) for i in list(batch1)+list(batch2)],True)
				if time<Dec_time:
					flag=1
					best_left=batch1
					best_right=batch2
					break
			if flag==1:
				break
		if flag==1:
			new_layers[i]=[best_left.union(best_right)]
			layer.remove(best_left)
			Layers_of_batches[i+1].remove(best_right)
	for key, layer in new_layers.items():
		Layers_of_batches.insert(key, layer)
	List_of_batches=[]
	for layer in Layers_of_batches:
		for batch in layer:
			List_of_batches.append(list(batch))
	group_list_of_batches=[]
	prev_batch=List_of_batches[0]
	for batch in List_of_batches[1:]:
		gar,time=Ent_Gen(qG,[(EP_list[i][0],EP_list[i][1]) for i in prev_batch+batch],True)
		if time<Dec_time:
			prev_batch+=batch
		else:
			group_list_of_batches.append(prev_batch)
			prev_batch=batch
	group_list_of_batches.append(prev_batch)
	actual_latency=0.0
	last_index=0
	prev_extra_time=0
	for batch in group_list_of_batches:
		#time=Iterative_Ent_Gen_NoDec(qg,[EP_list[i] for i in batch],False)
		#print([EP_list[i] for i in batch])
		time1=0
		time=0
		try:
			gar,time=Ent_Gen(qG,[(EP_list[i][0],EP_list[i][1]) for i in batch],True)
		except:
			time=Iterative_Ent_Gen_NoDec(qg,[(EP_list[i][0],EP_list[i][1]) for i in batch],False)
		print(time)

		actual_latency+=time
	print("Actual latency ", actual_latency)
	print("Total time ", datetime.datetime.now() - start_time)
	return group_list_of_batches,actual_latency,datetime.datetime.now() - start_time 	
		

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
	
def Greedy_Ent_Gen_Dec_par( qg, EP_list, Dec_time, dist_matrix, parts, DAG2, Bin_gates, easymode=False):
	n=len(EP_list)
	start_time=datetime.datetime.now()
	G=nx.DiGraph()
	G.add_nodes_from(range(n))
	for i in range(n):
		q1,q2,gar=EP_list[i]
		for j,p in enumerate(EP_list[i+1:]):
			p1,p2,gar=p
			if p1==q1 or p1==q2 or p2==q1 or p2==q2:
				G.add_edge(i,j+i+1)	
	#G=nx.transitive_reduction(G)
	print("created EP dag " , datetime.datetime.now() - start_time)
	EP_set=set(range(n))
	total_latency=0.0
	Keep_track_of_EPs=[]
	done=set()
	#print(EP_list)
	while EP_set:
		ggar,full_latency=Ent_Gen(qg,[EP_list[i][:-1] for i in EP_set],True)
		#print("full latency ", full_latency)
		if full_latency<=Dec_time:
			#print("All good ", EP_set)
			total_latency+=full_latency
			Keep_track_of_EPs.append(EP_set)
			break
		temp_EP_set=EP_set.copy()
		curr_latency=np.inf
		#print("Entering second while ")
		while curr_latency>=Dec_time:
			#print("in second while")
			min_benefit=np.inf
			best_set={next(iter((temp_EP_set)))}
			#print("entering for")
			for EP_bunch in grouper(temp_EP_set,5, 0.0):
				for EP in EP_bunch:
					des=set(nx.descendants(G, EP) | {EP})
					temp_temp_EP_set=temp_EP_set-des
				if not temp_temp_EP_set:
					continue
				#print("number of EPs ", len(temp_temp_EP_set))
				gar,latency=Ent_Gen(qg,[EP_list[i][:-1] for i in temp_temp_EP_set],True)
				#print("latency ", latency)
				if min_benefit>(latency/len(temp_temp_EP_set)):
					min_benefit=latency/len(temp_temp_EP_set)
					best_set=temp_EP_set-des
				if latency<=Dec_time:
					break
			#print("best benefit ", min_benefit)
			#print("best set ", best_set)
			curr_latency=min_benefit*len(best_set)
			temp_EP_set=best_set
			#print("current latency ", curr_latency)
		
		total_latency+=curr_latency
		EP_set-=temp_EP_set
		Keep_track_of_EPs.append(temp_EP_set)
	#print("EP batches ", Keep_track_of_EPs)
	#print("Total expected latency Greedy ", total_latency)
	if easymode:
		return Keep_track_of_EPs, total_latency, 0
	actual_latency=0.0
	last_index=0
	prev_extra_time=0
	for batch in Keep_track_of_EPs:
		time1=0
		time=0
		try:
			gar,time1=Ent_Gen(qg,[(EP_list[i][0],EP_list[i][1]) for i in batch], True)
		except:
			time=Iterative_Ent_Gen_NoDec(qg,[(EP_list[i][0],EP_list[i][1]) for i in batch],False)
		actual_latency+=time+time1
	print("Actual latency ", actual_latency)
	print("Total time ", datetime.datetime.now() - start_time)
	return Keep_track_of_EPs,actual_latency,datetime.datetime.now() - start_time

def disjoint_paths(layer, ng):
	ng_copy=ng.copy()
	solution=[]
	for s,t in layer:
		if not nx.has_path(ng_copy, s,t):
			break
		path=nx.shortest_path(ng_copy,s,t)
		path_edges=[(v,path[i+1])for i,v in enumerate(path[:-1])]
		
		solution.append((s,t))
		ng_copy.remove_edges_from(path_edges)
			
	return solution

def disjoint_paths2(layer, qg, dec):
	graph_dp=qg.clone()
	solution_cost=0.0
	solution=[]
	ng=qGraph_to_Networkx(graph_dp, True)
	#print(ng.edges(data=True))
	for s,t in layer:
		if not nx.is_connected(qGraph_to_Networkx(graph_dp)):
			break
		naive_shortest = hop_shortest_path(graph_dp, dec)
		naive_shortest_path = naive_shortest.get_shortest_path(graph_dp.V[s], graph_dp.V[t])
		if not naive_shortest_path:
			break
		latency= naive_shortest_path.avr_ent_time
		path= hop_shortest_path._bfs(adj_lst=naive_shortest._adj_lst, src=s, dst=t)
		path_edges=[(v,path[i+1])for i,v in enumerate(path[:-1])]
		#for edge in path_edges:
		#	graph_dp.del_edge(edge[0],edge[1])
		if solution_cost<=latency:
			solution_cost=latency
		solution.append((s,t))
	del graph_dp, naive_shortest
	return solution, solution_cost
	
def disjoint_paths3(layer, qg, dec, interval_time,capacity_per_edge):
	graph_dp=qg.clone()
	solution_cost=0.0
	solution=[]
	ng=qGraph_to_Networkx(graph_dp, True)
	#print(ng.edges(data=True))
	#print(capacity_per_edge)
	index=0
	trees=[]
	srcs=[]
	dsts=[]
	for s,t in layer:
		if not nx.is_connected(qGraph_to_Networkx(graph_dp)):
			break
		naive_shortest = hop_shortest_path(graph_dp, dec)
		naive_shortest_path = naive_shortest.get_shortest_path(graph_dp.V[s], graph_dp.V[t])
		trees.append(naive_shortest_path)
		srcs.append(graph_dp.V[s])
		dsts.append(graph_dp.V[t])
		if not naive_shortest_path:
			break
		latency= naive_shortest_path.avr_ent_time
		path= hop_shortest_path._bfs(adj_lst=naive_shortest._adj_lst, src=s, dst=t)
		path_edges=[tuple(sorted((v,path[i+1]))) for i,v in enumerate(path[:-1])]
		for edge in path_edges:
			capacity_per_edge[edge]-=1
			if capacity_per_edge[edge]<=0:
				graph_dp.del_edge(edge[0],edge[1])
		if solution_cost<=latency:
			solution_cost=latency
		solution.append((s,t))
		index+=1
	not_solution=layer[index:]
	new_latency=0.0
	del graph_dp, naive_shortest
	return trees, srcs, dsts, solution, not_solution, solution_cost
		

def Caleffi(layers, EP_list,ng, qg, dec):
	total_latency=0.0
	start_time=datetime.datetime.now()
	#layers=[[EP_list[i] for i in sorted(generation)] for generation in nx.topological_generations(DAG)]
	print("num layers ", len(layers))
	#print("layers ", layers)
	l=0
	Set_collections=[]
	diameter = nx.diameter(ng)
	for node in sorted(ng.nodes()):
		start_end_nodes = sorted([(node, k) for k,v in nx.shortest_path_length(ng, node).items() if v == diameter])
		if start_end_nodes:
			s,t=start_end_nodes[0]
			break
	naive_shortest = hop_shortest_path(qg, dec)
	naive_shortest_path = naive_shortest.get_shortest_path(qg.V[s], qg.V[t])
	interval_time=naive_shortest_path.avr_ent_time
	capacity_per_edge={}
	for edge in qg.E:
		#print("capacity ", edge.max_channel_capacity)
		capacity_per_edge[tuple(sorted([edge.this.id,edge.other.id]))]=edge.max_channel_capacity*interval_time
	#print("interval time ", interval_time)
	Trees=[]
	Srcs=[]
	Dsts=[]
	for layer in layers:
		set_layer=layer[:]
		#print("layer ", l)
		while set_layer:
			#print("dis paths enter ", set_layer)
			trees,srcs,dsts,S_bar, remains,latency=disjoint_paths3(set_layer, qg, max(dec,0.05), interval_time, capacity_per_edge.copy())
			Set_collections.append(S_bar)
			Trees+=trees
			Srcs+=srcs
			Dsts+=dsts
			print("dis_paths out ", S_bar)
			#gar,latency=Ent_Gen(qG,S_bar,False,True)
			#latency=Iterative_Ent_Gen_NoDec(qg, S_bar, True)
			total_latency+=latency
			set_layer=remains[:]
		l+=1
	try:
		print("stuck in protocol Caleffi")
		ours_protocol = TreeProtocol(graph=graph_dp, trees=Trees,sources=Srcs, destinations=Dsts,duration=3, IS_TELEPORTING=False)
		new_latency=0.0
		for tree_id, rate in enumerate(ours_protocol.success_rate):
			new_latency+=1.0/rate
		total_latency=new_latency
		del ours_protocol
	except:
		pass
	print("Caleffi latency ", total_latency)
	print("Caleffi sets ", Set_collections, len(Set_collections), l)
	
	return Set_collections,total_latency,datetime.datetime.now() - start_time

def qGraph_to_Networkx(qG: qGraph, withCaps=False):
	G = nx.Graph()
	nodes = {}
	edges = []
	edges2width = {}
	widths = []
	for node in qG.V:
		nodes[node.id] = node.loc.x, node.loc.y
	if withCaps:
		for qchannel in qG.E:
			u, v = qchannel.this.id, qchannel.other.id
			edges.append((min(u,v), max(u,v), optimal_linear_programming._edge_capacity(qchannel)))
			edges2width[(min(u,v), max(u,v))] = qchannel.channels_num
		G.add_weighted_edges_from(edges)
		
	else:
		for qchannel in qG.E:
			u, v = qchannel.this.id, qchannel.other.id
			edges.append((min(u,v), max(u,v)))
			edges2width[(min(u,v), max(u,v))] = qchannel.channels_num
		G.add_edges_from(edges)
	return G


class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
	def signal_handler(signum, frame):
		raise TimeoutException("Timed out!")
	signal.signal(signal.SIGALRM, signal_handler)
	signal.setitimer(signal.ITIMER_REAL,seconds)
	try:
		yield
	finally:
		signal.alarm(0)


def Ent_Gen(graph0: qGraph, EP_list,NoLP=True,Sim=False ):
	srcs=[]
	dsts=[]
	if Sim == True:
		NoLP=True
	graph_lp=graph0.clone()
	#print("EP format ", EP_list[0])
	Ent_Counter=collections.Counter(EP_list)
	EP_set=set(EP_list)
	teleportation_bsm_rate = 1.0
	Condensed_EP_list=list(EP_set)
	for src,dst in Condensed_EP_list:
		srcs.append(src)
		dsts.append(dst)
	start_time = datetime.datetime.now()
	lp_srcs = [graph_lp.V[src] for src in srcs]
	lp_dsts = [graph_lp.V[dst] for dst in dsts]
	rate_per_EP=defaultdict(int)
	finished=set()
	total_time=0.0
	#print("timed out ")
	while EP_set:
		#print("EPs left ", len(EP_set))
		Condensed_EP_list=list(EP_set)
		lp_srcs = []
		lp_dsts = []
		for src,dst in Condensed_EP_list:
			lp_srcs.append(graph_lp.V[src])
			lp_dsts.append(graph_lp.V[dst])
		max_flow_ours = max_flow(graph_lp, lp_srcs, lp_dsts,
										decoherence_time=1000
										, multiple_pairs_search='fair')
		for pair_id in range(len(lp_dsts)):
			pair_res = max_flow_ours.get_flows(pair_id)
			for _, tree, _ in pair_res:
				#print("tree time ", pair_id, tree.avr_ent_time)
				rate_per_EP[(lp_srcs[pair_id].id,lp_dsts[pair_id].id)]+= teleportation_bsm_rate/tree.avr_ent_time
				finished.add((lp_srcs[pair_id].id,lp_dsts[pair_id].id))
		EP_set-=finished					
		del max_flow_ours
		del graph_lp
		graph_lp=graph0.clone()
		total_time+=max([(1.0/v) * Ent_Counter[k] for k, v in rate_per_EP.items()])
		rate_per_EP=defaultdict(int)
		pass
	value=sum([(1.0/v) * Ent_Counter[k] for k, v in rate_per_EP.items()])
	
	return rate_per_EP, total_time

def Ent_Gen_old(graph0: qGraph, EP_list,NoLP=True,Sim=False ):
	srcs=[]
	dsts=[]
	if Sim == True:
		NoLP=True
	graph_lp=graph0.clone()
	Ent_Counter=collections.Counter(EP_list)
	Condensed_EP_list=list(set(EP_list))
	teleportation_bsm_rate = 1.0
	for src,dst in Condensed_EP_list:
		srcs.append(src)
		dsts.append(dst)
	EP_set=set(EP_list)
	#print("\n\n-->LP")
	#print(len(srcs))
	#G=qGraph_to_Networkx(graph_lp)
	#print(len(G.nodes()), len(G.edges()))
	start_time = datetime.datetime.now()
	lp_srcs = [graph_lp.V[src] for src in srcs]
	lp_dsts = [graph_lp.V[dst] for dst in dsts]
	rate_per_EP=defaultdict(int)
	NoLP=True
	Sim=False
	try:
		if NoLP:
			raise TimeoutException()
		with time_limit(0.05):
			lp = optimal_linear_programming(graph_lp, lp_srcs, lp_dsts)
			#alg_result = (teleportation_bsm_rate ) * lp.max_flow
		#print("Algorithm (LP) results")
		#print("Total flow algorithm (LP): {:.3f} q/s\ntime:".format(alg_result),datetime.datetime.now() - start_time)
		for pair_id, trees in enumerate(lp.trees):
			for tree in trees:
				#print(f"Tree{tree_num} (rate:"f" {(teleportation_bsm_rate) / tree.avr_ent_time} EPs/s), "f"between nodes {lp_srcs[pair_id].id} (loc: {lp_srcs[pair_id].loc}) and "f"{lp_dsts[pair_id].id} (loc: {lp_dsts[pair_id].loc})")
				#tree_node.print_tree(tree)
				rate_per_EP[(lp_srcs[pair_id].id,lp_dsts[pair_id].id)]+= teleportation_bsm_rate/tree.avr_ent_time 
		
		del lp
	except TimeoutException as e:
		#print("timed out ")
		max_flow_ours = max_flow(graph_lp, lp_srcs, lp_dsts,
									 decoherence_time=1000
									 , multiple_pairs_search='fair')
		
		#alg_result = (teleportation_bsm_rate) * max_flow_ours.get_flow_total
		#print("Algorithm (max-flow) results")
		#print("Total flow algorithm (max-flow): {:.2f} q/s\ntime:".format(alg_result),datetime.datetime.now() - start_time)
		if Sim:
			print("in Sim")
			ours_all_trees = []
			ours_protocol_all_srcs, ours_protocol_all_dsts = [], []
			tree_num = 0
			map_tree_to_pairs={}
			for pair_id in range(len(lp_dsts)):
				pair_res = max_flow_ours.get_flows(pair_id)
				for _, tree, _ in pair_res:
					ours_all_trees.append(tree)
					ours_protocol_all_srcs.append(lp_srcs[pair_id])
					ours_protocol_all_dsts.append(lp_dsts[pair_id])
					map_tree_to_pairs[tree_num]=pair_id
					tree_num+=1
			print("stuck in protocol")
			ours_protocol = TreeProtocol(graph=graph_lp, trees=ours_all_trees,sources=ours_protocol_all_srcs, destinations=ours_protocol_all_dsts,duration=3, IS_TELEPORTING=False)
			for tree_id, rate in enumerate(ours_protocol.success_rate):
				pair_id=map_tree_to_pairs[tree_id]
				rate_per_EP[(lp_srcs[pair_id].id,lp_dsts[pair_id].id)]+= rate
			print("done")
			del ours_protocol
		else:
			for pair_id in range(len(lp_dsts)):
				pair_res = max_flow_ours.get_flows(pair_id)
				for _, tree, _ in pair_res:
					#print("tree time ", pair_id, tree.avr_ent_time)
					rate_per_EP[(lp_srcs[pair_id].id,lp_dsts[pair_id].id)]+= teleportation_bsm_rate/tree.avr_ent_time
			
			
		
		del max_flow_ours
		del graph_lp
		pass
	value=sum([(1.0/v) * Ent_Counter[k] for k, v in rate_per_EP.items()])
	
	return rate_per_EP, value

def Ent_Gen2(graph0: qGraph, EP_list,NoLP=False ):
	srcs=[]
	dsts=[]
	graph_lp=graph0.clone()
	Ent_Counter=collections.Counter(EP_list)
	Condensed_EP_list=list(set(EP_list))
	teleportation_bsm_rate = 1.0
	for src,dst in Condensed_EP_list:
		srcs.append(src)
		dsts.append(dst)
	#print("\n\n-->LP")
	#print(len(srcs))
	#G=qGraph_to_Networkx(graph_lp)
	#print(len(G.nodes()), len(G.edges()))
	start_time = datetime.datetime.now()
	lp_srcs = [graph_lp.V[src] for src in srcs]
	lp_dsts = [graph_lp.V[dst] for dst in dsts]
	rate_per_EP=defaultdict(int)
	try:
		if NoLP:
			raise TimeoutException()
		with time_limit(0.05):
			lp = optimal_linear_programming(graph_lp, lp_srcs, lp_dsts)
			#alg_result = (teleportation_bsm_rate ) * lp.max_flow
		#print("Algorithm (LP) results")
		#print("Total flow algorithm (LP): {:.3f} q/s\ntime:".format(alg_result),datetime.datetime.now() - start_time)
		for pair_id, trees in enumerate(lp.trees):
			for tree in trees:
				#print(f"Tree{tree_num} (rate:"f" {(teleportation_bsm_rate) / tree.avr_ent_time} EPs/s), "f"between nodes {lp_srcs[pair_id].id} (loc: {lp_srcs[pair_id].loc}) and "f"{lp_dsts[pair_id].id} (loc: {lp_dsts[pair_id].loc})")
				#tree_node.print_tree(tree)
				rate_per_EP[(lp_srcs[pair_id].id,lp_dsts[pair_id].id)]+= teleportation_bsm_rate/tree.avr_ent_time 
		
		del lp
	except TimeoutException as e:
		#print("timed out ")
		max_flow_ours = max_flow(graph_lp, lp_srcs, lp_dsts,
									 decoherence_time=100
									 , multiple_pairs_search='best_paths')
		
		#alg_result = (teleportation_bsm_rate) * max_flow_ours.get_flow_total
		#print("Algorithm (max-flow) results")
		#print("Total flow algorithm (max-flow): {:.2f} q/s\ntime:".format(alg_result),datetime.datetime.now() - start_time)
		for pair_id in range(len(lp_dsts)):
			pair_res = max_flow_ours.get_flows(pair_id)
			for _, tree, _ in pair_res:
				rate_per_EP[(lp_srcs[pair_id].id,lp_dsts[pair_id].id)]+= teleportation_bsm_rate/tree.avr_ent_time 
		del max_flow_ours
		pass

		
	return rate_per_EP, sum([(1.0/v) * Ent_Counter[k] for k, v in rate_per_EP.items()])
def flatten(l):
	return [item for sublist in l for item in sublist]
	
def Complete_value( dist_matrix, parts,EP_gate_map, DAG2, min_index, max_index):
	"""
	index1=EP_gate_map[min_index]
	index2=EP_gate_map[max_index]
	Bin_gates=DAG2[index1:index2]
	extra_cost=0.0
	for q1,q2 in Bin_gates:
		cost=dist_matrix[parts[int(q1)],parts[int(q2)]]
		if cost<10.0:
			extra_cost+=cost
	"""
	return 0
	

def DP_Ent_Gen_Dec_par(qg:qGraph, Par_order, Dec_time, dist_matrix, parts, DAG2):
	start_time = datetime.datetime.now()
	last=len(Par_order)
	S=[np.inf for i in range(last+1)]
	V=[[] for i in range(last+1)]
	for index in range(0,last,20):
		print(index)
		if index==0:
			continue
		#print("EP list ",[(item[0],item[1]) for item in Par_order[0:index]])
		garb,S[index]=Ent_Gen(qg, [(item[0],item[1]) for item in Par_order[0:index]] ,True)
		#S[0,index]=Iterative_Ent_Gen_NoDec(qg, Par_order[0:index],True)+ Complete_value(dist_matrix, parts, EP_gate_map, DAG2,0, index)
		print(S[index])
		#print(Complete_value(dist_matrix, parts, EP_gate_map, DAG2,0, index))
		if S[index]>Dec_time:
			#print("too much 1")
			S[index]=np.inf
		V[index].append((0,index))
	S[0]=0.0
	print("initialized")
	for index2 in range(0,last,20):
		if index2==0:
			continue
		min_cost=np.inf
		best_index=0
		V[0]=[]
		for index1 in range(0,index2,20):
			print(index1,index2, S[index1], S[index2])
			garb,ij_cost=Ent_Gen(qg,[(item[0],item[1]) for item in Par_order[index1:index2]],True)
			if ij_cost>Dec_time:
				print("too much 2 ", index1, index2)
				S[index2]=min_cost
				V[index2]=V[index1]
				V[index2].append((index1,index2))
				continue

			new_cost=S[index1]+ij_cost
			if min_cost>new_cost:
				#print("getting updated ", new_cost)
				min_cost=new_cost
				best_index=index1
		print("best index ", best_index)
		print("interals ", V[best_index])
		S[index2]=min_cost
		V[index2]=V[best_index][:]
		V[index2].append((best_index,index2))
	print("end ", last-last%20)
	BestValue=S[last-last%20]
	BestSequence=V[last-last%20]+[(last-last%20,last-1)]
	print("number of EPs ", last, " dec time ", Dec_time)
	print(BestValue)
	print(BestSequence)
	actual_latency=0
	batches=[]
	for i,j in BestSequence:
		gar,time=Ent_Gen(qg,[(item[0],item[1]) for item in Par_order[i:j]],True)
		#print(time)
		batches.append([(item[0],item[1]) for item in Par_order[i:j]])
		actual_latency+=time
	print("Actual latency ", actual_latency)
	print("Total time ", datetime.datetime.now() - start_time)
	#print(Ent_Gen(qg,EP_list))
	return batches,actual_latency

def DP_Ent_Gen_Dec_Tot(qg:qGraph, EP_list, Dec_time):
	start_time = datetime.datetime.now()
	last=len(EP_list)
	S=np.ones((last+1,last+1))*np.inf
	V=[[[] for i in range(last+1)] for i in range(last+1)]
	for index in range(last):
		if index==0:
			continue
		garb,S[0,index]=Ent_Gen(qg, EP_list[0:index])
		#S[0,index]=Iterative_Ent_Gen_NoDec(qg, EP_list[0:index])
		print(S[0,index])
		if S[0,index]>Dec_time:
			print("too much 1")
			S[0,index]=np.inf
		V[0][index].append((0,index))
	for index1 in range(last):
		if index1==0:
			continue
		for index2 in range(index1+1,last):
			min_cost=np.inf
			best_index=0
			print(index1,index2, S[index,index1])
			#S[index1,index2]=Iterative_Ent_Gen_NoDec(qg, EP_list[index1:index2])
			#V[index1][index2].append((index1,index2))
			garb,ij_cost=Ent_Gen(qg, EP_list[index1:index2])
			#ij_cost=Iterative_Ent_Gen_NoDec(qg, EP_list[index1:index2])
			#print("set latency ", ij_cost)
			if ij_cost>Dec_time:
				print("too much 2 ", index1, index2)
			for index in range(0,index1):
				if S[index,index1]==np.inf:
					#print("continuing..")
					continue
				new_cost=S[index,index1]+ij_cost
				if min_cost>new_cost:
					#print("getting updated ", new_cost)
					min_cost=new_cost
					best_index=index
			#print("best index ", best_index)
			S[index1,index2]=min_cost
			V[index1][index2]+=V[best_index][index1]
			V[index1][index2].append((index1,index2))
	BestValue=math.inf
	BestSequence=[]
	print("number of EPs ", last, " dec time ", Dec_time)
	for entry in range(last):
		if BestValue>S[entry, last-1]:
			BestValue=S[entry, last-1]
			BestSequence=V[entry][last-1]	
	print(BestValue)
	print(BestSequence)
	actual_latency=0
	for i,j in BestSequence:
		time=Iterative_Ent_Gen_NoDec(qG, EP_list[i,j])
		actual_latency+=time
	print("Actual latency ", actual_latency)
	print("Total running time ", datetime.datetime.now() - start_time)
	#print(Ent_Gen(qg,EP_list))
	return

def Iterative_Ent_Gen_NoDec(qG:qGraph,EP_list, NoLP=False):
	Ent_Counter=collections.Counter(EP_list)
	Condensed_EP_list=list(set(EP_list))
	frac_to_next_EP={k:1.0 for k in Condensed_EP_list}
	total_latency=0.0
	#print("Ent counter ", Ent_Counter)
	prev_Condensed_EP_list=Condensed_EP_list[:]
	EP_rate_map,gar=Ent_Gen(qG,Condensed_EP_list, NoLP)
	Expected_finish_times={}
	while any(Ent_Counter.values()):
		#print("latency ", total_latency)
		#print("counter ", Ent_Counter)
		for key in Condensed_EP_list:
			if EP_rate_map[key]==0:
				Expected_finish_times[key]=np.inf
				continue
			Expected_finish_times[key]=frac_to_next_EP[key]/EP_rate_map[key]
		min_finish_time=min(Expected_finish_times.values())
		#print(min_finish_time)
		for key,value in frac_to_next_EP.items():
			if value==np.inf:
				continue
			#print(frac_to_next_EP[key], Expected_finish_times[key])
			new_value=frac_to_next_EP[key]-(min_finish_time/Expected_finish_times[key])
			#print(new_value)
			if new_value<=0:
				Ent_Counter[key]-=1
				if Ent_Counter[key]<=0:
					Condensed_EP_list.remove(key)
					Ent_Counter.pop(key,None)
					frac_to_next_EP[key]=np.inf
					Expected_finish_times.pop(key,None)
				else:
					frac_to_next_EP[key]=1.0
			else:
				frac_to_next_EP[key]=new_value
		total_latency+=min_finish_time
		if not Condensed_EP_list:
			#print("No EPs, Counter ", Ent_Counter)
			break
		if prev_Condensed_EP_list!=Condensed_EP_list:
			EP_rate_map,garb=Ent_Gen(qG,Condensed_EP_list, NoLP)
		prev_Condensed_EP_list=Condensed_EP_list[:]
	#print("Overall latency ", total_latency)
	if total_latency==0.0:
		return 0.001
	return total_latency

def find_ge(a, low, high):
	return [i for i in a if i >= low and i < high]  

def Compute_Subcircuit_Cost(qg:qGraph , Bin_gates,  node_map,parts, Dec_time , dist_matrix, easymode=True):
	EP_list=[]
	#qg=qg1.clone()
	#Bin_gates=[sorted(gate) for gate in gates if len(gate)==2]
	#print("parts ", parts)
	#print(node_map)
	#print("binary gates ", Bin_gates)
	for i,(q1,q2) in enumerate(Bin_gates):
		if dist_matrix[parts[int(q1)],parts[int(q2)]]>=10.0:
			EP_list.append(tuple(sorted([node_map[parts[int(q1)]],node_map[parts[int(q2)]]])+[i]))
	#print("EP list ", EP_list)
	if EP_list==[]:
		return 0
	if easymode:
		garb,value=Ent_Gen(qg,[(item[0],item[1]) for item in EP_list],True)
		#garb,value,garb2=Greedy_Ent_Gen_Dec_par(qg, EP_list, Dec_time, dist_matrix, parts,None, Bin_gates, True) 
	else:
		garb,value,garb2=Greedy_DP_Ent_Gen(qg, EP_list, Dec_time, dist_matrix, parts,None, Bin_gates) 
	#print("value latency ", value)
	return value


def StitchAdjacent(partition1,partition2, qubits, index, comm_cost, node_map):
	B = nx.Graph()
	EP_list=[]
	#print("parts1 ", partition)
	numparts=len(set(node_map.values()))
	#print("parts2 ", partition2)
	parts1=[[] for i in range(numparts)]
	parts2=[[] for i in range(numparts)]
	for i in range(numparts):
		parts1[node_map[partition1[i]]].append(i)
		parts2[node_map[partition2[i]]].append(i)

	#print("parts1 ", parts1)
	#print("parts2 ", parts2)
	partition={}
	for qubit in qubits:
		flag=0
		for i in range(0, numparts):
			if int(qubit) in parts1[i]:
				flag=1
				partition[int(qubit)]=i
		if flag==0:
			parts1[0].append(int(qubit))
			partition[int(qubit)]=0
	#print("parts1 ", parts1)
	partitionr={}
	for qubit in qubits:
		flag=0
		for i in range(0, numparts):
			if int(qubit) in parts2[i]:
				flag=1
				partitionr[int(qubit)]=i
		if flag==0:
			parts2[0].append(int(qubit))
			partitionr[int(qubit)]=0
	for qubit in qubits:
		if partition[int(qubit)]!=partitionr[int(qubit)]:
			EP_list.append(tuple(sorted([partition[int(qubit)],partitionr[int(qubit)]])+[0]))
			EP_list.append(tuple(sorted([partition[int(qubit)],partitionr[int(qubit)]])+[0]))
			
	"""
	B.add_nodes_from(range(0,len(parts1)), bipartite=0)
	B.add_nodes_from([str(ind) for ind in range(0,len(parts2))], bipartite=1)
	
	for i, part1 in enumerate(parts1):
		for j,part2 in enumerate(parts2):
			if part1==part2:
				B.add_edge(i, str(j), weight = 0)
			else:
				B.add_edge(i, str(j), weight = -1)
	#print("matching ", nx.max_weight_matching(B, maxcardinality=True))
	matching=list(nx.max_weight_matching(B, maxcardinality=True))
	
	for i, entry in enumerate(matching):
		if isinstance(entry[0], str):
			matching[i]=(matching[i][1],matching[i][0])
	temp_right_parts=[[] for i in range(numparts)]
	Perm_map={}
	for left,right in matching:
		right=int(right)
		left=int(left)
		right_part=parts2[right]
		temp_right_parts[left]=right_part
		Perm_map[left]=right
	unseen=set(range(numparts))
	temp_left_parts=parts1[:]
	while unseen:
		NextIndex=unseen.pop()
		while temp_left_parts[NextIndex]!=temp_right_parts[NextIndex]:
			MovingQubit=-1
			for q in temp_right_parts[NextIndex]:
				if q not in temp_left_parts[NextIndex]:
					MovingQubit=q
					break
			if MovingQubit==-1:
				#print("Problem!! ", NextIndex, temp_right_parts, temp_left_parts)
				break
			#else:
			#	print("Found a qubit")
			#print(MovingQubit, partition)
			EP_list.append(tuple(sorted([partition[int(MovingQubit)],NextIndex])+[0]))
			#teleportations.append((MovingQubit, Perm_map[NextIndex], index, comm_cost[partition[MovingQubit], NextIndex]))
			temp_left_parts[NextIndex].append(MovingQubit)
			temp_left_parts[partition[MovingQubit]].remove(MovingQubit)
			temp=partition[MovingQubit]
			partition[MovingQubit]=NextIndex
			NextIndex=temp
	
	"""
	
	return EP_list
	

def DP_teleportations(qg: qGraph, dist_matrix, CZgates, qubits, gates, node_map, Dec_time, last):
	S=np.ones((last+1,last+1))*np.inf
	V=[[[] for i in range(last+1)] for i in range(last+1)]
	Bin_gates=[sorted(gate) for gate in gates if len(gate)==2]
	SubCircuitParts=[[[] for i in range(last+1)] for i in range(last+1)]
	SubCircuitValues=np.ones((last+1,last+1))*np.inf
	print("hi ", len(CZgates))
	for entry in CZgates[1:]:
		if entry ==0:
			continue
		#print(entry)
		Mydata=Circuit_to_Graph(qubits, gates[0:entry])
		#print(MinQAP3(qubits, Mydata, dist_matrix))
		SubCircuitParts[0][entry]=MinQAP3(qubits, Mydata, dist_matrix)
		SubCircuitValues[0,entry]=Compute_Subcircuit_Cost(qg, gates[0:entry], node_map, SubCircuitParts[0][entry], Dec_time, dist_matrix)
		S[0,entry]=SubCircuitValues[0,entry]
		V[0][entry].append((0,entry))
	
	for l,index1 in enumerate(CZgates):
		print(index1)
		if index1==0:
			continue			
		for index2 in CZgates[l+1:]:
			Mydata=Circuit_to_Graph(qubits, gates[index1:index2])
			SubCircuitParts[index1][index2]=MinQAP3(qubits, Mydata, dist_matrix)
			#print(SubCircuitParts[index1][index2])
			SubCircuitValues[index1,index2]=Compute_Subcircuit_Cost(qg, gates[index1:index2], node_map, SubCircuitParts[index1][index2], Dec_time, dist_matrix)
	#print("computed best parts for each")
	for l,index1 in enumerate(CZgates):
		if index1==0:
			continue			
		for index2 in CZgates[l+1:]:
			
			best_index=0
			min_cost=np.inf

			for index in find_ge(CZgates, 0, index1):
				print(index, index1, index2)
				if S[index,index1]==np.inf:
					continue
				#print("stitching parts ,", SubCircuitParts[index][index1],SubCircuitParts[index1][index2] )
				EP_list=StitchAdjacent(SubCircuitParts[index][index1],SubCircuitParts[index1][index2], qubits, 0, dist_matrix, node_map)
				#print("stitching EPs ", EP_list)
				garb,stitch_cost=Ent_Gen(qg,[(item[0],item[1]) for item in EP_list],True)
				#garb,stitch_cost, garb2=Greedy_Ent_Gen_Dec_par(qg, EP_list, Dec_time, dist_matrix, SubCircuitParts[index][index1], None, Bin_gates, True)	
				#stitch_cost=0
				new_cost=stitch_cost+S[index,index1]+SubCircuitValues[index1,index2]+SubCircuitValues[index,index1]
				if min_cost>new_cost:
					best_index=index
					min_cost=new_cost

			#print("best index for ",(index1,index2), " ", index,  " " , min_cost )
			S[index1,index2]=min_cost
			V[index1][index2]+=V[best_index][index1]
			V[index1][index2].append((index1,index2))
	BestValue=math.inf
	BestSequence=[]
	for entry in CZgates:
		if BestValue>S[entry, last]:
			BestValue=S[entry, last]
			BestSequence=V[entry][last]
	print("Best Sequence ", BestSequence)
	print("Best value ", BestValue)
	actual_latency=0
	batches=[]
	i=0
	for index1,index2 in BestSequence:
		time1=Compute_Subcircuit_Cost(qg, gates[index1:index2], node_map, SubCircuitParts[index1][index2], Dec_time, dist_matrix, False)
		print("time1 ", time1)
		actual_latency+=time1
		if len(BestSequence)>i:
			EP_list=StitchAdjacent(SubCircuitParts[index1][index2],SubCircuitParts[BestSequence[i][0]][BestSequence[i][1]], qubits, 0, dist_matrix, node_map)
			#print(EP_list)
			if EP_list:
				garb,stitch_cost, garb2=Greedy_DP_Ent_Gen(qg, EP_list, Dec_time, dist_matrix, SubCircuitParts[index1][index2],None, Bin_gates) 
			else:
				stitch_cost=0
					
			actual_latency+=stitch_cost
			i+=1
	print("Best value ", actual_latency)
	return BestSequence, actual_latency

def nextR(l, i, r):
	nextlist=[entry for entry in l if len(entry)==3 and entry[-1]>i]
	nextlist=nextlist[:r+1]
	return nextlist


def newBenefit(qubit, part, nextr):
	benefit=0
	for gate in nextr:
		if gate[0] in part and gate[1]==qubit:
			benefit+=1
		elif gate[1] in part and gate[0] == qubit:
			benefit+=1
			
	return benefit
def PreferenceOrder(partition, num, nextr, qubit):
	partCount=dict.fromkeys(range(num),0)
	for gate in nextr:
		if gate[0]==qubit:
			partCount[partition[gate[1]]]+=1
		else:
			partCount[partition[gate[0]]]+=1
	partCount = sorted(partCount, key=partCount.get, reverse=True)
	return partCount

def ComputeOutgoing(qubits, gates):
	RelevantGates=[tuple(sorted(elem, key=int)) for elem in gates if len(elem)>1 and elem[0] in qubits and elem[-1] in qubits]
	outgoingEdges=dict.fromkeys(qubits,0)
	for edge in RelevantGates:
		outgoingEdges[edge[0]]+=1
		outgoingEdges[edge[1]]+=1
	return outgoingEdges	
	
	
def BestPartitionModified(partition,parts, CZgates, gaps, nextr1, nextr2, qubit1, qubit2): ## CZgates: CZ gates that lie before maxR
	bestPart=-1
	highestBenefit=-np.inf
	minCutQubits=[]
	r=len(nextr1)
	#print(qubit1,qubit2)
	for i, part in enumerate(parts):
		#print(part)
		gapsNeeded=2-gaps[i]
		cutQubits=[]
		if partition[qubit1]==i or partition[qubit2]==i:
			gapsNeeded=1-gaps[i]
			gapsNeeded=max(gapsNeeded,0)
		outgoingEdges=ComputeOutgoing([qubit for qubit in part if qubit!=qubit1 and qubit!=qubit2], CZgates)		
		#print(outgoingEdges)
		if not outgoingEdges:
			continue
		if gapsNeeded==2:
			poorQubit1=min(outgoingEdges, key=outgoingEdges.get)
			outgoingEdges[poorQubit1]=np.inf
			poorQubit2=min(outgoingEdges, key=outgoingEdges.get)
			outgoingEdges[poorQubit1]=0
			outgoingEdges[poorQubit2]=0
			cutQubits+=[poorQubit1,poorQubit2]
		elif gapsNeeded==1:
			poorQubit1=min(outgoingEdges, key=outgoingEdges.get)
			outgoingEdges[poorQubit1]=0
			cutQubits+=[poorQubit1]
		CurrentBenefit=newBenefit(qubit1,list(set(part)-set(cutQubits)),nextr1)+newBenefit(qubit2,list(set(part)-set(cutQubits)),nextr2)+newBenefit(qubit1,[qubit2],nextr1)-2*r*gapsNeeded
		if CurrentBenefit>highestBenefit:
			bestPart=i
			minCutQubits=cutQubits
			highestBenefit=CurrentBenefit
			
	StaticBenefit=newBenefit(qubit1, parts[partition[qubit1]], nextr1)+newBenefit(qubit2, parts[partition[qubit2]], nextr2)
	if StaticBenefit>highestBenefit:
		return -1, []
	
	return bestPart, minCutQubits


def BestWithGaps(order,gaps, forbidden):
	for entry in order:
		if entry==forbidden:
			continue
		if gaps[entry]>0:
			return entry
	print("Mistake! ", order, gaps)
	return 0

def getNonempty(gaps, forbidden):
	flag=0
	first=-1
	second=-1
	for part, gap in gaps.items():
		if part in forbidden:
			continue
		if gap>1:
			first=part
			second=part
			break
		elif gap==1 and first==-1:
			first =part
		elif gap>=1:
			second=part
			break
	return first, second
	
def Online_tele(MapQtoG, qG, comm_cost, CZindices, qubits, gates, node_map, dec):
	r=10
	Mydata=Circuit_to_Graph(qubits, gates)
	InitPartition=MinQAP3(qubits, Mydata, comm_cost)
	numparts=len(set(node_map.values()))
	parts=[[] for i in range(numparts)]
	for i in range(len(qubits)):
		parts[node_map[InitPartition[i]]].append(str(i))
	gaps={}
	cap={}
	print(qubits)
	#print(gates)
	print(parts)
	partition={}
	for qubit in qubits:
		flag=0
		for i in range(0, numparts):
			if qubit in parts[i]:
				flag=1
				partition[qubit]=i
		if flag==0:
			parts[0].append(qubit)
			partition[qubit]=0
	print("partition ", partition)
	for i,part in enumerate(parts):
		cap[i]=max(len(part),2)
		gaps[i]=cap[i]-len(part)
	parts_copy=[part[:] for part in parts]	
	teleportations=[]
	EP_list=[]
	telegates=0
	preferenceOrder=defaultdict(list)
	for qubit in qubits:
		nextr=nextR(MapQtoG[qubit], 0, r)
		partCount=defaultdict(int)
		preferenceOrder[qubit]=PreferenceOrder(partition, numparts, nextr, qubit)	
	for index in CZindices:
		qubit1=gates[index][0]
		qubit2=gates[index][1]
		nextr1=nextR(MapQtoG[qubit1],index,r)
		nextr2=nextR(MapQtoG[qubit2],index,r)
		maxR=max(max([entry[-1] for entry in nextr1]),max([entry[-1] for entry in nextr1]))
		preferenceOrder[qubit1]=PreferenceOrder(partition, numparts, nextr1, qubit1)	
		preferenceOrder[qubit2]=PreferenceOrder(partition, numparts, nextr2, qubit2)	
		cutQubits=[]
		if partition[qubit1]!=partition[qubit2]:
			#print("here")
			RelevantIndices=find_ge(CZindices, index, maxR)
			bestpart, cutQubits =BestPartitionModified(partition,parts_copy, [gates[i] for i in RelevantIndices], gaps, nextr1, nextr2, qubit1, qubit2)
			#print("gaps ", gaps)
			if bestpart==-1:
				EP_list.append(tuple(sorted([partition[qubit2], partition[qubit1]])+[index]))
				telegates+=1
				continue
			print("best part ", bestpart)
			print(index, parts_copy, gates[index], bestpart, gaps[bestpart], cutQubits)
			
			
			if bestpart == partition[qubit1]:
				if cutQubits!=[]:
					print(bestpart, parts_copy[bestpart], cutQubits, gates[index] )
					poorQubit=cutQubits[0]
					otherpart=BestWithGaps(preferenceOrder[poorQubit],gaps, bestpart)
					teleportations.append((poorQubit, otherpart, index, comm_cost[otherpart, bestpart]))
					
					EP_list.append(tuple(sorted([partition[poorQubit],otherpart])+[index]))
					parts_copy[otherpart].append(poorQubit)
					parts_copy[bestpart].remove(poorQubit)
					partition[poorQubit]=otherpart
					gaps[otherpart]-=1
					gaps[bestpart]+=1
				teleportations.append((qubit2, bestpart, index, comm_cost[partition[qubit2], bestpart]))
				EP_list.append(tuple(sorted([partition[qubit2], bestpart])+[index]))
				parts_copy[partition[qubit2]].remove(qubit2)
				parts_copy[bestpart].append(qubit2)
				gaps[partition[qubit2]]+=1
				gaps[bestpart]-=1
				partition[qubit2]=bestpart

			elif bestpart == partition[qubit2]:
				if cutQubits!=[]:
					poorQubit=cutQubits[0]
					otherpart=BestWithGaps(preferenceOrder[poorQubit],gaps, bestpart)
					teleportations.append((poorQubit, otherpart, index, comm_cost[otherpart, bestpart]))
					
					EP_list.append(tuple(sorted([partition[poorQubit],otherpart])+[index]))
					parts_copy[otherpart].append(poorQubit)
					parts_copy[bestpart].remove(poorQubit)
					partition[poorQubit]=otherpart
					gaps[otherpart]-=1
					gaps[bestpart]+=1
				teleportations.append((qubit1, bestpart, index, comm_cost[partition[qubit1], bestpart]))
				EP_list.append(tuple(sorted([partition[qubit1], bestpart])+[index]))
				
				parts_copy[partition[qubit1]].remove(qubit1)
				parts_copy[bestpart].append(qubit1)
				gaps[partition[qubit1]]+=1
				gaps[bestpart]-=1
				partition[qubit1]=bestpart
				
			elif len(cutQubits)==1:
				poorQubit=cutQubits[0]
				otherpart=BestWithGaps(preferenceOrder[poorQubit],gaps, bestpart)
				teleportations.append((poorQubit, otherpart, index, comm_cost[otherpart, bestpart]))
				EP_list.append(tuple(sorted([partition[poorQubit],otherpart])+[index]))
				
				parts_copy[otherpart].append(poorQubit)
				parts_copy[bestpart].remove(poorQubit)
				partition[poorQubit]=otherpart
				gaps[otherpart]-=1
				gaps[bestpart]+=1
				teleportations.append((qubit1, bestpart, index,comm_cost[partition[qubit1], bestpart]))
				teleportations.append((qubit2, bestpart, index, comm_cost[partition[qubit2], bestpart]))
				EP_list.append(tuple(sorted([partition[qubit1], bestpart])+[index]))
				EP_list.append(tuple(sorted([partition[qubit2], bestpart])+[index]))
				
				parts_copy[partition[qubit2]].remove(qubit2)
				parts_copy[bestpart].append(qubit2)
				parts_copy[partition[qubit1]].remove(qubit1)
				parts_copy[bestpart].append(qubit1)
				gaps[partition[qubit2]]+=1
				gaps[bestpart]-=1
				gaps[partition[qubit1]]+=1
				gaps[bestpart]-=1
				partition[qubit2]=bestpart
				partition[qubit1]=bestpart
			elif len(cutQubits)==2:
				poorQubit=cutQubits[0]
				otherpart=BestWithGaps(preferenceOrder[poorQubit],gaps, bestpart)
				teleportations.append((poorQubit, otherpart, index, comm_cost[otherpart, bestpart]))
				EP_list.append(tuple(sorted([partition[poorQubit],otherpart])+[index]))
				
				parts_copy[otherpart].append(poorQubit)
				parts_copy[bestpart].remove(poorQubit)
				partition[poorQubit]=otherpart
				gaps[otherpart]-=1
				gaps[bestpart]+=1
				poorQubit=cutQubits[1]
				otherpart=BestWithGaps(preferenceOrder[poorQubit],gaps, bestpart)
				teleportations.append((poorQubit, otherpart, index, comm_cost[otherpart, bestpart]))
				EP_list.append(tuple(sorted([partition[poorQubit],otherpart])+[index]))
				
				parts_copy[otherpart].append(poorQubit)
				parts_copy[bestpart].remove(poorQubit)
				partition[poorQubit]=otherpart
				gaps[otherpart]-=1
				gaps[bestpart]+=1
				
				teleportations.append((qubit1, bestpart, index, comm_cost[partition[qubit1], bestpart]))
				teleportations.append((qubit2, bestpart, index, comm_cost[partition[qubit2], bestpart]))
				
				EP_list.append(tuple(sorted([partition[qubit1], bestpart])+[index]))
				EP_list.append(tuple(sorted([partition[qubit2], bestpart])+[index]))
				parts_copy[partition[qubit2]].remove(qubit2)
				parts_copy[bestpart].append(qubit2)
				parts_copy[partition[qubit1]].remove(qubit1)
				parts_copy[bestpart].append(qubit1)
				gaps[partition[qubit2]]+=1
				gaps[bestpart]-=1
				gaps[partition[qubit1]]+=1
				gaps[bestpart]-=1
				partition[qubit2]=bestpart
				partition[qubit1]=bestpart
			else:
				teleportations.append((qubit1, bestpart, index, comm_cost[partition[qubit1], bestpart]))
				teleportations.append((qubit2, bestpart, index, comm_cost[partition[qubit2], bestpart]))
				EP_list.append(tuple(sorted([partition[qubit1], bestpart])+[index]))
				EP_list.append(tuple(sorted([partition[qubit2], bestpart])+[index]))
				
				parts_copy[partition[qubit2]].remove(qubit2)
				parts_copy[bestpart].append(qubit2)
				parts_copy[partition[qubit1]].remove(qubit1)
				parts_copy[bestpart].append(qubit1)
				gaps[partition[qubit2]]+=1
				gaps[bestpart]-=1
				gaps[partition[qubit1]]+=1
				gaps[bestpart]-=1
				partition[qubit2]=bestpart
				partition[qubit1]=bestpart
			#print(teleportations)
							
	print(partition)
	cost=0
	#print("EP list ", EP_list)
	
	batches,actual_latency, garb=Greedy_Ent_Gen_Dec_par(qG, EP_list, dec, comm_cost, InitPartition,  None , gates)
	#Greedy_DP_Ent_Gen(qG, EP_list, dec, comm_cost, InitPartition,None, gates)
	print("actual latency ", actual_latency)
	print("number of teles ", len(teleportations), telegates)
	return batches, actual_latency


def gnp_random_connected_graph(n, p, label=None):
	edges = combinations(range(n), 2)
	G = nx.Graph()
	if label:
		edges = combinations([i for i in range(n)], 2)
		for i in range(n):
			G.add_node(i, Label=label)
	else:
		G.add_nodes_from(range(n))
	if p <= 0:
		return G
	if p >= 1:
		return nx.complete_graph(n, create_using=G)
	for _, node_edges in groupby(edges, key=lambda x: x[0]):
		node_edges = list(node_edges)
		random_edge = random.choice(node_edges)
		G.add_edge(*random_edge, weight=0.1)
		for e in node_edges:
			if random.random() < p:
				G.add_edge(*e)	
	return G	

def Create_qG(num_nodes):
	area_side = 10  # a square side of side x side cells
	cell_size = 10000  # side value of each square side (m)
	min_node_degree, max_node_degree = 2, 4  # number of edges from one node to its neighbors
	min_channel_num, max_channel_num = 5, 5  # number of parallel channels a connection would have
	default_number_of_src_dst_pair = 3
	default_decoherence_time = 1.5  # seconds
	default_atomic_bsm_rate, default_optical_bsm_rate = 1.0, 1.0
	default_edge_density = 1.0
	default_num_nodes = 10
	default_distance_range = [15, 20]
	min_memory_size, max_memory_size = 20, 20
	qG=create_random_graph(area_side=area_side, cell_size=cell_size, number_nodes=num_nodes,min_memory_size=min_memory_size, max_memory_size=max_memory_size,
									  min_node_degree=min_node_degree,max_node_degree=max_node_degree,
									  min_channel_num=min_channel_num, max_channel_num=max_channel_num)
	'''							  
	qG=create_random_waxman(area_side=area_side, cell_size=cell_size,number_nodes=default_num_nodes,
									  min_memory_size=min_memory_size, max_memory_size=max_memory_size,
									  min_channel_num=min_channel_num, max_channel_num=max_channel_num,
									  atomic_bsm_success_rate= default_atomic_bsm_rate,
									  edge_density=default_edge_density)
	'''
	connected_graph(qG, min_channel_num=min_channel_num, max_channel_num=max_channel_num)
	
	return qG

def averageLen(lst):
    lengths = [len(i) for i in lst]
    return 0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths)) 

	


def printing(qubits, gates, NumEPs, greedy_val, num_nodes, filename,greed_time, batch_size_greedy,Greedy_batch_time):
	print(filename, " numqubits ", len(qubits), "num EPs ", NumEPs)
	f=open(filename, "w")
	f.write("Original Circuit: ")
	f.write('\n')
	f.write("qubits: "+ str(qubits))
	f.write('\n')
	f.write("gates: "+ str(len(gates)))
	f.write('\n')
	f.write("num of CZ: " + str(len([gate for gate in gates if len(gate)>1])))
	f.write('\n')
	f.write("number of nodes: "+str(num_nodes))
	f.write('\n')
	f.write("Greedy_latency: "+str(greedy_val))
	f.write('\n')
	f.write("Greedy_time: "+str(greed_time))
	f.write('\n')
	f.write("Greedy_batch_size: "+str(batch_size_greedy))
	f.write('\n')
	f.write("Greedy_batch_latency: "+str(Greedy_batch_time))
	f.write('\n')

greedy=0.0
caleffi=0.0


		
		
