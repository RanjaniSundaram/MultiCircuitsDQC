import sys
from unittest.mock import MagicMock
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
from MinQAP import *
import random as rn
from Circuit import Circuit
import heapq
import cProfile
import pstats

class NetworkGraph:
    def __init__(self, numMems, num):
        self.numMems = numMems
        self.num = num
        self.pieces = self.create_pieces()
        self.qG= None
        self.node_map = {}  
        self.network_graph = self.create_network_graph()
        self.main_network_graph= self.create_main_network_graph()
    
    def create_pieces(self):
        pieces = []
        a = self.numMems
        if a == self.num:
            pieces = [1] * self.num
        else:
            for idx in range(self.num - 1):
                #pieces.append(randint(1, a - sum(pieces) - self.num + idx))
                pieces.append(self.numMems//self.num)
            pieces.append(a - sum(pieces))
        print(pieces)
        return pieces

    def create_network_graph(self):
        self.qG = Create_qG(self.num)
        return qGraph_to_Networkx(self.qG)

    def create_main_network_graph(self):
        main_network_graph = nx.Graph()
        for node in sorted(self.network_graph.nodes()):
            temp_graph = gnp_random_connected_graph(self.pieces[node], 0.9, str(node))
            main_network_graph = nx.disjoint_union(main_network_graph, temp_graph)

        for subnode, att in main_network_graph.nodes(data=True):
            self.node_map[subnode] = int(att['Label'])
        for edge in self.network_graph.edges():
            left_nodes = [x for x, y in main_network_graph.nodes(data=True) if y['Label'] == str(edge[0])]
            right_nodes = [x for x, y in main_network_graph.nodes(data=True) if y['Label'] == str(edge[1])]
            for left in left_nodes:
                for right in right_nodes:
                    main_network_graph.add_edge(left, right, weight=10)

        return main_network_graph
    
    def AddDummyNodes(self,n):
        G = nx.Graph()
        for i in range(n):
            G.add_node(self.numMems+i, Label=str(self.numMems+i))
            if i>=1:
                G.add_edge(self.numMems+i-1, self.numMems+i, weight=10000000)
        NewNetworkGraph=nx.disjoint_union(self.main_network_graph, G)
        NewNetworkGraph.add_edge(0,self.numMems, weight=10)
        #fig, axes = plt.subplots(5,5,dpi=100)

        elarge = [(u, v) for (u, v, d) in NewNetworkGraph.edges(data=True)]
        esmall = [(u, v) for (u, v, d) in NewNetworkGraph.edges(data=True)]
        pos = nx.spring_layout(NewNetworkGraph, seed=7)
        nx.draw_networkx_nodes(NewNetworkGraph, pos, node_size=10)

        # edges
        nx.draw_networkx_edges(NewNetworkGraph, pos, edgelist=elarge, width=1)
        #nx.draw_networkx_edges(NewNetworkGraph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed")
        edge_labels = nx.get_edge_attributes(NewNetworkGraph, "weight")
        nx.draw_networkx_edge_labels(NewNetworkGraph, pos, edge_labels)
        plt.savefig("biggraph.png") 

        return NewNetworkGraph

    def __repr__(self):
        return f"NetworkGraph(numqubits={self.numqubits}, num={self.num}, pieces={self.pieces})"



dec=1.0
M=1000
num=50

def ReadInput(folder):
    Circuits=[]
    path='MultiCircuits/'+folder
    for filename in os.listdir(path):
        Circuits.append(Circuit(path+'/'+filename))
    return Circuits
        
        

def MinQAPSmallCircuit(circ, network_graph):
    '''
    circ: class for a circuit
    network_graph: nx.Graph() with memory as unit
    '''
    global _distance_matrix
    memory_list=network_graph.nodes()
    flow_matrix=np.zeros((len(memory_list),len(memory_list)))
    for entry in circ.MyData:
        flow_matrix[int(entry[0]), int(entry[1])]= int(entry[2])
        flow_matrix[int(entry[1]), int(entry[0])]= int(entry[2])
    res = quadratic_assignment(flow_matrix, _distance_matrix, options={"maximize":False})
    assigned_memory={}
    for qubit,location in enumerate(res['col_ind']): 
        assigned_memory[qubit]=location
        
    return assigned_memory, _distance_matrix, res.fun



def MinQAP_qG(circ, network_graph):
    '''
    circ: class for a circuit
    network_graph: nx.Graph() with computer/node as unit
    '''
    global _distance_matrix
    node_list=network_graph.nodes()
    flow_matrix=np.zeros((len(node_list),len(node_list)))
    for entry in circ.qubit_partition:
        flow_matrix[int(entry[0]), int(entry[1])]= int(entry[2])
        flow_matrix[int(entry[1]), int(entry[0])]= int(entry[2])
    res = quadratic_assignment(flow_matrix, _distance_matrix, options={"maximize":False})
    assigned_memory={}
    for qubit,location in enumerate(res['col_ind']): 
        assigned_memory[qubit]=location
        
    return res.fun

def MinQAP_qG_Naive(circ):
    sum_upper = sum(int(weight) for p1, p2, weight in circ.qubit_partition)
    return sum_upper


def MinQAP_qG_less_Naive(partition_list):
    global _distance_matrix
    flow_matrix=np.zeros((50,50))
    for partition in partition_list:
        k = 0
        for entry in partition:
            flow_matrix[k+int(entry[0]), k+int(entry[1])]= int(entry[2])
            flow_matrix[k+int(entry[1]), k+int(entry[0])]= int(entry[2])
        k += 5
    res = quadratic_assignment(flow_matrix, _distance_matrix, options={"maximize":False})
    return res.fun


def LatencyofCircuitonNetwork(circ, network):
    assigned_memory,distance_matrix,latency=MinQAPSmallCircuit(circ, network.main_network_graph)
    #latency=CallGreedy(assigned_memory,distance_matrix,circ, network)
    
    return latency
    
def CallGreedy(parts,dist_matrix,circ, network):
    EP_list=[]
    Bin_gates=[sorted(gate) for i,gate in enumerate(circ.gates) if len(gate)==2]
    for i,(q1,q2) in enumerate(Bin_gates):
        if dist_matrix[parts[int(q1)],parts[int(q2)]]>=10.0:
            EP_list.append(tuple(sorted([network.node_map[parts[int(q1)]],network.node_map[parts[int(q2)]]])+[i]))
    # print("in Greedy")
    greedy_batches,greedy_val,greed_time=Greedy_Ent_Gen_Dec_par(network.qG, EP_list, dec, 0,0,0,0)
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
    BigCircuit.latency=LatencyofCircuitonNetwork(BigCircuit, network)
    return BigCircuit


def blackbox(circuit_index, circuits_list, network):
    """
    Computes the latency of a set of circuits running on a network.
    Uses caching to avoid redundant computations.
    """
    global blackbox_cache
    key = tuple(sorted(circuit_index))  # Ensure consistent ordering for the cache

    # Check if the result is already cached
    if key in blackbox_cache:
        return blackbox_cache[key]

    # Compute the latency if not cached
    memory_number = network.numMems
    total_qubit_num = sum(len(circuits_list[i].qubits) for i in circuit_index)
    
    if total_qubit_num > memory_number:
        blackbox_cache[key] = np.inf  # Store the result to avoid re-computation
        return np.inf

    if len(circuit_index) == 1:
        latency = LatencyofCircuitonNetwork(circuits_list[circuit_index[0]], network)
    else:
        combined_circuit = CombineCircuits([circuits_list[i] for i in circuit_index], network)
        latency = combined_circuit.latency

    # Store the result in the cache
    blackbox_cache[key] = latency
    return latency

def blackbox1(circuit_index, circuits_list, network):
    """
    Computes the latency of a set of circuits running on a network.
    Uses caching to avoid redundant computations.
    """
    global blackbox_cache
    key = tuple(sorted(circuit_index))  # Ensure consistent ordering for the cache

    # Check if the result is already cached
    if key in blackbox_cache:
        return blackbox_cache[key]
    
    if len(circuit_index) > 10:
        blackbox_cache[key] = np.inf  # Store the result to avoid re-computation
        return np.inf

    combined_circuit_latency = [MinQAP_qG_Naive(circuits_list[i]) for i in key]
    latency = max(combined_circuit_latency)

    # Store the result in the cache
    blackbox_cache[key] = latency
    return latency

def blackbox11(circuit_index, circuits_list, network):
    """
    Computes the latency of a set of circuits running on a network.
    Uses caching to avoid redundant computations.
    """
    global blackbox_cache
    key = tuple(sorted(circuit_index))  # Ensure consistent ordering for the cache

    # Check if the result is already cached
    if key in blackbox_cache:
        return blackbox_cache[key]
    
    if len(circuit_index) > 10:
        blackbox_cache[key] = np.inf  # Store the result to avoid re-computation
        return np.inf

    partition_list = [circuits_list[i].qubit_partition for i in key]
    combined_circuit_latency = MinQAP_qG_less_Naive(partition_list)

    # Store the result in the cache
    blackbox_cache[key] = combined_circuit_latency
    return combined_circuit_latency


def min_k_DP(circuits, network, K=None):
    """
    Finds an optimal batching strategy using dynamic programming.
    Keeps track of at most K best collections.
    """
    n = len(circuits)
    if K is None:
        K = int(np.sqrt(n))
    DP = [[] for _ in range(n + 1)]
    # Base case: First circuit starts in its own batch
    DP[0].append(([[0]], [blackbox([0], circuits, network)]))  
    
    # DP[i] contains at most K entries, each entry is like: [[batches], [times]],
    # the first list contains the batch collections and second is the time for the 
    # corresponding batch from blackbox 
    for i in range(1, n):
        print(i)
        new_solutions = []
        
        for batch_collection, batch_time in DP[i-1]:
            # batch_collection is like [[0],[1,2],[3]]
            # batch_time is like [time1, time2, time3]
            
            # Option 1: Add c_i to an existing batch
            for j in range(len(batch_collection)):
                new_collection = [b[:] for b in batch_collection]  # Fast copy
                new_collection[j].append(i)
                new_time = batch_time[:]  # Fast copy
                new_time[j] = blackbox(new_collection[j], circuits, network)
                new_solutions.append((new_collection, new_time))
            
            # Option 2: Create a new batch with only c_i
            new_collection = batch_collection + [[i]]
            new_time = batch_time[:] + [blackbox([i], circuits, network)]
            new_solutions.append((new_collection, new_time))
        
        # Maintain only K best solutions
        if len(new_solutions) > K:
            DP[i] = heapq.nsmallest(K, new_solutions, key=lambda x: sum(x[1]))
        else:
            DP[i] = new_solutions
    
    # Extract the best solution
    best_batching, best_time_list = DP[n-1][0]
    best_time = sum(best_time_list)

    return best_time, best_batching




def order_DP(circuits, network):
    n = len(circuits)
    DP = np.full((n, n), np.inf)  # DP table for latency
    history = [[[] for _ in range(n)] for _ in range(n)]  # History table for batches

    # Base case: Single circuit batches
    for i in range(n):
        DP[i][i] = circuits[i].latency
        history[i][i] = [[i]]

    # Fill DP table
    for length in range(2, n + 1):  # Subproblem sizes
        for i in range(n - length + 1):
            j = i + length - 1

            # Try all possible splits
            for k in range(i, j):
                current_cost = DP[i][k] + DP[k + 1][j]
                if current_cost < DP[i][j]:
                    DP[i][j] = current_cost
                    history[i][j] = [batch[:] for batch in history[i][k] + history[k + 1][j]]

            # Also consider the case where the entire range is one batch
            entire_batch_cost = blackbox(list(range(i, j + 1)), circuits, network)
            if entire_batch_cost < DP[i][j]:
                DP[i][j] = entire_batch_cost
                history[i][j] = [list(range(i, j + 1))]

    return float(DP[0][n - 1]), history[0][n - 1]



def merge_batches(circuits, network):
    """Merge batches based on maximum gain until no positive gain remains."""
    n = len(circuits)
    batches = [[i] for i in range(n)]  # Initialize each circuit as a batch
    time_cache = {tuple(batch): circuits[batch[0]].latency for batch in batches}  
    # Cache blackbox results
    
    while True:
        best_gain = float('-inf')
        best_pair = None
        best_combined_batch_time = None
        
        for (i, j) in combinations(range(len(batches)), 2):
            batch_i = tuple(batches[i])
            batch_j = tuple(batches[j])
            merged_batch = batch_i + batch_j
            combined_batch_time = blackbox(list(merged_batch), circuits, network)
            gain = time_cache[batch_i] + time_cache[batch_j] - combined_batch_time
            
            if gain > best_gain:
                best_gain = gain
                best_pair = (i, j)
                best_combined_batch_time = combined_batch_time
        
        if best_gain <= 0:
            break  # Stop if no positive gain remains
        
        i, j = best_pair
        merged_batch = tuple(batches[i] + batches[j])
        print(i,j)
        
        # Update the cache with the new merged batch
        time_cache[merged_batch] = best_combined_batch_time
        time_cache.pop(tuple(batches[i]))
        time_cache.pop(tuple(batches[j]))
        batches[i] = list(merged_batch)  # Merge the best pair
        del batches[j]  # Remove the merged batch
    
    return sum(time_cache.values()), batches



def optimal_DP(circuits, network, _get_all_time=False):
    n = len(circuits)
    batch_time = {}
    best_batch_collection = {}
    two_partition = {}
    for k in range(1, n+1):  # each iteration works on circuit list of length k
        for circuit_index in itertools.combinations(range(n), k):  # circuit indices
            batch_time[circuit_index] = blackbox(circuit_index, circuits, network)
            best_batch_collection[circuit_index] = [list(circuit_index)]
            two_partition[circuit_index] = []
            for sub_num in range(1,math.floor(k/2)+1): # k = (sub_num)+(k-sub_num)
                for subset in itertools.combinations(circuit_index, sub_num): # choose sub_num-circuit from k-circuit
                    if k%2==0 and sub_num==k/2: # avoid (subset1,subset2)==(subset2,subset1)
                        if min(circuit_index) in subset:
                            two_partition[circuit_index].append(subset)
                        else:
                            continue
                    else:
                        two_partition[circuit_index].append(subset)

    for k in range(2, n+1):  # each iteration works on k-circuits
        for target_circuit_index in itertools.combinations(range(n), k):
            for subset1 in two_partition[target_circuit_index]:
                subset2 = tuple(sorted(set(target_circuit_index) - set(subset1)))
                combined_time = batch_time[subset1] + batch_time[subset2]
                if combined_time < batch_time[target_circuit_index]:
                    batch_time[target_circuit_index] = combined_time
                    best_batch_collection[target_circuit_index]\
                    = best_batch_collection[subset1] + best_batch_collection[subset2]
                    
        
    best_time, best_batch = batch_time[tuple(range(n))], best_batch_collection[tuple(range(n))]
    
    if _get_all_time:
        return best_time, best_batch, batch_time
    else:
        return best_time, best_batch, None



def SetCover1(circuits, network):
    batch_collection = [] 
    Total_latency = 0  

    remainCircuits_index = set(range(len(circuits)))  

    while remainCircuits_index:
        # Start a new batch
        currentBatch = []

        # Select the circuit with the lowest latency
        best_circ_idx = min(remainCircuits_index, key=lambda i: circuits[i].latency)
        remainCircuits_index.remove(best_circ_idx)
        currentBatch.append(best_circ_idx)
        current_batch_time = circuits[best_circ_idx].latency

        # Try to add more circuits to the batch
        while remainCircuits_index:
            best_circ_idx = None
            best_latency = np.inf

            # Check each remaining circuit to find the best one to add
            for i in remainCircuits_index:
                NewBigCircuit_time = blackbox(currentBatch + [i], circuits, network)
                if NewBigCircuit_time < best_latency:
                    best_latency = NewBigCircuit_time
                    best_circ_idx = i

            # If no valid circuit can be added, start a new batch
            if best_circ_idx is None:
                break

            # Check if adding the best circuit satisfies the latency condition
            if best_latency <= current_batch_time + circuits[best_circ_idx].latency:
                currentBatch.append(best_circ_idx)
                current_batch_time = best_latency
                remainCircuits_index.remove(best_circ_idx)
            else:
                break
            
        # Store the finished batch and update total latency
        batch_collection.append(currentBatch)
        Total_latency += current_batch_time

    return batch_collection, Total_latency


def ApproximateGreedy(Circuits, network):
    """
    Greedy approximation for Set Cover with execution time minimization.
    - Iteratively picks the best subset with the lowest latency per circuit covered.
    """
    ToBeCovered = set(range(len(Circuits)))
    batch_collection = [] 
    Total_latency = 0 

    while ToBeCovered:
        best_value = np.inf
        best_subset = None

        for circ_index in ToBeCovered:
            # Compute the best batch that includes 'circ'
            subset = find_best_subset(circ_index, ToBeCovered, Circuits, network)  # Efficient subset selection
            if not subset:
                continue

            subset_time = blackbox(list(subset), Circuits, network)
            valuation = subset_time / len(subset)  # Latency per circuit

            if valuation < best_value:
                best_value = valuation
                best_subset = subset

        # if best_subset is None:
        #     break  # Should not happen if input is valid

        ToBeCovered -= best_subset  # Remove covered circuits
        batch_collection.append(list(best_subset))  # Store batch
        Total_latency += best_value*len(best_subset)  # Update total execution time

    return batch_collection, Total_latency

def find_best_subset(seed, ToBeCovered, Circuits, network):
    """
    Efficiently finds a good subset of circuits to batch with `seed`.
    - Instead of generating all subsets, it incrementally builds a batch.
    """
    batch = {seed}
    remaining = ToBeCovered - {seed}

    while remaining:
        best_latency_gain = np.inf
        best_circuit_index = None

        for circ_index in remaining:
            new_batch = batch | {circ_index}
            new_batch_time = blackbox(new_batch, Circuits, network)
            latency_gain = new_batch_time / len(new_batch)  

            if latency_gain < best_latency_gain:
                best_latency_gain = latency_gain
                best_circuit_index = circ_index

        if best_circuit_index is None:
            break  

        batch.add(best_circuit_index)
        remaining.remove(best_circuit_index)

    return batch

def main():
    method_list = ['min_k_DP', 'merge_batches', 'SetCover1', 'order_DP']
    random.seed(102)
    np.random.seed(102)
    network = NetworkGraph(M,num) # a class for network
    global _distance_matrix
    _distance_matrix=nx.floyd_warshall_numpy(network.main_network_graph)
    
    ################## change the input here
    used_method_index = [0]  
    used_circ_index = [14]
    ##################
    
    start_time = time.time()
    time_list = []
    time_dic = {method_list[i]: [] for i in used_method_index}
    for folder_num in used_circ_index:  
        circuit_list = ReadInput('dir'+str(folder_num)) # a list, with entry being class for circuit
        global blackbox_cache 
        blackbox_cache = {}
        end_time = time.time()
        spent_time = end_time - start_time
        print('prepare ready',spent_time)
        time_list.append(spent_time)
        start_time = end_time
        for i in range(len(circuit_list)):
            circuit_list[i].latency = blackbox([i], circuit_list, network)
        for i in used_method_index:
            method = method_list[i]
            # if method == 'optimal_DP':
            #     best_time, best_batch, all_batch_time = optimal_DP(circuit_list, network, False)
            #     time_dic['optimal_DP'].append(round(best_time, 4))
                #print(best_time, best_batch)
            if method == 'min_k_DP':
                best_time, best_batch = min_k_DP(circuit_list, network)
            if method == 'order_DP':
                best_time, best_batch = order_DP(circuit_list, network)
            if method == 'merge_batches':
                best_time, best_batch = merge_batches(circuit_list, network)
            if method == 'SetCover1':
                best_batch, best_time = SetCover1(circuit_list, network)
            if method == 'ApproximateGreedy':
                best_batch, best_time = ApproximateGreedy(circuit_list, network)
            time_dic[method].append(round(best_time, 1))
            end_time = time.time()
            spent_time = end_time - start_time
            print(method,spent_time)
            time_list.append(spent_time)
            start_time = end_time
            #print(best_batch)
        #print(folder_num)

    print(time_dic)
    print(time_list)
    print(sum(time_list))



if __name__ == "__main__":
    #cProfile.run('main()', 'profile_results')

    # Print results
    with open("profile_output.txt", "w") as f:
        stats = pstats.Stats("profile_results", stream=f)
        stats.strip_dirs().sort_stats("tottime").print_stats(10)  # Show top 10
































