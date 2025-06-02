
from MinQAP import*
from NetworkGraph import NetworkGraph
from sklearn.cluster import SpectralClustering
from scipy.sparse import csgraph


class Circuit:
    def __init__(self, filepath, qubits=None, gates=None):
        if not filepath:
            self.secondaryinit(qubits, gates)
            return
        self.filepath = filepath
        self.qubits, self.gates, self.MapQtoG = get_circuit(filepath)
        self.MyData = Circuit_to_Graph(self.qubits, self.gates)
        self.locations=[]
        self.latency=np.inf
        #self.CheckMap()
        self.get_qubit_partition(5)
    def secondaryinit(self, qubits, gates):
        self.filepath = None
        self.qubits, self.gates= qubits, gates
        self.MyData = Circuit_to_Graph(self.qubits, self.gates)
        self.locations=[]
        self.latency=np.inf
        #self.CheckMap()
        self.get_qubit_partition(5)

    def CheckMap(self):
        for qubit in self.qubits:
            flag=1
            for entry in self.MyData:
                if entry[0]== qubit or entry[1]==qubit:
                    flag=0
                    break
            if flag:
                self.MyData.append([0,int(qubit), 1])
                
                      
    def get_qubit_partition(self, partition_num):
        # if len(self.qubits) == 0:
        #     return
        G = nx.Graph()
        for u, v, w in self.MyData:
            G.add_edge(int(u), int(v), weight=int(w))
        adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        adj_matrix += 0.0001  # Adding a small value to make the graph weakly connected

        sc = SpectralClustering(n_clusters=partition_num, affinity='precomputed', assign_labels='kmeans')
        labels = sc.fit_predict(adj_matrix)
        node_to_partition = {node: labels[i] for i, node in enumerate(sorted(G.nodes()))}
        
        partition_weights = defaultdict(int)
        for u, v, data in G.edges(data=True):
            pu, pv = node_to_partition[u], node_to_partition[v]
            if pu != pv:  # Only count inter-partition edges
                partition_weights[(min(pu, pv), max(pu, pv))] += data['weight']
        self.qubit_to_partition = node_to_partition
        self.qubit_partition = [[str(p1), str(p2), str(weight)] for (p1, p2), weight in partition_weights.items()]
        self.qubit_partition_weight_sum = sum(int(entry[2]) for entry in self.qubit_partition)
            

    def __repr__(self):
        return f"Circuit(qubits={self.qubits}, gates={self.gates}, MyData={self.MyData})"
    
