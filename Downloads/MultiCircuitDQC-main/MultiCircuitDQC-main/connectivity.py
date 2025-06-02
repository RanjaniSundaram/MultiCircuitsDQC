import sys
from MinQAP import *
import random as rn
from Circuit import Circuit
from NetworkGraph import NetworkGraph


dec=1.0
M=100
num=10

def has_inf(matrix):
    for row in matrix:
        for entry in row:
            if math.isinf(entry):
                return True
    return False  

# search for i where the graph is unconnected
for i in range(100):
    random.seed(100+i)
    np.random.seed(100+i)
    qG = Create_qG(num)
    network = qGraph_to_Networkx(qG)
    distance_matrix=nx.floyd_warshall_numpy(network)
    if has_inf(distance_matrix):
        print(i)
        # check
        print(distance_matrix)


