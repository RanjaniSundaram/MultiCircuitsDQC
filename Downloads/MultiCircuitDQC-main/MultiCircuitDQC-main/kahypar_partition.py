import os
import kahypar as kahypar

num_nodes = 7
num_nets = 2

hyperedge_indices = [0,2,6,9,12]
hyperedges = [0,2,0,1,3,4,3,4,6,2,5,6]

node_weights = [1,2,3,4,5,6,7]
edge_weights = [11,22,33,44]

k=2

hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

context = kahypar.Context()
context.loadINIconfiguration("./kahypar_config.ini")

context.setK(k)
context.setEpsilon(0.03)
context.suppressOutput(True)


kahypar.partition(hypergraph, context)

n=hypergraph.numBlocks()
parts=[[x for x in range(num_nodes) if hypergraph.blockID(x)==y] for y in range(n)]
print(parts)
