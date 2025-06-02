'''
Topology for a graph
'''

import networkx as nx
import matplotlib.pyplot as plt

class Topology:
    def __init__(self):
        self.filename = ''
        self.nodes = {}         # index -> (x, y)
        self.edges = []         # an element is a tuple (index1, index2)
        self.memories = {}        # index -> memory of a node
        self.num_channels = {}  # (index1, index2) -> # of channels
    
    def read_file(self, filename: 'str', unit: int = 1000):
        '''read a file that has the topology, the file format is like the "toy1" example
        Args:
            filename: the name of the topology file
            unit:     km -> m
        '''
        self.filename = filename
        with open(filename, 'r') as f:
            node = True
            for line in f:
                line = line.strip()
                if line == '-':
                    node = False
                    continue
                if node:
                    idx, memory, x, y = line.split(' ')
                    idx    = int(idx)
                    memory = int(memory)
                    x   = float(x) * unit
                    y   = float(y) * unit
                    self.nodes[idx] = (x, y)
                    self.memories[idx] = memory
                else:
                    idx1, idx2, num_channel = line.split(' ')
                    idx1, idx2 = int(idx1), int(idx2)
                    idx1, idx2 = min(idx1, idx2), max(idx1, idx2)
                    self.edges.append((idx1, idx2))
                    self.num_channels[(idx1, idx2)] = int(num_channel)

    def generate_random_topology(self):
        '''generate a random graph topology
        '''
        pass # TODO

    def visualize(self, savefig=False):
        '''visulaize the topology using package networkx
        '''
        G = nx.Graph()
        G.add_edges_from(self.edges)
        widths = []
        for u, v in G.edges:
            u, v = min(u, v), max(u, v)
            widths.append(self.num_channels[(u, v)])
        fig, ax = plt.subplots(figsize=(10,10))
        nx.draw_networkx(G, self.nodes, width=widths)
        if savefig:
            plt.savefig(self.filename)
        else:
            plt.show()


if __name__ == '__main__':

    # topo = Topology()
    # topo.read_file('toy1')
    # topo.visualize()

    topo = Topology()
    topo.read_file('toy5-multichannel')
    topo.visualize(savefig=True)
