from routing_algorithms.ghz_steiner_tree import ghz_steiner_tree
from commons.qNode import qNode
from commons.qGraph import qGraph
from commons.qChannel import qChannel
from commons.tree_node import tree_node
from collections import defaultdict
from random import randint
from commons.ghz_fusion_nodes import general_fusion_node


class ghz_star_expansion:
    """This class is designed to do multipartite routing using star expansion algorithm where a Steiner tree is
    first computed. Then, over a random leaf node (a node with the degree of one), the star expansion is recursively
    applied till the final GHZ state is obtained.
        The algorithm is summarized as:
            1- Compute Steiner tree
            2- Create a left(or right) skewed special fusion tree over a random leaf node.
    """
    def __init__(self, G: qGraph, end_nodes: set[qNode], decoherence_time: float):
        self._G = G
        self._end_nodes = list(end_nodes)
        self._decoherence_time = decoherence_time
        end_nodes_id = [node.id for node in end_nodes]

        _, self._steiner_tree = ghz_steiner_tree.approx_steiner_tree(edges=G.E, end_nodes=list(end_nodes),
                                                                     decoherence_time=decoherence_time)
        adj_list = defaultdict(list)
        steiner_tree_nodes = set()
        for edge in self._steiner_tree:
            adj_list[edge.this.id].append(edge.other.id)
            adj_list[edge.other.id].append(edge.this.id)
            steiner_tree_nodes.add(edge.this.id)
            steiner_tree_nodes.add(edge.other.id)

        leaves = [node_id for node_id, adj in adj_list.items() if len(adj) == 1]
        root_id = leaves[randint(0, len(leaves) - 1)]

        a_nodes = [adj_list[root_id][0]]
        a_nodes_seen = set()
        prev_fusion_node = None
        non_end_nodes = set()
        edges_processed = set()
        while a_nodes:
            a_node = a_nodes.pop()
            # only non-leaf nodes and those that weren't processed and should not be the selected root
            if a_node not in a_nodes_seen and len(adj_list[a_node]) > 1 and a_node != root_id:
                a_nodes_seen.add(a_node)
                a_nodes += adj_list[a_node]
                if a_node not in end_nodes_id:
                    non_end_nodes.add(a_node)
                if prev_fusion_node is None:
                    end_nodes = set(adj_list[a_node])
                    if a_node in end_nodes_id:
                        end_nodes.add(a_node)
                    edges_processed = edges_processed.union(set([(a_node, u) for u in adj_list[a_node]]))
                    fusion_node = general_fusion_node(
                        node=G.V[a_node],
                        all_nodes=set(adj_list[a_node]).union({a_node}),
                        end_nodes=end_nodes, is_retain=a_node in end_nodes_id,
                        avr_ent_time=(max([G.get_edge(a_node, u).avr_ent_time for u in adj_list[a_node]]) +
                                      G.V[a_node].fusion_time) *
                                     (1.0 / G.V[a_node].fusion_success_rate) * len(adj_list[a_node]) / 2)
                else:
                    end_nodes = set(adj_list[a_node]).union(prev_fusion_node.end_nodes).difference(non_end_nodes)
                    new_edges = set()
                    for u in adj_list[a_node]:
                        if (a_node, u) not in edges_processed and (u, a_node) not in edges_processed:
                            new_edges.add((a_node, u))
                    edges_processed = edges_processed.union(new_edges)
                    if a_node in end_nodes_id:
                        end_nodes.add(a_node)
                    elif a_node not in end_nodes_id and a_node in end_nodes:
                        end_nodes.remove(a_node)
                    fusion_node = general_fusion_node(
                        node=G.V[a_node],
                        all_nodes=set(adj_list[a_node]).union(prev_fusion_node.all_nodes).union({a_node}),
                        end_nodes=end_nodes,
                        sub1=prev_fusion_node,
                        is_retain=a_node in end_nodes_id,
                        avr_ent_time=(max(prev_fusion_node.avr_ent_time,
                                          max([G.get_edge(a_node, u).avr_ent_time for _, u in new_edges])) +
                                      G.V[a_node].fusion_time) *
                                     (1.0 / G.V[a_node].fusion_success_rate) * len(adj_list[a_node]) / 2)
                prev_fusion_node = fusion_node
        self._fusion_tree = prev_fusion_node

    @property
    def fusion_tree(self):
        return self._fusion_tree

    @property
    def steiner_tree(self):
        return self._steiner_tree
