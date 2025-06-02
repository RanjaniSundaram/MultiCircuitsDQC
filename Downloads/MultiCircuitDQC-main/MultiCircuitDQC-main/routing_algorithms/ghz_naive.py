from routing_algorithms.optimal_linear_programming import optimal_linear_programming as lp
from routing_algorithms.max_flow import max_flow
from commons.qGraph import qGraph
from commons.qNode import qNode


class ghz_star_multi_pair:
    """This class is designed to do multipartite routing (GHZ) in a naive way. The algorithm try to find the best
    central node where the GHZ state is teleported through many bi-partite entanglement. Every node can be a candidate
    for the central node; the best one is found in a brute-force manner. Many bi-partite entanglement is solved
    through LP multi-pair multi-path algorithm."""

    def __init__(self, G: qGraph, end_nodes: set[qNode], decoherence_time: float):
        self._trees = None  # paths for best central node
        self._avr_ent_time = float('inf')

        for central_node in G.V:
            srcs, dsts = [], []
            for end_node in end_nodes:
                if end_node.id != central_node.id:
                    srcs.append(central_node)
                    dsts.append(end_node)
            # central_lp = lp(G=G, sources=srcs, destinations=dsts, node_capacity_division_constant=len(srcs))
            # central_multi_pair_multi_path = lp(G=G, sources=srcs, destinations=dsts, max_min=True)
            central_multi_pair_multi_path = max_flow(G=G, srcs=srcs, dsts=dsts, decoherence_time=decoherence_time,
                                                     multiple_pairs_search='fair', alternate=False)
            max_avr_ent = 0
            for node_trees in central_multi_pair_multi_path.trees:
                if len(node_trees) == 0:
                    max_avr_ent = float('inf')
                    break
                max_avr_ent = max(max_avr_ent, 1.0 / sum([1.0 / tree.avr_ent_time for tree in node_trees]))
            # TODO ignoring classical time for now
            if (max_avr_ent + central_node.bsm_time) / (central_node.bsm_success_rate ** len(srcs)) < \
                    self._avr_ent_time:
                self._avr_ent_time = (max_avr_ent + central_node.bsm_time) / (central_node.bsm_success_rate ** len(srcs))
                self._trees = [[] for _ in range(len(dsts))]
                for idx in range(len(dsts)):
                    self._trees[idx] = central_multi_pair_multi_path.trees[idx]

    @property
    def avr_ent_time(self):
        return self._avr_ent_time

    @property
    def bi_trees(self):
        return self._trees


