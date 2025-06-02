import functools
import sys

sys.path.append("..")

from commons.qGraph import qGraph
from commons.qNode import qNode
from commons.qChannel import qChannel
from typing import List
from routing_algorithms.dp_shortest_path import dp_shortest_path
from routing_algorithms.dp_alternate import DP_Alt
# from routing_algorithms.naive_shortest_path import hop_shortest_path as dp_shortest_path
from commons.tree_node import tree_node
from commons.tools import get_classical_time
from predistributed_algorithms.super_link import SL

COST_BUDGET_DEFAULT = 1e4  # eps/s
NAIVE_V2 = True  # if true, we use a different kind of naive where an SL is acceptable for a (s,d) pair if it's a
# sub-path of the (s,d)'s shortest path
# import cProfile as profile


class greedy_predist:

    def __init__(self, G: qGraph, srcs: List[qNode], dsts: List[qNode], latency_threshold: float, max_sl_latency: float,
                 IS_LATENCY_OBJECTIVE: bool, decoherence_time: float, FIXED_LATENCY: bool = False, NAIVE: bool = False,
                 cost_budget: float = COST_BUDGET_DEFAULT, NON_SHORTEST_SLS: bool = False,
                 NO_DELETION_GREEDY: bool = False, no_deletion_version: str = ""):
        if len(srcs) != len(dsts):
            raise ValueError("The number of sources and destinations are not equal!")
        self._sd_path_with_sls = [[] for _ in range(len(srcs))]
        self._best_sls = []
        self._average_latency = float('inf')

        n = len(G.V)
        shortest_paths = dp_shortest_path(G, decoherence_time=decoherence_time)
        shortest_paths_non_sl = shortest_paths # dp_shortest_path(G, decoherence_time=decoherence_time)

        # for testing
        # shortest_paths.get_shortest_path_without_update(G.V[3],
        #                                                 G.V[5]).avr_ent_time = 1.0 / \
        #                                                                                     greedy_predist.SL_RATE
        # self._best_sls += [SL(G.V[5], G.V[8], shortest_paths.get_shortest_path(G.V[5], G.V[8], True),
        #                       self.SL_RATE, 0)
        #                    ]
        # self._sd_path_with_sls.append([[shortest_paths.get_shortest_path_without_update(G.V[0], G.V[5]),
        #                                 (0, False, False)]
        #                                ])
        # # self._sd_path_with_sls.append([[shortest_paths.get_shortest_path_without_update(G.V[1], G.V[4]), (1, False),
        # #                                 shortest_paths.get_shortest_path_without_update(G.V[6], G.V[8])]
        # #                                ])
        # return
        # for testing

        # check which pairs have latency greater than latency threshold
        if not IS_LATENCY_OBJECTIVE:
            # objective = cost & constraint = latency
            sd_latency_targets = [latency_threshold] * len(srcs) if FIXED_LATENCY else \
                [shortest_paths.get_shortest_path_without_update(src, dst).avr_ent_time * latency_threshold
                 for src, dst in zip(srcs, dsts)]
            uncovered_sd_pairs = set([i for i in range(len(srcs)) if
                                      shortest_paths.get_shortest_path_without_update(srcs[i], dsts[i]).avr_ent_time >
                                      sd_latency_targets[i]])
            for sd_idx in set(list(range(len(srcs)))) - uncovered_sd_pairs:  # use sp for those that do not need SL
                self._sd_path_with_sls[sd_idx].append([shortest_paths_non_sl.get_shortest_path(srcs[sd_idx],
                                                                                               dsts[sd_idx])])
        else:
            # objective = latency & constraint = cost
            uncovered_sd_pairs = set([i for i in range(len(srcs))])

        # first try to satisfy remaining (S,d) pairs with good SLs (sl's latency < max_latency_threshold)
        if not NON_SHORTEST_SLS:
            if not NO_DELETION_GREEDY:
                new_sls, uncovered_sd_pairs, self._average_latency = \
                    greedy_predist._general_module_new(NAIVE, shortest_paths, shortest_paths_non_sl,
                                                       G,  srcs, dsts,
                                                       max_sl_latency=max_sl_latency,
                                                       uncovered_sd_pairs=uncovered_sd_pairs,
                                                       sd_latency_targets=sd_latency_targets if not
                                                       IS_LATENCY_OBJECTIVE else [],
                                                       sl_index_offset=0,
                                                       sd_path_with_sls=self._sd_path_with_sls,
                                                       IS_LATENCY_OBJECTIVE=IS_LATENCY_OBJECTIVE,
                                                       cost_budget=cost_budget,
                                                       decoherence_time=decoherence_time)
                self._best_sls += new_sls
            else:
                # non-efficient version of greedy, no-deletion, latency (sls selected based on latency not latency/cost)
                self._best_sls, self._average_latency, _ = \
                    greedy_predist._general_module(NAIVE=False, shortest_paths=shortest_paths,
                                                   shortest_paths_non_sl=shortest_paths_non_sl,
                                                   G=G, srcs=srcs, dsts=dsts, max_sl_latency=max_sl_latency,
                                                   uncovered_sd_pairs=uncovered_sd_pairs,
                                                   sd_latency_targets=sd_latency_targets if not IS_LATENCY_OBJECTIVE
                                                   else [], sl_index_offset=0, IS_LATENCY_OBJECTIVE=IS_LATENCY_OBJECTIVE,
                                                   sd_path_with_sls=self._sd_path_with_sls,
                                                   cost_budget=cost_budget,
                                                   no_deletion_greedy_version=no_deletion_version)
        else:
            self._best_sls, self._sd_path_with_sls, self._average_latency = \
                greedy_predist._non_shortest_sls_module(G, srcs, dsts, max_sl_latency, cost_budget,
                                                        decoherence_time, shortest_paths)
        if not IS_LATENCY_OBJECTIVE:
            # use bad SLs (violating the constraint)
            new_sls, _ = greedy_predist._general_module(NAIVE, shortest_paths, shortest_paths_non_sl, G, srcs, dsts,
                                                        max_sl_latency=float('inf'),
                                                        uncovered_sd_pairs=uncovered_sd_pairs,
                                                        sd_latency_targets=sd_latency_targets,
                                                        sl_index_offset=len(self._best_sls),
                                                        sd_path_with_sls=self._sd_path_with_sls,
                                                        IS_LATENCY_OBJECTIVE=False, cost_budget=cost_budget)
            self._best_sls += new_sls

    @property
    def average_latency(self) -> float:
        return self._average_latency

    @staticmethod
    def candidate_sls(orig_graph: qGraph, residual_graph: qGraph, residual_dp_alt: DP_Alt, sl_nodes: set[int],
                      src_id: int, dst_id: int, shortest: dp_shortest_path, shortest_cost: List[List[float]],
                      max_sl_latency: float, non_shortest_sls: bool = True) -> List[SL]:
        src, dst = orig_graph.get_node(src_id), orig_graph.get_node(dst_id)
        sl_sp = shortest.get_shortest_path_without_update(src, dst)
        sls = [SL(src, dst, sl_sp, shortest_cost[src_id][dst_id])]  # shortest path
        if non_shortest_sls and set(tree_node.tree_nodes(sl_sp)).intersection(sl_nodes):
            # there is an intersection, return non-shortest path
            sl_non_sp = residual_dp_alt.get_shortest_path(residual_graph.get_node(src_id),
                                                          residual_graph.get_node(dst_id), True)
            sls.append(SL(src, dst, sl_non_sp, greedy_predist.sl_cost(sl_non_sp, max_sl_latency)))
        return sls

    @staticmethod
    def _non_shortest_sls_module(G: qGraph, srcs: List[qNode], dsts: List[qNode], max_sl_latency: float,
                                 cost_budget: float, decoherence_time: float, shortest_paths: dp_shortest_path):
        best_sls, n = [], len(G.V)
        current_sl_set = set()
        sl_costs = greedy_predist._exhaustive_greedy(G, shortest_paths, max_sl_latency)
        best_total_latency, best_sd_paths, _ = greedy_predist. \
            sd_latency_with_sl_residual(srcs=srcs, dsts=dsts, orig_graph=G, shortest_paths=shortest_paths, sls=best_sls)
        # pr = profile.Profile()
        # pr.enable()
        while True:
            current_best_sl_metric = 0
            current_best_sls = None
            current_best_total_latency = 0
            current_best_sd_paths = []
            sl_nodes = set()
            residual_graph = G.clone()
            for sl in best_sls:
                greedy_predist.residual_graph(residual_graph, residual_graph.get_node(sl.src.id),
                                              residual_graph.get_node(sl.dst.id), [sl.tree], True)
                sl_nodes = sl_nodes.union(set(tree_node.tree_nodes(sl.tree)))
            residual_dp_alt = DP_Alt(residual_graph, decoherence_time)
            for i in range(n):
                for j in range(i + 1, n):
                    if f"{G.V[i].id}-{G.V[j].id}" in current_sl_set:  # these pair already selected
                        continue
                    for candid_idx, sl_candid in enumerate(greedy_predist.candidate_sls(G, residual_graph,
                                                                                        residual_dp_alt, sl_nodes,
                                                                                        i, j, shortest_paths, sl_costs,
                                                                                        max_sl_latency)):
                        if sl_candid.cost == float('inf') or not sl_candid.tree:
                            continue
                        if candid_idx == 0:  # shortest_path, get the maximum subset of prev sls
                            current_sls = greedy_predist.merge_sl(best_sls, sl_candid)
                        else:  # non-shortest path
                            current_sls = best_sls + [sl_candid]

                        if sum([sl.cost for sl in current_sls]) > cost_budget:
                            continue
                        current_total_latency, current_sd_paths, current_sls = greedy_predist. \
                            sd_latency_with_sl_residual(srcs, dsts, orig_graph=G, shortest_paths=shortest_paths,
                                                        sls=current_sls)
                        if current_total_latency == float('inf') or current_total_latency > best_total_latency:
                            continue  # no useful sls

                        current_sl_metric = (best_total_latency - current_total_latency) / \
                                            (sum([sl.cost for sl in current_sls]) - sum([sl.cost for sl in best_sls]) + 1e-9)
                        if current_total_latency < float('inf') and best_total_latency > current_total_latency and \
                                ((current_sl_metric < 0 and current_sl_metric < current_best_sl_metric) or
                                 (current_sl_metric > 0 and 0 <= current_best_sl_metric < current_sl_metric)):
                            current_best_sl_metric = current_sl_metric
                            current_best_sls = current_sls
                            current_best_total_latency = current_total_latency
                            current_best_sd_paths = current_sd_paths
            if current_best_sl_metric == 0:
                # no sl found
                break
            best_sls = current_best_sls
            best_total_latency = current_best_total_latency
            best_sd_paths = current_best_sd_paths
            current_sl_set = set([f"{sl.src.id}-{sl.dst.id}" for sl in best_sls])
            current_cost = sum([sl.cost for sl in best_sls])
            print(f"Total latency = {best_total_latency}s, cost = {current_cost}, # of SLs = {len(current_sl_set)}"
                  f", selected sls: [{'#'.join([sl.iid for sl in best_sls])}]")
        return best_sls, best_sd_paths, best_total_latency / len(srcs)

    @staticmethod
    def _general_module_new(NAIVE: bool, shortest_paths: dp_shortest_path, shortest_path_non_sl: dp_shortest_path,
                            G: qGraph, srcs: List[qNode],
                            dsts: List[qNode], max_sl_latency: float, uncovered_sd_pairs: set[int],
                            sd_latency_targets: List[float], sl_index_offset: int,
                            sd_path_with_sls: List[List], IS_LATENCY_OBJECTIVE: bool,
                            cost_budget: float, decoherence_time: float) -> [List[SL], set[int], float]:
        """Only the case where LATENCY is objective has been implemented."""
        best_sls, n = [], len(G.V)
        if IS_LATENCY_OBJECTIVE:
            if not NAIVE:
                sl_costs = greedy_predist._exhaustive_greedy(G, shortest_paths, max_sl_latency)
            else:
                sl_costs = greedy_predist._naive_greedy(G, shortest_paths, srcs, dsts, max_sl_latency)
        else:
            pass

        current_cost = 0.0
        best_total_latency, best_sd_paths, _ = greedy_predist. \
            sd_latency_with_sl_residual(srcs=srcs, dsts=dsts, orig_graph=G, shortest_paths=shortest_paths, sls=best_sls,
                                        is_naive=NAIVE)
        current_sl_set = set()
        while current_cost < cost_budget:

            current_best_sl_metric = 0
            current_best_sls = None
            current_best_total_latency = 0
            current_best_sd_paths = []
            for i in range(n):
                for j in range(i + 1, n):
                    if sl_costs[i][j] == float('inf') or \
                            (IS_LATENCY_OBJECTIVE and f"{G.V[i].id}-{G.V[j].id}" in current_sl_set):
                        continue
                    current_sls = greedy_predist.merge_sl(best_sls, SL(G.V[i], G.V[j],
                                                                       shortest_paths.get_shortest_path_without_update(
                                                                           G.V[i], G.V[j]), sl_costs[i][j]))
                    if sum([sl.cost for sl in current_sls]) > cost_budget:
                        continue
                    current_total_latency, current_sd_paths, current_sls = greedy_predist.\
                        sd_latency_with_sl_residual(srcs, dsts, orig_graph=G, shortest_paths=shortest_paths,
                                                    sls=current_sls, is_naive=NAIVE)
                    current_sl_metric = (best_total_latency - current_total_latency) / \
                                        (sum([sl.cost for sl in current_sls]) - sum([sl.cost for sl in best_sls]) + 1e-9)
                    if current_total_latency < float('inf') and best_total_latency > current_total_latency and \
                            ((current_sl_metric < 0 and current_sl_metric < current_best_sl_metric) or
                             (current_sl_metric > 0 and 0 <= current_best_sl_metric < current_sl_metric)):
                        current_best_sl_metric = current_sl_metric
                        current_best_sls = current_sls
                        current_best_total_latency = current_total_latency
                        current_best_sd_paths = current_sd_paths
            if current_best_sl_metric == 0:
                # no sl found
                break
            best_sls = current_best_sls
            best_total_latency = current_best_total_latency
            best_sd_paths = current_best_sd_paths
            current_sl_set = set([f"{sl.src.id}-{sl.dst.id}" for sl in best_sls])
            current_cost = sum([sl.cost for sl in best_sls])
            print(f"Total latency = {best_total_latency}s, cost = {current_cost}, # of SLs = {len(current_sl_set)}")
        for sd_idx, path in enumerate(best_sd_paths):
            sd_path_with_sls[sd_idx] = path
        return best_sls, uncovered_sd_pairs, best_total_latency / len(srcs)

    @staticmethod
    def _general_module(NAIVE: bool, shortest_paths: dp_shortest_path, shortest_paths_non_sl: dp_shortest_path,
                        G: qGraph, srcs: List[qNode],
                        dsts: List[qNode], max_sl_latency: float, uncovered_sd_pairs: set[int],
                        sd_latency_targets: List[float], sl_index_offset: int,
                        sd_path_with_sls: List[List], IS_LATENCY_OBJECTIVE: bool,
                        cost_budget: float, no_deletion_greedy_version: str) -> [List[SL], set[int]]:
        best_sls, n, current_uncovered_sd_pairs = [], len(G.V), set(uncovered_sd_pairs)
        if not IS_LATENCY_OBJECTIVE:
            if not NAIVE:
                sl_costs = greedy_predist._exhaustive_greedy(G, shortest_paths, max_sl_latency)
            else:
                sl_costs = greedy_predist._naive_greedy(G, shortest_paths, srcs, dsts, max_sl_latency)
        else:
            if not NAIVE:
                sl_costs = greedy_predist._exhaustive_greedy(G, shortest_paths, max_sl_latency)
            else:
                sl_costs = greedy_predist._naive_greedy(G, shortest_paths, srcs, dsts, max_sl_latency)

        current_cost = 0
        current_sl_set = set()
        current_tot_latency, _ = greedy_predist.sd_latency_with_sl(srcs, dsts, shortest_paths, best_sls,
                                                                   IS_NAIVE=NAIVE)
        best_total_latency = current_tot_latency
        while (not IS_LATENCY_OBJECTIVE and current_uncovered_sd_pairs) or (IS_LATENCY_OBJECTIVE and current_cost
                                                                            < cost_budget):
            # different loop condition
            # if IS_LATENCY_OBJECTIVE:
            #     current_tot_latency, best_sls = greedy_predist.sd_latency_with_sl(srcs, dsts, shortest_paths, best_sls,
            #                                                                       IS_NAIVE=NAIVE)
            best_sl_metric = 0
            best_sl_idx = None
            best_sd_candidates = None
            for i in range(n):
                for j in range(i + 1, n):
                    if sl_costs[i][j] == float('inf') or \
                            (IS_LATENCY_OBJECTIVE and (sl_costs[i][j] + current_cost > cost_budget or
                             f"{G.V[i].id}-{G.V[j].id}" in current_sl_set or
                              any([not tree_node.is_disjoint(sl.tree,
                                                             shortest_paths.get_shortest_path_without_update(G.V[i],
                                                                                                             G.V[j]))
                                  for sl in best_sls]))):
                        continue
                    if not IS_LATENCY_OBJECTIVE:
                        sd_improvement = 0
                        # number of (S,d) pairs whose latency will be less than latency_threshold if cost is objective
                        # otherwise is the cumulative time decrease by SL
                        sd_candidates = []
                        for sd_idx in current_uncovered_sd_pairs:
                            latency_without_sl = shortest_paths.get_shortest_path_without_update(srcs[sd_idx],
                                                                                                 dsts[
                                                                                                     sd_idx]).avr_ent_time
                            new_latency, _ = greedy_predist.latency_with_sl(
                                shortest_paths.get_shortest_path_without_update(srcs[sd_idx], G.V[i]),
                                srcs[sd_idx], G.V[i], G.V[j],
                                shortest_paths.get_shortest_path_without_update(G.V[j], dsts[sd_idx]), dsts[sd_idx])
                            new_latency_inverse, _ = greedy_predist.latency_with_sl(
                                shortest_paths.get_shortest_path_without_update(srcs[sd_idx], G.V[j]),
                                srcs[sd_idx], G.V[j], G.V[i],
                                shortest_paths.get_shortest_path_without_update(G.V[i], dsts[sd_idx]), dsts[sd_idx])
                            if min(new_latency, new_latency_inverse) < sd_latency_targets[sd_idx]:
                                sd_improvement += 1
                                sd_candidates.append(sd_idx)
                        if sd_improvement / sl_costs[i][j] > best_sl_metric:
                            best_sl_metric = sd_improvement / sl_costs[i][j]
                            best_sl_idx = (i, j)
                            best_sd_candidates = sd_candidates
                    else:
                        # new latency with adding this sl
                        tot_latency, _ = greedy_predist.sd_latency_with_sl(srcs, dsts, shortest_paths,
                                                                           best_sls +
                                                                           [SL(G.V[i],
                                                                               G.V[j],
                                                                               shortest_paths.get_shortest_path_without_update(
                                                                                   G.V[i], G.V[j]), sl_costs[i][j])],
                                                                           IS_NAIVE=NAIVE)
                        if (no_deletion_greedy_version != "latency" and
                            (current_tot_latency - tot_latency) / sl_costs[i][j] > best_sl_metric) or \
                                (no_deletion_greedy_version == "latency" and
                                 current_tot_latency - tot_latency > best_sl_metric):
                            best_sl_metric = current_tot_latency - tot_latency
                            if no_deletion_greedy_version != "latency":
                                best_sl_metric /= sl_costs[i][j]
                            best_sl_idx = (i, j)
                            best_total_latency = tot_latency

            if best_sl_metric == 0:
                break  # no more solution exists
            best_sls.append(SL(G.V[best_sl_idx[0]], G.V[best_sl_idx[1]],
                               shortest_paths.get_shortest_path(G.V[best_sl_idx[0]], G.V[best_sl_idx[1]], True),
                               sl_costs[best_sl_idx[0]][best_sl_idx[1]]))
            if not IS_LATENCY_OBJECTIVE:
                current_uncovered_sd_pairs -= set(best_sd_candidates)
            else:
                current_cost = sum([sl.cost for sl in best_sls])
            current_sl_set = set([f"{sl.src.id}-{sl.dst.id}" for sl in best_sls])
            current_tot_latency = best_total_latency
            print(f"Total latency = {best_total_latency}s, cost = {current_cost}, # of SLs = {len(current_sl_set)}"
                  f", selected sls: [{'#'.join([sl.iid for sl in best_sls])}]")
        # sls assignment to (s, d) pairs
        for sd_idx in range(len(srcs)):
            if not IS_LATENCY_OBJECTIVE and sd_idx not in uncovered_sd_pairs - current_uncovered_sd_pairs:
                continue
            src, dst = srcs[sd_idx], dsts[sd_idx]
            path = []  # only one path for now
            best_path = (float('inf'), -1, None)
            for sl_idx, sl in enumerate(best_sls):
                if NAIVE and NAIVE_V2 and not tree_node.is_sub_tree(
                        main_tree=shortest_paths.get_shortest_path_without_update(src, dst),
                        sub_tree=shortest_paths.get_shortest_path_without_update(sl.src, sl.dst)):
                    continue
                latency = greedy_predist.latency_with_sl(shortest_paths.get_shortest_path_without_update(src,
                                                                                                         sl.src),
                                                         src, sl.src, sl.dst,
                                                         shortest_paths.get_shortest_path_without_update(sl.dst,
                                                                                                         dst), dst)
                if latency[0] < best_path[0]:
                    best_path = (latency[0], sl_idx + sl_index_offset, latency[1], False)
                # if we use SL inversely
                latency = greedy_predist.latency_with_sl(shortest_paths.get_shortest_path_without_update(src,
                                                                                                         sl.dst),
                                                         src, sl.dst, sl.src,
                                                         shortest_paths.get_shortest_path_without_update(sl.src,
                                                                                                         dst), dst)
                if latency[0] < best_path[0]:
                    best_path = (latency[0], sl_idx + sl_index_offset, latency[1], True)
            if best_path[0] < shortest_paths.get_shortest_path_without_update(src, dst).avr_ent_time:
                # using an SL is useful
                if not best_path[3]:
                    sl_src, sl_dst = best_sls[best_path[1] - sl_index_offset].src, \
                                     best_sls[best_path[1] - sl_index_offset].dst
                else:
                    sl_src, sl_dst = best_sls[best_path[1] - sl_index_offset].dst, \
                                     best_sls[best_path[1] - sl_index_offset].src
                if src is not sl_src:
                    path.append(shortest_paths_non_sl.get_shortest_path(src, sl_src))
                path.append(best_path[1:])
                if dst is not sl_dst:
                    path.append(shortest_paths_non_sl.get_shortest_path(sl_dst, dst))
                sd_path_with_sls[sd_idx].append(path)
            else:
                # use shortest path without sl
                sd_path_with_sls[sd_idx].append([shortest_paths_non_sl.get_shortest_path(src, dst)])
        return best_sls, best_total_latency/len(srcs), current_uncovered_sd_pairs

    @staticmethod
    def _exhaustive_greedy(G: qGraph, shortest_paths: dp_shortest_path, max_sl_latency: float) -> List[List[float]]:
        n = len(G.V)
        sl_costs = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sl_costs[i][j] = greedy_predist.sl_cost(shortest_paths.get_shortest_path_without_update(G.V[i], G.V[j]),
                                                        max_sl_latency)
        return sl_costs

    @staticmethod
    def _naive_greedy(G: qGraph, shortest_paths: dp_shortest_path, srcs: List[qNode], dsts: List[qNode],
                      max_sl_Latency: float) -> \
            List[List[float]]:
        n = len(G.V)
        sl_costs = [[float('inf')] * n for _ in range(n)]
        used_nodes, used_edges = set(), set()
        for src, dst in zip(srcs, dsts):
            nodes, edges = tree_node.used_nodes_edges(shortest_paths.get_shortest_path_without_update(src, dst))
            used_nodes, used_edges = used_nodes.union(nodes), used_edges.union(edges)
        for i in range(n):
            for j in range(i + 1, n):
                sl_used_node, sl_used_edges = tree_node.used_nodes_edges(
                    shortest_paths.get_shortest_path_without_update(G.V[i], G.V[j]))
                if len(sl_used_node - used_nodes) > 0 or len(sl_used_edges - used_edges) > 0:
                    continue
                sl_costs[i][j] = greedy_predist.sl_cost(shortest_paths.get_shortest_path_without_update(G.V[i], G.V[j]),
                                                        max_sl_Latency)

        return sl_costs

    @property
    def get_final_paths(self) -> List:
        return self._sd_path_with_sls

    @property
    def get_sls(self) -> List[SL]:
        return self._best_sls

    @staticmethod
    def sl_cost(node: tree_node, max_sl_latency: float, success_eps: float = 1.0) -> float:
        if node is None or node.data is None or node.avr_ent_time > max_sl_latency:
            return float('inf')
        if isinstance(node.data, qChannel):
            return success_eps / node.data.link_success_rate
        return greedy_predist.sl_cost(node.left, max_sl_latency * node.data.bsm_success_rate,
                                      success_eps / node.data.bsm_success_rate) + \
               greedy_predist.sl_cost(node.right, max_sl_latency * node.data.bsm_success_rate,
                                      success_eps / node.data.bsm_success_rate)

    @staticmethod
    def sl_cost2(node: tree_node, max_sl_latency: float) -> float:
        return float('inf') if (node.data is None or node.avr_ent_time > max_sl_latency) else node.avr_ent_time

    @staticmethod
    def sl_cost1(node: tree_node, max_sl_latency: float) -> float:
        """A function to calculate a super link's cost which is sum of all the underlying EP rates based on the
        super link's rate"""
        if node.data is None or node.avr_ent_time > max_sl_latency:
            return float('inf')
        if isinstance(node.data, qChannel):
            return node.data.avr_ent_time
            # return node.data.max_channel_capacity
            # return sl_rate
        # children_time = (1.0 / sl_rate * node.data.bsm_success_rate - node.data.bsm_time - node.classical_time) / 1.5
        return greedy_predist.sl_cost(node.left, max_sl_latency) + greedy_predist.sl_cost(node.right, max_sl_latency)
        # return greedy_predist.sl_cost(node.left, 1.0 / children_time) + \
        #        greedy_predist.sl_cost(node.right, 1.0 / children_time)

    @staticmethod
    def latency_with_sl(p1: tree_node, p1_left_node: qNode,
                        sl_left_node: qNode, sl_right_node: qNode,
                        p2: tree_node, p2_right_node: qNode) -> (float, bool):
        """Calculate new latency assuming sl end points have enough EPs to do BSM for the whole path p1 ~ sl ~ p2
        gets connected"""
        # SL is useless
        if p1_left_node.id == sl_right_node.id or p2_right_node.id == sl_left_node.id or p1 is None or p2 is None:
            return (float('inf'), True)
        # check if there is cycle when adding SL
        p1_nodes, _ = tree_node.used_nodes_edges(p1)
        p2_nodes, _ = tree_node.used_nodes_edges(p2)
        if sl_right_node.id in p1_nodes or sl_left_node.id in p2_nodes:
            return (float('inf'), True)
        # p1 = sl
        if p1_left_node.id == sl_left_node.id and p2_right_node.id == sl_right_node.id:
            return (0, False)

        if p1_left_node.id == sl_left_node.id:
            return ((p2.avr_ent_time + sl_right_node.bsm_time +
                     max(get_classical_time(sl_right_node.loc.distance(p1_left_node.loc)),
                         get_classical_time(sl_right_node.loc.distance(p2_right_node.loc)))) / \
                    sl_right_node.bsm_success_rate, False)
        # p2 = sl
        if sl_right_node.id == p2_right_node.id:
            return ((p1.avr_ent_time + sl_left_node.bsm_time +
                     max(get_classical_time(sl_left_node.loc.distance(p1_left_node.loc)),
                         get_classical_time(sl_left_node.loc.distance(sl_right_node.loc))))
                    / sl_left_node.bsm_success_rate, True)
        # sl is neither p1 nor p2
        # first do sl's left and then sl's right (sl's right as root)
        latency = (p1.avr_ent_time + sl_left_node.bsm_time +
                   max(get_classical_time(sl_left_node.loc.distance(p1_left_node.loc)),
                       get_classical_time(sl_left_node.loc.distance(sl_right_node.loc)))) \
                  / sl_left_node.bsm_success_rate
        latency = (1.5 * max(latency, p2.avr_ent_time) + sl_right_node.bsm_time +
                   max(get_classical_time(sl_right_node.loc.distance(p1_left_node.loc)),
                       get_classical_time(
                           sl_right_node.loc.distance(p2_right_node.loc)))) / sl_right_node.bsm_success_rate
        new_latency = latency

        # first do sl's right then sl's left (sl's left as root)
        latency = (p2.avr_ent_time + sl_right_node.bsm_time +
                   max(get_classical_time(sl_right_node.loc.distance(p2_right_node.loc)),
                       get_classical_time(
                           sl_right_node.loc.distance(sl_left_node.loc)))) / sl_right_node.bsm_success_rate
        latency = (1.5 * max(latency, p1.avr_ent_time) + sl_left_node.bsm_time +
                   max(get_classical_time(sl_left_node.loc.distance(p2_right_node.loc)),
                       get_classical_time(sl_left_node.loc.distance(p1_left_node.loc)))) / sl_left_node.bsm_success_rate

        return (min(latency, new_latency), new_latency < latency)

    @staticmethod
    def sd_latency_with_sl(srcs: List[qNode], dsts: List[qNode], shortest_paths: dp_shortest_path,
                           sls: List[SL], IS_NAIVE:bool=False, NAIVE_V2:bool=NAIVE_V2) -> (float, List[SL]):
        """This function returns the total latency of (s,d)-pairs with(or without) a set of SLs and
        returns a list of only USED SLs."""
        total_latency = 0
        new_sls = []
        used_sl = set()
        for src, dst in zip(srcs, dsts):
            sd_latency = shortest_paths.get_shortest_path_without_update(src, dst).avr_ent_time
            best_sl = ""
            for sl in sls:
                if IS_NAIVE and NAIVE_V2 and not tree_node.is_sub_tree(
                        main_tree=shortest_paths.get_shortest_path_without_update(src, dst),
                        sub_tree=shortest_paths.get_shortest_path_without_update(sl.src, sl.dst)):
                    continue
                latency_sl, _ = greedy_predist.latency_with_sl(shortest_paths.get_shortest_path_without_update(src,
                                                                                                               sl.src),
                                                               src, sl.src, sl.dst,
                                                               shortest_paths.get_shortest_path_without_update(sl.dst,
                                                                                                               dst),
                                                               dst)
                latency_sl_inv, _ = greedy_predist.latency_with_sl(
                    shortest_paths.get_shortest_path_without_update(src, sl.dst), src, sl.dst, sl.src,
                    shortest_paths.get_shortest_path_without_update(sl.src, dst), dst)
                if min(latency_sl, latency_sl_inv) < sd_latency:
                    sd_latency = min(latency_sl, latency_sl_inv)
                    best_sl = f"{sl.src.id}-{sl.dst.id}"
            total_latency += sd_latency
            if best_sl != "":
                used_sl.add(best_sl)

        for sl in sls:
            if f"{sl.src.id}-{sl.dst.id}" in used_sl:
                new_sls.append(sl)
        return total_latency, new_sls

    @staticmethod
    def merge_sl(current_sls: List[SL], new_sl: SL) -> List[SL]:
        """A function to merge a new SL with current SL set with some consideration (e.g. being disjoint,
        or sharing resource)"""
        if not current_sls or len(current_sls) == 0:
            return [new_sl]
        new_sls = []
        # sorted(current_sls, key=lambda x: x.cost, reverse=True)
        for sl in current_sls:
            if tree_node.is_disjoint(sl.tree, new_sl.tree):
                new_sls.append(sl)
        return new_sls + [new_sl]

    @staticmethod
    def sd_latency_with_sl_residual(srcs: List[qNode], dsts: List[qNode], orig_graph: qGraph,
                                    shortest_paths: dp_shortest_path,
                                    sls: List[SL], is_naive: bool = False) -> (float, List):
        """Calculate sum of (s-d) latency given a set of SL. return useful SLs"""
        total_latency = 0.0
        sd_paths = [[] for _ in range(len(srcs))]
        # residual_graph = orig_graph.clone()
        # for sl in sls:
        #     greedy_predist.residual_graph(residual_graph, sl.src, sl.dst, [sl.tree])
        #
        # shortest_paths = dp_shortest_path(residual_graph, decoherence_time=decoherence_time)
        new_sls = []
        used_sls = {}
        for sd_idx in range(len(srcs)):
            src, dst = orig_graph.get_node(srcs[sd_idx].id), orig_graph.get_node(dsts[sd_idx].id)
            # src, dst = srcs[sd_idx], dsts[sd_idx]
            path = []  # only one path for now
            best_path = (float('inf'), -1, None)
            for sl_idx, sl in enumerate(sls):
                sl_src, sl_dst = orig_graph.get_node(sl.src.id), orig_graph.get_node(sl.dst.id)
                # sl_src, sl_dst = sl.src, sl.dst
                if is_naive and not tree_node.is_sub_tree(
                        main_tree=shortest_paths.get_shortest_path_without_update(src, dst),
                        sub_tree=shortest_paths.get_shortest_path_without_update(sl_src, sl_dst)):
                    continue

                latency = greedy_predist.latency_with_sl(shortest_paths.get_shortest_path_without_update(src, sl_src),
                                                         src, sl_src, sl_dst,
                                                         shortest_paths.get_shortest_path_without_update(sl_dst, dst),
                                                         dst)
                if latency[0] < best_path[0]:
                    best_path = (latency[0], sl_idx, latency[1], False)
                # if we use SL inversely
                latency = greedy_predist.latency_with_sl(shortest_paths.get_shortest_path_without_update(src, sl_dst),
                                                         src, sl_dst, sl_src,
                                                         shortest_paths.get_shortest_path_without_update(sl_src,
                                                                                                         dst), dst)
                if latency[0] < best_path[0]:
                    best_path = (latency[0], sl_idx, latency[1], True)
            sp = shortest_paths.get_shortest_path_without_update(src, dst)
            if best_path[0] < (float('inf') if sp is None else sp.avr_ent_time):
                if best_path[1] not in used_sls:
                    used_sls[best_path[1]] = len(new_sls)
                    new_sls.append(sls[best_path[1]])
                new_idx = used_sls[best_path[1]]
                best_path = (best_path[0], new_idx, best_path[2], best_path[3])

                # using an SL is useful
                if not best_path[3]:
                    sl_src, sl_dst = orig_graph.get_node(new_sls[best_path[1]].src.id), \
                                     orig_graph.get_node(new_sls[best_path[1]].dst.id)
                    # sl_src, sl_dst = sls[best_path[1]].src, sls[best_path[1]].dst
                else:
                    sl_src, sl_dst = orig_graph.get_node(new_sls[best_path[1]].dst.id), \
                                     orig_graph.get_node(new_sls[best_path[1]].src.id)
                    # sl_src, sl_dst = sls[best_path[1]].dst, sls[best_path[1]].src
                if src is not sl_src:
                    # src to sl shortest path
                    path.append(shortest_paths.get_shortest_path(src, sl_src, True))
                path.append(best_path[1:])
                if dst is not sl_dst:
                    # sl to dst shortest path
                    path.append(shortest_paths.get_shortest_path(sl_dst, dst, True))
                sd_paths[sd_idx].append(path)
                total_latency += best_path[0]
            else:
                # use shortest path without sl
                sp = shortest_paths.get_shortest_path(src, dst)
                sd_paths[sd_idx].append([sp])
                if not sp:
                    total_latency += float('inf')
                else:
                    total_latency += sd_paths[sd_idx][-1][-1].avr_ent_time

        # some sls are useless
        # TODO Some SLs may not be used; we should remove them and repeat
        return total_latency, sd_paths, new_sls

    @staticmethod
    def residual_graph(graph: qGraph, src: qNode, dst: qNode, trees: List[tree_node], non_shortest: bool = False):
        def _helper(graph: qGraph, src: qNode, dst: qNode, node: tree_node, non_shortest: bool = False):
            if node is None:
                return
            if isinstance(node.data, qChannel):
                this, other = graph.get_node(node.data.this.id), graph.get_node(node.data.other.id)
                channel = graph.get_edge(node.data.this.id, node.data.other.id)
                if not non_shortest:
                    channel_flow = 1.0 / node.avr_ent_time
                    channel.current_flow += channel_flow  # update channel's capacity
                    # updating nodes' capacity
                    this.used_capacity += channel_flow / channel.link_success_rate
                    other.used_capacity += channel_flow / channel.link_success_rate
                else:
                    channel.current_flow = channel.max_channel_capacity
                    if this.id == src.id:
                        this.used_capacity = this.max_capacity
                    else:
                        this.used_capacity += this.max_capacity / 2
                    if other.id == dst.id:
                        other.used_capacity = other.max_capacity
                    else:
                        other.used_capacity += other.max_capacity / 2
                # updating memories
                this.memory -= 1
                if this.memory <= 1 and this.id != src.id and this.id != dst.id:
                    this.memory = 0  # intermediate nodes should have at least two memories
                other.memory -= 1

                return
            _helper(graph, src, dst, node.left)  # go left
            _helper(graph, src, dst, node.right)  # go right

        for tree in trees:
            if isinstance(tree, tree_node):
                _helper(graph, src, dst, tree, non_shortest)