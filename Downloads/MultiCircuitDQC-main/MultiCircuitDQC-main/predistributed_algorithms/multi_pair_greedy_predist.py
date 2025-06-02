import functools
import sys

sys.path.append("..")

from commons.qGraph import qGraph
from commons.qNode import qNode
from commons.qChannel import qChannel
from typing import List
from routing_algorithms.dp_shortest_path import dp_shortest_path
from routing_algorithms.dp_alternate import DP_Alt
from commons.tree_node import tree_node
from commons.tools import get_classical_time
from predistributed_algorithms.super_link import SL
from routing_algorithms.max_flow import max_flow
from .greedy_predist import greedy_predist, COST_BUDGET_DEFAULT
import cProfile as profile


class multi_pair_greedy_predist:

    def __init__(self, G: qGraph, srcs: List[qNode], dsts: List[qNode], latency_threshold: float, IS_NAIVE: bool,
                 max_sl_latency: float, IS_LATENCY_OBJECTIVE: bool, decoherence_time: float,
                 FIXED_LATENCY: bool = False, cost_budget: float = COST_BUDGET_DEFAULT,
                 NON_SHORTEST_SLS: bool = True):
        if len(srcs) != len(dsts):
            raise ValueError("The number of sources and destinations are not equal!")
        self._sd_path_with_sls = [[] for _ in range(len(srcs))]
        self._best_sls = []
        self._average_latency = float('inf')

        shortest_paths = dp_shortest_path(G, decoherence_time=decoherence_time)  # only used for SLs tree

        if not IS_LATENCY_OBJECTIVE:
            # objective = cost & constraint = latency
            # holds a list for each sd, each for a path, latency after assigning an SL
            sd_latency_targets = [latency_threshold] * len(srcs) if FIXED_LATENCY else \
                [shortest_paths.get_shortest_path_without_update(src, dst).avr_ent_time * latency_threshold
                 for src, dst in zip(srcs, dsts)]
            uncovered_sd_pairs = set([i for i in range(len(srcs)) if
                                      shortest_paths.get_shortest_path_without_update(srcs[i], dsts[i]).avr_ent_time >
                                      sd_latency_targets[i]])
            if len(uncovered_sd_pairs) < len(srcs):
                shortest_paths_non_sl = dp_shortest_path(G, decoherence_time=decoherence_time)
                for sd_idx in set(list(range(len(srcs)))) - uncovered_sd_pairs:  # use sp for those that do not need SL
                    self._sd_path_with_sls[sd_idx].append([shortest_paths_non_sl.get_shortest_path(srcs[sd_idx],
                                                                                                   dsts[sd_idx])])
        else:
            # objective = latency & constraint = cost
            uncovered_sd_pairs = set([i for i in range(len(srcs))])

        if IS_NAIVE:
            sd_trees = []
            path_all_pair_sp = []
            sd_latencies = []
            updated_latencies = []
            for src, dst in zip(srcs, dsts):
                graph_tmp = G.clone()
                multi_pair = max_flow(G=graph_tmp, srcs=[graph_tmp.get_node(src.id)], dsts=[graph_tmp.get_node(dst.id)],
                                      decoherence_time=decoherence_time, multiple_pairs_search="best",
                                      PATH_ALL_PAIR=IS_NAIVE)
                sd_latencies.append(1.0 / multi_pair.get_flow_total)
                sd_trees.append(multi_pair.get_flows(0))
                updated_latencies.append([1.0 / sd_trees[-1][i][0] for i in range(len(sd_trees[-1]))])
                path_all_pair_sp.append(multi_pair.get_path_all_pair_shortest_path(0))
            uncovered_sd_pairs = self._naive_approach(srcs=srcs, dsts=dsts, sl_shortest_paths=shortest_paths,
                                                      sl_graph=G,
                                                      sd_latency_targets=sd_latency_targets if not IS_LATENCY_OBJECTIVE
                                                      else [],
                                                      updated_latency=updated_latencies,
                                                      sd_trees=sd_trees, path_all_pair_sp=path_all_pair_sp,
                                                      uncovered_sd_pairs=uncovered_sd_pairs,
                                                      max_sl_latency=max_sl_latency,
                                                      is_latency_objective=IS_LATENCY_OBJECTIVE,
                                                      cost_budget=cost_budget)
            if not IS_LATENCY_OBJECTIVE:
                # another round of finding "bad" SLs with high latency
                if uncovered_sd_pairs:
                    self._best_sls, self._sd_path_with_sls = multi_pair_greedy_predist._drop_unused_sls(self._best_sls,
                                                                                                        self._sd_path_with_sls)
                    uncovered_sd_pairs = self._naive_approach(srcs=srcs, dsts=dsts, sl_shortest_paths=shortest_paths,
                                                              sl_graph=G, sd_latency_targets=sd_latency_targets,
                                                              updated_latency=updated_latencies,
                                                              sd_trees=sd_trees, path_all_pair_sp=path_all_pair_sp,
                                                              uncovered_sd_pairs=uncovered_sd_pairs,
                                                              max_sl_latency=float('inf'),
                                                              is_latency_objective=False,
                                                              cost_budget=cost_budget)
                self._best_sls, self._sd_path_with_sls = multi_pair_greedy_predist._drop_unused_sls(self._best_sls,
                                                                                                    self._sd_path_with_sls)
                for sd_idx in uncovered_sd_pairs:  # assigning just whole slp path
                    self._sd_path_with_sls[sd_idx].append([(len(self._best_sls), False, False)])
                    sd_shortest_path = shortest_paths.get_shortest_path(srcs[sd_idx], dsts[sd_idx])
                    self._best_sls.append(SL(srcs[sd_idx], dsts[sd_idx],
                                             sd_shortest_path,
                                             greedy_predist.sl_cost(sd_shortest_path, float('inf'))))

        else:
            if False:
                uncovered_sd_pairs = self._iter_approach(srcs=srcs, dsts=dsts, sl_shortest_paths=shortest_paths,
                                                         sd_latency_targets=sd_latency_targets
                                                         if not IS_LATENCY_OBJECTIVE else [],
                                                         orig_graph=G,
                                                         uncovered_sd_pairs=uncovered_sd_pairs,
                                                         max_sl_latency=max_sl_latency,
                                                         is_latency_objective=IS_LATENCY_OBJECTIVE,
                                                         cost_budget=cost_budget,
                                                         decoherence_time=decoherence_time)
                if not IS_LATENCY_OBJECTIVE and uncovered_sd_pairs:
                    # another round of finding "bad" SLS with high latencies
                    self._best_sls, self._sd_path_with_sls = multi_pair_greedy_predist._drop_unused_sls(self._best_sls,
                                                                                                        self._sd_path_with_sls)
                    uncovered_sd_pairs = self._iter_approach(srcs=srcs, dsts=dsts, sl_shortest_paths=shortest_paths,
                                                             sd_latency_targets=sd_latency_targets, orig_graph=G,
                                                             uncovered_sd_pairs=uncovered_sd_pairs,
                                                             max_sl_latency=float('inf'),
                                                             is_latency_objective=IS_LATENCY_OBJECTIVE,
                                                             cost_budget=cost_budget,
                                                             decoherence_time=decoherence_time)
                if not IS_LATENCY_OBJECTIVE and len(uncovered_sd_pairs):
                    print(f"{len(uncovered_sd_pairs)} of (s,d) pairs couldn't be satisfied!")
            else:
                self._best_sls, self._sd_path_with_sls, self._average_latency = multi_pair_greedy_predist.\
                    _iter_advanced_approach_v2(G=G, srcs=srcs, dsts=dsts, max_sl_latency=max_sl_latency,
                                               cost_budget=cost_budget, decoherence_time=decoherence_time,
                                               shortest_paths=shortest_paths,
                                               non_shortest_sls=NON_SHORTEST_SLS)

    @property
    def average_latency(self) -> float:
        return self._average_latency

    def _iter_approach(self, srcs: List[qNode], dsts: List[qNode], sl_shortest_paths: dp_shortest_path,
                       sd_latency_targets: List[float], orig_graph: qGraph, uncovered_sd_pairs: set[int],
                       max_sl_latency: float, is_latency_objective: bool, cost_budget: float, decoherence_time: float) \
            -> set[int]:
        n = len(orig_graph.V)

        picked_sls = {idx: f"{sl.src.id}_{sl.dst.id}" for idx, sl in enumerate(self._best_sls)}
        current_uncovered_sd_pairs = set(uncovered_sd_pairs)
        sl_costs = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sl_costs[i][j] = greedy_predist.sl_cost(
                    sl_shortest_paths.get_shortest_path_without_update(orig_graph.V[i], orig_graph.V[j]),
                    max_sl_latency)

        residual_graphs = {sd_index: orig_graph.clone() for sd_index in current_uncovered_sd_pairs}
        best_used_sls = {sd_index: set() for sd_index in current_uncovered_sd_pairs}

        sd_latencies = {sd_index: sl_shortest_paths.get_shortest_path_without_update(srcs[sd_index],
                                                                                     dsts[sd_index]).avr_ent_time for
                        sd_index in current_uncovered_sd_pairs}
        NO_PATH_FOUND = False
        current_cost = 0.0
        while current_uncovered_sd_pairs:
            # find shortest paths from srcs -> dsts based on their own residual graph
            all_pair_sps = {i: dp_shortest_path(residual_graphs[i], decoherence_time=decoherence_time)
                            for i in current_uncovered_sd_pairs}
            iter_uncovered_sd_pairs = set(current_uncovered_sd_pairs)  # those sd pairs that were not assigned an SL
            while iter_uncovered_sd_pairs:
                max_latency_reduction, best_sl_index, best_sd_candidates = 0, -1, None
                for i in range(n):
                    for j in range(i + 1, n):
                        if sl_costs[i][j] == float('inf') or (is_latency_objective and sl_costs[i][j] + current_cost >
                                                              cost_budget):
                            continue
                        sl_left_node, sl_right_node = orig_graph.get_node(i), orig_graph.get_node(j)
                        sl_latency_reduction, sl_sd_candidates = 0, {}
                        for sd_id in iter_uncovered_sd_pairs:
                            sd_sps = all_pair_sps[sd_id]
                            src_id, dst_id = srcs[sd_id].id, dsts[sd_id].id
                            if f"{i}_{j}" in best_used_sls[sd_id]:
                                continue
                            src, dst = residual_graphs[sd_id].get_node(src_id), residual_graphs[sd_id].get_node(dst_id)
                            sl_left_res_node = residual_graphs[sd_id].get_node(i)
                            sl_right_res_node = residual_graphs[sd_id].get_node(j)
                            # src -> sl_left -> sl_right -> dst
                            norm_late = greedy_predist.latency_with_sl(sl_left_node=sl_left_node,
                                                                       sl_right_node=sl_right_node,
                                                                       p1_left_node=srcs[sd_id],
                                                                       p2_right_node=dsts[sd_id],
                                                                       p1=sd_sps.get_shortest_path_without_update(
                                                                           src, sl_left_res_node),
                                                                       p2=sd_sps.get_shortest_path_without_update(
                                                                           sl_right_res_node, dst))
                            # src -> sl_right -> sl_left -> dst
                            inv_late = greedy_predist.latency_with_sl(sl_left_node=sl_right_node,
                                                                      sl_right_node=sl_left_node,
                                                                      p1_left_node=srcs[sd_id],
                                                                      p2_right_node=dsts[sd_id],
                                                                      p1=sd_sps.get_shortest_path_without_update(
                                                                          src, sl_right_res_node),
                                                                      p2=sd_sps.get_shortest_path_without_update(
                                                                          sl_left_res_node, dst))
                            INVERTED = inv_late[0] < norm_late[0]
                            best_late = inv_late if INVERTED else norm_late
                            if len(self._sd_path_with_sls[sd_id]) == 0:
                                # the first path add for this sd
                                if best_late[0] < sd_latencies[sd_id]:
                                    sl_latency_reduction += (sd_latencies[sd_id] - best_late[0]) / sd_latencies[sd_id]
                                    sl_sd_candidates[sd_id] = (*best_late, INVERTED)
                            else:
                                # not the second path, new latency should be calculated
                                sp_without_sl = sd_sps.get_shortest_path_without_update(src, dst)
                                if (sp_without_sl is not None and best_late[0] < sp_without_sl.avr_ent_time) or \
                                        (sp_without_sl is None and best_late[0] != float('inf')):
                                    # this sl is useful
                                    late_with_sl = 1.0 / (1.0 / sd_latencies[sd_id] + 1.0 / best_late[0]) \
                                        if best_late[0] != 0 else 0
                                    sl_latency_reduction += (sd_latencies[sd_id] - late_with_sl) / sd_latencies[sd_id]
                                    sl_sd_candidates[sd_id] = (*best_late, INVERTED)
                        # check if we need to update the best picked sl
                        if sl_latency_reduction / sl_costs[i][j] > max_latency_reduction:
                            best_sl_index = (i, j)
                            max_latency_reduction = sl_latency_reduction / sl_costs[i][j]
                            best_sd_candidates = sl_sd_candidates
                # create an SL if the new one was not previously created
                if best_sl_index == -1:  # no SL was found
                    NO_PATH_FOUND = True
                    break
                best_sl_key = f"{best_sl_index[0]}_{best_sl_index[1]}"
                if best_sl_key not in picked_sls:
                    picked_sls[best_sl_key] = len(self._best_sls)
                    sl_left_node, sl_right_node = orig_graph.get_node(best_sl_index[0]), orig_graph.get_node(
                        best_sl_index[1])
                    # sl_shortest_paths.get_shortest_path_without_update(sl_left_node,  # updating SL rate
                    #                                                    sl_right_node).avr_ent_time = 1.0 /\
                    #                                                                                  greedy_predist.SL_RATE
                    self._best_sls.append(SL(src=sl_left_node, dst=sl_right_node,
                                             tree=sl_shortest_paths.get_shortest_path(sl_left_node, sl_right_node,
                                                                                      True),
                                             cost=sl_costs[best_sl_index[0]][best_sl_index[1]]))
                    current_cost += self._best_sls[-1].cost

                for sd_id, late in best_sd_candidates.items():
                    iter_uncovered_sd_pairs.remove(sd_id)  # each sd path will be given one SL each round
                    best_used_sls[sd_id].add(best_sl_key)  # each SL will be used once for each sd
                    if len(self._sd_path_with_sls[sd_id]) == 0:
                        sd_latencies[sd_id] = late[0]  # no path before
                    else:  # updating latency
                        sd_latencies[sd_id] = 1.0 / (1.0 / sd_latencies[sd_id] + 1.0 / late[0]) if late[0] != 0 else 0
                    src, dst = srcs[sd_id], dsts[sd_id]
                    # constructing the path with SL
                    path = []
                    if not late[2]:
                        sl_src, sl_dst = self._best_sls[picked_sls[best_sl_key]].src, self._best_sls[
                            picked_sls[best_sl_key]].dst
                    else:  # USING SL inverted
                        sl_src, sl_dst = self._best_sls[picked_sls[best_sl_key]].dst, self._best_sls[
                            picked_sls[best_sl_key]].src
                    if src is not sl_src:  # use get-shortest_path in case to do throttling
                        path.append(all_pair_sps[sd_id].get_shortest_path(
                            residual_graphs[sd_id].get_node(src.id), residual_graphs[sd_id].get_node(sl_src.id)))
                    path.append((picked_sls[best_sl_key], late[1], late[2]))
                    if dst is not sl_dst:
                        path.append(all_pair_sps[sd_id].get_shortest_path(
                            residual_graphs[sd_id].get_node(sl_dst.id), residual_graphs[sd_id].get_node(dst.id)))
                    self._sd_path_with_sls[sd_id].append(path)

                    if not is_latency_objective and sd_latencies[sd_id] < sd_latency_targets[sd_id]:
                        current_uncovered_sd_pairs.remove(sd_id)

                    greedy_predist.residual_graph(residual_graphs[sd_id], src=residual_graphs[sd_id].get_node(src.id),
                                                  dst=residual_graphs[sd_id].get_node(dst.id), trees=path)

            if NO_PATH_FOUND:
                break
        if not is_latency_objective:
            for sd_idx in current_uncovered_sd_pairs:
                self._sd_path_with_sls[sd_idx] = []
            return current_uncovered_sd_pairs
        # latency is objective
        # add as many as possible paths without SLs based on residual graph
        for sd_id in uncovered_sd_pairs:
            multi_pair = max_flow(G=residual_graphs[sd_id], srcs=[residual_graphs[sd_id].get_node(srcs[sd_id].id)],
                                  dsts=[residual_graphs[sd_id].get_node(dsts[sd_id].id)],
                                  decoherence_time=decoherence_time, multiple_pairs_search="best")
            for tree in multi_pair.get_flows(0):
                self._sd_path_with_sls[sd_id].append([tree[1]])
        return []  # return none (all sd pairs where given some(or none) based on cost budget when latency is objective)

    def _naive_approach(self, srcs: List[qNode], dsts: List[qNode],
                        sl_shortest_paths: dp_shortest_path, sl_graph: qGraph,
                        sd_latency_targets: List[float], updated_latency: List[List[float]],
                        sd_trees: List[List[List]],
                        path_all_pair_sp: List[List[List[dict]]], uncovered_sd_pairs: set[int],
                        max_sl_latency: float, is_latency_objective: bool,
                        cost_budget: float) -> set[int]:
        """First do DP-iTER and then tries to find the best SLs along their paths"""
        # check which pairs have latency greater than latency threshold
        n = len(sl_graph.V)
        current_uncovered_sd_pairs = set(uncovered_sd_pairs)
        non_assigned_path = {sd_idx: set(range(len(sd_trees[sd_idx]))) for sd_idx in uncovered_sd_pairs}
        picked_sls = {idx: f"{sl.src.id}_{sl.dst.id}" for idx, sl in enumerate(self._best_sls)}
        best_used_sls = {sd_idx: set() for sd_idx in
                         uncovered_sd_pairs}  # to avoid using an SL multiple time for an (s-d) pair

        sl_costs = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sl_costs[i][j] = greedy_predist.sl_cost(
                    sl_shortest_paths.get_shortest_path_without_update(sl_graph.V[i], sl_graph.V[j]),
                    max_sl_latency)

        current_cost = 0.0
        while current_uncovered_sd_pairs:
            # if any([len(non_assigned_path[sd_idx]) == 0 for sd_idx in uncovered_sd_pairs]):
            #     break  # all paths of an s-d were assigned an SL but it hasn't been satisfied; no solution
            max_latency_reduction, best_sl_index, best_sd_candidates = 0, -1, None
            for i in range(n):
                for j in range(i + 1, n):
                    if sl_costs[i][j] == float('inf') or (is_latency_objective and sl_costs[i][j] + current_cost >
                                                          cost_budget):
                        continue
                    sl_left_node, sl_right_node = sl_graph.get_node(i), sl_graph.get_node(j)
                    latency_reduction, sd_candidates = 0, {}
                    for sd_id in current_uncovered_sd_pairs:
                        if f"{i}_{j}" in best_used_sls[sd_id]:
                            continue
                        sd_best_reduction, best_path_id = 0, None
                        src_id, dst_id = srcs[sd_id].id, dsts[sd_id].id
                        for path_id in non_assigned_path[sd_id]:
                            if (i in path_all_pair_sp[sd_id][path_id][0] and j in path_all_pair_sp[sd_id][path_id][
                                0]) or \
                                    ((i == src_id or i == dst_id) and j in path_all_pair_sp[sd_id][path_id][0]) or \
                                    ((j == src_id or j == dst_id) and i in path_all_pair_sp[sd_id][path_id][0]):
                                # src -> sl_left -> sl_right -> dst
                                norm_late = greedy_predist.latency_with_sl(sl_left_node=sl_left_node,
                                                                           sl_right_node=sl_right_node,
                                                                           p1_left_node=srcs[sd_id],
                                                                           p2_right_node=dsts[sd_id],
                                                                           p1=path_all_pair_sp[sd_id][path_id][0][i]
                                                                           if i in path_all_pair_sp[sd_id][path_id][0]
                                                                           else None,
                                                                           p2=path_all_pair_sp[sd_id][path_id][1][j]
                                                                           if j in path_all_pair_sp[sd_id][path_id][1]
                                                                           else None)
                                # src -> sl_right -> sl_left -> dst
                                inv_late = greedy_predist.latency_with_sl(sl_left_node=sl_right_node,
                                                                          sl_right_node=sl_left_node,
                                                                          p1_left_node=srcs[sd_id],
                                                                          p2_right_node=dsts[sd_id],
                                                                          p1=path_all_pair_sp[sd_id][path_id][0][j]
                                                                          if j in path_all_pair_sp[sd_id][path_id][0]
                                                                          else None,
                                                                          p2=path_all_pair_sp[sd_id][path_id][1][i]
                                                                          if i in path_all_pair_sp[sd_id][path_id][1]
                                                                          else None)
                                INVERTED = inv_late[0] < norm_late[0]
                                new_sd_latencies = updated_latency[sd_id][:path_id] + [min(inv_late[0], norm_late[0])] \
                                                   + updated_latency[sd_id][path_id + 1:]
                                if 1.0 / sum([1 / x for x in updated_latency[sd_id]]) - \
                                        1.0 / sum([1 / x for x in new_sd_latencies]) > sd_best_reduction:
                                    # if 1.0 / sd_trees[sd_id][path_id][0] - min(inv_late[0], norm_late[0]) >\
                                    #         sd_best_reduction:
                                    sd_best_reduction = 1.0 / sum([1 / x for x in updated_latency[sd_id]]) - \
                                                        1.0 / sum([1 / x for x in new_sd_latencies])
                                    best_path_id = (path_id, *(inv_late if INVERTED else norm_late), INVERTED)
                        latency_reduction += sd_best_reduction
                        if best_path_id is not None:
                            sd_candidates[sd_id] = best_path_id
                    if latency_reduction / sl_costs[i][j] > max_latency_reduction:
                        max_latency_reduction = latency_reduction / sl_costs[i][j]
                        best_sl_index = (i, j)
                        best_sd_candidates = sd_candidates
            # time to update uncovered sd pairs
            if best_sl_index == -1:
                break
            # create the best SL if not previously created
            best_sl_key = f"{best_sl_index[0]}_{best_sl_index[1]}"
            if best_sl_key not in picked_sls:
                picked_sls[best_sl_key] = len(self._best_sls)
                sl_left_node, sl_right_node = sl_graph.get_node(best_sl_index[0]), sl_graph.get_node(best_sl_index[1])
                # sl_shortest_paths.get_shortest_path_without_update(sl_left_node,
                #                                                    sl_right_node).avr_ent_time = 1.0 / \
                #                                                                                  greedy_predist.SL_RATE
                self._best_sls.append(SL(src=sl_left_node, dst=sl_right_node,
                                         tree=sl_shortest_paths.get_shortest_path(sl_left_node, sl_right_node, True),
                                         cost=sl_costs[best_sl_index[0]][best_sl_index[1]]))
                current_cost += self._best_sls[-1].cost

            for sd_id, value in best_sd_candidates.items():
                updated_latency[sd_id] = updated_latency[sd_id][:value[0]] + [value[1]] + \
                                         updated_latency[sd_id][value[0] + 1:]
                src, dst = srcs[sd_id], dsts[sd_id]
                non_assigned_path[sd_id].remove(value[0])  # remove this path because it's assigned an sL
                best_used_sls[sd_id].add(best_sl_key)  # to avoid using an SL in sd twice
                path = []
                if not value[3]:
                    sl_src, sl_dst = self._best_sls[picked_sls[best_sl_key]].src, self._best_sls[
                        picked_sls[best_sl_key]].dst
                else:
                    sl_src, sl_dst = self._best_sls[picked_sls[best_sl_key]].dst, self._best_sls[
                        picked_sls[best_sl_key]].src
                if src is not sl_src:
                    path.append(path_all_pair_sp[sd_id][value[0]][0][sl_src.id])
                path.append((picked_sls[best_sl_key], value[2], value[3]))
                if dst is not sl_dst:
                    path.append(path_all_pair_sp[sd_id][value[0]][1][sl_dst.id])
                self._sd_path_with_sls[sd_id].append(path)

                if (not is_latency_objective and 1.0 / sum([1.0 / x for x in updated_latency[sd_id]]) <
                    sd_latency_targets[sd_id]) or (is_latency_objective and len(non_assigned_path[sd_id]) == 0):
                    current_uncovered_sd_pairs.remove(sd_id)

        # check if there are paths with no SL assigned
        for sd_id in uncovered_sd_pairs - current_uncovered_sd_pairs:
            for path_id in non_assigned_path[sd_id]:
                self._sd_path_with_sls[sd_id].append([sd_trees[sd_id][path_id][1]])
        if not is_latency_objective:
            for sd_id in current_uncovered_sd_pairs:
                self._sd_path_with_sls[sd_id] = []
        else:
            for sd_id in current_uncovered_sd_pairs:
                for path_id in non_assigned_path[sd_id]:
                    self._sd_path_with_sls[sd_id].append([sd_trees[sd_id][path_id][1]])
        return current_uncovered_sd_pairs

    @staticmethod
    def _iter_advanced_approach_v2(G: qGraph, srcs: List[qNode], dsts: List[qNode], max_sl_latency: float,
                                   cost_budget: float, decoherence_time: float, shortest_paths: dp_shortest_path,
                                   non_shortest_sls: bool):
        """Shortest-path calculations is located outside SL-for loop"""
        best_sls, n = [], len(G.V)
        current_sl_set = set()
        sl_costs = greedy_predist._exhaustive_greedy(G, shortest_paths, max_sl_latency)
        best_total_latency, best_sd_paths, _, _ = multi_pair_greedy_predist. \
            _total_latency_with_sls(orig_graph=G, srcs=srcs, dsts=dsts, sls=[], decoherence_time=decoherence_time)
        # residual_graphs = [G.clone() for _ in range(len(srcs))]
        # pr = profile.Profile()
        # pr.enable()
        while True:
            current_best_sl_metric = 0
            current_best_sls = None
            current_best_total_latency = 0
            current_best_sd_paths = []
            sl_nodes = set()
            sls_residual_graph = G.clone()
            # residual_routing_algos = [DP_Alt(res_graph, decoherence_time) for res_graph in residual_graphs]
            # calculating shortest-path here
            # all_sd_shortest_paths = [[[residual_routing_algos[sd_idx].get_shortest_path_by_id(srcs[sd_idx].id, i)
            #                           for i in range(n)],
            #                          [residual_routing_algos[sd_idx].get_shortest_path_by_id(i, dsts[sd_idx].id)
            #                           for i in range(n)]] for sd_idx in range(len(srcs))]
            for sl in best_sls:
                greedy_predist.residual_graph(sls_residual_graph, sls_residual_graph.get_node(sl.src.id),
                                              sls_residual_graph.get_node(sl.dst.id), [sl.tree], True)
                sl_nodes = sl_nodes.union(set(tree_node.tree_nodes(sl.tree)))
            residual_dp_alt = DP_Alt(sls_residual_graph, decoherence_time)
            for i in range(n):
                for j in range(i + 1, n):
                    if f"{G.V[i].id}-{G.V[j].id}" in current_sl_set:  # these pair already selected
                        continue
                    for candid_idx, sl_candid in enumerate(greedy_predist.candidate_sls(G, sls_residual_graph,
                                                                                        residual_dp_alt, sl_nodes,
                                                                                        i, j, shortest_paths, sl_costs,
                                                                                        max_sl_latency,
                                                                                        non_shortest_sls)):
                        if sl_candid.cost == float('inf') or not sl_candid.tree:
                            continue
                        if candid_idx == 0:  # shortest_path, get the maximum subset of prev sls
                            current_sls = greedy_predist.merge_sl(best_sls, sl_candid)
                        else:  # non-shortest path
                            current_sls = best_sls + [sl_candid]

                        if sum([sl.cost for sl in current_sls]) > cost_budget:
                            continue
                        # current_total_latency, current_sd_paths, current_sls = multi_pair_greedy_predist. \
                        #     _total_latency_with_sls_approx_v2(srcs=srcs, dsts=dsts,
                        #                                       all_sd_shortest_paths=all_sd_shortest_paths,
                        #                                       old_sls=best_sls, new_sls=current_sls,
                        #                                       old_sd_paths=best_sd_paths,
                        #                                       decoherence_time=decoherence_time,
                        #                                       orig_graph=G)
                        current_total_latency, current_sd_paths, current_sls, _ = multi_pair_greedy_predist. \
                            _total_latency_with_sls(orig_graph=G, srcs=srcs, dsts=dsts, sls=current_sls,
                                                    decoherence_time=decoherence_time)

                        if current_total_latency == float('inf') or current_total_latency >= best_total_latency:
                            continue  # not useful sl

                        current_sl_metric = (best_total_latency - current_total_latency) / \
                                            (sum([sl.cost for sl in current_sls]) - sum(
                                                [sl.cost for sl in best_sls]) + 1e-9)
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
            # run non-approx version once at the end
            best_sls = current_best_sls
            best_total_latency = current_best_total_latency
            best_sd_paths = current_best_sd_paths
            current_sl_set = set([f"{sl.src.id}-{sl.dst.id}" for sl in best_sls])
            current_cost = sum([sl.cost for sl in best_sls])
            # residual_graphs = [G.clone() for _ in range(len(srcs))]
            # multi_pair_greedy_predist._calculate_residual_graphs_v2(residual_graphs, sd_paths=best_sd_paths,
            #                                                         sls=best_sls)
            # _, best_sd_paths, best_sls, residual_graphs = multi_pair_greedy_predist. \
            #     _total_latency_with_sls(orig_graph=G, srcs=srcs, dsts=dsts, sls=best_sls,
            #                             decoherence_time=decoherence_time)
            print(f"Total latency = {round(best_total_latency, 3)}s, cost = {round(current_cost, 3)}, "
                  f"# of SLs = {len(current_sl_set)}, selected sls: [{'#'.join([sl.iid for sl in best_sls])}]")
            # pr.disable()
            # pr.dump_stats('stat.pstat')
        best_total_latency, best_sd_paths, best_sls, _ = multi_pair_greedy_predist. \
            _total_latency_with_sls(orig_graph=G, srcs=srcs, dsts=dsts, sls=best_sls,
                                    decoherence_time=decoherence_time, max_sd_paths=10, is_dp=True)
        print(f"Final: Total latency = {round(best_total_latency, 3)}s, cost = {round(current_cost, 3)}, "
              f"# of SLs = {len(current_sl_set)}, selected sls: [{'#'.join([sl.iid for sl in best_sls])}]")
        return best_sls, best_sd_paths, best_total_latency / len(srcs)

    @staticmethod
    def _iter_advanced_approach(G: qGraph, srcs: List[qNode], dsts: List[qNode], max_sl_latency: float,
                                cost_budget: float, decoherence_time: float, shortest_paths: dp_shortest_path):
        best_sls, n = [], len(G.V)
        current_sl_set = set()
        sl_costs = greedy_predist._exhaustive_greedy(G, shortest_paths, max_sl_latency)
        best_total_latency, best_sd_paths, _ = multi_pair_greedy_predist.\
            _total_latency_with_sls(orig_graph=G, srcs=srcs, dsts=dsts, sls=[], decoherence_time=decoherence_time)
        routing_algos = [{} for _ in range(len(srcs))]
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
                        current_total_latency, current_sd_paths, current_sls = multi_pair_greedy_predist.\
                            _total_latency_with_sls_approx(orig_graph=G, srcs=srcs, dsts=dsts, current_sls=best_sls,
                                                           new_sls=current_sls, current_sd_paths=best_sd_paths,
                                                           decoherence_time=decoherence_time,
                                                           routing_algos=routing_algos)

                        if current_total_latency == float('inf') or current_total_latency >= best_total_latency:
                            continue  # not useful sl

                        current_sl_metric = (best_total_latency - current_total_latency) / \
                                            (sum([sl.cost for sl in current_sls]) - sum(
                                                [sl.cost for sl in best_sls]) + 1e-9)
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
            print(f"Total latency = {round(best_total_latency, 3)}s, cost = {round(current_cost, 3)},"
                  f" # of SLs = {len(current_sl_set)}")
            # pr.disable()
            # pr.dump_stats('stat.pstat')
        # run non-approx version once at the end
        best_total_latency, best_sd_paths, best_sls = multi_pair_greedy_predist.\
            _total_latency_with_sls(orig_graph=G, srcs=srcs, dsts=dsts, sls=best_sls, decoherence_time=decoherence_time,
                                    max_sd_paths=10)
        return best_sls, best_sd_paths, best_total_latency / len(srcs)

    @staticmethod
    def _drop_unused_sls(sls: List[SL], sd_paths_with_sls: List[List[List]]) -> [List[SL], List[List[List]]]:
        used_sl_idx = set()
        for sd_paths in sd_paths_with_sls:
            for trees in sd_paths:
                for tree_part in trees:
                    if not isinstance(tree_part, tree_node):
                        used_sl_idx.add(tree_part[0])
        if len(sls) == len(used_sl_idx):  # all sls are being used
            return sls, sd_paths_with_sls
        # some SLS not being used
        sl_map, new_sls = {}, []
        for old_idx, sl in enumerate(sls):
            if old_idx in used_sl_idx:
                sl_map[old_idx] = len(new_sls)
                new_sls.append(sl)
        # updating sls' paths assignment
        new_sd_paths_with_sls = []
        for sd_paths in sd_paths_with_sls:
            new_sd_paths = []
            for trees in sd_paths:
                new_path = []
                for tree_part in trees:
                    if isinstance(tree_part, tree_node):
                        new_path.append(tree_part)
                    else:
                        new_path.append((sl_map[tree_part[0]], tree_part[1], tree_part[2]))
                new_sd_paths.append(new_path)
            new_sd_paths_with_sls.append(new_sd_paths)
        return new_sls, new_sd_paths_with_sls

    @staticmethod
    def _total_latency_with_sls_approx_v2(srcs: List[qNode], dsts: List[qNode],
                                          all_sd_shortest_paths: List[List[List[tree_node]]],
                                          old_sls: List[SL], new_sls: List[SL], old_sd_paths: List[List],
                                          decoherence_time: float, orig_graph: qGraph) -> \
            (float, List[SL], List[List]):
        new_sd_paths = [[] for _ in range(len(srcs))]
        sd_rate = [0] * len(srcs)
        # find deleted SLs, remove all sd-paths related to it, return the resources to residual_graphs
        old_sl_dict = {sl: idx for idx, sl in enumerate(old_sls)}
        sl_map = {}
        for idx, new_sl in enumerate(new_sls):
            if new_sl in old_sl_dict:
                sl_map[old_sl_dict[new_sl]] = idx
        sd_affected = [False] * len(srcs)  # check if sd was previously assigned an sl that has been removed
        for sd_idx in range(len(srcs)):
            for path in old_sd_paths[sd_idx]:
                new_path = []
                if len(path) > 1 or (
                        len(path) == 1 and not isinstance(path[0], tree_node)):  # has previously an sl assigned
                    for sub_path in path:
                        if not isinstance(sub_path, tree_node):
                            if sub_path[0] not in sl_map:  # this SL was removed
                                new_path = []
                                sd_affected[sd_idx] = True
                                break
                            else:
                                new_path.append((sl_map[sub_path[0]], sub_path[1], sub_path[2]))
                        else:
                            new_path.append(sub_path)
                    if len(new_path) == 0:
                        continue
                    new_sd_paths[sd_idx].append(new_path)
                    # calculating latency
                    if len(new_path) == 3:
                        # sl in the middle
                        latency, _ = greedy_predist.latency_with_sl(new_path[0], srcs[sd_idx],
                                                                    new_path[0].rightest_node, new_path[2].leftest_node,
                                                                    new_path[2], dsts[sd_idx])
                    elif len(new_path) == 2 and isinstance(new_path[0], tree_node):
                        # sl is the rightmost
                        latency, _ = greedy_predist.latency_with_sl(new_path[0], srcs[sd_idx],
                                                                    new_path[0].rightest_node, dsts[sd_idx],
                                                                    None, dsts[sd_idx])
                    elif len(new_path) == 2 and isinstance(new_path[1], tree_node):
                        # sl is the leftmost
                        latency, _ = greedy_predist.latency_with_sl(None, srcs[sd_idx], srcs[sd_idx],
                                                                    new_path[1].leftest_node, new_path[1],
                                                                    dsts[sd_idx])
                    elif len(new_path) == 1:
                        latency = 0
                    sd_rate[sd_idx] += 1 / latency if latency > 0 else float('inf')

        # update shortest paths for an sd that has been updated (one of its sl is removed)
        updated_all_sd_shortest_paths = []
        for sd_idx in range(len(srcs)):
            sl = new_sls[-1]
            updated_all_sd_shortest_paths.append(all_sd_shortest_paths[sd_idx])
            if sd_affected[sd_idx]:  # an sd has been removed
                tmp_graph = orig_graph.clone()
                # re-calculate the residual
                for path in new_sd_paths[sd_idx]:
                    for sub_path in path:
                        if isinstance(sub_path, tree_node):
                            target_path = sub_path
                        else:
                            target_path = new_sls[sub_path[0]].tree
                        greedy_predist.residual_graph(tmp_graph, target_path.leftest_node, target_path.rightest_node,
                                                      [target_path])
                # find 4 shortest paths src->sl_src, src->sl_dst, sl_src->dst, dl_dst->dst
                routing_algo = DP_Alt(tmp_graph, decoherence_time=decoherence_time)
                updated_all_sd_shortest_paths[sd_idx][0][sl.src.id] = routing_algo.\
                    get_shortest_path_without_update_by_id(srcs[sd_idx].id, sl.src.id)
                updated_all_sd_shortest_paths[sd_idx][0][sl.dst.id] = routing_algo. \
                    get_shortest_path_without_update_by_id(srcs[sd_idx].id, sl.dst.id)
                updated_all_sd_shortest_paths[sd_idx][1][sl.src.id] = routing_algo. \
                    get_shortest_path_without_update_by_id(sl.src.id, dsts[sd_idx].id)
                updated_all_sd_shortest_paths[sd_idx][1][sl.dst.id] = routing_algo. \
                    get_shortest_path_without_update_by_id(sl.dst.id, dsts[sd_idx].id)

        # check if new sl is helping any sd-path
        sl = new_sls[-1]
        for sd_idx in range(len(srcs)):
            sp_no_sl = updated_all_sd_shortest_paths[sd_idx][0][dsts[sd_idx].id]
            latency_without_sl = sp_no_sl.avr_ent_time if sp_no_sl else float('inf')
            latency_with_sl, is_first, inverted = multi_pair_greedy_predist._single_path_latency_with_sl_v2(
                src=srcs[sd_idx], dst=dsts[sd_idx], sl_src=sl.src,
                sl_dst=sl.dst, shortest_paths=updated_all_sd_shortest_paths[sd_idx])
            if latency_with_sl < latency_without_sl:
                sd_rate[sd_idx] += 1 / latency_with_sl if latency_with_sl > 0 else float('inf')
                # generate the path
                path = []
                sl_src, sl_dst = sl.src, sl.dst
                if inverted:
                    sl_src, sl_dst = sl_dst, sl_src
                if srcs[sd_idx].id != sl_src.id:
                    # src to sl shortest path
                    path.append(updated_all_sd_shortest_paths[sd_idx][0][sl_src.id])
                path.append((len(new_sls) - 1, is_first, inverted))
                if dsts[sd_idx].id != sl_dst.id:
                    # sl to dst shortest path
                    path.append(updated_all_sd_shortest_paths[sd_idx][1][sl_dst.id])
                new_sd_paths[sd_idx].append(path)
            elif len(new_sd_paths[sd_idx]) == 0:
                # no path with sl assigned, add shortest path
                sd_rate[sd_idx] = 1 / latency_without_sl
                new_sd_paths[sd_idx].append([sp_no_sl])
        return sum([1 / rate if rate > 0 else float('inf') for rate in sd_rate]), new_sd_paths, new_sls

    @staticmethod
    def _calculate_residual_graphs_v2(residual_graphs: List[qGraph], sd_paths: List[List], sls: List[SL]):
        for sd_idx in range(len(residual_graphs)):
            for path in sd_paths[sd_idx]:
                if len(path) > 1 or (len(path) == 1 and not isinstance(path[0], tree_node)):
                    for sub_path in path:
                        if isinstance(sub_path, tree_node):
                            target_path = sub_path
                            src, dst = target_path.leftest_node, target_path.rightest_node
                        else:
                            target_path = sls[sub_path[0]].tree
                            src, dst = target_path.leftest_node, target_path.rightest_node
                            if sub_path[2]:
                                src, dst = dst, src
                        greedy_predist.residual_graph(residual_graphs[sd_idx], src, dst, [target_path])



    @staticmethod
    def _total_latency_with_sls_approx(orig_graph: qGraph, srcs: List[qNode], dsts: List[qNode],
                                       current_sls: List[SL], new_sls: List[SL], current_sd_paths: List[List],
                                       decoherence_time: float, routing_algos: List[dict[str, DP_Alt]]) -> \
            (float, List[SL], List[List]):
        residual_graphs = [orig_graph.clone() for _ in range(len(srcs))]
        new_sd_paths = [[] for _ in range(len(srcs))]
        sd_rate = [0] * len(srcs)
        sd_sl_assignment = [[] for _ in range(len(srcs))]
        # find deleted SLs, remove all sd-paths related to it, return the resources to residual_graphs
        curren_sl_dict = {sl: idx for idx, sl in enumerate(current_sls)}
        sl_map = {}
        for idx, new_sl in enumerate(new_sls):
            if new_sl in curren_sl_dict:
                sl_map[curren_sl_dict[new_sl]] = idx

        for sd_idx in range(len(srcs)):
            for path in current_sd_paths[sd_idx]:
                new_path = []
                if len(path) > 1 or (len(path) == 1 and not isinstance(path[0], tree_node)):  # has previously an sl assigned
                    for sub_path in path:
                        if not isinstance(sub_path, tree_node):
                            if sub_path[0] not in sl_map:  # this SL was removed
                                new_path = []
                                break
                            else:
                                new_path.append((sl_map[sub_path[0]], sub_path[1], sub_path[2]))
                                sd_sl_assignment[sd_idx].append(sl_map[sub_path[0]])
                        else:
                            new_path.append(sub_path)
                    if len(new_path) == 0:
                        continue
                    new_sd_paths[sd_idx].append(new_path)
                    for sub_idx, sub_path in enumerate(new_path):
                        if isinstance(sub_path, tree_node):
                            target_tree = sub_path
                        else:
                            target_tree = current_sls[sub_path[0]].tree
                        greedy_predist.residual_graph(residual_graphs[sd_idx], target_tree.leftest_node,
                                                      target_tree.rightest_node, [target_tree])
                    # calculating latency
                    if len(new_path) == 3:
                        # sl in the middle
                        latency, _ = greedy_predist.latency_with_sl(new_path[0], srcs[sd_idx],
                                                                    new_path[0].rightest_node, new_path[2].leftest_node,
                                                                    new_path[2], dsts[sd_idx])
                    elif len(new_path) == 2 and isinstance(new_path[0], tree_node):
                        # sl is the rightmost
                        latency, _ = greedy_predist.latency_with_sl(new_path[0], srcs[sd_idx],
                                                                    new_path[0].rightest_node, dsts[sd_idx],
                                                                    None, dsts[sd_idx])
                    elif len(new_path) == 2 and isinstance(new_path[1], tree_node):
                        # sl is the leftmost
                        latency, _ = greedy_predist.latency_with_sl(None, srcs[sd_idx], srcs[sd_idx],
                                                                    new_path[1].leftest_node, new_path[1],
                                                                    dsts[sd_idx])
                    elif len(new_path) == 1:
                        latency = 0
                    sd_rate[sd_idx] += 1/latency if latency > 0 else float('inf')

        #check if new sl is helping any sd-path
        sl = new_sls[-1]
        for sd_idx in range(len(srcs)):
            assigned_sls_ids = "#".join([f"{new_sls[sl_id].src.id}-{new_sls[sl_id].dst.id}"
                                         for sl_id in sd_sl_assignment[sd_idx]])
            assigned_sls_ids = assigned_sls_ids if assigned_sls_ids != "" else "#"
            if assigned_sls_ids in routing_algos[sd_idx]:
                routing_alg = routing_algos[sd_idx][assigned_sls_ids]
            else:
                routing_alg = DP_Alt(residual_graphs[sd_idx], decoherence_time=decoherence_time)
                routing_algos[sd_idx][assigned_sls_ids] = routing_alg
            sp = routing_alg.get_shortest_path_without_update_by_id(srcs[sd_idx].id, dsts[sd_idx].id)
            latency_without_sl = sp.avr_ent_time if sp else float('inf')
            latency_with_sl, is_first, inverted = multi_pair_greedy_predist._single_path_latency_with_sl(
                residual_graph=residual_graphs[sd_idx], src=srcs[sd_idx], dst=dsts[sd_idx], sl_src=sl.src,
                sl_dst=sl.dst, shortest_paths=routing_alg)
            if latency_with_sl < latency_without_sl:
                sd_rate[sd_idx] += 1 / latency_with_sl if latency_with_sl > 0 else float('inf')
                # generate the path
                path = []
                sl_src, sl_dst = residual_graphs[sd_idx].get_node(sl.src.id), \
                                 residual_graphs[sd_idx].get_node(sl.dst.id)
                if inverted:
                    sl_src, sl_dst = sl_dst, sl_src
                if srcs[sd_idx].id != sl_src.id:
                    # src to sl shortest path
                    src = residual_graphs[sd_idx].get_node(srcs[sd_idx].id)
                    path.append(routing_alg.get_shortest_path(src, sl_src, False))
                path.append((len(new_sls) - 1, is_first, inverted))
                if dsts[sd_idx].id != sl_dst.id:
                    # sl to dst shortest path
                    dst = residual_graphs[sd_idx].get_node(dsts[sd_idx].id)
                    path.append(routing_alg.get_shortest_path(sl_dst, dst, False))
                new_sd_paths[sd_idx].append(path)
            elif len(new_sd_paths[sd_idx]) == 0:
                # no path with sl assigned, add shortest path
                sd_rate[sd_idx] = 1 / latency_without_sl
                new_sd_paths[sd_idx].append([sp])
        return sum([1 / rate if rate > 0 else float('inf') for rate in sd_rate]), new_sd_paths, new_sls

    @staticmethod
    def _total_latency_with_sls(orig_graph: qGraph, srcs: List[qNode], dsts: List[qNode], sls: List[SL],
                                decoherence_time: float, max_sd_paths: int = 1, is_dp: bool = False) -> (float, List[SL], List[List]):
        sd_paths = [[] for _ in range(len(srcs))]
        sd_rate = [0] * len(srcs)
        # sls.sort(key=lambda sl: sl.cost, reverse=False)
        new_sls, sls_used = [], {}  # drop unused sls
        residual_graphs = [orig_graph.clone() for _ in range(len(srcs))]
        if is_dp:
            routing_algs = [dp_shortest_path(residual_graph, decoherence_time=decoherence_time)
                            for residual_graph in residual_graphs]
        else:
            routing_algs = [DP_Alt(residual_graph, decoherence_time=decoherence_time)
                            for residual_graph in residual_graphs]

        for sd_idx in range(len(srcs)):
            assigned_sls = set()
            while True:
                routing_alg = routing_algs[sd_idx]
                sp = routing_alg.get_shortest_path_without_update_by_id(srcs[sd_idx].id, dsts[sd_idx].id)
                best_current_latency = sp.avr_ent_time if sp else float('inf')
                best_current_sl_idx, best_side_info = -1, None
                for sl_idx, sl in enumerate(sls):
                    if sl_idx in assigned_sls:  # each sl is assigned once for each sd
                        continue
                    latency_with_sl, is_first, inverted = multi_pair_greedy_predist._single_path_latency_with_sl(
                        residual_graph=residual_graphs[sd_idx], src=srcs[sd_idx], dst=dsts[sd_idx], sl_src=sl.src,
                        sl_dst=sl.dst, shortest_paths=routing_alg)
                    if latency_with_sl < best_current_latency:
                        best_current_sl_idx, best_current_latency = sl_idx, latency_with_sl
                        best_side_info = (is_first, inverted)
                if best_current_sl_idx != -1:
                    assigned_sls.add(best_current_sl_idx)  # add to avoid using it later again
                    # sl is useful
                    if best_current_sl_idx not in sls_used:
                        # add to used sls
                        sls_used[best_current_sl_idx] = len(new_sls)
                        new_sls.append(sls[best_current_sl_idx])
                    sl = sls[best_current_sl_idx]
                    is_first, inverted = best_side_info
                    sd_rate[sd_idx] += 1 / best_current_latency if best_current_latency > 0 else float('inf')
                    # generate the path
                    path = []
                    sl_src, sl_dst = residual_graphs[sd_idx].get_node(sl.src.id), \
                                     residual_graphs[sd_idx].get_node(sl.dst.id)
                    if inverted:
                        sl_src, sl_dst = sl_dst, sl_src
                    if srcs[sd_idx].id != sl_src.id:
                        # src to sl shortest path
                        src = residual_graphs[sd_idx].get_node(srcs[sd_idx].id)
                        path.append(routing_alg.get_shortest_path(src, sl_src, False))
                        greedy_predist.residual_graph(residual_graphs[sd_idx], src, sl_src, [path[-1]])
                    path.append((sls_used[best_current_sl_idx], is_first, inverted))
                    greedy_predist.residual_graph(residual_graphs[sd_idx], sl_src, sl_dst,
                                                  [new_sls[sls_used[best_current_sl_idx]].tree], non_shortest=False)
                    if dsts[sd_idx].id != sl_dst.id:
                        # sl to dst shortest path
                        dst = residual_graphs[sd_idx].get_node(dsts[sd_idx].id)
                        path.append(routing_alg.get_shortest_path(sl_dst, dst, False))
                        greedy_predist.residual_graph(residual_graphs[sd_idx], sl_dst, dst, [path[-1]])
                    sd_paths[sd_idx].append(path)
                    if is_dp:
                        routing_algs[sd_idx] = dp_shortest_path(residual_graphs[sd_idx],
                                                                decoherence_time=decoherence_time)
                    else:
                        routing_algs[sd_idx] = DP_Alt(residual_graphs[sd_idx], decoherence_time=decoherence_time)
                else:
                    break  # no more useful sl
        # add new paths from residual
        for sd_idx in range(len(srcs)):
            src, dst = residual_graphs[sd_idx].get_node(srcs[sd_idx].id), \
                       residual_graphs[sd_idx].get_node(dsts[sd_idx].id)
            while len(sd_paths[sd_idx]) < max_sd_paths:
                if is_dp:
                    routing_alg = dp_shortest_path(residual_graphs[sd_idx], decoherence_time)
                else:
                    routing_alg = DP_Alt(residual_graphs[sd_idx], decoherence_time)
                sp = routing_alg.get_shortest_path(src, dst)
                if not sp:
                    break
                sd_rate[sd_idx] += 1 / sp.avr_ent_time
                sd_paths[sd_idx].append([sp])
                greedy_predist.residual_graph(residual_graphs[sd_idx], src, dst, [sp])

        return sum([1/rate if rate > 0 else float('inf') for rate in sd_rate]), sd_paths, new_sls, residual_graphs

    @staticmethod
    def _single_path_latency_with_sl_v2(src: qNode, dst: qNode, sl_src: qNode, sl_dst: qNode,
                                        shortest_paths: List[List[tree_node]]) -> (float, bool, bool):
        latency = greedy_predist.latency_with_sl(shortest_paths[0][sl_src.id],
                                                 src, sl_src, sl_dst,
                                                 shortest_paths[1][sl_dst.id], dst)
        # if we use SL inversely
        latency_inv = greedy_predist.latency_with_sl(shortest_paths[0][sl_dst.id],
                                                     src, sl_dst, sl_src,
                                                     shortest_paths[1][sl_src.id], dst)
        return (latency[0], latency[1], False) if latency[0] < latency_inv[0] else (latency_inv[0], latency_inv[1],
                                                                                    True)

    @staticmethod
    def _single_path_latency_with_sl(residual_graph: qGraph, src: qNode, dst: qNode, sl_src: qNode, sl_dst: qNode,
                                     shortest_paths) -> (float, bool, bool):
        src_res, dst_res = residual_graph.get_node(src.id), residual_graph.get_node(dst.id)
        slsrc_res, sldst_res = residual_graph.get_node(sl_src.id), residual_graph.get_node(sl_dst.id)
        latency = greedy_predist.latency_with_sl(shortest_paths.get_shortest_path_without_update_by_id(src_res.id,
                                                                                                       slsrc_res.id),
                                                 src_res, slsrc_res, sldst_res,
                                                 shortest_paths.get_shortest_path_without_update_by_id(sldst_res.id,
                                                                                                       dst_res.id),
                                                 dst_res)
        # if we use SL inversely
        latency_inv = greedy_predist.latency_with_sl(shortest_paths.get_shortest_path_without_update_by_id(src_res.id,
                                                                                                           sldst_res.id),
                                                     src_res, sldst_res, slsrc_res,
                                                     shortest_paths.get_shortest_path_without_update_by_id(slsrc_res.id,
                                                                                                           dst_res.id),
                                                     dst_res)
        return (latency[0], latency[1], False) if latency[0] < latency_inv[0] else (latency_inv[0], latency_inv[1], True)

    @property
    def get_final_paths(self) -> List:
        return self._sd_path_with_sls

    @property
    def get_sls(self) -> List[SL]:
        return self._best_sls
