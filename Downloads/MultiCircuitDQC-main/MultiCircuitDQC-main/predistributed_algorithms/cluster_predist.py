import sys

sys.path.append("..")

import random
import routing_algorithms.dp_shortest_path
from commons.qGraph import qGraph
from commons.qNode import qNode
from commons.qChannel import qChannel
from typing import List
from routing_algorithms.dp_shortest_path import dp_shortest_path
from commons.tree_node import tree_node
from commons.tools import get_classical_time, SL
from predistributed_algorithms.greedy_predist import *
# import cProfile as profile


#
# def compare_sl(sl1: SL, sl2: SL):
#     if (sl1.src is sl2.src and sl1.dst is sl2.dst) or (sl1.src.id == sl2.src.id and sl1.dst.id == sl2.dst.id):
#         return True
#     return False
#
#
# def check_sl_in_list(sl, sl_list):
#     for temp_sl in sl_list:
#         if compare_sl(temp_sl, sl):
#             return True
#     return False
#
#
# def find_sl_index(sl, sl_list):
#     for i in range(len(sl_list)):
#         temp_sl = sl_list[i]
#         if compare_sl(temp_sl, sl):
#             return i
#     return -1

class cluster_predist:
    def __init__(self, G: qGraph, srcs: List[qNode], dsts: List[qNode],
                 latency_threshold: float, max_sl_latency: float, IS_LATENCY_OBJECTIVE: bool, decoherence_time: float,
                 FIXED_LATENCY: bool = False, cost_budget: float = COST_BUDGET_DEFAULT):
        # print("Start clustering")
        srcs_len = len(srcs)
        dsts_len = len(dsts)
        self._average_latency = float('inf')
        if srcs_len != dsts_len:
            raise ValueError("THe number of sources and destinations are not equal!")
        n = len(G.V)

        shortest_paths = dp_shortest_path(G, decoherence_time=decoherence_time)
        self._sd_path_with_sls = []
        self._best_sls = []

        if not IS_LATENCY_OBJECTIVE:
            sd_latency_targets = [latency_threshold] * len(srcs) if FIXED_LATENCY else \
                [shortest_paths.get_shortest_path_without_update(src, dst).avr_ent_time * latency_threshold
                 for src, dst in zip(srcs, dsts)]

        # The following store temporary best centroids and clusters which every sd pair being satisfied.
        temp_best_centroids = None
        temp_best_clusters = None
        temp_best_cost = float('inf')
        best_reduction = 0

        # The following store back centroids and clusters which has the least number of unsatisfied pairs.
        # Only updated when some centroids and clusters at some number k being dropped.
        # Only used when all Ks are being dropped.
        back_up_centroids = None
        back_up_clusters = None
        back_up_unsatisfied_sd_index = None  # A list storing the indices of s d pairs which haven't being satisfied.
        min_num_unsatisfied_pairs = len(srcs)
        back_up_cost = float('inf')

        # Generate all SLs.
        n = len(G.V)
        sl_costs = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sl_costs[i][j] = greedy_predist.sl_cost(shortest_paths.get_shortest_path_without_update(G.V[i], G.V[j]),
                                                        max_sl_latency)
        # pr = profile.Profile()
        # pr.enable()
        for k in range(1, srcs_len + 1):
            # Structures
            # centroids: [SL1_idx, SL2_idx,......,SLk_idx] --> k terms
            # clusters: [cluster1, cluster2, ....., clusterK] --> k terms
            # each cluster: [(index of s,d pair, reversed or not , unchanged boolean, Left Tree, Right Tree, latency), ....] --> # of terms unknown

            centroids, clusters = self.modified_k_means(K=k, srcs=srcs, dsts=dsts,
                                                        shortest_paths=shortest_paths, sl_costs=sl_costs,
                                                        orig_latency=[shortest_paths.get_shortest_path_without_update
                                                                      (src, dst).avr_ent_time for src, dst in
                                                                      zip(srcs, dsts)],
                                                        is_latency_objective=IS_LATENCY_OBJECTIVE,
                                                        cost_budget=cost_budget,
                                                        orig_graph=G)
            if not IS_LATENCY_OBJECTIVE:
                # Following part check each (s,d) pairs' latency has been satisfied.
                unsatisfied_sd_index = []
                # print("When checking unsatisfication, there are ", len(clusters), " clusters")
                for cluster in clusters:
                    for pair in cluster:
                        if pair[-1] > sd_latency_targets[pair[0]]:
                            unsatisfied_sd_index.append(pair[0])
                # print("And there are ", count_pair, " numbers of (s,d) pairs.")

                # Now calculate the total cost of this clustering.
                total_cost = sum([centroid.cost for centroid in centroids])
                # print("The total cost of k=", k, " is ", total_cost)

                # We need to ignore the clusters and centroids if there are any unsatisfied (s,d) pairs.
                # But we also need to do backup to prevent the case that all Ks being ignored.
                num_unsatisfied = len(unsatisfied_sd_index)
                if num_unsatisfied != 0:
                    # Before we drop this case, we do some backup.
                    if num_unsatisfied < min_num_unsatisfied_pairs or \
                            (num_unsatisfied == min_num_unsatisfied_pairs and total_cost < back_up_cost):
                        back_up_centroids = centroids
                        back_up_clusters = clusters
                        back_up_unsatisfied_sd_index = unsatisfied_sd_index
                        min_num_unsatisfied_pairs = num_unsatisfied
                        back_up_cost = total_cost
                    # print("Drop when k == ", k, " Due to having ", len(unsatisfied_sd_index), " unsatisfied s,d pairs")
                    continue

                # Update the temp best solutions.
                # TODO should be reduction/cost here too??
                if total_cost < temp_best_cost:
                    temp_best_cost = total_cost
                    temp_best_centroids = centroids
                    temp_best_clusters = clusters
            else:
                # latency is objective
                if sum([sl_costs[centroids[idx][0]][centroids[idx][1]] for idx in range(k) if len(clusters) > 0]) \
                        < cost_budget:
                    # this k satisfies the budget
                    k_reduction = sum([shortest_paths.get_shortest_path_without_update(srcs[pair[0]],
                                                                                       dsts[pair[0]]).avr_ent_time -
                                       pair[-1] for cluster in clusters for pair in cluster])
                    if k_reduction > best_reduction:
                        best_reduction = k_reduction
                        temp_best_clusters = clusters
                        temp_best_centroids = centroids
                        temp_best_cost = sum([sl_costs[i][j] for i, j in centroids])

        if IS_LATENCY_OBJECTIVE:
            # generating final sls and path with sls
            total_latency = 0.0
            sd_pair_not_assigned = set([sd_idx for sd_idx in range(srcs_len)])
            self._sd_path_with_sls = [[] for _ in range(srcs_len)]
            if temp_best_centroids and temp_best_clusters:
                self._best_sls = [SL(src=G.get_node(temp_best_centroids[cluster_idx][0]),
                                     dst=G.get_node(temp_best_centroids[cluster_idx][1]),
                                     tree=shortest_paths.get_shortest_path(
                                         src=G.get_node(temp_best_centroids[cluster_idx][0]),
                                         dst=G.get_node(temp_best_centroids[cluster_idx][1]),
                                         NON_THROTTLING=True),
                                     cost=sl_costs[temp_best_centroids[cluster_idx][0]][
                                         temp_best_centroids[cluster_idx][1]])
                                  for cluster_idx in range(len(temp_best_centroids)) if
                                  len(temp_best_clusters[cluster_idx]) > 0]
                temp_best_clusters = [cluster for cluster in temp_best_clusters if len(cluster) > 0]

                for i, cluster in enumerate(temp_best_clusters):
                    for pair in cluster:
                        sd_idx = pair[0]
                        total_latency += pair[-1]
                        sd_pair_not_assigned.remove(sd_idx)
                        packet = [pair[3], (i, pair[2], pair[1]), pair[4]]
                        if not pair[1]:
                            sl_src, sl_dst = self._best_sls[i].src, self._best_sls[i].dst
                        else:
                            sl_src, sl_dst = self._best_sls[i].dst, self._best_sls[i].src
                        new_path = []
                        if sl_src is not srcs[pair[0]]:
                            new_path.append(pair[3])
                        new_path.append(packet[1])
                        if sl_dst is not dsts[pair[0]]:
                            new_path.append(pair[4])
                        self._sd_path_with_sls[sd_idx].append(new_path)
            # use simple tree without sls for those that are left
            for sd_idx in sd_pair_not_assigned:
                self._sd_path_with_sls[sd_idx].append([shortest_paths.get_shortest_path(srcs[sd_idx],
                                                                                        dsts[sd_idx])])
                total_latency += self._sd_path_with_sls[sd_idx][-1][-1].avr_ent_time
            self._average_latency = total_latency / len(srcs)

        else:
            # TODO this has not been updated due to sl_cost design change
            # This is the end of the for loop which loops k from 1 to number of s,d pairs.
            if temp_best_cost != float('inf') and temp_best_clusters is not None and temp_best_centroids is not None:
                # Reformat the clusters.
                self._best_sls = temp_best_centroids
                reformat_clusters = [[] for _ in range(srcs_len)]
                for i in range(0, len(temp_best_clusters)):
                    cluster = temp_best_clusters[i]
                    for pair in cluster:
                        sd_index = pair[0]
                        packet = [pair[3], (i, pair[2], pair[1]), pair[4]]
                        if not pair[1]:
                            sl_src, sl_dst = self._best_sls[i].src, self._best_sls[i].dst
                        else:
                            sl_src, sl_dst = self._best_sls[i].dst, self._best_sls[i].src
                        new_path = []
                        if sl_src is not srcs[pair[0]]:
                            new_path.append(pair[3])
                        new_path.append(packet[1])
                        if sl_dst is not dsts[pair[0]]:
                            new_path.append(pair[4])
                        reformat_clusters[sd_index].append(new_path)
                self._sd_path_with_sls = reformat_clusters
                # print("Result:")
                # print("The number of SLs is: ", len(self._best_sls), "and the cost is: ", temp_best_cost)
            else:
                print("All Ks have been dropped. Picking the previous K with least unsatisfied pairs.")
                if back_up_centroids is None or back_up_clusters is None:
                    print("Error!!! Should have back ups but not shown up.")
                else:
                    reformat_clusters = [[] for _ in range(srcs_len)]
                    for i in range(0, len(back_up_clusters)):
                        cluster = back_up_clusters[i]
                        for pair in cluster:
                            sd_index = pair[0]
                            packet = [pair[3], (i, pair[2], pair[1]), pair[4]]
                            if not pair[1]:
                                sl_src, sl_dst = back_up_centroids[i].src, back_up_centroids[i].dst
                            else:
                                sl_src, sl_dst = back_up_centroids[i].dst, back_up_centroids[i].src
                            new_path = []
                            if sl_src is not srcs[pair[0]]:
                                new_path.append(pair[3])
                            new_path.append(packet[1])
                            if sl_dst is not dsts[pair[0]]:
                                new_path.append(pair[4])
                            reformat_clusters[sd_index].append(new_path)
                    back_up_clusters = reformat_clusters
                    # Should drop unused SLs (centroid)
                    used_centroids = set()
                    for sd_index in range(len(srcs)):
                        if sd_index not in back_up_unsatisfied_sd_index:
                            for path_part in back_up_clusters[sd_index][0]:
                                if not isinstance(path_part, tree_node):
                                    used_centroids.add(path_part[0])
                    new_centroids, idx_map = [], {}
                    for old_idx in used_centroids:
                        idx_map[old_idx] = len(new_centroids)
                        new_centroids.append(back_up_centroids[old_idx])
                    back_up_centroids = new_centroids
                    # updating path assignment
                    for sd_index in range(len(srcs)):
                        if sd_index not in back_up_unsatisfied_sd_index:
                            path = []
                            for path_part in back_up_clusters[sd_index][0]:
                                if isinstance(path_part, tree_node):
                                    path.append(path_part)
                                else:
                                    path.append((idx_map[path_part[0]], path_part[1], path_part[2]))
                            back_up_clusters[sd_index][0] = path
                    new_all_sls = [SL(G.V[i], G.V[j], shortest_paths.get_shortest_path_without_update(G.V[i], G.V[j]),
                                      greedy_predist.sl_cost(
                                          shortest_paths.get_shortest_path_without_update(G.V[i], G.V[j])
                                          , float('inf'))) for i in range(n) for j in range(i + 1, n)]
                    # even SLs higher than TIMESlOT constraint
                    for pair_index in back_up_unsatisfied_sd_index:
                        s = srcs[pair_index]
                        d = dsts[pair_index]

                        best_sl = None
                        best_sd_info = None
                        best_cost = float('inf')

                        for sl in new_all_sls:
                            # Calculate the latency of using this SL non-reversely
                            ltree1 = shortest_paths.get_shortest_path_without_update(s, sl.src)
                            rtree1 = shortest_paths.get_shortest_path_without_update(sl.dst, d)
                            latency1, unchanged_bool1 = greedy_predist.latency_with_sl(
                                ltree1, s,
                                sl.src, sl.dst,
                                rtree1, d
                            )

                            ltree2 = shortest_paths.get_shortest_path_without_update(s, sl.dst)
                            rtree2 = shortest_paths.get_shortest_path_without_update(sl.src, d)
                            latency2, unchanged_bool2 = greedy_predist.latency_with_sl(
                                ltree2, s,
                                sl.dst, sl.src,
                                rtree2, d
                            )

                            if latency1 <= latency2:
                                latency = latency1
                                reverse_bool = False
                                unchanged_bool = unchanged_bool1
                                ltree = ltree1
                                rtree = rtree1
                            else:
                                latency = latency2
                                reverse_bool = True
                                unchanged_bool = unchanged_bool2
                                ltree = ltree2
                                rtree = rtree2

                            if latency < sd_latency_targets[pair_index] and sl.cost < best_cost:
                                # Create the tuple and list.
                                # Here we make a temporary tuple, we would add the index of sl when sl is determined.
                                best_sl = sl
                                best_sd_info = [ltree, (-1, unchanged_bool, reverse_bool), rtree]
                                best_cost = sl.cost
                        # For one sd pair, we know an sl works with min cost. Now update it.
                        # sl_index = find_sl_index(best_sl, back_up_centroids)
                        if sl_index == -1:
                            sl_index = len(back_up_centroids)
                            back_up_centroids.append(best_sl)
                        new_tuple = (sl_index, best_sd_info[1][1], best_sd_info[1][2])
                        if not best_sd_info[1][2]:
                            sl_src, sl_dst = best_sl.src, best_sl.dst
                        else:
                            sl_src, sl_dst = best_sl.dst, best_sl.src
                        new_path = []
                        if sl_src is not s:
                            new_path.append(best_sd_info[0])
                        new_path.append(new_tuple)
                        if sl_dst is not d:
                            new_path.append(best_sd_info[2])
                        back_up_clusters[pair_index] = [new_path]
                    # After all unsatisfied sd pairs being satisfied.
                    self._best_sls = back_up_centroids
                    self._sd_path_with_sls = back_up_clusters
                    # print("Result:")
                    # print("The number of SLs is: ", len(self._best_sls))
                    # print("Finish recover.")
        # pr.disable()
        # pr.dump_stats('cluster_stat.pstat')

    @staticmethod
    def modified_k_means(K: int, srcs: List[qNode], dsts: List[qNode], shortest_paths: dp_shortest_path,
                         orig_graph: qGraph,
                         sl_costs: List[List[float]], orig_latency: List[float], is_latency_objective: bool,
                         cost_budget: float):
        # Structures
        # centroids: [SL1, SL2,......,SLk] --> k terms
        # clusters: [cluster1, cluster2, ....., clusterK] --> k terms
        # each cluster: [(index of s,d pair, reversed or not , unchanged boolean, Left Tree, Right Tree, latency), ....] --> # of terms unknown

        # Initialize the centroids by randomly picking k SLs as the centroid.
        n = len(sl_costs)
        if not is_latency_objective:
            sl_indices = [(i, j) for i in range(n) for j in range(n) if sl_costs[i][j] != float('inf')]
            random_indices = random.sample(range(len(sl_indices)), K)
            centroids = [sl_indices[ri] for ri in random_indices]
        else:
            sl_indices = [(i, j) for i in range(n) for j in range(n) if sl_costs[i][j] != float('inf') and
                          sl_costs[i][j] < cost_budget]
            centroids = [sl_indices[ri] for ri in random.sample(range(len(sl_indices)), K)]
            while False:
                # to pick SLs such that the sum of costs is less than cost_budget
                current_cost = 0.0
                centroids = []
                while len(centroids) != K:
                    all_sls_lower_budget = [sl for sl in all_sl if sl.cost < cost_budget - current_cost]
                    if len(all_sls_lower_budget) == 0:
                        # no sl left to pick, start over
                        break
                    centroids.append(all_sls_lower_budget[random.randint(0, len(all_sls_lower_budget) - 1)])
                    current_cost += centroids[-1].cost
                if len(centroids) == K:
                    # found an acceptable solution
                    break

        best_centroids = None
        best_clusters = None
        best_clusters_total_reduction_per_cost = -float(
            'inf')  # The sum the distance from each s,d pair to its corresponding centroid
        best_clusters_wins = 0
        old_clusters = [[] for _ in range(K)]
        while True:
            # Initialize the clusters.
            clusters = [[] for _ in range(K)]
            # Part1: Assignment. Assign each (s,d) pair to the closest centroid.
            # At this step centroids are given. Need to modify the clusters.
            for sd_idx in range(len(srcs)):
                # First we get the s,d pair nodes
                s = srcs[sd_idx]
                d = dsts[sd_idx]

                # It is going to traverse through all centroids and calculate the min "distance"
                # The "distance" means the latency between (s,d) pairs when using the centroid SL.
                min_dist = float('inf')
                best_tuple = None
                best_centroid_index = 0
                for k in range(K):
                    i, j = centroids[k]
                    sl_src, sl_dst = orig_graph.get_node(i), orig_graph.get_node(j)
                    dist_with_sl = cluster_predist._distance_with_sl(s, d, shortest_paths, sl_src, sl_dst)

                    # Update the min_dist and best_tuple
                    if dist_with_sl[-1] < min_dist:
                        min_dist = dist_with_sl[-1]
                        best_tuple = (sd_idx, *dist_with_sl)
                        best_centroid_index = k
                if (not is_latency_objective and min_dist != float('inf')) or \
                        (is_latency_objective and min_dist < float('inf')):#orig_latency[sd_idx]):
                    clusters[best_centroid_index].append(
                        best_tuple)  # Add this s,d pair info to the corresponding cluster.

            # Part2: Determine if it is necessary to stop.
            # After the Assignment part, sum the total "distance" from each cluster and compare.
            if not is_latency_objective:
                cluster_reduction_per_cost = sum([sum([(orig_latency[pair[0]] - pair[-1]) / orig_latency[pair[0]]
                                                       for pair in cluster]) / sl_costs[centroids[idx][0]][
                                                      centroids[idx][1]]
                                                  for idx, cluster in enumerate(clusters)])
            else:
                # raw reduction not percentage
                cluster_reduction_per_cost = sum([sum([(orig_latency[pair[0]] - pair[-1]) if pair else 0
                                                       for pair in cluster]) / sl_costs[centroids[idx][0]][
                                                      centroids[idx][1]]
                                                  for idx, cluster in enumerate(clusters)])
            # for cluster in clusters:
            #     for pair in cluster:
            #         dist_sum += pair[-1]  # The last element in the tuple is the latency.

            # Store the best value. -- including the best centroids and clusters.
            # If it continues to win for 5 times, break the loop.
            if len([True for cluster in clusters if len(cluster)]) > 0:
                if cluster_reduction_per_cost > best_clusters_total_reduction_per_cost:
                    best_centroids = centroids
                    best_clusters = clusters
                    best_clusters_total_reduction_per_cost = cluster_reduction_per_cost
                    best_clusters_wins = 1  # This is the case a new clusters wins. So set the # of wins to 1
                else:
                    best_clusters_wins += 1  # This is the case a cluster continues to win.

            # Doesn't improve for 5 times stop doing the clustering.
            if best_clusters_wins == 5:
                break

            # Part3: Update Stage
            # If it doesn't win, go into the update stage.
            # At this time, clusters are given, need to modify the centroids.
            if any([set([sd_pair[0] for sd_pair in clusters[k] if sd_pair]) !=
                    set([sd_pair[0] for sd_pair in old_clusters[k] if sd_pair]) for k in range(K)]):
                for k in range(K):  # Traverse through each cluster.
                    if len(clusters[k]) == 0:
                        continue  # an empty cluster, we will randomly pick one later
                    best_sl_total_reduction_per_cost = -float('inf')
                    best_sl_for_cluster_k = None
                    for i in range(n):  # For each cluster, we try for all SLs.
                        for j in range(i + 1, n):
                            if sl_costs[i][j] == float('inf') or (
                                    is_latency_objective and sl_costs[i][j] > cost_budget):
                                continue
                            # SLS have to be disjoint
                            if any([not tree_node.is_disjoint(
                                    shortest_paths.get_shortest_path_without_update(orig_graph.get_node(i),
                                                                                    orig_graph.get_node(j)),
                                    shortest_paths.get_shortest_path_without_update(orig_graph.get_node(ii),
                                                                                    orig_graph.get_node(jj)))
                                    for (ii, jj) in [centroids[kk] for kk in range(k)]]):
                                continue
                            sum_sl_reduction = 0
                            # The following FOR loop calculates the total sum of latency from one cluster to an SL.
                            for pair in clusters[k]:
                                src = srcs[pair[0]]
                                dst = dsts[pair[0]]
                                dist_with_sl = cluster_predist._distance_with_sl(src, dst, shortest_paths,
                                                                                 orig_graph.get_node(i),
                                                                                 orig_graph.get_node(j))
                                # The distance of this pair using the SL would be the smaller one.
                                if not is_latency_objective:
                                    sum_sl_reduction += (orig_latency[pair[0]] - dist_with_sl[-1]) / orig_latency[
                                        pair[0]]
                                else:
                                    sum_sl_reduction += (orig_latency[pair[0]] - dist_with_sl[-1])

                            if sum_sl_reduction > best_sl_total_reduction_per_cost:  # / sl_costs[i][j] > best_sl_total_reduction_per_cost:
                                best_sl_total_reduction_per_cost = sum_sl_reduction  # / sl_costs[i][j]
                                best_sl_for_cluster_k = (i, j)
                    # Below: k represent the index of each cluster. Thus it applies to the index of centroids
                    centroids[k] = best_sl_for_cluster_k

            old_clusters = clusters
            empty_clusters_count = sum([len(cluster) == 0 for cluster in clusters])
            if empty_clusters_count > 0:
                if not is_latency_objective:
                    sl_indices = [(i, j) for i in range(n) for j in range(i + 1, n) if sl_costs[i][j] != float('inf')
                                  and (i, j) not in centroids]
                else:
                    sl_indices = [(i, j) for i in range(n) for j in range(i + 1, n) if sl_costs[i][j] < cost_budget
                                  and (i, j) not in centroids]
                rand_centroids = [sl_indices[ri] for ri in random.sample(range(len(sl_indices)), empty_clusters_count)]
                cnt = 0
                for k in range(K):
                    if len(clusters[k]) > 0:
                        continue
                    centroids[k] = rand_centroids[cnt]
                    cnt += 1

        if is_latency_objective:
            # check if the set of clusters satisfies the cost budget
            while sum([sl_costs[best_centroids[k][0]][best_centroids[k][1]] for k in range(K)
                       if len(clusters[k]) > 0]) > cost_budget:
                # replacing the SLs with high cost by selecting a part of that SL
                # each time, a link is subtracted
                new_centroids_found = False
                for cluster_idx, centroid in enumerate(best_centroids):
                    if len(clusters[cluster_idx]) == 0:
                        continue
                    i, j = centroid
                    if sl_costs[i][j] > cost_budget / K:
                        changed, new_centroid = cluster_predist._shorten_sl(srcs, dsts, centroid,
                                                                            best_clusters[cluster_idx], shortest_paths,
                                                                            orig_graph, orig_latency, sl_costs)
                        new_centroids_found |= changed
                        best_centroids[cluster_idx] = new_centroid
                if not new_centroids_found:
                    break
            # reassign (s,d) pairs and creates the final sls
            best_clusters = [[] for _ in range(K)]
            for sd_idx in range(len(srcs)):
                src, dst = srcs[sd_idx], dsts[sd_idx]
                min_dist, best_cluster_idx = float('inf'), None
                best_tuple = None
                for cluster_idx in range(K):
                    i, j = best_centroids[cluster_idx]
                    dist_with_sl = cluster_predist._distance_with_sl(src, dst, shortest_paths, orig_graph.get_node(i),
                                                                     orig_graph.get_node(j))
                    if dist_with_sl[-1] < min_dist:
                        best_cluster_idx = cluster_idx
                        min_dist = dist_with_sl[-1]
                        best_tuple = dist_with_sl
                if min_dist < orig_latency[sd_idx]:
                    best_clusters[best_cluster_idx].append((sd_idx, *best_tuple))
        return best_centroids, best_clusters

    @staticmethod
    def _shorten_sl(srcs: List[qNode], dsts: List[qNode], current_centroid: tuple[int, int], cluster: List,
                    shortest_path: dp_shortest_path, orig_graph: qGraph, orig_latency: List[float],
                    sl_costs: List[List[float]]):
        def sl_nodes(node: tree_node) -> List[int]:
            if node is None or node.data is None:
                return []
            if isinstance(node.data, qChannel):
                return [node.data.this.id, node.data.other.id]
            return sl_nodes(node.left) + sl_nodes(node.right)

        nodes = sl_nodes(shortest_path.get_shortest_path_without_update(orig_graph.get_node(current_centroid[0]),
                                                                        orig_graph.get_node(current_centroid[1])))
        if len(nodes) <= 2:
            # no new sl
            return [False, current_centroid]

        # two options: remove the leftmost or rightmost link
        left_sl = sorted((nodes[0], nodes[-2]))  # rightmost link removed
        right_sl = sorted((nodes[1], nodes[-1]))  # leftmost link removed
        sl_candidates = [left_sl, right_sl]
        best_metric = -float('inf')
        best_new_centroid = None
        for i, j in sl_candidates:
            if sl_costs[i][j] == float('inf'):
                continue
            reduction_with_sl = sum([orig_latency[pair[0]] -
                                     cluster_predist._distance_with_sl(src=srcs[pair[0]],
                                                                       dst=dsts[pair[0]],
                                                                       shortest_paths=shortest_path,
                                                                       sl_src=orig_graph.get_node(i),
                                                                       sl_dst=orig_graph.get_node(j))[-1]
                                     for pair in cluster])
            if reduction_with_sl / sl_costs[i][j] > best_metric:
                best_metric = reduction_with_sl / sl_costs[i][j]
                best_new_centroid = (i, j)
        return [False, current_centroid] if best_metric == -float('inf') else [True, best_new_centroid]

    @staticmethod
    def _distance_with_sl(src: qNode, dst: qNode, shortest_paths: dp_shortest_path, sl_src: qNode,
                          sl_dst: qNode, per_cost: bool = False):
        # calculate latency as distance w.r.t the sl (in both directions)
        left_tree_node1 = shortest_paths.get_shortest_path_without_update(src, sl_src)
        right_tree_node1 = shortest_paths.get_shortest_path_without_update(sl_dst, dst)
        temp_dist1, bool1 = greedy_predist.latency_with_sl(left_tree_node1, src, sl_src, sl_dst,
                                                           right_tree_node1, dst)
        # Case 2, reversing
        left_tree_node2 = shortest_paths.get_shortest_path_without_update(src, sl_dst)
        right_tree_node2 = shortest_paths.get_shortest_path_without_update(sl_src, dst)
        temp_dist2, bool2 = greedy_predist.latency_with_sl(left_tree_node2, src, sl_dst, sl_src,
                                                           right_tree_node2, dst)

        return (False, bool1, left_tree_node1, right_tree_node1, temp_dist1) if temp_dist1 <= temp_dist2 else \
            (True, bool2, left_tree_node2, right_tree_node2, temp_dist2)

    @property
    def get_sls(self) -> List[SL]:
        return self._best_sls

    @property
    def get_final_paths(self) -> List:
        return self._sd_path_with_sls

    @property
    def average_latency(self) -> float:
        return self._average_latency
