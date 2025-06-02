import sys

sys.path.append("..")

from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB
from commons.qGraph import qGraph
from commons.qNode import qNode
from commons.qChannel import qChannel
from commons.Point import Point
from collections import namedtuple, defaultdict
from typing import List
from commons.tree_node import tree_node

h_node = namedtuple("h_node", {"type", "v_nodes", "idd"})  # type={"start", "term", "gen", "avail", "prod"}
h_edge = namedtuple("h_edge", {"type", "head", "tail", "idd"})  # type={"single", "multiple"}
START_TYPE, GEN_TYPE, AVAIL_TYPE, PROD_TYPE, TERM_TYPE = "start", "gen", "avail", "prod", "term"
ZERO_APPX = 1e-7
CHILD_TO_PARENT_ENT_RATIO = 1e-1


class optimal_linear_programming:

    def __init__(self, G: qGraph, sources: List[qNode], destinations: List[qNode], solver_type: str = "gurobi",
                 node_capacity_division_constant: int = 1, max_min: bool = False):
        self._hv, self._he = [], []
        self._v_to_ind = {v: ind for ind, v in enumerate(G.V)}
        self._he_to_ind = {}
        self._hv_to_ind = {}
        self._trees = []
        self._node_capacity_division_constant = node_capacity_division_constant
        self._max_min = max_min
        self._solver_type = solver_type.lower()
        self.create_hyper_graph(G, sources, destinations)
        if solver_type == "gurobi":
            self._solver = self.solve(G, sources, destinations)
        else:
            self._solver = self.solve_orig(G, sources, destinations)

        self._construct_tree(G, sources, destinations)

    def create_hyper_graph(self, G: qGraph, sources: List[qNode], destinations: List[qNode]):
        # ********* create hyper nodes **************
        self._hv.append(h_node(type=START_TYPE, v_nodes=None, idd=START_TYPE))
        self._hv.append(h_node(type=TERM_TYPE, v_nodes=None, idd=TERM_TYPE))
        # create gen nodes
        for e in G.E:
            nodes = sorted([self._v_to_ind[e.this], self._v_to_ind[e.other]])
            self._hv.append(h_node(type=GEN_TYPE, v_nodes=(nodes[0], nodes[1]),
                                   idd="{type}-({this},{other})".format(type=GEN_TYPE, this=nodes[0], other=nodes[1])))
        # create avail nodes
        self._hv += [h_node(type=AVAIL_TYPE, idd="{type}-({n1},{n2})".format(type=AVAIL_TYPE, n1=i, n2=j),
                            v_nodes=(i, j)) for i in range(len(G.V)) for j in range(i + 1, len(G.V))]
        # create prod nodes
        self._hv += [h_node(type=PROD_TYPE, idd="{type}-({n1},{n2})".format(type=PROD_TYPE, n1=i, n2=j), v_nodes=(i, j))
                     for i in range(len(G.V)) for j in range(i + 1, len(G.V))]
        for ind, hv in enumerate(self._hv):
            self._hv_to_ind[hv.idd] = ind

        # *********** create hyper edges *************
        # ** start to gen
        for e in G.E:
            nodes = sorted([self._v_to_ind[e.this], self._v_to_ind[e.other]])
            gen_id = GEN_TYPE + "-({this},{other})".format(this=nodes[0], other=nodes[1])
            self._he.append(h_edge(type="s", head=self._hv_to_ind[START_TYPE],
                                   tail=self._hv_to_ind[gen_id],
                                   idd=START_TYPE + "-to-" + gen_id))
        # ** gen to avail
        for e in G.E:
            nodes = sorted([self._v_to_ind[e.this], self._v_to_ind[e.other]])
            gen_id = GEN_TYPE + "-({this},{other})".format(this=nodes[0], other=nodes[1])
            avail_id = AVAIL_TYPE + "-({this},{other})".format(this=nodes[0],
                                                               other=nodes[1])
            self._he.append(h_edge(type="s", head=self._hv_to_ind[gen_id], tail=self._hv_to_ind[avail_id],
                                   idd=GEN_TYPE + "-to-" + avail_id))
        # ** prod to avail
        ids = [[i, j] for i in range(len(G.V)) for j in range(i + 1, len(G.V))]
        for idd in ids:
            prod_id = PROD_TYPE + "-({n1},{n2})".format(n1=idd[0], n2=idd[1])
            avail_id = AVAIL_TYPE + "-({n1},{n2})".format(n1=idd[0], n2=idd[1])
            self._he.append(h_edge(type="s", head=self._hv_to_ind[prod_id], tail=self._hv_to_ind[avail_id],
                                   idd=PROD_TYPE + "-to-" + AVAIL_TYPE + "-({n1}, {n2})".format(n1=idd[0], n2=idd[1])))

        # ** avail-avail to prod
        for avail1_idx in range(len(ids)):
            for avail2_idx in range(avail1_idx + 1, len(ids)):
                if len(set(ids[avail1_idx] + ids[avail2_idx])) == 3:
                    avail1_id = AVAIL_TYPE + "-({n1},{n2})".format(n1=ids[avail1_idx][0], n2=ids[avail1_idx][1])
                    avail2_id = AVAIL_TYPE + "-({n1},{n2})".format(n1=ids[avail2_idx][0], n2=ids[avail2_idx][1])
                    prod = sorted(list(set(ids[avail1_idx] + ids[avail2_idx]) -
                                       set(ids[avail1_idx]).intersection(set(ids[avail2_idx]))))
                    prod_id = PROD_TYPE + "-({n1},{n2})".format(n1=prod[0], n2=prod[1])
                    self._he.append(h_edge(type="m", head=[self._hv_to_ind[avail1_id], self._hv_to_ind[avail2_id]],
                                           tail=self._hv_to_ind[prod_id],
                                           idd="{avail_tp}-(({n1},{n2}),({n3},{n4}))-to-{prod_tp}-({n5},{n6})".format(
                                               avail_tp=AVAIL_TYPE, prod_tp=PROD_TYPE,
                                               n1=ids[avail1_idx][0], n2=ids[avail1_idx][1], n3=ids[avail2_idx][0],
                                               n4=ids[avail2_idx][1], n5=prod[0], n6=prod[1])))

        # ** avail to term
        for src, dst in zip(sources, destinations):
            avail = sorted([self._v_to_ind[src], self._v_to_ind[dst]])
            avail_id = AVAIL_TYPE + "-({n1},{n2})".format(n1=avail[0], n2=avail[1])
            self._he.append(h_edge(type="s", head=self._hv_to_ind[avail_id], tail=self._hv_to_ind[TERM_TYPE],
                                   idd=avail_id + "-to-" + TERM_TYPE))

        for idx, edge in enumerate(self._he):
            self._he_to_ind[edge.idd] = idx

    def solve(self, G: qGraph, sources: List[qNode], destinations: List[qNode]):
        n = len(self._he)
        model = gp.Model("Flow")

        for idd in self._he_to_ind:
            self._he_to_ind[idd] = model.addVar(lb=0, name=str(idd))

        # create c (obj) (avail - term)
        obj = gp.LinExpr()
        list_of_obj = []
        for src, dst in zip(sources, destinations):
            avail = sorted([self._v_to_ind[src], self._v_to_ind[dst]])
            e_id = AVAIL_TYPE + "-({n1},{n2})-to-".format(n1=avail[0], n2=avail[1]) + TERM_TYPE
            obj += self._he_to_ind[e_id]
            list_of_obj.append(self._he_to_ind[e_id])
        model.setObjective(obj, GRB.MAXIMIZE)
        if self._max_min:
            max_min = model.addVar(lb=0, name="general_min_var")
            model.addGenConstrMin(max_min, list_of_obj, name="general_mini_cons")

        # A_ub (ineqs_lhs), b_ub (ineqs_rhs)
        # original edges capacities and original nodes capacities
        ineqs_lhs = []
        ineqs_rhs = []
        node_edges = defaultdict(list)
        for e in G.E:
            # ineq_lhs = [0] * n
            nodes = sorted([self._v_to_ind[e.this], self._v_to_ind[e.other]])
            e_id = START_TYPE + "-to-" + GEN_TYPE + "-({this},{other})".format(this=nodes[0], other=nodes[1])
            # ineq_lhs[self._he_to_ind[e_id]] = 1
            # ineqs_lhs.append(ineq_lhs)
            # ineqs_rhs.append(optimal_linear_programming._edge_capacity(e))
            lhs = gp.LinExpr()
            lhs += self._he_to_ind[e_id]
            model.addConstr(lhs <= optimal_linear_programming._edge_capacity(e, self._node_capacity_division_constant))

            # for nodes capacities
            edge_to_node_const = 1.0 / (e.this.gen_success_rate * e.other.gen_success_rate *
                                        e.channel_success_rate ** 2 * e.optical_bsm_rate)
            node_edges[self._v_to_ind[e.this]].append((self._he_to_ind[e_id], edge_to_node_const))
            node_edges[self._v_to_ind[e.other]].append((self._he_to_ind[e_id], edge_to_node_const))

        # create ineqs for nodes
        for v_id, v_values in node_edges.items():
            lhs = gp.LinExpr()
            for v_value in v_values:
                lhs += v_value[1] * v_value[0]
            model.addConstr(lhs <= G.V[v_id].max_capacity)

        # A_eq (eqs_lhs), b_eq
        # hyper nodes equalities
        eqs_hv_nodes = [v_idx for v_idx, v in enumerate(self._hv)
                        if v.idd not in [START_TYPE, TERM_TYPE]]  # all hv nodes without start and term
        idx_to_eqs_hv_nodes = {eq_node: idx for idx, eq_node in enumerate(eqs_hv_nodes)}
        eqs_lhs = [gp.LinExpr() for _ in range(len(eqs_hv_nodes))]
        eqs_rhs = [0] * len(eqs_hv_nodes)
        for he in self._he:
            he_idx = self._he_to_ind[he.idd]
            if he.type == "s":
                if he.head in idx_to_eqs_hv_nodes:
                    eqs_lhs[idx_to_eqs_hv_nodes[he.head]] += -1 * he_idx  # from head
                if he.tail in idx_to_eqs_hv_nodes:
                    eqs_lhs[idx_to_eqs_hv_nodes[he.tail]] += he_idx  # to tail
            else:  # he.type = "m"
                for h in he.head:  # from head
                    if h in idx_to_eqs_hv_nodes:
                        eqs_lhs[idx_to_eqs_hv_nodes[h]] += -1 * he_idx
                node = G.V[set(self._hv[he.head[0]].v_nodes).intersection(set(self._hv[he.head[1]].v_nodes)).pop()]
                loss_factor = node.bsm_success_rate * 2 / 3
                if he.tail in idx_to_eqs_hv_nodes:
                    eqs_lhs[idx_to_eqs_hv_nodes[he.tail]] += loss_factor * he_idx
        for eq in eqs_lhs:
            model.addConstr(eq == 0)

        model.Params.OutputFlag = 0
        model.optimize()
        return model

    def solve_orig(self, G: qGraph, sources: List[qNode], destinations: List[qNode]):
        n = len(self._he)
        # create c (obj) (avail - term)
        obj = [0] * n
        for src, dst in zip(sources, destinations):
            avail = sorted([self._v_to_ind[src], self._v_to_ind[dst]])
            e_id = AVAIL_TYPE + "-({n1},{n2})-to-".format(n1=avail[0], n2=avail[1]) + TERM_TYPE
            obj[self._he_to_ind[e_id]] = -1

        # A_ub (ineqs_lhs), b_ub (ineqs_rhs)
        # original edges capacities and original nodes capacities
        bounds = [(0, float('inf'))] * n
        ineqs_lhs = []
        ineqs_rhs = []
        node_edges = defaultdict(list)
        for e in G.E:
            # ineq_lhs = [0] * n
            nodes = sorted([self._v_to_ind[e.this], self._v_to_ind[e.other]])
            e_id = START_TYPE + "-to-" + GEN_TYPE + "-({this},{other})".format(this=nodes[0], other=nodes[1])
            # ineq_lhs[self._he_to_ind[e_id]] = 1
            # ineqs_lhs.append(ineq_lhs)
            # ineqs_rhs.append(optimal_linear_programming._edge_capacity(e))
            bounds[self._he_to_ind[e_id]] = (0, optimal_linear_programming._edge_capacity(
                e, self._node_capacity_division_constant))
            # for nodes capacities
            edge_to_node_const = 1.0 / (e.this.gen_success_rate * e.other.gen_success_rate *
                                        e.channel_success_rate ** 2 * e.optical_bsm_rate)
            node_edges[self._v_to_ind[e.this]].append((self._he_to_ind[e_id], edge_to_node_const))
            node_edges[self._v_to_ind[e.other]].append((self._he_to_ind[e_id], edge_to_node_const))

        # create ineqs for nodes
        for v_id, v_values in node_edges.items():
            ineq_lhs = [0] * n
            for v_value in v_values:
                ineq_lhs[v_value[0]] = v_value[1]
            ineqs_lhs.append(ineq_lhs)
            # node capacity
            ineqs_rhs.append(G.V[v_id].max_capacity)

        # A_eq (eqs_lhs), b_eq
        # hyper nodes equalities
        eqs_hv_nodes = [v_idx for v_idx, v in enumerate(self._hv)
                        if v.idd not in [START_TYPE, TERM_TYPE]]  # all hv nodes without start and term
        idx_to_eqs_hv_nodes = {eq_node: idx for idx, eq_node in enumerate(eqs_hv_nodes)}
        eqs_lhs = [[0] * n for _ in range(len(eqs_hv_nodes))]
        eqs_rhs = [0] * len(eqs_hv_nodes)
        for he in self._he:
            he_idx = self._he_to_ind[he.idd]
            if he.type == "s":
                if he.head in idx_to_eqs_hv_nodes:
                    eqs_lhs[idx_to_eqs_hv_nodes[he.head]][he_idx] = -1  # from head
                if he.tail in idx_to_eqs_hv_nodes:
                    eqs_lhs[idx_to_eqs_hv_nodes[he.tail]][he_idx] = 1  # to tail
            else:  # he.type = "m"
                for h in he.head:  # from head
                    if h in idx_to_eqs_hv_nodes:
                        eqs_lhs[idx_to_eqs_hv_nodes[h]][he_idx] = -1
                node = G.V[set(self._hv[he.head[0]].v_nodes).intersection(set(self._hv[he.head[1]].v_nodes)).pop()]
                if he.tail in idx_to_eqs_hv_nodes:
                    eqs_lhs[idx_to_eqs_hv_nodes[he.tail]][he_idx] = node.bsm_success_rate * 2 / 3

        return linprog(c=obj,
                       A_ub=ineqs_lhs, b_ub=ineqs_rhs,
                       A_eq=eqs_lhs, b_eq=eqs_rhs, bounds=bounds)

    @property
    def max_flow(self):
        if self._solver_type == "gurobi":
            return self._solver.objVal
        else:
            return -self._solver.fun

    def _construct_tree(self, G: qGraph, sources: List[qNode], destinations: List[qNode]):
        for src, dest in zip(sources, destinations):
            src_id, dst_id = self._v_to_ind[src], self._v_to_ind[dest]
            lp_src_id, lp_dst_id = sorted([src_id, dst_id])
            var = (self._he_to_ind["{avail}-({src},{dst})-to-{term}".format(
                avail=AVAIL_TYPE, src=lp_src_id, dst=lp_dst_id, term=TERM_TYPE)])
            if self._solver_type == "gurobi":
                val = var.X
            else:
                val = self._solver.x[var]
            trees, _ = self._tree_helper(G, src_id, dst_id, val, {src_id, dst_id})
            self._trees.append(trees)
            # self._trees.append([optimal_linear_programming._fix_tree_order(tree, src, dest) for tree in trees])

    def _tree_helper(self, G: qGraph, src_id: int, dst_id: int, parent_ent_rate: float, tree_nodes: set):
        lp_src_id, lp_dst_id = sorted([src_id, dst_id])
        trees = []
        nodes = []
        # direct edge from src to dst
        if "{start}-to-{gen}-({left},{right})".format(start=START_TYPE, gen=GEN_TYPE, left=lp_src_id, right=lp_dst_id) \
                in self._he_to_ind:
            ent_rate = parent_ent_rate
            if ent_rate == float('inf'):
                var = (self._he_to_ind["{start}-to-{gen}-({left},{right})".format(
                    start=START_TYPE, gen=GEN_TYPE, left=lp_src_id, right=lp_dst_id)])
                if self._solver_type == "gurobi":
                    ent_rate = var.X
                else:
                    ent_rate = self._solver.x[var]
            trees.append(tree_node(data=qChannel(G.V[src_id], G.V[dst_id]), avr_ent_time=1.0 / ent_rate))
            nodes.append({src_id, dst_id})
            return trees, nodes
        for node, root_id in self._v_to_ind.items():
            if root_id in tree_nodes:
                continue
            left_nodes, right_nodes = sorted([lp_src_id, root_id]), sorted([root_id, lp_dst_id])
            he_root_edge = "{avail_tp}-(({left1},{left2}),({right1},{right2}))-to-{prod_tp}-({src},{dst})".format(
                avail_tp=AVAIL_TYPE, left1=left_nodes[0], left2=left_nodes[1],
                right1=right_nodes[0], right2=right_nodes[1], src=lp_src_id, dst=lp_dst_id, prod_tp=PROD_TYPE)
            if self._solver_type == "gurobi":
                tree_rate = (self._he_to_ind[he_root_edge]).X
            else:
                tree_rate = self._solver.x[(self._he_to_ind[he_root_edge])]
            if tree_rate > ZERO_APPX and tree_rate / parent_ent_rate > CHILD_TO_PARENT_ENT_RATIO:
                tree_nodes.add(root_id)
                left_sub_trees, left_sub_tree_nodes = self._tree_helper(G, src_id, root_id, tree_rate, tree_nodes)
                right_sub_trees, right_sub_tree_nodes = self._tree_helper(G, root_id, dst_id, tree_rate, tree_nodes)

                tree_nodes.remove(root_id)
                for left_idx, left_sub_tree in enumerate(left_sub_trees):
                    for right_idx, right_sub_tree in enumerate(right_sub_trees):
                        intersect = left_sub_tree_nodes[left_idx].intersection(right_sub_tree_nodes[right_idx])
                        # merging if and only if root is the only common nodes from left and right children
                        if len(intersect) == 1 and root_id in intersect:
                            trees.append(tree_node(data=node, left=left_sub_tree, right=right_sub_tree,
                                                   avr_ent_time=(1.5 * max(left_sub_tree.avr_ent_time,
                                                                           right_sub_tree.avr_ent_time)) /
                                                                node.bsm_success_rate))
                            nodes.append(left_sub_tree_nodes[left_idx].union(right_sub_tree_nodes[right_idx]))
        return trees, nodes

    @staticmethod
    def _fix_tree_order(node: tree_node, leftest_node: qNode, rightest_node: qNode):
        if isinstance(node.data, qChannel):
            if node.data.this is leftest_node:
                return node
            return tree_node(data=qChannel(leftest_node, rightest_node), avr_ent_time=node.avr_ent_time)
        new_left_subtree = optimal_linear_programming._fix_tree_order(node.left,
                                                                      leftest_node=leftest_node,
                                                                      rightest_node=node.data)
        new_right_subtree = optimal_linear_programming._fix_tree_order(node.right,
                                                                       leftest_node=node.data,
                                                                       rightest_node=rightest_node)
        return tree_node(data=node.data, left=new_left_subtree, right=new_right_subtree)

    @property
    def trees(self) -> List[List[tree_node]]:
        max_rate = max([max([1.0/tree.avr_ent_time for tree in trees]) for trees in self._trees if len(trees) > 0])
        return [[tree for tree in trees if 1.0/tree.avr_ent_time > 0.02 * max_rate] for trees in self._trees]

    @staticmethod
    def _edge_capacity(e: qChannel, division_constant: int = 1) -> float:
        # use nodes' generation rate. min gives us maximum possible rate between two nodes
        # different than qChannel capacities, full node capacities
        # division_constant is used when one node is a common for many trees.
        # Setting this constant let all get some share
        max_flow_nodes = min(e.this.max_capacity / division_constant, e.other.max_capacity / division_constant) * \
                         e.this.gen_success_rate * e.other.gen_success_rate * e.channel_success_rate ** 2 * \
                         e.optical_bsm_rate
        return min(max_flow_nodes, e.max_channel_capacity)


if __name__ == "__main__":
    V = [qNode(20, Point(0, 0)), qNode(30, Point(0, 5e3)), qNode(20, Point(0, 10e3)), qNode(20, Point(0, 15e3)),
         qNode(20, Point(10e3, 10e3))]
    E = [qChannel(V[2], V[3], 5), qChannel(V[2], V[1], 5), qChannel(V[1], V[0], 5), qChannel(V[0], V[4]),
         qChannel(V[4], V[3])]
    graph = qGraph(V, E)
    opt_lin_prog = optimal_linear_programming(graph, [graph.V[3]], [graph.V[0]])
    print("gurobi")
    print(opt_lin_prog.max_flow)
    print(opt_lin_prog.trees)
