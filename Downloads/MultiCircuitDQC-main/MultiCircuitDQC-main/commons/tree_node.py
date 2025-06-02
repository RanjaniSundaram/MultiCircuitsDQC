from .qNode import qNode
from .qChannel import qChannel
from typing import List
import functools


class tree_node:
    data, left, right = None, None, None  # node where bsm happens for its left and right subtrees
    avr_ent_time, one_round_ent_time = float('inf'), float('inf')  # average total time, average successful time
    classical_time = 0  # maximum of classical time duration between node and its children
    left_avr_ent_time, right_avr_ent_time = float('inf'), float('inf')

    def __init__(self, data=None, left: 'tree_node' = None, right: "tree_node" = None,
                 avr_ent_time: float = float('inf'), one_round_ent_time: float = float('inf'),
                 classical_time: float = 0,
                 left_avr_ent_time: float = float('inf'), right_avr_ent_time: float = float('inf')):
        # left_edge: qChannel = None, right_edge: qChannel = None,
        self.data, self.left, self.right = data, left, right
        # self.left_edge = left_edge  # left_edge = (left -> node)
        # self.right_edge = right_edge  # right_edge = (node -> right)
        self.avr_ent_time, self.one_round_ent_time = avr_ent_time, one_round_ent_time
        self.classical_time = classical_time
        self.left_avr_ent_time, self.right_avr_ent_time = left_avr_ent_time, right_avr_ent_time
        # self.IS_SL, self.SL_RATE = IS_SL, SL_RATE

    def clone(self, node: "tree_node"):
        # TODO not completed. needs recursive
        return tree_node(self.data)

    def __str__(self):
        '''Sample output: for a tree path 0--6--3--5
            Tree Node:
            middle = 3
            left   = 6
            right  = 3--5
            avr_ent_time = 0.017752431413124176
            one_round_ent_time = 0.0036126495037156186
        '''
        describe = 'Tree Node:\n'
        if isinstance(self.data, qNode):
            describe += f'middle = {self.data.id}\n'
        if isinstance(self.data, qChannel):
            describe += f'middle = {self.data.this.id}--{self.data.other.id}\n'
        if self.left and isinstance(self.left.data, qNode):
            describe += f'left   = {self.left.data.id}\n'
        if self.left and isinstance(self.left.data, qChannel):
            describe += f'left   = {self.left.data.this.id}--{self.left.data.other.id}\n'
        if self.right and isinstance(self.right.data, qNode):
            describe += f'right  = {self.right.data.id}\n'
        if self.right and isinstance(self.right.data, qChannel):
            describe += f'right  = {self.right.data.this.id}--{self.right.data.other.id}\n'
        describe += f'avr_ent_time = {self.avr_ent_time:.5e}\none_round_ent_time = {self.one_round_ent_time:.5e}'
        return describe

    def clone(self) -> 'tree_node':
        if isinstance(self.data, qChannel):
            return tree_node(self.data, avr_ent_time=self.avr_ent_time, one_round_ent_time=self.one_round_ent_time,
                             classical_time=self.classical_time)
        return tree_node(data=self.data, left=self.left.clone(), right=self.right.clone(),
                         avr_ent_time=self.avr_ent_time, one_round_ent_time=self.one_round_ent_time,
                         classical_time=self.classical_time, left_avr_ent_time=self.left_avr_ent_time,
                         right_avr_ent_time=self.right_avr_ent_time)

    @staticmethod
    def print_tree2(node: 'tree_node'):
        if node is None or node.data is None:
            return ""
        if isinstance(node.data, qChannel):
            return f"({node.data.this.id} -- {node.data.other.id}: rate:{1.0/node.avr_ent_time:.4f} 1/s)"
        left_subtree = tree_node.print_tree2(node.left)
        right_subtree = tree_node.print_tree2(node.right)
        return f"{node.data.id}\n({left_subtree} , {right_subtree})"

    @staticmethod
    def print_tree(node: 'tree_node', level=0):
        if node is not None and node.data is not None:
            if isinstance(node.data, qChannel):
                print(" " * 10 * level + "->", f"({node.data.this.id} -- {node.data.other.id}: "
                                               f"rate: {float('inf') if node.avr_ent_time== 0 else 1.0/node.avr_ent_time:.4f} 1/s)")
                return
            tree_node.print_tree(node.right, level + 1)
            print(" " * 10 * level + '->' + str(node.data.id),
                  f": rate: {float('inf') if node.avr_ent_time == 0 else 1.0/node.avr_ent_time:.4f} 1/s")
            tree_node.print_tree(node.left, level + 1)

    @staticmethod
    def print_tree_str(node: 'tree_node', level=0):
        if node is not None and node.data is not None:
            if isinstance(node.data, qChannel):
                return f"\n" + f" " * 10 * level + "->" + f"({node.data.this.id} --" \
                                                 f" {node.data.other.id}: rate: " \
                                                          f"{float('inf') if node.avr_ent_time == 0 else 1.0 / node.avr_ent_time:.4f} 1/s)"

            tree = tree_node.print_tree_str(node.right, level + 1)
            tree += f"\n" + f" " * 10 * level + '->' + str(node.data.id) + \
                    f": rate: {float('inf') if node.avr_ent_time == 0 else 1.0 / node.avr_ent_time:.4f} 1/s"
            tree += tree_node.print_tree_str(node.left, level + 1)
            return tree

    @staticmethod
    @functools.lru_cache(10240)
    def used_nodes_edges(node: 'tree_node', join_character: str = "-"):
        if node is None:
            return set(), set()
        if isinstance(node.data, qChannel):
            return {node.data.this.id, node.data.other.id}, {f"{node.data.this.id}{join_character}{node.data.other.id}",
                                                             f"{node.data.other.id}{join_character}{node.data.this.id}"}
        left_nodes, left_edges = tree_node.used_nodes_edges(node.left, join_character)
        right_nodes, right_edges = tree_node.used_nodes_edges(node.right, join_character)
        return left_nodes.union(right_nodes), left_edges.union(right_edges)

    @staticmethod
    def max_node_occurrences(node: 'tree_node'):
        def _helper(node: 'tree_node', counter: dict):
            if node is None:
                return
            if isinstance(node.data, qChannel):
                if node.data.this.id not in counter:
                    counter[node.data.this.id] = 0
                counter[node.data.this.id] += 1
                if node.data.other.id not in counter:
                    counter[node.data.other.id] = 0
                counter[node.data.other.id] += 1
                return
            _helper(node.left, counter)
            _helper(node.right, counter)
        counter = {}
        _helper(node, counter)
        return max([value for value in counter.values()]) if len(counter) > 0 else 0

    @staticmethod
    @functools.lru_cache(10240)
    def tree_nodes(tree: 'tree_node') -> List[int]:
        """Returns a list of nodes id in pre-order traversal"""
        if tree is None or tree.data is None:
            return []
        if isinstance(tree.data, qChannel):
            return [tree.data.this.id, tree.data.other.id]
        return tree_node.tree_nodes(tree.left) + tree_node.tree_nodes(tree.right)

    @staticmethod
    def is_sub_tree(main_tree: 'tree_node', sub_tree: 'tree_node') -> bool:
        """This function checks if sub_tree is a sub-tree of main_tree.
        """
        def helper(main_nodes: List[int], sub_nodes: List[int]):
            if len(main_nodes) == 0 or len(sub_nodes) > len(main_nodes):  # nothing can't be superset of anything
                return False
            if len(sub_nodes) == 0:  # nothing is subset of everything
                return True
            lo, hi = 0, len(main_nodes) - 1
            while lo <= hi and main_nodes[lo] != sub_nodes[0]:
                lo += 1
            while hi >= lo and main_nodes[hi] != sub_nodes[-1]:
                hi -= 1
            if hi - lo + 1 != len(sub_nodes):
                return False
            for idx in range(len(sub_nodes)):
                if sub_nodes[idx] != main_nodes[idx+lo]:
                    return False
            return True
        main_nodes, sub_nodes = tree_node.tree_nodes(main_tree), tree_node.tree_nodes(sub_tree)
        return helper(main_nodes, sub_nodes) or helper(main_nodes, list(reversed(sub_nodes)))

    @property
    def leftest_node(self):
        if self.data is None:
            return None
        cur = self
        while cur.left is not None:
            cur = cur.left
        return cur.data.this

    @property
    def rightest_node(self):
        if self.data is None:
            return None
        cur = self
        while cur.right is not None:
            cur = cur.right
        return cur.data.other

    @staticmethod
    @functools.lru_cache(10240)
    def is_disjoint(tree1: 'tree_node', tree2: 'tree_node'):
        """Check if two trees are disjoint"""
        def _helper(tree: 'tree_node'):
            # can be used for edge-level
            if isinstance(tree.data, qChannel):
                return {f"{tree.data.this.id}-{tree.data.other.id}", f"{tree.data.other.id}-{tree.data.this.id}"}
            links = _helper(tree.left)
            links.add(_helper(tree.right))
            return links
        if False:
            # edge-level
            tree1_links = _helper(tree1)
            tree2_links = _helper(tree2)
            for tree1_link in tree1_links:
                if tree1_link in tree2_links:
                    return False
            return True
        else:
            # node-level (edge-level is implicitly assumed)
            tree1_nodes = set(tree_node.tree_nodes(tree1))
            tree2_nodes = set(tree_node.tree_nodes(tree2))
            return False if tree1_nodes.intersection(tree2_nodes) else True
