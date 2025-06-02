import sys

sys.path.append("..")
from commons.qNode import qNode
from commons.tree_node import tree_node


class SL:
    """class SL (Super Link)"""
    def __init__(self, src: qNode, dst: qNode, tree: tree_node, cost: float):
        self._src, self._dst, self._tree, self._cost = src, dst, tree, cost
        self._id = "-".join([str(node_id) for node_id in tree_node.tree_nodes(tree)])

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def tree(self):
        return self._tree

    @property
    def cost(self):
        return self._cost

    @property
    def iid(self):
        return self._id

    def __eq__(self, other):
        return self.iid == other.iid

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return self.iid

    def __hash__(self):
        return hash(self.iid)
