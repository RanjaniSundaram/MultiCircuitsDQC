from commons.qNode import qNode


class fusion_retain_node:
    """A node that holds fusion-retain-only GHZ structure"""

    def __init__(self, node: qNode = None, all_nodes: set = None,
                 end_nodes: set = None, avr_ent_time: float = float('inf'),
                 sub1: 'fusion_retain_node' = None, sub2: 'fusion_retain_node' = None):
        self.node = node
        self.all_nodes = all_nodes
        self.end_nodes, self.avr_ent_time = end_nodes, avr_ent_time
        self.sub1, self.sub2 = sub1, sub2


class general_fusion_node(fusion_retain_node):
    """A node that hols general GHZ fusion structure"""

    def __init__(self, node: qNode = None, all_nodes: set = None,
                 end_nodes: set = None, avr_ent_time: float = float('inf'),
                 sub1: 'fusion_retain_node' = None, sub2: 'fusion_retain_node' = None, is_retain: bool = False):
        super().__init__(node, all_nodes, end_nodes, avr_ent_time, sub1, sub2)
        self.is_retain = is_retain


def used_resources(fusion_node: fusion_retain_node):
    if fusion_node is None:
        return set(), set()
    if fusion_node.node is None:
        # reached to channel
        return set(fusion_node.all_nodes), set(["_".join([str(node) for node in nodes]) for nodes in
                                                [list(fusion_node.all_nodes), reversed(list(fusion_node.all_nodes))]])
    sub1_nodes, sub1_edges = used_resources(fusion_node.sub1)
    sub2_nodes, sub2_edges = used_resources(fusion_node.sub2)
    return sub1_nodes.union(sub2_nodes), sub1_edges.union(sub2_edges)


def print_fusion_tree(fusion_node: fusion_retain_node, level=0):
    if fusion_node is not None:
        if fusion_node.node is None:
            print(" " * 10 * level + "->",
                  f"({'--'.join([str(num) for num in fusion_node.all_nodes])}"
                  f"rate: {float('inf') if fusion_node.avr_ent_time== 0 else 1.0/fusion_node.avr_ent_time:.4f} 1/s)")
            return
        print_fusion_tree(fusion_node.sub1, level + 1)
        print(" " * 10 * level + '->' + str(fusion_node.node.id),
              f"GHZ-{'-'.join([str(num) for num in sorted(list(fusion_node.end_nodes))])}, "
              f"all nodes: {fusion_node.all_nodes}"
              f"rate: {float('inf') if fusion_node.avr_ent_time == 0 else 1.0/fusion_node.avr_ent_time:.4f} 1/s")
        print_fusion_tree(fusion_node.sub2, level + 1)


def print_fusion_tree_str(fusion_node: fusion_retain_node, level=0):
    if fusion_node is not None:
        if fusion_node.node is None:
            return f"\n" + f" " * 10 * level + "->" + \
                   f"({'--'.join([str(num) for num in fusion_node.all_nodes])} rate: " \
                   f"{float('inf') if fusion_node.avr_ent_time== 0 else 1.0/fusion_node.avr_ent_time:.4f} 1/s)"

        tree = print_fusion_tree_str(fusion_node.sub1, level + 1)
        tree += f"\n" + f" " * 10 * level + '->' + str(fusion_node.node.id) + \
                f"GHZ-{'-'.join([str(num) for num in sorted(list(fusion_node.end_nodes))])}," \
                f"all nodes: {fusion_node.all_nodes} " \
                f"rate: {float('inf') if fusion_node.avr_ent_time== 0 else 1.0/fusion_node.avr_ent_time:.4f} 1/s"
        tree += print_fusion_tree_str(fusion_node.sub2, level + 1)
        return tree
