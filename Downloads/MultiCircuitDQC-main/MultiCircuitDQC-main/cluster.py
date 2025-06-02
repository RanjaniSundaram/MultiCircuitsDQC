import networkx as nx
import numpy as np

def form_groups(node_dist_matrix, node_labels, k):
        """
        Form k groups of 5 nodes each from the given distance matrix, ensuring minimal internal distances.
        
        Parameters:
        - node_dist_matrix (np.ndarray): Distance matrix from Floyd-Warshall algorithm.
        - node_labels (list): List of node labels corresponding to the matrix indices.
        - k (int): Number of groups to form.
        
        Returns:
        - list: List of groups, each group is a list of 5 node labels.
        """
        n = node_dist_matrix.shape[0]
        available_indices = list(range(n))
        groups = []
        
        for _ in range(k):
            best_sum = float('inf')
            best_group = None
            
            for u in available_indices:
                # Get available candidates excluding u
                candidates = [v for v in available_indices if v != u]
                
                # Sort candidates by their distance to u
                sorted_candidates = sorted(candidates, key=lambda v: node_dist_matrix[u, v])
                
                # Check if there are enough candidates to form a group
                if len(sorted_candidates) < 4:
                    continue  # This should not happen if k is <= 10 for 50 nodes
                
                # Form the candidate group
                closest = sorted_candidates[:4]
                group = [u] + closest
                
                # Calculate sum of all pairwise distances in the group
                current_sum = 0
                for i in range(5):
                    for j in range(i+1, 5):
                        current_sum += node_dist_matrix[group[i], group[j]]
                
                # Update best group if current is better
                if current_sum < best_sum:
                    best_sum = current_sum
                    best_group = group
            
            if best_group is None:
                raise ValueError("Not enough nodes to form the requested number of groups.")
            
            # Convert indices to node labels and add to groups
            groups.append(sorted([node_labels[idx] for idx in best_group]))
            
            # Remove selected nodes from available indices
            available_indices = [v for v in available_indices if v not in best_group]
        
        return groups


def get_group_matrices_memory(network):
    """
    Get adjacency matrices for each group in the main network graph.
    
    Parameters:
    - network (NetworkGraph): The network containing the main graph.

    Returns:
    - dict: A dictionary where keys are group indices and values are adjacency matrices for memory.
    """
    # Step 1: Compute shortest path matrix and labels
    node_dist_matrix = nx.floyd_warshall_numpy(network.network_graph)
    node_labels = list(network.network_graph.nodes())
    memory_dist_matrix = nx.floyd_warshall_numpy(network.main_network_graph)

    group_matrices = {}

    # Step 2: Iterate through k (1 to 10)
    for k in range(1, 11):
        groups = form_groups(node_dist_matrix, node_labels, k)
        groups_matrix = []


        for group_idx, group in enumerate(groups):
            # Expand group indices: Each node index maps to 20 memory nodes
            expanded_indices = []
            for node in group:
                expanded_indices.extend(range(node * 20, (node + 1) * 20))

            # Extract the corresponding submatrix
            submatrix = memory_dist_matrix[np.ix_(expanded_indices, expanded_indices)]
            groups_matrix.append(submatrix)
            
            # Store the adjacency matrix
            group_matrices[k] = groups_matrix

    return groups, group_matrices

def get_group_matrices_computer(network):
    """
    Get adjacency matrices for each group in the main network graph.
    
    Parameters:
    - network (NetworkGraph): The network containing the main graph.

    Returns:
    - dict: A dictionary where keys are group indices and values are adjacency matrices for computer.
    """
    # Step 1: Compute shortest path matrix and labels
    node_dist_matrix = nx.floyd_warshall_numpy(network.network_graph)
    node_labels = list(network.network_graph.nodes())

    group_matrices = {}

    # Step 2: Iterate through k (1 to 10)
    for k in range(1, 11):
        groups = form_groups(node_dist_matrix, node_labels, k)
        groups_matrix = []


        for group_idx, group in enumerate(groups):
            # Extract the corresponding submatrix
            submatrix = node_dist_matrix[np.ix_(group, group)]
            groups_matrix.append(submatrix)
            
            # Store the adjacency matrix
            group_matrices[k] = groups_matrix

    return groups, group_matrices

