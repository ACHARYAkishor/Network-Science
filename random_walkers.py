#imports
import numpy as np
import networkx as nx

def random_walk(G, stop, W, p_stay=0, weights=None):
    """
    Performs a random walk on a graph.
    
    Parameters:
    G (networkx.classes.graph.Graph): The graph on which the random walk is performed.
    stop (int): The number of time steps to run the random walk.
    W (dict): A dictionary with the number of walkers at each node.
    p_stay (float, optional): The probability of staying at the current node. Defaults to 0.
    weights (dict, optional): A dictionary with edge weights, used for calculating probabilities. 
                               If None, all edges have equal weight. Defaults to None.
                               
    Returns:
    dict: Updated W with the final distribution of walkers.
    """
    
    # Prepare for weighted graph
    if weights is not None:
        nx.set_edge_attributes(G, weights, 'weight')

    t = 0
    N = G.number_of_nodes()
    nodes = list(G.nodes())

    while t < stop:
        temp = {node: 0 for node in nodes} # creating temprory empty dict 
        for source in nodes:
            # Retrieve neighbors
            neighbors = list(G.neighbors(source))
            if not neighbors:
                # If node is isolated, move on
                continue

            # Calculate probabilities
            if weights is None:
                # Unweighted case: probability 1/degree, considering p_stay
                degree = G.degree(source)
                prob = [(1 - p_stay) / degree for _ in neighbors]
            else:
                # Weighted case: probability proportional to edge weight, considering p_stay
                total_weight = sum(G[source][neighbor]['weight'] for neighbor in neighbors)
                prob = [(1 - p_stay) * G[source][neighbor]['weight'] / total_weight for neighbor in neighbors]

            prob.append(p_stay)  # Probability of staying at the current node

            # Perform the random walk
            targets = neighbors + [source]  # Include the current node as a possible target (stay)
            output = np.random.multinomial(W[source], prob)

            # Update temp with the results
            for target, walkers in zip(targets, output):
                temp[target] += walkers

        # Update W with the new distribution
        W = temp

        t += 1

    return W
