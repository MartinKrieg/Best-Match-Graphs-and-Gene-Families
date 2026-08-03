import networkx as nx
import random
from utils import visualizeGraph

def generateHybridNetwork(geneTree : nx.DiGraph, rootID: int, num_hybridizations, geneColors):
    network = geneTree.copy()

    if len(network.edges()) < 2:
        raise ValueError("The input network must have at least two edges to perform hybridization.")

    successful_events = 0
    attempts = 0
    max_attempts = num_hybridizations * 10

    print(f"-> Iteratively inserting {num_hybridizations} hybridization events...")
    while successful_events < num_hybridizations and attempts < max_attempts:
        attempts += 1
        edges = list(network.edges())

        # 1. Randomly select two distinct edges in the network
        e1, e2 = random.sample(edges, 2)
        u1, v1 = e1
        u2, v2 = e2

        # 2. Generate unique identifiers for the new hybridization points
        h1 = f"h_{u1}_{v1}_evt{successful_events}"
        h2 = f"h_{u2}_{v2}_evt{successful_events}"

        # 3. Subdivide edge 1 (u1 -> v1) by inserting vertex h1
        network.remove_edge(u1, v1)
        network.add_edge(u1, h1)
        network.add_edge(h1, v1)

        # 4. Subdivide edge 2 (u2 -> v2) by inserting vertex h2
        network.remove_edge(u2, v2)
        network.add_edge(u2, h2)
        network.add_edge(h2, v2)

        # 5. Create the hybridization link (try h1 -> h2 first)
        network.add_edge(h1, h2)

        if nx.is_directed_acyclic_graph(network):
            successful_events += 1
            continue

        # If it created a cycle, try reversing the direction of the flow (h2 -> h1)
        network.remove_edge(h1, h2)
        network.add_edge(h2, h1)

        if nx.is_directed_acyclic_graph(network):
            successful_events += 1
            continue

        # If both directions violate the DAG property, roll back this attempt entirely
        network.remove_edge(h2, h1)
        network.remove_node(h1)
        network.remove_node(h2)
        network.add_edge(u1, v1)
        network.add_edge(u2, v2)

    print(
        f" Done! Added {successful_events} hybridization events after {attempts} attempts."
    )
    visualizeGraph(
        network,
        f"Final Phylogenetic Network (+{successful_events} Hybridizations)",
        rootID,
        geneColors
    )
    return network, rootID