import networkx as nx
import random
import asymmetree.treeevolve as te
from asymmetree.visualization.tree_vis import visualize, assign_colors
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.util import visualize_graph


def generate_hybrid_network(num_leaves, num_hybridizations):
    print(f"-> Generating base tree with {num_leaves} leaves using AsymmeTree...")
    S = te.species_tree_n_age(
        n=num_leaves,
        age=1.0,
        model="BDP",
        innovation=True,
        birth_rate=1.0,
        death_rate=0.5,
        contraction_probability=0.2,
    )
    T = te.dated_gene_tree(
        S,
        dupl_rate=0.7,
        loss_rate=0.7,
        hgt_rate=0.7,
        gc_rate=0.7,
        dupl_polytomy=0.5,
        replace_prob=0.5,
        transfer_distance_bias="inverse",
    )

    _, gene_colors = assign_colors(S, T)
    visualize(T, color_dict=gene_colors, save_as="./plots/hybrid_network/genetree.pdf")

    graph, root_id = T.to_nx()
    visualize_graph(graph,"tes", root_id)
    network = graph.copy()

    successful_events = 0
    attempts = 0
    max_attempts = num_hybridizations * 10

    print(f"-> Iteratively inserting {num_hybridizations} hybridization events...")
    while successful_events < num_hybridizations and attempts < max_attempts:
        attempts += 1
        edges = list(network.edges())

        if len(edges) < 2:
            print("Not enough edges left to hybridize.")
            break

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

        # Phylogenetic networks must be DAGs (no directed cycles)
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
    visualize_graph(
        network,
        f"2. Final Phylogenetic Network (+{successful_events} Hybridizations)",
        root_id,
    )
    return network, root_id
