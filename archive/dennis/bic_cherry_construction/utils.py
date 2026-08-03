import networkx as nx
import random

def extract_sigma_from_graph(G: nx.DiGraph) -> dict:
    return {node: G.nodes[node]["color"] for node in G.nodes()}

def generate_random_colored_digraph(
    n_nodes: int = 10,
    edge_prob: float = 0.3,
    colors: list[str] = None
) -> nx.DiGraph:
    """Generates a random DiGraph with random 'color' and 'size' node attributes."""
    
    if colors is None:
        colors = ["red", "blue", "green", "yellow", "purple"]

    # 1. Generate a random directed graph (Erdős-Rényi model)
    G = nx.gnp_random_graph(n_nodes, edge_prob, directed=True)

    dag = nx.DiGraph([(u, v) for u, v in G.edges() if u < v])

    # 2. Attach random size and color attributes to each node
    for node in dag.nodes():
        dag.nodes[node]["id"] = node
        dag.nodes[node]["color"] = random.choice(colors)

    return dag