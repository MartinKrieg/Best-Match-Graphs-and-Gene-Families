import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(G, title, root_id):
    """
    Plots a DAG/phylogenetic network with a strict top-to-bottom hierarchy.
    """
    plt.figure(figsize=(11, 8))
    
    # --- STEP 1: Compute Hierarchical Layout Coordinates ---
    pos = {}
    try:
        # Groups nodes into sequential layers based on DAG flow
        generations = list(nx.topological_generations(G))
        num_layers = len(generations)
        
        for depth, layer in enumerate(generations):
            # Sort to keep leaf names or IDs consistently grouped
            nodes_in_layer = sorted(list(layer), key=lambda n: str(n))
            num_nodes = len(nodes_in_layer)
            
            for i, node in enumerate(nodes_in_layer):
                # Evenly space X coordinates across the layer with side margins
                x = (i + 0.5) / num_nodes
                # Map Y coordinates from 1.0 (top) down to 0.0 (bottom)
                y = 1.0 - (depth / (num_layers - 1)) if num_layers > 1 else 0.5
                pos[node] = (x, y)
                
    except nx.NetworkXUnfeasible:
        # Fallback to spring layout just in case a cycle accidentally sneaked in
        print("Warning: Graph contains a cycle! Falling back to spring layout.")
        pos = nx.spring_layout(G, seed=42)

    # --- STEP 2: Color Code Nodes based on Roles ---
    node_colors = []
    for node in G.nodes():
        if node == root_id:
            node_colors.append("skyblue")       # Root
        elif G.out_degree(node) == 0:
            node_colors.append("lightgreen")    # Leaves
        elif str(node).startswith("h_"):
            node_colors.append("coral")         # Hybridization points
        else:
            node_colors.append("lightgrey")     # Standard internal nodes

    # --- STEP 3: Draw the Hierarchy ---
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    
    # Draw directed edges with highly visible arrow styles
    nx.draw_networkx_edges(
        G, pos, 
        arrowstyle="-|>", 
        arrowsize=18, 
        edge_color="dimgray", 
        width=1.5,
        connectionstyle="arc3,rad=0.05"  # Slighly curves edges to untangle overlapping lines
    )
    
    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("./plots/hybrid_network/hybrid_network.pdf")
    
    