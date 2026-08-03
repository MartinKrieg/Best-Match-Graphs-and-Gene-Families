import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx


def visualizeGraph(G, title, root_id, geneColors):
    """Plots a DAG/phylogenetic network with an optimized top-to-bottom hierarchy

    and an explicit color legend[cite: 4].
    """
    plt.figure(figsize=(12, 8))

    # --- STEP 1: Compute Hierarchical Layout Coordinates ---
    pos = {}
    try:
        # Groups nodes into sequential layers based on DAG flow[cite: 4]
        generations = list(nx.topological_generations(G))
        num_layers = len(generations)

        for depth, layer in enumerate(generations):
            if depth == 0:
                # The first layer (usually just the root) sorted deterministically[cite: 4]
                nodes_in_layer = sorted(list(layer), key=lambda n: str(n))
            else:
                # BETTER ORDERING: Sort nodes based on the average X position of their parents.[cite: 4]
                # This naturally untangles edge crossings and centers hybrid nodes between their parents.[cite: 4]
                def get_parent_barycenter(node):
                    parents = list(G.predecessors(node))
                    placed_parents = [pos[p][0] for p in parents if p in pos]
                    if placed_parents:
                        return sum(placed_parents) / len(placed_parents)
                    return 0.5  # Fallback if no parents are placed yet[cite: 4]

                # Primary sort by parent positioning, secondary sort by name to break ties[cite: 4]
                nodes_in_layer = sorted(
                    list(layer),
                    key=lambda n: (get_parent_barycenter(n), str(n)),
                )

            num_nodes = len(nodes_in_layer)
            for i, node in enumerate(nodes_in_layer):
                # Evenly space X coordinates across the layer with side margins[cite: 4]
                x = (i + 0.5) / num_nodes if num_nodes > 0 else 0.5
                # Map Y coordinates from 1.0 (top) down to 0.0 (bottom)[cite: 4]
                y = 1.0 - (depth / (num_layers - 1)) if num_layers > 1 else 0.5
                pos[node] = (x, y)

    except nx.NetworkXUnfeasible:
        # Fallback to spring layout just in case a cycle accidentally sneaked in[cite: 4]
        print("Warning: Graph contains a cycle! Falling back to spring layout.")
        pos = nx.spring_layout(G, seed=42)

    # --- STEP 2: Color Code Nodes based on Roles ---
    node_colors = []
    for node in G.nodes():
        node_str = str(node)
        if node == root_id:
            node_colors.append("skyblue")  # Root[cite: 4]
        elif G.out_degree(node) == 0:
            # --- ROBUST COLOR RESOLUTION FOR RAW TREE POINTERS ---
            leaf_color = None
            node_data = G.nodes[node]
            
            # 1. Try resolving via internal network metadata attributes ('label' or 'name')[cite: 3]
            label = node_data.get('label') or node_data.get('name')
            if label is not None:
                if label in geneColors:
                    leaf_color = geneColors[label]
                elif str(label) in geneColors:
                    leaf_color = geneColors[str(label)]
            
            # 2. Try matching the raw integer address directly against the original Node keys[cite: 3]
            if leaf_color is None:
                for key in geneColors.keys():
                    # Check if the memory address matches the key instance
                    if id(key) == node:
                        leaf_color = geneColors[key]
                        break
                    # Check if the key object holds a matching numerical index field[cite: 3]
                    if hasattr(key, 'index') and key.index == node:
                        leaf_color = geneColors[key]
                        break
                    # Double check if the object's label string matches our leaf metadata[cite: 3]
                    if label is not None and hasattr(key, 'label') and str(key.label) == str(label):
                        leaf_color = geneColors[key]
                        break
            
            # 3. Direct identifier fallback lookup[cite: 3]
            if leaf_color is None:
                leaf_color = geneColors.get(node) or geneColors.get(node_str)
            
            # Apply original color if discovered; otherwise, fall back to default lightgreen[cite: 3, 4]
            if leaf_color is not None:
                node_colors.append(leaf_color)
            else:
                node_colors.append("lightgreen")
        elif node_str.startswith("h_"):
            node_colors.append("coral")  # Hybridization points[cite: 4]
        else:
            node_colors.append("lightgrey")  # Standard internal nodes[cite: 4]

    # --- STEP 3: Draw the Hierarchy ---
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=700, edgecolors="black"
    )
    
    # Generate clean labels for the display text
    display_labels = {}
    for node in G.nodes():
        if G.out_degree(node) == 0:
            # If the leaf node has a clean string label (like 'x1'), use it; otherwise show the ID
            label_text = G.nodes[node].get('label') or G.nodes[node].get('name')
            display_labels[node] = str(label_text) if label_text is not None else str(node)
        else:
            display_labels[node] = str(node)
            
    nx.draw_networkx_labels(G, pos, labels=display_labels, font_size=8, font_weight="bold")

    # Draw directed edges with highly visible arrow styles[cite: 4]
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=18,
        edge_color="dimgray",
        width=1.5,
        connectionstyle="arc3,rad=0.05",  # Slightly curves edges to untangle overlapping lines[cite: 4]
    )

    # --- STEP 4: Add Legend for Color Scheme ---
    legend_handles = [
        mpatches.Patch(
            facecolor="skyblue", edgecolor="black", label="Root Origin"
        ),
        mpatches.Patch(
            facecolor="lightgrey", edgecolor="black", label="Internal Node"
        ),
        mpatches.Patch(
            facecolor="coral", edgecolor="black", label="Hybridization Point"
        )
    ]
    plt.legend(
        handles=legend_handles,
        loc="upper right",
        title="Node Classifications",
        frameon=True,
        shadow=True,
        facecolor="white",
        edgecolor="lightgrey",
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("./plots/hybrid_network/hybrid_network.png")


def visualizeHierarchicalNetwork(N, title, root_id, geneColors):
    plt.figure(figsize=(10, 7))
    pos = {}
    generations = list(nx.topological_generations(N))
    num_layers = len(generations)

    # Positionen berechnen
    for depth, layer in enumerate(generations):
        nodes_in_layer = sorted(list(layer), key=lambda n: str(n))
        num_nodes = len(nodes_in_layer)
        for i, node in enumerate(nodes_in_layer):
            x = (i + 0.5) / num_nodes

            # PHYLOGENETISCHER FIX: Alle Blätter (out_degree == 0) auf die unterste Ebene (y=0) zwingen
            if N.out_degree(node) == 0:
                y = 0.0
            else:
                # Innere Knoten proportional zu ihrer topologischen Tiefe verteilen
                y = 1.0 - (depth / (num_layers - 1)) if num_layers > 1 else 0.5

            pos[node] = (x, y)

    # Knotenfarben bestimmen
    node_colors = []
    for node in N.nodes():
        node_str = str(node)
        if node == root_id:
            node_colors.append("skyblue")
        elif N.out_degree(node) == 0:
            # Robust lookup against the original geneColors mapping (handles raw keys and string variations)
            leaf_color = (
                geneColors.get(node)
                or geneColors.get(node_str)
                or (
                    geneColors.get(int(node))
                    if node_str.isdigit()
                    else None
                )
            )
            # Apply original color if found; otherwise, use lightgreen as a fallback
            if leaf_color is not None:
                node_colors.append(leaf_color)
            else:
                node_colors.append("lightgreen")
        # FIX 1b: Erkennt nun q_-Knoten (Task 1a) UND _ext-Knoten (Task 1b) als Erweiterungen
        elif node_str.startswith("q_") or node_str.endswith("_ext"):
            node_colors.append("salmon")
        else:
            node_colors.append("lightgrey")

    # Knoten zeichnen
    nx.draw_networkx_nodes(
        N, pos, node_color=node_colors, node_size=900, edgecolors="black"
    )

    # Labels generieren
    labels = {}
    for node in N.nodes():
        if N.out_degree(node) == 0:
            node_str = str(node)
            val = geneColors.get(node_str, "Unknown")

            if isinstance(val, (list, tuple)):
                labels[node] = f"{node}"
            else:
                # FIX: Der 'continue'-Bug ist behoben. Text-Farben werden jetzt korrekt ausgegeben.
                labels[node] = f"{node}\n({val})"
        else:
            labels[node] = node

    # Labels und Kanten zeichnen
    nx.draw_networkx_labels(
        N, pos, labels=labels, font_size=8, font_weight="bold"
    )

    nx.draw_networkx_edges(
        N,
        pos,
        arrowstyle="-|>",
        arrowsize=15,
        edge_color="dimgray",
        width=1.2,
        # rad=0.05 sorgt dafür, dass parallel verlaufende Kanten zu den Hybriden gut sichtbar sind
        connectionstyle="arc3,rad=0.05",
    )

    plt.title(title, fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(f"./plots/big_cherry/{title}.png")
