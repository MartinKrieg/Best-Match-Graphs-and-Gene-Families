import networkx as nx
from asymmetree.analysis import best_matches
import matplotlib.pyplot as plt 
def generateBmg(geneTree:nx.DiGraph, geneColors):
    bmg = best_matches.bmg_from_tree(geneTree)

    # 2. Farben aus der Variable genecolors passend zu den BMG-Knoten zuordnen
    node_colors = []
    for node in bmg.nodes():
        if node in geneColors:
            node_colors.append(geneColors[node])
        # Sicherheits-Check: Falls die Knoten-ID im Graphen ein String "10" statt der Zahl 10 ist
        elif isinstance(node, str) and node.isdigit() and int(node) in geneColors:
            node_colors.append(geneColors[int(node)])
        else:
            node_colors.append("lightblue")  # Fallback, falls eine ID fehlt

    # 3. Visualisierung des BMG mit den passenden Farben
    plt.figure(figsize=(8, 8))
    nx.draw(
        bmg,
        with_labels=True,
        node_color=node_colors,  # Nutzt deine exakten RGBA-Farben
        edge_color="gray",
        node_size=2000,
        font_size=10,
        font_weight="bold",
        arrows=True,
    )

    plt.title("Best Matches Graph (BMG) with Species Coloring ($\\sigma$)")
    plt.savefig("./plots/big_cherry/bmg.png")

    # --- FIX: Map colors comprehensively across leaf labels and indices ---
    leaf_colors = {}
    for node in geneTree.leaves():
        # Get the clean string identifier of the leaf node (e.g., "x1")
        label = getattr(node, "label", None)
        if label is None:
            label = getattr(node, "name", str(node))
        label_key = str(label)

        # Safely find the corresponding value in gene_colors using valid lookup keys
        color_val = None
        for lookup_key in [node, getattr(node, "index", None), label, label_key]:
            if lookup_key in geneColors:
                color_val = geneColors[lookup_key]
                break

        if color_val is not None:
            # Handle numpy arrays safely without accidentally breaking down regular strings
            if hasattr(color_val, "tolist"):
                color_val = color_val.tolist()

            # If the numpy conversion turned a string into a list of characters, rejoin it
            if isinstance(color_val, list) and all(
                isinstance(c, str) and len(c) == 1 for c in color_val
            ):
                color_val = "".join(color_val)

            # Direct mapping: "x1" -> "Species_A"
            leaf_colors[label_key] = color_val

    return bmg, leaf_colors