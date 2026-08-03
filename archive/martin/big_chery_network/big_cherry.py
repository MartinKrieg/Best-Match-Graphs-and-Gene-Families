import networkx as nx
import matplotlib.pyplot as plt
from asymmetree.analysis import best_matches
import asymmetree.treeevolve as te
from asymmetree.visualization.tree_vis import visualize, assign_colors


def build_base_bic_cherry(G, geneColors):
    X = list(G.nodes())
    N = nx.DiGraph()
    root_id = "rho"
    N.add_node(root_id, type="root")
    p_map = {}
    p_list = []

    if len(X) == 2:
        x, y = X[0], X[1]
        N.add_nodes_from([x, y])
        N.add_edges_from([(root_id, x), (root_id, y)])
        p_map[(x, y)] = root_id
        p_map[(y, x)] = root_id
    else:
        N.add_nodes_from(X)
        for i in range(len(X)):
            for j in range(len(X)):
                x, y = X[i], X[j]

                x_color = geneColors.get(str(x), None)
                y_color = geneColors.get(str(y), None)

                if x_color is not None and y_color is not None and x_color != y_color:
                    p_node_1 = f"p_{x}_{y}"
                    p_node_2 = f"p_{y}_{x}"
                    if p_node_1 not in p_list and p_node_2 not in p_list:
                        p_list.append(p_node_1)
                        p_list.append(p_node_2)
                        N.add_node(p_node_1, type="parent")
                        N.add_edge(root_id, p_node_1)
                        N.add_edge(p_node_1, x)
                        N.add_edge(p_node_1, y)
                        p_map[(x, y)] = p_node_1
                        p_map[(y, x)] = p_node_1

    return N, root_id, p_map

def extend_bic_cherry_network(N, root_id, p_map, G, sigma):
    X = list(G.nodes())

    for x in X:
        for y in X:
            if x != y and (f'{x}' in sigma.keys() and f'{y}' in sigma.keys()) and sigma[f'{x}'] != sigma[f'{y}']:
                # Wenn (x,y) ein Non-Arc im BMG ist
                if not G.has_edge(x, y):

                    # In Aufgabe (a) betrachten wir ALLE z der gleichen Farbe wie y
                    for z in X:
                        if z != x and (f'{z}' in sigma.keys() and f'{y}' in sigma.keys()) and sigma[f'{z}'] == sigma[f'{y}']:

                            p_xy = p_map.get((x, y))
                            p_xz = p_map.get((x, z))  # Der innere Partner-Elternknoten

                            # Wichtig: Die Kante darf nur aufgebrochen werden, wenn sie noch existiert
                            if p_xy and p_xz and N.has_edge(p_xy, x):
                                if nx.has_path(N, p_xz, p_xy):
                                    continue

                                q_node = f"q_{x}_{z}_for_{y}_extension"
                                N.add_node(q_node, type="extension")

                                # 1. Fehlerbehebung: Alte direkte Kante UNBEDINGT entfernen!
                                N.remove_edge(p_xy, x)

                                # 2. Fehlerbehebung: q_node sauber dazwischenschalten (Subdivision)
                                N.add_edge(p_xy, q_node)
                                N.add_edge(q_node, x)

                                # 3. Fehlerbehebung: Verbindung zum inneren Knoten p_xz statt zum Blatt z
                                N.add_edge(q_node, p_xz)

    return N


def visualize_hierarchical_network(N, title, root_id, sigma):
    plt.figure(figsize=(10, 7))

    pos = {}
    generations = list(nx.topological_generations(N))
    num_layers = len(generations)

    for depth, layer in enumerate(generations):
        nodes_in_layer = sorted(list(layer), key=lambda n: str(n))
        num_nodes = len(nodes_in_layer)
        for i, node in enumerate(nodes_in_layer):
            x = (i + 0.5) / num_nodes
            y = 1.0 - (depth / (num_layers - 1)) if num_layers > 1 else 0.5
            pos[node] = (x, y)

    node_colors = []
    for node in N.nodes():
        if node == root_id:
            node_colors.append("skyblue")
        elif N.out_degree(node) == 0:
            node_colors.append("lightgreen")
        elif str(node).startswith("q_"):
            node_colors.append("salmon")
        else:
            node_colors.append("lightgrey")

    nx.draw_networkx_nodes(
        N, pos, node_color=node_colors, node_size=800, edgecolors="black"
    )

    # --- FIX: Safe label checking for missing keys and value types ---
    labels = {}
    for node in N.nodes():
        if N.out_degree(node) == 0:
            node_str = str(node)
            val = sigma.get(node_str, "Unknown")

            if isinstance(val, (list, tuple)):
                # Safely format color lists/tuples of floats
                formatted_val = ", ".join(f"{c:.2f}" for c in val[:3])
                labels[node] = f"{node}\n({formatted_val})"
            else:
                # Handle raw strings (e.g. "Species_A") or "Unknown" fallback directly
                # labels[node] = f"{node}\n({val})"
                continue
        else:
            labels[node] = node

    nx.draw_networkx_labels(N, pos, labels=labels, font_size=7, font_weight="bold")

    nx.draw_networkx_edges(
        N,
        pos,
        arrowstyle="-|>",
        arrowsize=15,
        edge_color="dimgray",
        width=1.2,
        connectionstyle="arc3,rad=0.03",
    )

    plt.title(title, fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"./plots/big_cherry/big_cherry_network_{title}.pdf")


def generate_bmg(num_leaves):
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
    visualize(T, color_dict=gene_colors, save_as="./plots/big_cherry/genetree.pdf")

    bmg = best_matches.bmg_from_tree(T)
    plt.figure(figsize=(8, 8))
    nx.draw(
        bmg,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=2000,
        font_size=10,
        font_weight="bold",
        arrows=True,  # BMGs sind gerichtet (Directed Graphs)
    )

    plt.title("Best Matches Graph (BMG)")
    plt.savefig("./plots/big_cherry/bmg.pdf")

    # --- FIX: Map colors comprehensively across leaf labels and indices ---
    leaf_colors = {}
    for node in T.leaves():
        # Get the clean string identifier of the leaf node (e.g., "x1")
        label = getattr(node, "label", None)
        if label is None:
            label = getattr(node, "name", str(node))
        label_key = str(label)

        # Safely find the corresponding value in gene_colors using valid lookup keys
        color_val = None
        for lookup_key in [node, getattr(node, "index", None), label, label_key]:
            if lookup_key in gene_colors:
                color_val = gene_colors[lookup_key]
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
