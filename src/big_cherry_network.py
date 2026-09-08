from collections import Counter
from itertools import combinations
import networkx as nx
import numpy as np


def getNodeColors(G, gene_colors) -> dict:
    """Bereitet die Farben platt vor und macht Listen sowie NumPy-Arrays

    für schnelle Vergleiche und Counter hashbar (als Tuples).
    """
    node_colors = {}
    for node in G.nodes():
        color = None
        # Flexibler Lookup für Ints, Strings oder Digits
        for key in (
            node,
            str(node),
            int(node) if isinstance(node, str) and node.isdigit() else None,
        ):
            if key in gene_colors:
                color = gene_colors[key]
                break

        # Konvertierung in Tuple, falls es sich um eine Liste oder ein NumPy-Array handelt
        if isinstance(color, np.ndarray):
            color = tuple(color.tolist())
        elif isinstance(color, list):
            color = tuple(color)

        node_colors[node] = color
    return node_colors


def checkSicorInhubProperty(G, gene_colors) -> bool:
    """Überprüft die Sicor-Inhub-Eigenschaft des Graphen."""
    node_colors = getNodeColors(G, gene_colors)
    color_counts = Counter(c for c in node_colors.values() if c is not None)

    for vertex in G.nodes():
        color = node_colors[vertex]
        if color is None:
            continue

        # Sicor: Farbe kommt exakt einmal im Graphen vor
        if color_counts[color] == 1:
            # Inhub: Alle Knoten anderer Farbe müssen eine Kante zu 'vertex' haben
            for v in G.nodes():
                if (
                    v != vertex
                    and node_colors[v] is not None
                    and node_colors[v] != color
                ):
                    if not G.has_edge(v, vertex):
                        return False
    return True


def buildBaseBigCherry(best_match_graph, gene_colors):
    """Initialisiert das BIC-Cherry Basisnetzwerk auf den Blättern."""
    if not checkSicorInhubProperty(best_match_graph, gene_colors):
        raise ValueError("The input graph does not satisfy the Sicor-Inhub property.")

    node_colors = getNodeColors(best_match_graph, gene_colors)
    X = list(best_match_graph.nodes())

    N = nx.DiGraph()
    root_id = "rho"
    N.add_node(root_id, type="root")
    # The leaf attributes carry the coloring, which the best match computation on the resulting network needs
    N.add_nodes_from(best_match_graph.nodes(data=True))
    p_map = {}

    # Sonderfall: Genau 2 Knoten
    if len(X) == 2:
        x, y = X[0], X[1]
        cx, cy = node_colors[x], node_colors[y]
        if cx is not None and cy is not None and cx != cy:
            N.add_edges_from([(root_id, x), (root_id, y)])
            p_map[(x, y)] = p_map[(y, x)] = root_id
        return N, root_id, p_map

    # Allgemeiner Fall
    for v1, v2 in combinations(X, 2):
        c1, c2 = node_colors[v1], node_colors[v2]
        if c1 is not None and c2 is not None and c1 != c2:
            nodes_sorted = sorted([str(v1), str(v2)])
            p_node = f"p_{nodes_sorted[0]}_{nodes_sorted[1]}"

            if not N.has_node(p_node):
                N.add_node(p_node, type="parent")
                N.add_edges_from([(root_id, p_node), (p_node, v1), (p_node, v2)])

            p_map[(v1, v2)] = p_map[(v2, v1)] = p_node

    return N, root_id, p_map


def extendBicCherryNetwork(
    base_big_cherry, base_parents, best_match_graph, gene_colors
):
    """Standard-Erweiterungsalgorithmus (Task 1a)."""
    network = base_big_cherry.copy()
    node_colors = getNodeColors(best_match_graph, gene_colors)
    X = list(best_match_graph.nodes())

    # Pre-Caching für y_prime, um die O(N^3) Schleife zu verhindern
    y_prime_map = {}
    for y in X:
        cy = node_colors[y]
        if cy is None:
            continue
        for v in X:
            if v != y and node_colors[v] == cy:
                y_prime_map[y] = v
                break

    for x in X:
        cx = node_colors[x]
        if cx is None:
            continue

        for y in X:
            if x == y:
                continue

            cy = node_colors[y]
            if cy is None or cx == cy:
                continue

            if not best_match_graph.has_edge(x, y):
                y_prime = y_prime_map.get(y)
                if y_prime is None:
                    continue

                p_xy = base_parents.get((x, y))
                if p_xy:
                    q_node = f"q_{x}_{y_prime}"
                    if not network.has_node(q_node):
                        network.add_node(q_node, type="extension")
                        network.add_edges_from([(q_node, x), (q_node, y_prime)])
                    network.add_edge(p_xy, q_node)

    return network


def extendBicCherryNetworkEdgeRestricted(
    base_big_cherry, base_parents, best_match_graph, gene_colors
):
    """Kanten-beschränkte Erweiterung (Task 1b)."""
    network = base_big_cherry.copy()
    node_colors = getNodeColors(best_match_graph, gene_colors)
    X = list(best_match_graph.nodes())

    for x in X:
        cx = node_colors[x]
        if cx is None:
            continue

        for y in X:
            if x == y:
                continue

            cy = node_colors[y]
            if cy is None or cx == cy:
                continue

            if not best_match_graph.has_edge(x, y):
                # Nutzen von .successors(x) reduziert die Suchmenge drastisch
                y_prime = None
                for v in best_match_graph.successors(x):
                    if v != y and node_colors[v] == cy:
                        y_prime = v
                        break

                if y_prime is None:
                    continue

                p_xy = base_parents.get((x, y))
                p_xz = base_parents.get((x, y_prime))

                if p_xy and p_xz:
                    network.add_edge(p_xy, p_xz)

    return network
