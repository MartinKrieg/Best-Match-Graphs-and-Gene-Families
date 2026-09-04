"""
Best matches and weak best matches in leaf-colored phylogenetic networks
"""

import networkx as nx
import numpy as np


def getRoot(network: nx.DiGraph):
    """Return the unique source of the network."""
    sources = [v for v in network.nodes() if network.in_degree(v) == 0]
    if len(sources) != 1:
        raise ValueError(
            f"a phylogenetic network needs exactly one root, found {len(sources)}"
        )
    return sources[0]


def getLeaves(network: nx.DiGraph) -> list:
    """Return the <=-minimal vertices, i.e. those without descendants."""
    return [v for v in network.nodes() if network.out_degree(v) == 0]


def computeAncestorSets(network: nx.DiGraph) -> dict:
    """Map every vertex v to {u : v <= u}, including v itself."""
    if not nx.is_directed_acyclic_graph(network):
        raise ValueError("best matches are only defined on acyclic networks")

    ancestors = {}
    for v in nx.topological_sort(network):
        anc = {v}
        for parent in network.predecessors(v):
            anc |= ancestors[parent]
        ancestors[v] = anc
    return ancestors


def minimalVertices(vertices, ancestors: dict) -> set:
    """Return the <=-minimal elements of a vertex set.

    w < z holds exactly when z is an ancestor of w and w != z, so z is minimal
    if no other element of the set has z among its ancestors.
    """
    vertices = set(vertices)
    return {
        z
        for z in vertices
        if not any(z in ancestors[w] for w in vertices if w != z)
    }


def computeLCA(vertices, ancestors: dict) -> set:
    """Return LCA(A), the <=-minimal common ancestors of the given vertices."""
    vertices = list(vertices)
    if not vertices:
        raise ValueError("LCA is undefined for the empty set")

    common = set(ancestors[vertices[0]])
    for v in vertices[1:]:
        common &= ancestors[v]

    if not common:
        raise ValueError(
            f"vertices {vertices} have no common ancestor, "
            "the network is not rooted at a single vertex"
        )
    return minimalVertices(common, ancestors)


def bestMatchGraphs(
    network: nx.DiGraph, sigma: dict = None, labels: dict = None
) -> tuple[nx.DiGraph, nx.DiGraph]:
    """Compute the strict and the weak best match graph of (N, sigma).

    Both graphs are returned together because they share the expensive part,
    the LCA of every leaf pair.

    Parameters
    ----------
    network : nx.DiGraph
        A rooted DAG. Leaves are the vertices without outgoing edges.
    sigma : dict, optional
        Leaf coloring, keyed by the vertices of the network. If omitted, the
        'reconc' node attribute is used, which is what AsymmeTree gene trees
        carry after Tree.to_nx().
    labels : dict, optional
        Names for the leaves in the resulting graphs, keyed by the vertices of
        the network. If omitted, the 'label' node attribute is used when every
        leaf has a unique one, otherwise the vertices themselves. The default
        makes the result directly comparable to
        asymmetree.analysis.best_matches.bmg_from_tree.

    Returns
    -------
    tuple of two nx.DiGraph
        The best match graph and the weak best match graph. Nodes carry the
        species as the 'color' attribute.
    """
    getRoot(network)  # rejects forests early, LCA would fail further down
    ancestors = computeAncestorSets(network)

    leaves = getLeaves(network)
    sigma = _resolveColors(network, leaves, sigma)
    labels = _resolveLabels(network, leaves, labels)

    leavesByColor = {}
    for x in leaves:
        leavesByColor.setdefault(sigma[x], []).append(x)

    lcaCache = {}

    def lcaOf(x, y):
        key = frozenset((x, y))
        if key not in lcaCache:
            lcaCache[key] = computeLCA(key, ancestors)
        return lcaCache[key]

    bmg = nx.DiGraph()
    wbmg = nx.DiGraph()
    for x in leaves:
        bmg.add_node(labels[x], color=sigma[x])
        wbmg.add_node(labels[x], color=sigma[x])

    for x in leaves:
        for color, candidates in leavesByColor.items():
            if color == sigma[x]:
                continue

            lcas = {y: lcaOf(x, y) for y in candidates}
            reachable = set().union(*lcas.values())
            q = minimalVertices(reachable, ancestors)

            for y in candidates:
                if lcas[y] & q:
                    wbmg.add_edge(labels[x], labels[y])

                # y is beaten if some competitor's LCA lies strictly below one
                # of the vertices in LCA(x, y).
                beaten = any(
                    v in ancestors[u] and u != v
                    for u in reachable
                    for v in lcas[y]
                )
                if not beaten:
                    bmg.add_edge(labels[x], labels[y])

    return bmg, wbmg


def bestMatchGraph(
    network: nx.DiGraph, sigma: dict = None, labels: dict = None
) -> nx.DiGraph:
    """Compute the strict best match graph of (N, sigma)."""
    return bestMatchGraphs(network, sigma=sigma, labels=labels)[0]


def weakBestMatchGraph(
    network: nx.DiGraph, sigma: dict = None, labels: dict = None
) -> nx.DiGraph:
    """Compute the weak best match graph of (N, sigma)."""
    return bestMatchGraphs(network, sigma=sigma, labels=labels)[1]


def colorsFromGraph(graph: nx.DiGraph) -> dict:
    """Read sigma off the 'color' vertex attribute of a colored graph.

    That attribute is what asymmetree.analysis.best_matches.bmg_from_tree sets,
    so a BMG already carries its own coloring and can hand it on to the
    constructions that consume (G, sigma).
    """
    sigma = {}
    for v in graph.nodes():
        color = graph.nodes[v].get("color")
        if color is None:
            raise ValueError(f"vertex {v} has no 'color' attribute")
        sigma[v] = _hashableColor(color)
    return sigma


def networkExplainsBmg(
    network: nx.DiGraph, bmg: nx.DiGraph, sigma: dict = None, weak: bool = False
) -> bool:
    """Whether the (weak) best match graph of the network is exactly the graph.

    The network counterpart of src.least_resolved_tree.explainsBmg and the
    acceptance test for every construction that claims to explain (G, sigma).
    The leaves of the network have to be the vertices of the graph, otherwise
    the two are trivially different.

    Parameters
    ----------
    network : nx.DiGraph
        The candidate explanation.
    bmg : nx.DiGraph
        The graph to be explained. Its 'color' attributes supply sigma.
    sigma : dict, optional
        Leaf coloring, keyed by the leaves of the network. Defaults to the
        coloring carried by the graph.
    weak : bool
        Compare the weak best match graph instead of the strict one.
    """
    if sigma is None:
        sigma = colorsFromGraph(bmg)

    strict, weakBmg = bestMatchGraphs(network, sigma=sigma)
    explained = weakBmg if weak else strict

    return set(explained.nodes()) == set(bmg.nodes()) and set(
        explained.edges()
    ) == set(bmg.edges())


def _hashableColor(color):
    """Colors are used as dict keys, so arrays and lists become tuples."""
    if isinstance(color, np.ndarray):
        return tuple(color.tolist())
    if isinstance(color, list):
        return tuple(color)
    return color


def _resolveColors(network: nx.DiGraph, leaves: list, sigma: dict) -> dict:
    if sigma is None:
        resolved = {}
        for x in leaves:
            color = network.nodes[x].get("reconc")
            if color is None:
                raise ValueError(
                    f"leaf {x} has no 'reconc' attribute, pass sigma explicitly"
                )
            resolved[x] = _hashableColor(color)
        return resolved

    missing = [x for x in leaves if x not in sigma]
    if missing:
        raise ValueError(f"no color assigned to the leaves {missing}")
    return {x: _hashableColor(sigma[x]) for x in leaves}


def _resolveLabels(network: nx.DiGraph, leaves: list, labels: dict) -> dict:
    if labels is None:
        candidates = {x: network.nodes[x].get("label") for x in leaves}
        usable = all(label is not None for label in candidates.values())
        unique = len(set(candidates.values())) == len(leaves)
        return candidates if usable and unique else {x: x for x in leaves}

    missing = [x for x in leaves if x not in labels]
    if missing:
        raise ValueError(f"no label assigned to the leaves {missing}")

    resolved = {x: labels[x] for x in leaves}
    if len(set(resolved.values())) != len(leaves):
        raise ValueError("leaf labels must be unique")
    return resolved
