"""
Task 2b: BIC-cherry+expansion explanations of the tree-BMGs from task 2a
"""

from typing import NamedTuple

import networkx as nx
from tralda.datastructures import Tree

from .big_cherry_network import (
    buildBaseBigCherry,
    extendBicCherryNetwork,
    extendBicCherryNetworkEdgeRestricted,
)
from .least_resolved_tree import buildTarget
from .network_best_matches import colorsFromGraph


def bicCherryExplanation(
    bmg: nx.DiGraph, sigma: dict = None, edgeRestricted: bool = False
) -> tuple[nx.DiGraph, str]:
    """
    The BIC-cherry+expansion network of (G, sigma) in one step.

    Parameters
    ----------
    bmg : nx.DiGraph
        A colored digraph with the sicor-in-hub property. Without sigma the
        'color' vertex attributes are used, which is the species that
        asymmetree sets when it builds a BMG and hence the real coloring; the
        RGBA values used for plotting are not needed here.
    sigma : dict, optional
        Leaf coloring, keyed by the vertices of the graph.
    edgeRestricted : bool
        Use variant (b) of task 1b, which only expands along arcs of the graph,
        instead of variant (a) of task 1a.

    Returns
    -------
    tuple of nx.DiGraph and str
        The network and its root.
    """
    if sigma is None:
        sigma = colorsFromGraph(bmg)

    base, root, parents = buildBaseBigCherry(bmg, sigma)
    extend = (
        extendBicCherryNetworkEdgeRestricted
        if edgeRestricted
        else extendBicCherryNetwork
    )
    return extend(base, parents, bmg, sigma), root


class TreeBmgExplanations(NamedTuple):
    """Everything task 2b asks for, for a single gene tree.

    Attributes
    ----------
    bmg : nx.DiGraph
        The tree-BMG of the gene tree, i.e. the input of task 2a.
    target : Tree
        The least resolved tree T* explaining it, i.e. the target of task 2a.
    variantA : nx.DiGraph
        The BIC-cherry+expansion explanation of task 1a.
    variantB : nx.DiGraph
        The edge-restricted explanation of task 1b.
    root : str
        The root shared by both networks.
    """

    bmg: nx.DiGraph
    target: Tree
    variantA: nx.DiGraph
    variantB: nx.DiGraph
    root: str


def explainTreeBmg(geneTree: Tree) -> TreeBmgExplanations:
    """
    Task 2b: the BIC-cherry+expansion explanations of the tree-BMG of a tree.

    The target T* of task 2a is carried along because tasks 2c to 2f edit the
    networks towards it, and comparing against it is only meaningful for the
    very BMG the networks were built from.
    """
    bmg, target = buildTarget(geneTree)
    sigma = colorsFromGraph(bmg)
    variantA, root = bicCherryExplanation(bmg, sigma)
    variantB, _ = bicCherryExplanation(bmg, sigma, edgeRestricted=True)
    return TreeBmgExplanations(bmg, target, variantA, variantB, root)


def missingArcs(bmg: nx.DiGraph, sigma: dict = None) -> list:
    """
    The ordered, differently colored non-arcs of (G, sigma).

    These are exactly the pairs (x, y) that trigger an expansion, so an empty
    result means both variants leave the base network untouched. A tree-BMG
    without missing arcs is the complete "one gene per species" case, on which
    every construction succeeds for trivial reasons; tests that want to say
    something about the expansion have to rule it out.
    """
    if sigma is None:
        sigma = colorsFromGraph(bmg)

    return [
        (x, y)
        for x in bmg.nodes()
        for y in bmg.nodes()
        if x != y and sigma[x] != sigma[y] and not bmg.has_edge(x, y)
    ]
