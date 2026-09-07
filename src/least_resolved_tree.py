"""
Tree-BMGs of AsymmeTree gene trees and their least resolved trees
"""

import networkx as nx
from asymmetree.analysis import best_matches
from tralda.datastructures import Tree
from utils import generateTreeBmg, generateLeastResolvedTree

def leastResolvedTreeFromBmg(bmg: nx.DiGraph) -> Tree:
    """
    T*, obtained from the informative triples of the graph via BUILD.

    A consistent triple set is necessary but not sufficient for the graph to be
    a tree-BMG: BUILD can succeed on a color-sink-free non-BMG and return a
    tree that does not explain it. The tree is therefore accepted only after
    checking that its own best match graph is the input graph again.

    Raises
    ------
    ValueError
        If the graph is not a tree-BMG, i.e. if the informative triples are
        inconsistent or the resulting tree does not explain the graph.
    """
    lrt = best_matches.is_bmg(bmg)
    if lrt is None:
        raise ValueError("the graph is not a tree-BMG, no tree explains it")
    return lrt


def treeBmg(geneTree: Tree) -> nx.DiGraph:
    """The best match graph explained by a leaf-colored gene tree."""
    return generateTreeBmg(geneTree)


def leastResolvedTree(geneTree: Tree) -> Tree:
    """T*, obtained by contracting the redundant edges of the gene tree."""
    return generateLeastResolvedTree(geneTree)


def buildTarget(geneTree: Tree) -> tuple[nx.DiGraph, Tree]:
    """Task 2a: the tree-BMG of a gene tree together with its target T*."""
    return treeBmg(geneTree), leastResolvedTree(geneTree)


def treeClusters(tree: Tree) -> set:
    """
    The set of leaf label sets below the vertices of the tree.

    Two phylogenetic trees on the same leaf set have the same topology exactly
    when their cluster systems coincide, which makes this a usable identity
    check for trees built by different routes.
    """
    leaves = tree.leaf_dict()
    return {frozenset(leaf.label for leaf in leaves[v]) for v in tree.preorder()}


def treesEqual(first: Tree, second: Tree) -> bool:
    """Whether both trees have the same topology on the same leaf set."""
    return treeClusters(first) == treeClusters(second)


def explainsBmg(tree: Tree, bmg: nx.DiGraph) -> bool:
    """Whether the best match graph of the tree is exactly the given graph."""
    explained = generateTreeBmg(tree)
    return set(explained.nodes()) == set(bmg.nodes()) and set(
        explained.edges()
    ) == set(bmg.edges())


def toNetwork(tree: Tree, rootName="rho") -> tuple[nx.DiGraph, str]:
    """
    Convert a tree into a DiGraph whose leaves are named by their labels.

    Tree.to_nx() names the vertices by the object ids of the tree nodes, which
    do not match the leaf names used by the BMG and by the BIC-cherry
    networks. Task 2e has to compare T* against those networks, so here the
    leaves keep their labels and only the inner vertices get generated names.
    The leaves also keep 'label' and 'reconc', so that the result can be fed
    straight into src.network_best_matches.
    """
    names = {}
    innerCount = 0
    for v in tree.preorder():
        if not v.children:
            names[v] = v.label
        elif v is tree.root:
            names[v] = rootName
        else:
            names[v] = f"v_{innerCount}"
            innerCount += 1

    if len(set(names.values())) != len(names):
        raise ValueError("leaf labels collide with the generated inner names")

    network = nx.DiGraph()
    for v in tree.preorder():
        network.add_node(names[v])
        if not v.children:
            network.nodes[names[v]]["label"] = v.label
            network.nodes[names[v]]["reconc"] = v.reconc

    for v in tree.preorder():
        for child in v.children:
            network.add_edge(names[v], names[child])

    return network, names[tree.root]
