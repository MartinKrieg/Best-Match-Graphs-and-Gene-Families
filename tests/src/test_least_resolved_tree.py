import random

import asymmetree.treeevolve as te
import networkx as nx
import numpy as np
import pytest
from tralda.datastructures import Tree, TreeNode

from src.least_resolved_tree import (
    buildTarget,
    explainsBmg,
    leastResolvedTree,
    leastResolvedTreeFromBmg,
    toNetwork,
    treeBmg,
    treeClusters,
    treesEqual,
)
from src.network_best_matches import bestMatchGraphs

SEEDS = [1, 2, 3, 4, 5]


def buildGeneTree(numSpecies, seed):
    """An AsymmeTree gene tree, built without any plotting side effects."""
    random.seed(seed)
    np.random.seed(seed)
    speciesTree = te.species_tree_n_age(n=numSpecies, age=1.0)
    return te.dated_gene_tree(speciesTree, dupl_rate=0.7)


@pytest.fixture
def redundant_tree():
    """The tree ((a1, a2), b) with a1, a2 in the same species.

    The best match graph cannot distinguish a1 from a2, so the edge above
    their parent is redundant and T* is the star on the three leaves.
    """
    root = TreeNode(label="", reconc=None, dist=1.0)
    parent = TreeNode(label="", reconc=None, dist=1.0)
    root.add_child(parent)
    parent.add_child(TreeNode(label="a1", reconc="A", dist=1.0))
    parent.add_child(TreeNode(label="a2", reconc="A", dist=1.0))
    root.add_child(TreeNode(label="b", reconc="B", dist=1.0))
    return Tree(root)


class TestLeastResolvedTree:
    """Tests for the target trees of task 2a."""

    def test_redundant_edge_is_contracted(self, redundant_tree):
        lrt = leastResolvedTree(redundant_tree)

        assert treeClusters(lrt) == {
            frozenset({"a1"}),
            frozenset({"a2"}),
            frozenset({"b"}),
            frozenset({"a1", "a2", "b"}),
        }

    def test_contraction_preserves_the_bmg(self, redundant_tree):
        bmg = treeBmg(redundant_tree)
        lrt = leastResolvedTree(redundant_tree)

        assert set(bmg.edges()) == {
            ("a1", "b"),
            ("a2", "b"),
            ("b", "a1"),
            ("b", "a2"),
        }
        assert explainsBmg(lrt, bmg)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_both_routes_yield_the_same_target(self, seed):
        geneTree = buildGeneTree(numSpecies=6, seed=seed)
        bmg, lrt = buildTarget(geneTree)

        assert treesEqual(lrt, leastResolvedTreeFromBmg(bmg))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_target_explains_the_bmg(self, seed):
        geneTree = buildGeneTree(numSpecies=6, seed=seed)
        bmg, lrt = buildTarget(geneTree)

        assert explainsBmg(lrt, bmg)
        assert explainsBmg(leastResolvedTreeFromBmg(bmg), bmg)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_target_is_not_more_resolved_than_the_gene_tree(self, seed):
        geneTree = buildGeneTree(numSpecies=6, seed=seed)
        bmg, lrt = buildTarget(geneTree)

        assert len(treeClusters(lrt)) <= len(treeClusters(geneTree))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_computing_the_target_twice_changes_nothing(self, seed):
        geneTree = buildGeneTree(numSpecies=6, seed=seed)
        lrt = leastResolvedTree(geneTree)

        assert treesEqual(lrt, leastResolvedTree(lrt))

    def test_non_bmg_is_not_explained_by_the_build_tree(self):
        """A color-sink-free graph that no tree explains.

        a1 and a2 each have a single best match, which forces the triples
        a1 b1|b2 and a2 b2|b1 and hence the tree ((a1, b1), (a2, b2)). That
        tree does not produce the edges b1 -> a2 and b2 -> a1, so BUILD
        succeeding is not by itself proof that the graph is a BMG.
        """
        G = nx.DiGraph()
        G.add_node("a1", color="A")
        G.add_node("a2", color="A")
        G.add_node("b1", color="B")
        G.add_node("b2", color="B")
        G.add_edges_from(
            [
                ("a1", "b1"),
                ("a2", "b2"),
                ("b1", "a1"),
                ("b1", "a2"),
                ("b2", "a1"),
                ("b2", "a2"),
            ]
        )
        lrt = leastResolvedTreeFromBmg(G)

        # BUILD succeeds here, but the resulting tree does not explain G,
        # which is what makes G a color-sink-free non-BMG.
        assert not explainsBmg(lrt, G)


class TestToNetwork:
    """Tests for the DiGraph view of the target tree."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_leaves_keep_the_names_used_by_the_bmg(self, seed):
        geneTree = buildGeneTree(numSpecies=6, seed=seed)
        bmg, lrt = buildTarget(geneTree)
        network, _ = toNetwork(lrt)

        leaves = {v for v in network.nodes() if network.out_degree(v) == 0}
        assert leaves == set(bmg.nodes())

    @pytest.mark.parametrize("seed", SEEDS)
    def test_network_best_matches_reproduce_the_bmg(self, seed):
        geneTree = buildGeneTree(numSpecies=6, seed=seed)
        bmg, lrt = buildTarget(geneTree)
        network, _ = toNetwork(lrt)

        strict, weak = bestMatchGraphs(network)

        assert set(strict.edges()) == set(bmg.edges())
        assert set(weak.edges()) == set(bmg.edges())

    def test_root_is_named_for_comparison_with_the_cherry_networks(
        self, redundant_tree
    ):
        network, root = toNetwork(leastResolvedTree(redundant_tree))

        assert root == "rho"
        assert set(network.edges()) == {
            ("rho", "a1"),
            ("rho", "a2"),
            ("rho", "b"),
        }
