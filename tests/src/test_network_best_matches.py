import random
from unittest.mock import patch

import asymmetree.treeevolve as te
import networkx as nx
import pytest
from asymmetree.analysis import best_matches

from src.generate_hybrid_network import generateHybridNetwork
from src.network_best_matches import (
    bestMatchGraphs,
    computeAncestorSets,
    computeLCA,
    getLeaves,
    getRoot,
    minimalVertices,
)


@pytest.fixture
def cherry_tree():
    """Tree ((a, c1), c2) with two leaves of the same color.

    c1 shares the lower ancestor p with a and is therefore the only best match
    of a in color C.
    """
    T = nx.DiGraph()
    T.add_edges_from([("r", "p"), ("r", "c2"), ("p", "a"), ("p", "c1")])
    sigma = {"a": "A", "c1": "C", "c2": "C"}
    return T, sigma


@pytest.fixture
def two_lca_network():
    """Network in which LCA(x, y) consists of two incomparable vertices.

    LCA(x, y) = {a, b} while LCA(x, y2) = {c} with c below b, so y is beaten by
    y2 under the strict definition but survives the weak one via a.
    """
    N = nx.DiGraph()
    N.add_edges_from(
        [
            ("r", "a"),
            ("r", "b"),
            ("b", "c"),
            ("a", "x"),
            ("a", "y"),
            ("b", "y"),
            ("c", "x"),
            ("c", "y2"),
        ]
    )
    sigma = {"x": "X", "y": "Y", "y2": "Y"}
    return N, sigma


def buildRandomNetwork(numSpecies, numHybridizations, seed):
    """Generate an AsymmeTree gene tree and hybridize it, without plotting."""
    random.seed(seed)
    speciesTree = te.species_tree_n_age(n=numSpecies, age=1.0)
    geneTree = te.dated_gene_tree(speciesTree, dupl_rate=0.7)
    nxGeneTree, rootID = geneTree.to_nx()

    with patch("src.generate_hybrid_network.visualizeGraph"):
        network, _ = generateHybridNetwork(
            nxGeneTree, rootID, numHybridizations, geneColors={}
        )
    return geneTree, nxGeneTree, network


class TestPartialOrder:
    """Tests for the reachability order and the LCA sets derived from it."""

    def test_ancestors_are_reflexive(self, cherry_tree):
        T, _ = cherry_tree
        ancestors = computeAncestorSets(T)

        assert all(v in ancestors[v] for v in T.nodes())

    def test_ancestors_accumulate_along_paths(self, cherry_tree):
        T, _ = cherry_tree
        ancestors = computeAncestorSets(T)

        assert ancestors["a"] == {"a", "p", "r"}
        assert ancestors["c2"] == {"c2", "r"}

    def test_lca_in_a_tree_is_a_single_vertex(self, cherry_tree):
        T, _ = cherry_tree
        ancestors = computeAncestorSets(T)

        assert computeLCA(["a", "c1"], ancestors) == {"p"}
        assert computeLCA(["a", "c2"], ancestors) == {"r"}

    def test_lca_in_a_network_can_have_several_vertices(self, two_lca_network):
        N, _ = two_lca_network
        ancestors = computeAncestorSets(N)

        assert computeLCA(["x", "y"], ancestors) == {"a", "b"}
        assert computeLCA(["x", "y2"], ancestors) == {"c"}

    def test_minimal_vertices_drops_comparable_ancestors(self, two_lca_network):
        N, _ = two_lca_network
        ancestors = computeAncestorSets(N)

        assert minimalVertices({"b", "c", "r"}, ancestors) == {"c"}

    def test_cycles_are_rejected(self):
        N = nx.DiGraph()
        N.add_edges_from([("r", "u"), ("u", "v"), ("v", "u")])

        with pytest.raises(ValueError):
            computeAncestorSets(N)

    def test_several_roots_are_rejected(self):
        N = nx.DiGraph()
        N.add_edges_from([("r1", "x"), ("r2", "y")])

        with pytest.raises(ValueError):
            getRoot(N)

    def test_leaves_are_the_vertices_without_descendants(self, two_lca_network):
        N, _ = two_lca_network

        assert set(getLeaves(N)) == {"x", "y", "y2"}


class TestBestMatchGraphs:
    """Tests for the strict and weak best match graphs of a network."""

    def test_lower_lca_wins_in_a_tree(self, cherry_tree):
        T, sigma = cherry_tree
        bmg, wbmg = bestMatchGraphs(T, sigma=sigma)

        assert set(bmg.edges()) == {("a", "c1"), ("c1", "a"), ("c2", "a")}
        assert set(wbmg.edges()) == set(bmg.edges())

    def test_colors_are_carried_over_to_the_graph(self, cherry_tree):
        T, sigma = cherry_tree
        bmg, _ = bestMatchGraphs(T, sigma=sigma)

        assert {v: bmg.nodes[v]["color"] for v in bmg.nodes()} == sigma

    def test_weak_can_be_strictly_larger_than_strict(self, two_lca_network):
        N, sigma = two_lca_network
        bmg, wbmg = bestMatchGraphs(N, sigma=sigma)

        assert set(bmg.edges()) == {("x", "y2"), ("y", "x"), ("y2", "x")}
        assert set(wbmg.edges()) == {
            ("x", "y"),
            ("x", "y2"),
            ("y", "x"),
            ("y2", "x"),
        }

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_tree_bmg_matches_asymmetree(self, seed):
        geneTree, nxGeneTree, _ = buildRandomNetwork(
            numSpecies=6, numHybridizations=0, seed=seed
        )
        expected = best_matches.bmg_from_tree(geneTree)
        bmg, wbmg = bestMatchGraphs(nxGeneTree)

        assert set(bmg.nodes()) == set(expected.nodes())
        assert set(bmg.edges()) == set(expected.edges())
        # On trees both definitions collapse to the same relation.
        assert set(wbmg.edges()) == set(expected.edges())

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_strict_best_matches_are_weak_best_matches(self, seed):
        _, _, network = buildRandomNetwork(
            numSpecies=6, numHybridizations=3, seed=seed
        )
        bmg, wbmg = bestMatchGraphs(network)

        assert set(bmg.edges()) <= set(wbmg.edges())

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_weak_best_match_graph_is_color_sink_free(self, seed):
        _, _, network = buildRandomNetwork(
            numSpecies=6, numHybridizations=3, seed=seed
        )
        _, wbmg = bestMatchGraphs(network)
        colors = {wbmg.nodes[v]["color"] for v in wbmg.nodes()}

        for x in wbmg.nodes():
            reached = {wbmg.nodes[y]["color"] for y in wbmg.successors(x)}
            assert reached == colors - {wbmg.nodes[x]["color"]}

    def test_uncolored_leaf_is_rejected(self, cherry_tree):
        T, sigma = cherry_tree
        del sigma["c2"]

        with pytest.raises(ValueError):
            bestMatchGraphs(T, sigma=sigma)

    def test_missing_reconciliation_is_rejected(self, cherry_tree):
        T, _ = cherry_tree

        with pytest.raises(ValueError):
            bestMatchGraphs(T)
