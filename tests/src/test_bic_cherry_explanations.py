"""
Tests for task 2b, the BIC-cherry+expansion explanations of tree-BMGs

The two counterexamples below are of minimal size, and variant (b) is not known
to fail at all: an exhaustive search over every leaf-colored phylogenetic tree
on up to six leaves, i.e. over every tree-BMG of that size, produced no graph
that variant (b) fails to explain, while variant (a) already fails on four
leaves for the weak and on five leaves for the strict best matches. Neither
minimum is attained by a single graph, so the fixtures are representatives
rather than the unique witnesses. The simulated gene families here cover the
larger sizes that such a search cannot reach.
"""

import itertools
import random

import asymmetree.treeevolve as te
import networkx as nx
import numpy as np
import pytest
from asymmetree.analysis import best_matches

from src.bic_cherry_explanations import (
    bicCherryExplanation,
    explainTreeBmg,
    missingArcs,
)
from src.least_resolved_tree import explainsBmg, treesEqual
from src.network_best_matches import (
    getLeaves,
    getRoot,
    networkExplainsBmg,
)

SEEDS = [1, 2, 3, 4, 5]

VARIANTS = [
    pytest.param(False, id="expansion"),
    pytest.param(True, id="edge-restricted"),
]

# One of the two smallest tree-BMGs on which variant (a) can get the weak BMG
# wrong. Several trees explain it; one is ((a1, a2), (a3, b)), so b is the only
# gene of its color and hence a sicor that every other gene points to, while b
# itself only has a3 as a best match. The missing arcs (b, a1) and (b, a2) are
# answered by an extension towards some other gene of color A, and picking a1
# or a2 there makes that gene a weak best match of b.
WEAK_COUNTEREXAMPLE = (
    {"a1": "A", "a2": "A", "a3": "A", "b": "B"},
    [("a1", "b"), ("a2", "b"), ("a3", "b"), ("b", "a3")],
)

# One of the smallest tree-BMGs on which variant (a) can get the strict BMG
# wrong; five leaves admit several such graphs. The tree is
# ((a1, b1), (a2, b2), b3), so b3 sits directly below the root and has both
# a1 and a2 as best matches. Answering the missing arc (a2, b1) with an
# extension towards b3 pushes a common ancestor of a2 and b3 below the root and
# costs b3 the arc to a1.
STRICT_COUNTEREXAMPLE = (
    {"a1": "A", "a2": "A", "b1": "B", "b2": "B", "b3": "B"},
    [
        ("a1", "b1"),
        ("b1", "a1"),
        ("a2", "b2"),
        ("b2", "a2"),
        ("b3", "a1"),
        ("b3", "a2"),
    ],
)


def buildGeneTree(numSpecies, seed):
    """An AsymmeTree gene tree, built without any plotting side effects.

    Loss leaves are pruned as in utils.generateGeneTree, since they carry an
    edge of the species tree as their 'reconc' and would add genes and species
    that are not observable.
    """
    random.seed(seed)
    np.random.seed(seed)
    speciesTree = te.species_tree_n_age(n=numSpecies, age=1.0)
    return te.prune_losses(te.dated_gene_tree(speciesTree, dupl_rate=0.7))


def coloredGraph(sigma, arcs):
    """A digraph whose vertices carry their color the way a BMG does."""
    G = nx.DiGraph()
    for v, color in sigma.items():
        G.add_node(v, color=color)
    G.add_edges_from(arcs)
    return G


def everyVertexOrder(sigma, arcs):
    """The same graph once per vertex order.

    Variant (a) may expand towards any gene of the right color. Which one it
    takes is decided by the order of the vertices, so running over all orders
    is how a test covers every admissible choice instead of the single one that
    a dict literal happens to produce.
    """
    for order in itertools.permutations(sigma):
        yield coloredGraph({v: sigma[v] for v in order}, arcs)


@pytest.fixture
def weak_counterexample():
    return coloredGraph(*WEAK_COUNTEREXAMPLE)


@pytest.fixture
def strict_counterexample():
    return coloredGraph(*STRICT_COUNTEREXAMPLE)


class TestCounterexamplesAreTreeBmgs:
    """The counterexamples only count if they belong to task 2b's input."""

    @pytest.mark.parametrize(
        "sigma, arcs", [WEAK_COUNTEREXAMPLE, STRICT_COUNTEREXAMPLE]
    )
    def test_a_tree_explains_them(self, sigma, arcs):
        assert best_matches.is_bmg(coloredGraph(sigma, arcs)) is not None


class TestExplanationStructure:
    """Both variants have to produce a phylogenetic network on the same leaves."""

    @pytest.mark.parametrize("edgeRestricted", VARIANTS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_result_is_a_dag_with_a_unique_root(self, seed, edgeRestricted):
        bmg = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed)).bmg
        network, root = bicCherryExplanation(bmg, edgeRestricted=edgeRestricted)

        assert nx.is_directed_acyclic_graph(network)
        assert getRoot(network) == root

    @pytest.mark.parametrize("edgeRestricted", VARIANTS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_leaves_are_the_vertices_of_the_graph(self, seed, edgeRestricted):
        bmg = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed)).bmg
        network, _ = bicCherryExplanation(bmg, edgeRestricted=edgeRestricted)

        assert set(getLeaves(network)) == set(bmg.nodes())

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_edge_restricted_variant_stays_smaller(self, seed):
        bmg = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed)).bmg
        expansion, _ = bicCherryExplanation(bmg)
        restricted, _ = bicCherryExplanation(bmg, edgeRestricted=True)

        # Variant (b) reuses the cherry vertices instead of inserting q_xz.
        assert restricted.number_of_nodes() <= expansion.number_of_nodes()

    def test_a_graph_without_the_sicor_in_hub_property_is_rejected(self):
        # b is the only gene of its color, so it has to be an in-hub, but the
        # arc (a, b) is missing.
        G = coloredGraph({"a": "A", "b": "B"}, [("b", "a")])

        with pytest.raises(ValueError, match="Sicor-Inhub"):
            bicCherryExplanation(G)


class TestEdgeRestrictedExplainsTreeBmgs:
    """Task 2b: the explanations built for the tree-BMGs of task 2a."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_strict_best_match_graph_is_recovered(self, seed):
        explanations = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed))

        assert networkExplainsBmg(explanations.variantB, explanations.bmg)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_weak_best_match_graph_is_recovered(self, seed):
        explanations = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed))

        assert networkExplainsBmg(
            explanations.variantB, explanations.bmg, weak=True
        )

    @pytest.mark.parametrize("numSpecies", [4, 6, 8])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_explanation_holds_for_larger_gene_families(self, seed, numSpecies):
        explanations = explainTreeBmg(buildGeneTree(numSpecies, seed=seed))

        assert networkExplainsBmg(explanations.variantB, explanations.bmg)
        assert networkExplainsBmg(
            explanations.variantB, explanations.bmg, weak=True
        )

    def test_the_seeds_actually_trigger_expansions(self):
        """Guard against a vacuous suite.

        Without a missing arc no expansion happens at all, the base BIC-cherry
        network explains the graph by itself and the tests above would hold for
        trivial reasons.
        """
        counts = [
            len(missingArcs(explainTreeBmg(buildGeneTree(6, seed)).bmg))
            for seed in SEEDS
        ]

        assert any(count > 0 for count in counts)

    @pytest.mark.parametrize(
        "sigma, arcs", [WEAK_COUNTEREXAMPLE, STRICT_COUNTEREXAMPLE]
    )
    def test_it_is_immune_to_the_choice_that_breaks_variant_a(self, sigma, arcs):
        """Variant (b) explains both counterexamples in every vertex order."""
        for bmg in everyVertexOrder(sigma, arcs):
            network, _ = bicCherryExplanation(bmg, edgeRestricted=True)

            assert networkExplainsBmg(network, bmg)
            assert networkExplainsBmg(network, bmg, weak=True)


class TestArbitraryChoiceIsNotEnough:
    """Why task 2b relies on variant (b).

    The expansion step of variant (a) may pick any y' of the right color. An
    extension [xy : xy'] puts a common ancestor of x and y' strictly below
    p_xy, which turns y' into a best match of x. That is only wanted when
    (x, y') is an arc of the graph, and restricting the choice to those arcs is
    exactly what variant (b) adds.
    """

    def test_some_choice_adds_a_weak_best_match(self):
        sigma, arcs = WEAK_COUNTEREXAMPLE
        results = {
            networkExplainsBmg(
                bicCherryExplanation(bmg)[0], bmg, weak=True
            )
            for bmg in everyVertexOrder(sigma, arcs)
        }

        assert False in results

    def test_some_choice_loses_a_strict_best_match(self):
        sigma, arcs = STRICT_COUNTEREXAMPLE
        results = {
            networkExplainsBmg(bicCherryExplanation(bmg)[0], bmg)
            for bmg in everyVertexOrder(sigma, arcs)
        }

        assert False in results

    def test_a_forced_choice_is_always_the_right_one(self):
        """With two genes per color there is nothing left to choose.

        The tree is ((a1, b1), (a2, b2)); every missing arc has exactly one
        candidate y', so both variants agree and variant (a) explains the graph
        in every vertex order.
        """
        sigma = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
        arcs = [("a1", "b1"), ("b1", "a1"), ("a2", "b2"), ("b2", "a2")]

        for bmg in everyVertexOrder(sigma, arcs):
            assert missingArcs(bmg)
            network, _ = bicCherryExplanation(bmg)

            assert networkExplainsBmg(network, bmg)
            assert networkExplainsBmg(network, bmg, weak=True)

    def test_it_fails_on_simulated_gene_families_too(self):
        """The counterexamples are not an artifact of hand-built graphs.

        Whether variant (a) survives a given gene tree depends on the sizes of
        its color classes, so the failure is asserted over the seed set rather
        than for every single seed.
        """
        failures = []
        for seed in SEEDS:
            explanations = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed))
            if not networkExplainsBmg(explanations.variantA, explanations.bmg):
                failures.append(seed)

        assert failures


class TestColoringSource:
    """sigma may come from the graph itself or be passed in explicitly."""

    @pytest.mark.parametrize("edgeRestricted", VARIANTS)
    def test_the_color_attribute_is_used_by_default(
        self, weak_counterexample, edgeRestricted
    ):
        bmg = weak_counterexample
        sigma = {v: bmg.nodes[v]["color"] for v in bmg.nodes()}

        implicit, _ = bicCherryExplanation(bmg, edgeRestricted=edgeRestricted)
        explicit, _ = bicCherryExplanation(
            bmg, sigma=sigma, edgeRestricted=edgeRestricted
        )

        assert set(implicit.edges()) == set(explicit.edges())

    def test_missing_arcs_are_the_pairs_that_trigger_an_expansion(
        self, weak_counterexample
    ):
        assert set(missingArcs(weak_counterexample)) == {
            ("b", "a1"),
            ("b", "a2"),
        }

    def test_a_complete_graph_needs_no_expansion(self):
        # One gene per species: every leaf is a sicor and the BMG is complete.
        bmg = coloredGraph({"a": "A", "b": "B"}, [("a", "b"), ("b", "a")])

        assert missingArcs(bmg) == []


class TestTargetIsCarriedAlong:
    """The T* of task 2a has to belong to the very BMG that was explained."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_target_explains_the_same_graph(self, seed):
        explanations = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed))

        assert explainsBmg(explanations.target, explanations.bmg)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_target_is_the_least_resolved_tree_of_the_graph(self, seed):
        explanations = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed))

        assert treesEqual(
            explanations.target, best_matches.is_bmg(explanations.bmg)
        )

    @pytest.mark.parametrize("edgeRestricted", VARIANTS)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_networks_are_not_trees_yet(self, seed, edgeRestricted):
        """The motivation for tasks 2c to 2f.

        The target is a tree, the explanation is not: a leaf hangs below one
        cherry vertex per differently colored partner, and that overlap is what
        the editing moves have to remove.
        """
        bmg = explainTreeBmg(buildGeneTree(numSpecies=6, seed=seed)).bmg
        network, _ = bicCherryExplanation(bmg, edgeRestricted=edgeRestricted)

        reticulations = [v for v in network if network.in_degree(v) > 1]

        assert reticulations, "nothing left for the editing tasks to do"
