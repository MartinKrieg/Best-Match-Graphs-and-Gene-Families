"""
Tests for task 2d: single 2c-moves and short combinations vs. the (weak) BMG.

A geometric 2c-move (pull-up, pull-down, contraction, redundant removal) that
stays a phylogenetic network need not keep the BMG. Task 2c therefore filters
by the weak BMG. These tests record that split on the BIC-cherry explanations
of tree-BMGs, which is the input task 2d is defined on, and check short
sequences against the BMG of the starting network.
"""

import random

import asymmetree.treeevolve as te
import networkx as nx
import numpy as np
import pytest

from src.bic_cherry_explanations import bicCherryExplanation, explainTreeBmg
from src.edit_move_checks import (
    checkMoveCombinations,
    checkSingleMoves,
    formatMoveSurvey,
    leafSet,
    surveyEditMoves,
)
from src.network_best_matches import bestMatchGraphs


def coloredGraph(sigma, arcs):
    G = nx.DiGraph()
    for v, color in sigma.items():
        G.add_node(v, color=color)
    G.add_edges_from(arcs)
    return G


def explainingNetwork(sigma, arcs, edgeRestricted=True):
    return bicCherryExplanation(
        coloredGraph(sigma, arcs), edgeRestricted=edgeRestricted
    )[0]


def buildGeneTree(numSpecies, seed):
    random.seed(seed)
    np.random.seed(seed)
    speciesTree = te.species_tree_n_age(n=numSpecies, age=1.0)
    return te.prune_losses(te.dated_gene_tree(speciesTree, dupl_rate=0.7))


# The 4-leaf tree-BMG of task 2b. Missing arcs exist, so the restricted
# expansion actually rewires the base network.
TREE_BMG_FOUR_LEAVES = (
    {"a1": "A", "a2": "A", "a3": "A", "b": "B"},
    [("a1", "b"), ("a2", "b"), ("a3", "b"), ("b", "a3")],
)

# Two genes per colour: every missing arc has a unique y', so both variants
# explain the graph and the restricted network is still small.
TREE_BMG_FORCED_CHOICE = (
    {"a1": "A", "a2": "A", "b1": "B", "b2": "B"},
    [("a1", "b1"), ("b1", "a1"), ("a2", "b2"), ("b2", "a2")],
)


@pytest.fixture
def restricted_four_leaves():
    return explainingNetwork(*TREE_BMG_FOUR_LEAVES)


@pytest.fixture
def dummy_chain():
    """An inner 1-in/1-out vertex, the contraction case of task 2c."""
    g = nx.DiGraph()
    g.add_edges_from([("r", "d"), ("d", "x"), ("r", "y")])
    g.nodes["x"]["color"] = "A"
    g.nodes["y"]["color"] = "B"
    return g


class TestSingleMovesOnTreeBmgExplanations:
    def test_geometric_moves_exist_when_the_expansion_rewired(self, restricted_four_leaves):
        results = checkSingleMoves(restricted_four_leaves)

        assert results

    def test_some_geometric_moves_keep_the_bmg(self, restricted_four_leaves):
        """Otherwise task 2c would have nothing it is allowed to commit."""
        results = checkSingleMoves(restricted_four_leaves)

        assert any(move.weak for move in results)
        assert any(move.strict for move in results)

    def test_some_geometric_moves_break_the_bmg(self, restricted_four_leaves):
        """Why task 2c has to filter: DAG and leaf set are not enough."""
        results = checkSingleMoves(restricted_four_leaves)

        assert any(not move.weak for move in results)
        assert any(not move.strict for move in results)

    def test_moves_that_keep_the_weak_bmg_keep_the_strict_one_too(
        self, restricted_four_leaves
    ):
        """On this tree-BMG explanation the two graphs of N coincide."""
        results = checkSingleMoves(restricted_four_leaves)

        assert all(move.strict for move in results if move.weak)

    def test_strict_and_weak_need_not_move_together(self):
        """The forced-choice explanation already separates the two filters."""
        network = explainingNetwork(*TREE_BMG_FORCED_CHOICE)
        results = checkSingleMoves(network)

        assert any(move.strict != move.weak for move in results)

    def test_contracting_a_dummy_vertex_keeps_both_bmgs(self, dummy_chain):
        results = checkSingleMoves(dummy_chain)
        contractions = [move for move in results if move.kind == "contract"]

        assert contractions
        assert all(move.strict and move.weak for move in contractions)


class TestCombinationsOnTreeBmgExplanations:
    def test_pairs_are_checked_against_the_starting_bmg(self, restricted_four_leaves):
        pairs = checkMoveCombinations(restricted_four_leaves, length=2, limit=25)

        assert pairs
        assert any(strict and weak for strict, weak in pairs)
        assert any(not (strict and weak) for strict, weak in pairs)

    def test_triples_are_checked_against_the_starting_bmg(self, restricted_four_leaves):
        triples = checkMoveCombinations(restricted_four_leaves, length=3, limit=15)

        assert triples
        assert any(strict and weak for strict, weak in triples)

    def test_a_combination_is_rejected_for_length_one(self, restricted_four_leaves):
        with pytest.raises(ValueError, match="at least two"):
            checkMoveCombinations(restricted_four_leaves, length=1)


class TestSurveyMatchesTheTask:
    def test_the_summary_counts_agree_with_the_single_move_list(
        self, restricted_four_leaves
    ):
        singles = checkSingleMoves(restricted_four_leaves)
        survey = surveyEditMoves(
            restricted_four_leaves, singleLimit=None, pairLimit=10, tripleLimit=5
        )

        assert survey.geometric == len(singles)
        assert survey.singleStrict == sum(m.strict for m in singles)
        assert survey.singleWeak == sum(m.weak for m in singles)
        text = formatMoveSurvey(survey)
        assert "singles  geometric=" in text
        assert "\npairs    geometric=" in text
        assert "\ntriples  geometric=" in text

    def test_simulated_restricted_explanations_can_be_surveyed(self):
        explanations = explainTreeBmg(buildGeneTree(numSpecies=4, seed=3))
        survey = surveyEditMoves(
            explanations.variantB, singleLimit=40, pairLimit=12, tripleLimit=6
        )

        assert survey.geometric > 0
        assert 0 <= survey.singleStrict <= survey.geometric
        assert 0 <= survey.singleWeak <= survey.geometric
        assert survey.pairGeometric <= 12
        assert survey.tripleGeometric <= 6

    def test_the_leaf_set_of_the_input_is_the_bmg(self, restricted_four_leaves):
        strict, weak = bestMatchGraphs(restricted_four_leaves)

        assert leafSet(restricted_four_leaves) == set(strict.nodes()) == set(weak.nodes())
