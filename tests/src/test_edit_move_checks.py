"""
Tests for task 2d: single 2c-moves and short combinations vs. the (weak) BMG.

A candidate 2c-move (pull-up, pull-down, contraction, redundant removal) that
stays a phylogenetic network need not keep the BMG. Task 2c therefore filters
by the weak BMG. These tests record that split on the BIC-cherry explanations
of tree-BMGs, which is the input task 2d is defined on, and check short
sequences against the BMG of the starting network.
"""

import random
from collections import Counter

import asymmetree.treeevolve as te
import networkx as nx
import numpy as np
import pytest

from src import edit_move_checks
from src.bic_cherry_explanations import bicCherryExplanation, explainTreeBmg
from src.edit_move_checks import (
    applyCandidateMove,
    candidateMoveApplicators,
    candidateMoves,
    checkEditingPath,
    checkMoveCombinations,
    checkSingleMoves,
    formatMoveSurvey,
    leafSet,
    surveyEditMoves,
)
from src.editing_operations import preservesPhylogeneticNetwork
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


@pytest.fixture
def redundant_pair():
    """'p' and 'q' share both parents and children, the removal case of 2c.

    The BIC-cherry explanations never produce this shape, so it needs its own
    fixture for the fourth move kind to be covered at all.
    """
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("r", "p"), ("r", "q"), ("r", "s"),
            ("p", "x"), ("p", "y"),
            ("q", "x"), ("q", "y"),
            ("s", "z"), ("s", "w"),
        ]
    )
    for leaf, color in {"x": "A", "y": "B", "z": "A", "w": "B"}.items():
        g.nodes[leaf]["color"] = color
    return g


class TestSingleMovesOnTreeBmgExplanations:
    def test_candidates_exist_when_the_expansion_rewired(self, restricted_four_leaves):
        results = checkSingleMoves(restricted_four_leaves)

        assert results

    def test_some_candidates_keep_the_bmg(self, restricted_four_leaves):
        """Otherwise task 2c would have nothing it is allowed to commit."""
        results = checkSingleMoves(restricted_four_leaves)

        assert any(move.weak for move in results)
        assert any(move.strict for move in results)

    def test_some_candidates_break_the_bmg(self, restricted_four_leaves):
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

    def test_removing_a_redundant_vertex_keeps_both_bmgs(self, redundant_pair):
        """An interchangeable vertex carries no best match information."""
        results = checkSingleMoves(redundant_pair)
        removals = [move for move in results if move.kind == "remove-redundant"]

        assert removals
        assert all(move.strict and move.weak for move in removals)

    def test_the_candidate_set_covers_all_four_move_kinds(
        self, restricted_four_leaves, dummy_chain, redundant_pair
    ):
        """Guards against a move kind silently dropping out of the survey."""
        kinds = set()
        for network in (restricted_four_leaves, dummy_chain, redundant_pair):
            kinds |= {kind for kind, _ in candidateMoveApplicators(network)}

        assert {"contract", "remove-redundant"} <= kinds
        assert len({k for k in kinds if k.startswith("pull-")}) == 4

    def test_contracting_a_dummy_vertex_keeps_both_bmgs(self, dummy_chain):
        results = checkSingleMoves(dummy_chain)
        contractions = [move for move in results if move.kind == "contract"]

        assert contractions
        assert all(move.strict and move.weak for move in contractions)


class TestCombinationsOnTreeBmgExplanations:
    def test_pairs_are_checked_against_the_starting_bmg(self, restricted_four_leaves):
        pairs = checkMoveCombinations(restricted_four_leaves, length=2, limit=25)

        assert pairs
        assert any(move.strict and move.weak for move in pairs)
        assert any(not (move.strict and move.weak) for move in pairs)

    def test_triples_are_checked_against_the_starting_bmg(self, restricted_four_leaves):
        triples = checkMoveCombinations(restricted_four_leaves, length=3, limit=15)

        assert triples
        assert any(move.strict and move.weak for move in triples)

    def test_a_combination_is_rejected_for_length_one(self, restricted_four_leaves):
        with pytest.raises(ValueError, match="at least two"):
            checkMoveCombinations(restricted_four_leaves, length=1)

    def test_each_sequence_reports_the_moves_it_used(self, restricted_four_leaves):
        """Task 2f needs to know which moves a failing sequence consisted of."""
        pairs = checkMoveCombinations(restricted_four_leaves, length=2, limit=25)

        assert all(len(move.kinds) == 2 for move in pairs)
        kinds = {kind for move in pairs for kind in move.kinds}
        assert kinds <= {
            "pull-up-tail",
            "pull-up-head",
            "pull-down-tail",
            "pull-down-head",
            "contract",
            "remove-redundant",
        }

    def test_no_sequence_leads_back_to_the_starting_network(
        self, restricted_four_leaves
    ):
        """Undoing a pull is not a modification, so it must not be counted.

        Without this filter a pull and its inverse show up as a pair that
        'preserves' both BMGs, which inflates the task 2d counts.
        """
        origin = frozenset(restricted_four_leaves.edges())
        reachable = _endStatesOfPairs(restricted_four_leaves)

        assert origin in _allPairEndStates(restricted_four_leaves)
        assert origin not in reachable

    def test_every_reported_end_network_is_distinct(self, restricted_four_leaves):
        # the limit has to exceed the number of distinct end states, otherwise
        # the walk stops early and the counts cannot be compared
        pairs = checkMoveCombinations(restricted_four_leaves, length=2, limit=10 ** 4)
        states = _endStatesOfPairs(restricted_four_leaves)

        assert len(pairs) == len(states)
        assert len(pairs) < len(_allPairEndStates(restricted_four_leaves))

    def test_the_limit_caps_the_number_of_sequences(self, restricted_four_leaves):
        pairs = checkMoveCombinations(restricted_four_leaves, length=2, limit=7)

        assert len(pairs) == 7


def _allPairEndStates(network):
    """Every 2-move end state, undeduplicated, as the naive walk would count."""
    states = []
    for _, first in candidateMoveApplicators(network):
        afterFirst = network.copy()
        if not first(afterFirst):
            continue
        for _, second in candidateMoveApplicators(afterFirst):
            afterSecond = afterFirst.copy()
            if not second(afterSecond):
                continue
            states.append(frozenset(afterSecond.edges()))
    return states


def _endStatesOfPairs(network):
    """The end states behind checkMoveCombinations(length=2), rebuilt."""
    origin = frozenset(network.edges())
    states = set()
    for signature in _allPairEndStates(network):
        if signature != origin:
            states.add(signature)
    return states


class TestGreedyEditingPath:
    """Task 2d along the path src.editing_operations actually commits to."""

    def test_the_path_is_reported_step_by_step(self, restricted_four_leaves):
        path = checkEditingPath(restricted_four_leaves, maxMoves=3)

        assert path
        assert len(path) <= 3
        assert [len(move.kinds) for move in path] == list(range(1, len(path) + 1))

    def test_the_first_step_agrees_with_a_single_move(self, restricted_four_leaves):
        """The greedy path starts inside the enumerated neighbourhood."""
        singles = checkSingleMoves(restricted_four_leaves)
        first = checkEditingPath(restricted_four_leaves, maxMoves=1)[0]

        assert any(
            move.strict == first.strict and move.weak == first.weak
            for move in singles
        )

    def test_the_path_stops_when_nothing_is_left_to_edit(self, dummy_chain):
        """The dummy chain admits one contraction and is least resolved after it.

        Asking for three moves therefore returns one, without an error.
        """
        path = checkEditingPath(dummy_chain, maxMoves=3)

        assert len(path) == 1
        assert path[0].strict and path[0].weak


class TestSurveyMatchesTheTask:
    def test_the_summary_counts_agree_with_the_single_move_list(
        self, restricted_four_leaves
    ):
        singles = checkSingleMoves(restricted_four_leaves)
        survey = surveyEditMoves(
            restricted_four_leaves, singleLimit=None, pairLimit=10, tripleLimit=5
        )

        assert survey.singles.total == len(singles)
        assert survey.singles.checked == len(singles)
        assert survey.singles.strict == sum(m.strict for m in singles)
        assert survey.singles.weak == sum(m.weak for m in singles)
        text = formatMoveSurvey(survey)
        assert "candidates: edits leaving a phylogenetic network" in text
        assert "\nsingles  candidates=" in text
        assert "\npairs    candidates=" in text
        assert "\ntriples  candidates=" in text
        assert "\npath     candidates=" in text

    def test_simulated_restricted_explanations_can_be_surveyed(self):
        explanations = explainTreeBmg(buildGeneTree(numSpecies=4, seed=3))
        survey = surveyEditMoves(
            explanations.variantB, singleLimit=40, pairLimit=12, tripleLimit=6
        )

        assert survey.singles.total > 0
        assert 0 <= survey.singles.strict <= survey.singles.checked
        assert 0 <= survey.singles.weak <= survey.singles.checked
        assert survey.pairs.checked <= 12
        assert survey.triples.checked <= 6

    def test_the_summary_counts_agree_with_the_greedy_path(
        self, restricted_four_leaves
    ):
        path = checkEditingPath(restricted_four_leaves, maxMoves=None)
        survey = surveyEditMoves(
            restricted_four_leaves, singleLimit=10, pairLimit=5, tripleLimit=3
        )

        assert survey.path.checked == len(path)
        assert survey.path.strict == sum(m.strict for m in path)
        assert survey.path.weak == sum(m.weak for m in path)
        assert survey.path.exhaustive

    def test_the_whole_path_is_walked_by_default(self, restricted_four_leaves):
        """Its first moves are not representative, so they are not the default.

        Every one of the first three keeps both BMGs here, while most of the path
        does not, which is the one place where a prefix flatters the operations
        rather than merely truncating them.
        """
        prefix = checkEditingPath(restricted_four_leaves, maxMoves=3)
        whole = surveyEditMoves(restricted_four_leaves, pairLimit=5, tripleLimit=3).path

        assert whole.exhaustive
        assert whole.checked > len(prefix)
        assert sum(m.strict for m in prefix) == len(prefix)
        assert whole.strict < whole.checked

    def test_the_leaf_set_of_the_input_is_the_bmg(self, restricted_four_leaves):
        strict, weak = bestMatchGraphs(restricted_four_leaves)

        assert leafSet(restricted_four_leaves) == set(strict.nodes()) == set(weak.nodes())


class TestTruncationIsVisible:
    """A capped walk must not look like a census of the neighbourhood."""

    def test_an_uncapped_survey_reports_itself_as_exhaustive(
        self, restricted_four_leaves
    ):
        survey = surveyEditMoves(
            restricted_four_leaves, singleLimit=None, pairLimit=None
        )

        assert survey.singles.exhaustive
        assert survey.pairs.exhaustive
        assert survey.singles.checked == survey.singles.total

    def test_a_capped_walk_says_so(self, restricted_four_leaves):
        survey = surveyEditMoves(restricted_four_leaves, singleLimit=5, pairLimit=5)

        assert not survey.singles.exhaustive
        assert not survey.pairs.exhaustive
        assert survey.singles.checked == 5

    def test_the_total_stays_exact_when_only_a_sample_is_checked(
        self, restricted_four_leaves
    ):
        """Enumerating candidates is cheap, so the denominator is never guessed."""
        everyMove = candidateMoves(restricted_four_leaves)
        survey = surveyEditMoves(restricted_four_leaves, singleLimit=5)

        assert survey.singles.total == len(everyMove)
        assert survey.singles.checked == 5
        assert survey.singles.total > survey.singles.checked

    def test_the_marker_appears_in_the_report_only_when_truncated(
        self, restricted_four_leaves
    ):
        capped = formatMoveSurvey(
            surveyEditMoves(restricted_four_leaves, singleLimit=5, pairLimit=5)
        )
        full = formatMoveSurvey(
            surveyEditMoves(
                restricted_four_leaves,
                singleLimit=None,
                pairLimit=None,
                tripleLimit=None,
                pathLimit=None,
            )
        )

        assert "candidates=5+" in capped
        assert "+" not in full.split("\n", 2)[2]

    def test_the_path_is_exhaustive_once_it_runs_out_of_moves(self, dummy_chain):
        survey = surveyEditMoves(dummy_chain, pathLimit=3)

        assert survey.path.checked == 1
        assert survey.path.exhaustive


class TestSamplingIsNotAPrefix:
    """A limit has to draw a sample, not the first few of a biased enumeration.

    candidateMoveApplicators walks network.edges() in insertion order, so a
    prefix concentrates on the first arcs of the network and the shares it
    reports are not estimates of the whole neighbourhood.
    """

    def test_a_capped_single_survey_is_not_the_enumeration_prefix(
        self, restricted_four_leaves
    ):
        prefixKinds = [kind for kind, _ in candidateMoves(restricted_four_leaves)[:8]]
        sampledKinds = [
            move.kind
            for move in checkSingleMoves(restricted_four_leaves, limit=8)
        ]

        assert len(sampledKinds) == 8
        assert sampledKinds != prefixKinds

    def test_the_sample_is_drawn_from_the_candidate_set(self, restricted_four_leaves):
        available = Counter(kind for kind, _ in candidateMoves(restricted_four_leaves))
        drawn = Counter(
            move.kind for move in checkSingleMoves(restricted_four_leaves, limit=8)
        )

        for kind, count in drawn.items():
            assert count <= available[kind]

    def test_the_seed_makes_the_draw_reproducible(self, restricted_four_leaves):
        first = checkSingleMoves(restricted_four_leaves, limit=8, seed=7)
        again = checkSingleMoves(restricted_four_leaves, limit=8, seed=7)
        other = checkSingleMoves(restricted_four_leaves, limit=8, seed=8)

        assert first == again
        assert first != other

    def test_an_uncapped_survey_keeps_the_enumeration_order(
        self, restricted_four_leaves
    ):
        """Shuffling only matters when sampling, so a full run stays stable."""
        kinds = [kind for kind, _ in candidateMoves(restricted_four_leaves)]
        checked = [
            move.kind for move in checkSingleMoves(restricted_four_leaves, limit=None)
        ]

        assert checked == kinds

    def test_capped_sequences_are_spread_over_the_neighbourhood(
        self, restricted_four_leaves
    ):
        capped = checkMoveCombinations(restricted_four_leaves, length=2, limit=8)
        exhaustive = checkMoveCombinations(
            restricted_four_leaves, length=2, limit=None
        )

        assert 0 < len(capped) <= 8
        assert len(exhaustive) > len(capped)
        assert [m.kinds for m in capped] != [m.kinds for m in exhaustive[: len(capped)]]

    def test_a_capped_share_estimates_the_exhaustive_one(self, restricted_four_leaves):
        """The point of sampling: the shares have to survive the truncation.

        A depth-first prefix used to report that no pair at all kept the BMG on
        this network, against a true share of about two in five, because the
        prefix never left the first branch. Averaging a few seeds keeps the
        tolerance here well inside sampling noise.
        """
        full = checkMoveCombinations(restricted_four_leaves, length=2, limit=None)
        truth = sum(m.strict for m in full) / len(full)

        estimates = []
        for seed in range(6):
            sample = checkMoveCombinations(
                restricted_four_leaves, length=2, limit=30, seed=seed
            )
            estimates.append(sum(m.strict for m in sample) / len(sample))

        assert 0 < truth < 1
        assert abs(sum(estimates) / len(estimates) - truth) < 0.1

    def test_a_capped_walk_does_not_apply_every_candidate(
        self, restricted_four_leaves, monkeypatch
    ):
        """The saving that makes an unbiased sample affordable on real networks.

        Thinning a layer after building it would need the whole layer first, and
        applying a candidate is the expensive step, so the limit has to bite on
        the candidates tried rather than on the networks kept.
        """
        applied = []
        original = edit_move_checks.applyCandidateMove

        def counting(network, apply):
            applied.append(1)
            return original(network, apply)

        monkeypatch.setattr(edit_move_checks, "applyCandidateMove", counting)

        checkMoveCombinations(restricted_four_leaves, length=3, limit=4)
        capped = len(applied)
        applied.clear()
        checkMoveCombinations(restricted_four_leaves, length=3, limit=None)
        full = len(applied)

        assert capped < full / 10


class TestMovesStayPhylogeneticNetworks:
    """Every candidate has to survive src.network_best_matches unchanged.

    A move that detaches a subnetwork leaves a second root behind. The BMG of
    such a graph is not defined, so counting it as a move that 'preserved' the
    BMG would be meaningless.
    """

    def test_no_single_move_produces_a_second_root(self, restricted_four_leaves):
        checked = 0
        for _, apply in candidateMoveApplicators(restricted_four_leaves):
            trial = applyCandidateMove(restricted_four_leaves, apply)
            if trial is None:
                continue
            checked += 1
            assert preservesPhylogeneticNetwork(
                leafSet(restricted_four_leaves), trial
            )

        assert checked

    def test_every_surveyed_network_has_a_well_defined_bmg(
        self, restricted_four_leaves
    ):
        """bestMatchGraphs raises on a forest, so reaching a verdict is the test."""
        for _, apply in candidateMoveApplicators(restricted_four_leaves):
            trial = applyCandidateMove(restricted_four_leaves, apply)
            if trial is None:
                continue
            bestMatchGraphs(trial)
