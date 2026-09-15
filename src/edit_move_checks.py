"""
Task 2d: do the 2c edit operations preserve the (weak) best match graph?

A candidate is any pull-up, pull-down, dummy contraction or redundant vertex
removal that leaves a phylogenetic network on the same leaf set, i.e. one that
passes src.editing_operations.preservesPhylogeneticNetwork. Whether it keeps the
best match graph is exactly what this module measures, so the BMG is never
consulted when a candidate is collected.

The candidates come from the very helpers src.editing_operations uses, called
with checkBMG=False so that nothing is filtered out. That way the survey
measures the move set task 2c really offers instead of a second implementation
of it.

Counting single moves is cheap, so their number is exact. Sequences of them are
not: the number of networks reachable in three moves runs into the millions on
the networks of task 2b, so those walks take a sample. Every count therefore
travels with the number of networks actually checked and a flag saying whether
the walk finished, because a share is only meaningful against the second and a
bare total invites reading a cut-off walk as a census.
"""

import math
import random
from typing import NamedTuple

import networkx as nx

from .editing_operations import (
    _tryContract,
    _tryMoveEdge,
    _tryRemoveRedundant,
    iterEditingSteps,
    redundantVertexGroups,
)
from .network_best_matches import bestMatchGraphs


class MoveResult(NamedTuple):
    """One candidate and whether it keeps the BMGs."""

    kind: str
    strict: bool
    weak: bool


class SequenceResult(NamedTuple):
    """A sequence of candidates and whether the end network keeps the BMGs."""

    kinds: tuple
    strict: bool
    weak: bool


class MoveCounts(NamedTuple):
    """How many candidates there are, how many were checked, how many survived.

    total and checked are kept apart on purpose. A single number cannot say
    whether 30 means "there are 30" or "we stopped at 30", and the shares are
    only meaningful against the number actually checked.

    exhaustive is False when the walk was cut short by a limit. The shares are
    then estimates from a random sample of a larger neighbourhood, and total is
    a lower bound rather than a census.
    """

    total: int
    checked: int
    strict: int
    weak: int
    exhaustive: bool

    def strictShare(self) -> float | None:
        return self.strict / self.checked if self.checked else None

    def weakShare(self) -> float | None:
        return self.weak / self.checked if self.checked else None


class MoveSurvey(NamedTuple):
    """Task 2d for one network: singles, pairs, triples and the greedy path."""

    singles: MoveCounts
    pairs: MoveCounts
    triples: MoveCounts
    path: MoveCounts

    def everyAcceptedMoveKeepsBothBmgs(self) -> bool:
        """Whether every single-move candidate that was checked kept both BMGs.

        Vacuous when the network admits no candidate at all, which happens on
        already least-resolved trees.
        """
        if self.singles.checked == 0:
            return True
        return (
            self.singles.strict == self.singles.checked
            and self.singles.weak == self.singles.checked
        )


def leafSet(network: nx.DiGraph) -> set:
    return {v for v in network.nodes() if network.out_degree(v) == 0}


def sameColoredGraph(first: nx.DiGraph, second: nx.DiGraph) -> bool:
    return set(first.nodes()) == set(second.nodes()) and set(first.edges()) == set(
        second.edges()
    )


def networkBmgs(network: nx.DiGraph, sigma: dict | None = None):
    """Strict and weak BMG of a leaf-colored network."""
    return bestMatchGraphs(network, sigma=sigma)


def candidateMoveApplicators(network: nx.DiGraph) -> list:
    """Callables that apply one 2c-move to the network they are given.

    The candidate sets are the ones src.editing_operations.pullingUpEditing and
    pullingDownEditing walk: contractions target 1-in/1-out inner vertices, and
    redundant vertices are inner vertices sharing parents and children with
    another inner vertex. Every applicator rolls back on its own if the move
    does not leave a phylogenetic network, and returns False in that case.
    """
    originalLeaves = leafSet(network)

    def relocate(u, v, newU, newV):
        return lambda N: _tryMoveEdge(N, u, v, newU, newV, None, originalLeaves, False)

    def contract(node):
        return lambda N: _tryContract(N, node, None, originalLeaves, False)

    def drop(node):
        return lambda N: _tryRemoveRedundant(N, node, None, originalLeaves, False)

    applicators = []

    for u, v in network.edges():
        for parentU in network.predecessors(u):
            applicators.append(("pull-up-tail", relocate(u, v, parentU, v)))
        for parentV in network.predecessors(v):
            if parentV != u:
                applicators.append(("pull-up-head", relocate(u, v, u, parentV)))
        for childU in network.successors(u):
            if childU != v:
                applicators.append(("pull-down-tail", relocate(u, v, childU, v)))
        for childV in network.successors(v):
            applicators.append(("pull-down-head", relocate(u, v, u, childV)))

    for node in network.nodes():
        if network.in_degree(node) == 1 and network.out_degree(node) == 1:
            applicators.append(("contract", contract(node)))

    for nodes in redundantVertexGroups(network):
        for node in nodes[1:]:
            applicators.append(("remove-redundant", drop(node)))

    return applicators


def applyCandidateMove(network: nx.DiGraph, apply) -> nx.DiGraph | None:
    """Apply a candidate to a copy; None if it does not leave a network."""
    trial = network.copy()
    if not apply(trial):
        return None
    return trial


def _keepsBmgs(
    network: nx.DiGraph, sigma: dict | None, originStrict, originWeak
) -> tuple[bool, bool]:
    """Whether network has the same strict and the same weak BMG as the origin."""
    strict, weak = networkBmgs(network, sigma=sigma)
    return (
        sameColoredGraph(strict, originStrict),
        sameColoredGraph(weak, originWeak),
    )


def candidateMoves(network: nx.DiGraph) -> list:
    """Every single-move candidate as a (kind, resulting network) pair.

    Applying a candidate is cheap, computing a best match graph is not, so the
    candidate set is always enumerated in full and only the sample that gets
    checked pays for a BMG. That is what lets the survey report an exact total
    even when it checks fewer of them.
    """
    moves = []
    for kind, apply in candidateMoveApplicators(network):
        trial = applyCandidateMove(network, apply)
        if trial is not None:
            moves.append((kind, trial))
    return moves


def _sample(items: list, limit: int | None, seed: int) -> tuple[list, bool]:
    """Cut items down to limit at random, and say whether all of them were kept.

    Taking the first few instead would bias the result: the enumeration follows
    the arcs of the network, so a prefix comes from the first few arcs only.
    """
    if limit is None or len(items) <= limit:
        return items, True
    return random.Random(seed).sample(items, limit), False


def checkSingleMoves(
    network: nx.DiGraph,
    sigma: dict | None = None,
    limit: int | None = None,
    seed: int = 0,
) -> list[MoveResult]:
    """Single-move candidates, with strict and weak BMG verdicts.

    limit=None checks every candidate. A smaller limit draws a random sample of
    that size, so the shares stay unbiased estimates of the whole neighbourhood.
    seed makes that draw reproducible.
    """
    return _surveySingleMoves(network, sigma, limit, seed)[0]


def _surveySingleMoves(
    network: nx.DiGraph, sigma: dict | None, limit: int | None, seed: int
) -> tuple[list[MoveResult], int, bool]:
    """(verdicts for the checked sample, total number of candidates, exhaustive)."""
    originStrict, originWeak = networkBmgs(network, sigma=sigma)
    everyMove = candidateMoves(network)
    drawn, exhaustive = _sample(everyMove, limit, seed)

    results = [
        MoveResult(kind, *_keepsBmgs(trial, sigma, originStrict, originWeak))
        for kind, trial in drawn
    ]
    return results, len(everyMove), exhaustive


def checkMoveCombinations(
    network: nx.DiGraph,
    length: int,
    sigma: dict | None = None,
    limit: int | None = 40,
    seed: int = 0,
) -> list[SequenceResult]:
    """Sequences of 'length' candidates in a row, compared to G(N).

    The BMGs of the final network are checked against those of the starting
    network, not against the intermediate ones. A sequence may therefore break
    a BMG even if each prefix would have been accepted by task 2c.

    Every end network is reported once. Sequences that lead back to the
    starting network are dropped, because undoing a pull is not a modification
    and would otherwise count as a sequence that trivially kept both BMGs.

    limit=None walks the whole neighbourhood, which grows roughly as the number
    of candidates to the power of length. A smaller limit keeps at most that
    many networks per depth, drawn at random; seed makes the draw reproducible.
    """
    return _surveyMoveCombinations(network, length, sigma, limit, seed)[0]


def _expandOnce(
    source: nx.DiGraph, kinds: tuple, keep: int | None, rng: random.Random
) -> tuple[list, bool]:
    """Up to keep networks one candidate move on from source, and whether it cut.

    keep=None applies every candidate. Otherwise the candidates are tried in a
    random order and the walk stops once it has enough, which is what keeps a
    capped survey affordable: the networks here admit a few thousand candidates
    and applying one means copying the graph, while comparing best match graphs
    afterwards is comparatively cheap.
    """
    applicators = candidateMoveApplicators(source)
    if keep is not None:
        rng.shuffle(applicators)

    children = []
    for kind, apply in applicators:
        if keep is not None and len(children) >= keep:
            return children, True
        trial = applyCandidateMove(source, apply)
        if trial is not None:
            children.append((kinds + (kind,), trial))
    return children, False


def _surveyMoveCombinations(
    network: nx.DiGraph,
    length: int,
    sigma: dict | None,
    limit: int | None,
    seed: int,
) -> tuple[list[SequenceResult], bool]:
    """(verdicts, whether the whole neighbourhood was walked).

    The walk goes depth by depth rather than depth-first. Under a limit a
    depth-first walk spends its whole budget inside the first branch it descends
    into, so every sequence it reports shares a prefix and the shares describe
    that one branch instead of the neighbourhood. Going by depth and drawing the
    survivors at random spreads them over the neighbourhood instead.

    The draw is split evenly over the networks of a depth, so each of them
    contributes its share of the next one rather than the first few crowding the
    rest out.
    """
    if length < 2:
        raise ValueError("combinations are sequences of at least two moves")

    rng = random.Random(seed)
    start = frozenset(network.edges())
    exhaustive = True

    layer = [((), network)]
    for depth in range(length):
        perSource = None if limit is None else max(1, math.ceil(limit / len(layer)))

        reached = {}
        for kinds, current in layer:
            children, cut = _expandOnce(current, kinds, perSource, rng)
            exhaustive = exhaustive and not cut
            for childKinds, trial in children:
                # two prefixes reaching the same network have the same future
                reached.setdefault(frozenset(trial.edges()), (childKinds, trial))

        if depth == length - 1:
            # undoing a pull is not a modification, so it is not a sequence
            reached.pop(start, None)

        layer = list(reached.values())
        if limit is not None and len(layer) > limit:
            layer = rng.sample(layer, limit)
            exhaustive = False

    originStrict, originWeak = networkBmgs(network, sigma=sigma)
    results = [
        SequenceResult(kinds, *_keepsBmgs(trial, sigma, originStrict, originWeak))
        for kinds, trial in layer
    ]
    return results, exhaustive


def checkEditingPath(
    network: nx.DiGraph,
    sigma: dict | None = None,
    maxMoves: int | None = 3,
) -> list[SequenceResult]:
    """Task 2d along the greedy path src.editing_operations actually walks.

    checkSingleMoves and checkMoveCombinations enumerate the neighbourhood;
    this follows the one trajectory editingNetwork commits to with its filter
    switched off. Entry k of the result is the network after k+1 moves.

    There is nothing to sample here, since the path is a single trajectory.
    maxMoves only decides how far along it to look.
    """
    return _surveyEditingPath(network, sigma, maxMoves)[0]


def _surveyEditingPath(
    network: nx.DiGraph, sigma: dict | None, maxMoves: int | None
) -> tuple[list[SequenceResult], bool]:
    """(verdicts, whether the path ended on its own rather than at maxMoves)."""
    originStrict, originWeak = networkBmgs(network, sigma=sigma)
    results = []
    exhaustive = True

    for step in iterEditingSteps(network, originWeak, checkBMG=False):
        results.append(
            SequenceResult(
                ("greedy-path",) * (len(results) + 1),
                *_keepsBmgs(step, sigma, originStrict, originWeak),
            )
        )
        if maxMoves is not None and len(results) >= maxMoves:
            exhaustive = False
            break

    return results, exhaustive


def _counts(results: list, total: int, exhaustive: bool) -> MoveCounts:
    return MoveCounts(
        total=total,
        checked=len(results),
        strict=sum(m.strict for m in results),
        weak=sum(m.weak for m in results),
        exhaustive=exhaustive,
    )


def surveyEditMoves(
    network: nx.DiGraph,
    sigma: dict | None = None,
    singleLimit: int | None = None,
    pairLimit: int | None = 500,
    tripleLimit: int | None = 500,
    pathLimit: int | None = None,
    seed: int = 0,
) -> MoveSurvey:
    """Task 2d for one network: singles, pairs, triples and the greedy path.

    Only the two sequence walks are limited by default. They grow roughly as the
    number of candidates to the power of their length, and 500 is enough for
    their shares to settle; whenever a limit bites, the MoveCounts of that row
    says exhaustive=False and its total is only a lower bound.

    The other two rows are exact. Enumerating single moves is cheap, and the
    greedy path is one trajectory that ends by itself. Cutting the path short
    would be worse than sampling a walk, because its first moves are not
    representative of it: on the tree-BMG network of task 2b the first three all
    keep the BMG while only one in nine does over the whole path.
    """
    singles, singleTotal, singlesDone = _surveySingleMoves(
        network, sigma, singleLimit, seed
    )
    pairs, pairsDone = _surveyMoveCombinations(network, 2, sigma, pairLimit, seed)
    triples, triplesDone = _surveyMoveCombinations(
        network, 3, sigma, tripleLimit, seed
    )
    path, pathDone = _surveyEditingPath(network, sigma, pathLimit)

    return MoveSurvey(
        singles=_counts(singles, singleTotal, singlesDone),
        # the sequence walks stop at the limit, so all they know is what they saw
        pairs=_counts(pairs, len(pairs), pairsDone),
        triples=_counts(triples, len(triples), triplesDone),
        path=_counts(path, len(path), pathDone),
    )


def formatMoveSurvey(survey: MoveSurvey) -> str:
    """One line per row. The denominators are what was checked, not what exists.

    So a row whose candidate count is larger than its denominator was sampled,
    and '+' says even that count is only a lower bound.
    """
    rows = (
        ("singles", survey.singles),
        ("pairs", survey.pairs),
        ("triples", survey.triples),
        ("path", survey.path),
    )
    lines = [
        "candidates: edits leaving a phylogenetic network on the same leaf set, "
        "the BMG is not consulted",
        "'+' marks a walk that stopped at its limit, so its total is a lower "
        "bound: the pair and triple rows then hold a random sample of that "
        "depth, the path row its first moves",
    ]
    for label, counts in rows:
        total = counts.total if counts.exhaustive else f"{counts.total}+"
        strictShare = counts.strictShare()
        weakShare = counts.weakShare()
        if strictShare is None:
            lines.append(f"{label:9}candidates={total} nothing to check")
            continue
        lines.append(
            f"{label:9}candidates={total} "
            f"strict={counts.strict}/{counts.checked} ({strictShare:.0%}) "
            f"weak={counts.weak}/{counts.checked} ({weakShare:.0%})"
        )
    return "\n".join(lines)
