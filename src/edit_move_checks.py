"""
Task 2d: do the 2c edit operations preserve the (weak) best match graph?

A geometric move is any pull-up, pull-down, dummy contraction or redundant
vertex removal that keeps a DAG on the same leaf set. This module
surveys the geometric candidates themselves and checks both the strict and the
weak BMG against the network before the move, including short sequences.
"""

from typing import NamedTuple

import networkx as nx

from .editing_operations import preserveNetworkLeaves
from .network_best_matches import bestMatchGraphs


class MoveResult(NamedTuple):
    """One geometrically valid candidate and whether it keeps the BMGs."""

    kind: str
    strict: bool
    weak: bool


class MoveSurvey(NamedTuple):
    """Counts for task 2d: singles, then sequences of two and three moves."""

    geometric: int
    singleStrict: int
    singleWeak: int
    pairGeometric: int
    pairStrict: int
    pairWeak: int
    tripleGeometric: int
    tripleStrict: int
    tripleWeak: int

    def everyAcceptedMoveKeepsBothBmgs(self) -> bool:
        """Whether every geometrically valid single move kept both BMGs.

        Vacuous when the network admits no geometric move, which happens on
        already least-resolved trees.
        """
        if self.geometric == 0:
            return True
        return self.singleStrict == self.geometric and self.singleWeak == self.geometric


def leafSet(network: nx.DiGraph) -> set:
    return {v for v in network.nodes() if network.out_degree(v) == 0}


def sameColoredGraph(first: nx.DiGraph, second: nx.DiGraph) -> bool:
    return set(first.nodes()) == set(second.nodes()) and set(first.edges()) == set(
        second.edges()
    )


def networkBmgs(network: nx.DiGraph, sigma: dict | None = None):
    """Strict and weak BMG of a leaf-colored network."""
    return bestMatchGraphs(network, sigma=sigma)


def _isPhylogenetic(network: nx.DiGraph, originalLeaves: set) -> bool:
    """DAG, same leaves, unique root — the 2c operations assume all three."""
    if not nx.is_directed_acyclic_graph(network):
        return False
    if not preserveNetworkLeaves(originalLeaves, network):
        return False
    sources = [v for v in network.nodes() if network.in_degree(v) == 0]
    return len(sources) == 1


def _relocateEdge(network: nx.DiGraph, u, v, newU, newV, originalLeaves: set) -> bool:
    if newU == newV or network.has_edge(newU, newV):
        return False
    network.remove_edge(u, v)
    network.add_edge(newU, newV)
    if not _isPhylogenetic(network, originalLeaves):
        network.remove_edge(newU, newV)
        network.add_edge(u, v)
        return False
    return True


def _contractDummy(network: nx.DiGraph, node, originalLeaves: set) -> bool:
    if network.in_degree(node) != 1 or network.out_degree(node) != 1:
        return False
    parent = next(network.predecessors(node))
    child = next(network.successors(node))
    if network.has_edge(parent, child):
        return False
    network.add_edge(parent, child)
    network.remove_node(node)
    if not _isPhylogenetic(network, originalLeaves):
        network.add_node(node)
        network.add_edge(parent, node)
        network.add_edge(node, child)
        network.remove_edge(parent, child)
        return False
    return True


def _removeRedundant(network: nx.DiGraph, node, originalLeaves: set) -> bool:
    if node not in network or network.out_degree(node) == 0:
        return False
    network.remove_node(node)
    if not _isPhylogenetic(network, originalLeaves):
        return False
    return True


def geometricMoveApplicators(network: nx.DiGraph) -> list:
    """Callables that apply one geometric 2c-move to a copy of the network.

    Pull-up/down follow the same candidate set as src.editing_operations.
    Contractions target 1-in/1-out inner vertices. Redundant vertices are
    inner vertices that share parents and children with another inner vertex.
    """
    originalLeaves = leafSet(network)
    applicators = []

    for u, v in network.edges():
        for parentU in network.predecessors(u):
            applicators.append(
                (
                    "pull-up-tail",
                    lambda N, u=u, v=v, p=parentU: _relocateEdge(
                        N, u, v, p, v, originalLeaves
                    ),
                )
            )
        for parentV in network.predecessors(v):
            if parentV != u:
                applicators.append(
                    (
                        "pull-up-head",
                        lambda N, u=u, v=v, p=parentV: _relocateEdge(
                            N, u, v, u, p, originalLeaves
                        ),
                    )
                )
        for childU in network.successors(u):
            if childU != v:
                applicators.append(
                    (
                        "pull-down-tail",
                        lambda N, u=u, v=v, c=childU: _relocateEdge(
                            N, u, v, c, v, originalLeaves
                        ),
                    )
                )
        for childV in network.successors(v):
            applicators.append(
                (
                    "pull-down-head",
                    lambda N, u=u, v=v, c=childV: _relocateEdge(
                        N, u, v, u, c, originalLeaves
                    ),
                )
            )

    for node in list(network.nodes()):
        if network.in_degree(node) == 1 and network.out_degree(node) == 1:
            applicators.append(
                (
                    "contract",
                    lambda N, node=node: _contractDummy(N, node, originalLeaves),
                )
            )

    signatures = {}
    for node in network.nodes():
        if network.in_degree(node) == 0 or network.out_degree(node) == 0:
            continue
        sig = (
            frozenset(network.predecessors(node)),
            frozenset(network.successors(node)),
        )
        signatures.setdefault(sig, []).append(node)
    for nodes in signatures.values():
        if len(nodes) < 2:
            continue
        for node in nodes[1:]:
            applicators.append(
                (
                    "remove-redundant",
                    lambda N, node=node: _removeRedundant(N, node, originalLeaves),
                )
            )

    return applicators


def applyGeometricMove(network: nx.DiGraph, apply) -> nx.DiGraph | None:
    """Apply a candidate to a copy; None if it is not geometrically valid."""
    trial = network.copy()
    if not apply(trial):
        return None
    return trial


def evaluateMove(network: nx.DiGraph, apply, sigma, originStrict, originWeak):
    """Apply a candidate to a copy and compare both BMGs to the origin."""
    trial = applyGeometricMove(network, apply)
    if trial is None:
        return None
    strict, weak = networkBmgs(trial, sigma=sigma)
    return (
        trial,
        sameColoredGraph(strict, originStrict),
        sameColoredGraph(weak, originWeak),
    )


def checkSingleMoves(
    network: nx.DiGraph, sigma: dict | None = None, limit: int | None = None
) -> list[MoveResult]:
    """Every geometrically valid 2c-move, with strict and weak BMG verdicts."""
    originStrict, originWeak = networkBmgs(network, sigma=sigma)
    results = []
    for kind, apply in geometricMoveApplicators(network):
        if limit is not None and len(results) >= limit:
            break
        outcome = evaluateMove(network, apply, sigma, originStrict, originWeak)
        if outcome is None:
            continue
        _, strict, weak = outcome
        results.append(MoveResult(kind, strict, weak))
    return results


def checkMoveCombinations(
    network: nx.DiGraph,
    length: int,
    sigma: dict | None = None,
    limit: int = 40,
) -> list[tuple[bool, bool]]:
    """Sequences of 'length' geometrically valid moves, compared to G(N).

    The BMGs of the final network are checked against those of the starting
    network, not against the intermediate ones. A sequence may therefore break
    a BMG even if each prefix would have been accepted by task 2c.
    """
    if length < 2:
        raise ValueError("combinations are sequences of at least two moves")
    originStrict, originWeak = networkBmgs(network, sigma=sigma)
    return _walkCombinations(
        network, length, sigma, originStrict, originWeak, limit
    )


def _walkCombinations(
    network, length, sigma, originStrict, originWeak, limit, depth=0
) -> list[tuple[bool, bool]]:
    found = []
    for _, apply in geometricMoveApplicators(network):
        if len(found) >= limit:
            break
        trial = applyGeometricMove(network, apply)
        if trial is None:
            continue
        if depth + 1 == length:
            strict, weak = networkBmgs(trial, sigma=sigma)
            found.append(
                (
                    sameColoredGraph(strict, originStrict),
                    sameColoredGraph(weak, originWeak),
                )
            )
        else:
            found.extend(
                _walkCombinations(
                    trial,
                    length,
                    sigma,
                    originStrict,
                    originWeak,
                    limit - len(found),
                    depth + 1,
                )
            )
    return found


def surveyEditMoves(
    network: nx.DiGraph,
    sigma: dict | None = None,
    singleLimit: int | None = 80,
    pairLimit: int = 30,
    tripleLimit: int = 20,
) -> MoveSurvey:
    """Task 2d for one network: singles, pairs and triples of 2c-moves."""
    singles = checkSingleMoves(network, sigma=sigma, limit=singleLimit)
    pairs = checkMoveCombinations(network, length=2, sigma=sigma, limit=pairLimit)
    triples = checkMoveCombinations(network, length=3, sigma=sigma, limit=tripleLimit)
    return MoveSurvey(
        geometric=len(singles),
        singleStrict=sum(m.strict for m in singles),
        singleWeak=sum(m.weak for m in singles),
        pairGeometric=len(pairs),
        pairStrict=sum(strict for strict, _ in pairs),
        pairWeak=sum(weak for _, weak in pairs),
        tripleGeometric=len(triples),
        tripleStrict=sum(strict for strict, _ in triples),
        tripleWeak=sum(weak for _, weak in triples),
    )


def formatMoveSurvey(survey: MoveSurvey) -> str:
    return (
        f"singles  geometric={survey.geometric} "
        f"strict={survey.singleStrict}/{survey.geometric} "
        f"weak={survey.singleWeak}/{survey.geometric}\n"
        f"pairs    geometric={survey.pairGeometric} "
        f"strict={survey.pairStrict}/{survey.pairGeometric} "
        f"weak={survey.pairWeak}/{survey.pairGeometric}\n"
        f"triples  geometric={survey.tripleGeometric} "
        f"strict={survey.tripleStrict}/{survey.tripleGeometric} "
        f"weak={survey.tripleWeak}/{survey.tripleGeometric}"
    )
