import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src import removingRedundantVertices
from src.heuristics.greedy_twin_distance_v1 import (
    allTwinDistances,
    applyMove,
    flattenAll,
    pairMoves,
    twinDistance,
)


def _signature(N: nx.DiGraph) -> frozenset:
  return frozenset(N.edges())


class Budget:

  def __init__(self, maxMoveChecks: int):
    self.remaining = maxMoveChecks
    self.lastReason = ""

  def exhausted(self) -> bool:
    return self.remaining <= 0

  def spend(self, n: int = 1) -> None:
    self.remaining -= n


def closePairVariants(N: nx.DiGraph, first, second, originBmg: nx.DiGraph, original_leaves: set,
                       budget: Budget, seen: set, maxSteps: int = 24,
                       plateauBudget: int = 2, branching: int = 6):
  yield from _descend(N, first, second, originBmg, original_leaves, budget, seen,
                       0, 0, maxSteps, plateauBudget, branching)


def _descend(N: nx.DiGraph, first, second, originBmg: nx.DiGraph, original_leaves: set,
             budget: Budget, seen: set, depth: int, plateau: int,
             maxSteps: int, plateauBudget: int, branching: int):
  """Calculates moves that actually decrease the distance between the pair (first, second)"""
  distance = twinDistance(N, first, second)
  if distance == 0:
    trial_network = N.copy()
    budget.spend()
    removingRedundantVertices(trial_network, originBmg, original_leaves)
    yield trial_network
    return

  if depth >= maxSteps:
    budget.lastReason = "maxSteps reached"
    return
  if budget.exhausted():
    budget.lastReason = "budget exhausted"
    return

  ranked = []
  for move in pairMoves(N, first, second):
    trial_network = N.copy()
    if not applyMove(trial_network, move, originBmg, original_leaves): # Continue if move would violate something
      continue
    candidate_distance = twinDistance(trial_network, first, second)
    if candidate_distance > distance:
      continue
    if candidate_distance == distance and plateau >= plateauBudget:
      continue
    ranked.append((candidate_distance, move, trial_network))
  ranked.sort(key=lambda e: (e[0], str(e[1])))

  if not ranked:
    budget.lastReason = "no improving move exists"
    return

  explored = 0
  for candidate_distance, move, trial_network in ranked:
    if explored >= branching:
      budget.lastReason = "branching exhausted"
      return
    if budget.exhausted():
      budget.lastReason = "budget exhausted"
      return
    signature = _signature(trial_network)
    if signature in seen:
      continue

    explored += 1
    seen.add(signature)
    yield from _descend(trial_network, first, second, originBmg, original_leaves, budget, seen,
                         depth + 1, plateau + 1 if candidate_distance == distance else 0,
                         maxSteps, plateauBudget, branching)


def _quality(N: nx.DiGraph) -> tuple:
  R = sum(max(0, N.in_degree(v) - 1) for v in N.nodes())
  return (R, N.number_of_nodes())


def _search(work: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set, budget: Budget,
            visited: set, maxPairAttempts: int, mergeVariants: int,
            maxSteps: int, plateauBudget: int, branching: int) -> nx.DiGraph:
  """
  Short explanation of what happens here:
  - Init of network
  - Return if quality (r(N)) equals 0 (important for recursion)
  - Calculating the pair-wise twin-distances and sorting them distance ascending
  - Iterate over distance-pairs (lowest distance first)
    - For the current distance-pair, it is tried to get this to distance 0
      - If not successful, the next distance-pair is considered
      - If successful, each variant (if multiple) is then used for the recursive call on _search.

  o Budget = Global variable contraining the maximum merges (removals), controls the total number of successful merges, s.t. at some point, no more can be started as budget is exhausted.
  o MaxSteps = Determines the maximum moves a pair (first, second) can use to reach distance 0
  o Branching = Top-k best movements for a pair (first, second) that are dived into.
  o Plateau(budgest) = When reached, no more non-improving moves are allowed until another distance-decreasing move was taken again.
  """
  best = work
  if _quality(work)[0] == 0:
    return work

  distances = allTwinDistances(work)
  ranked_pairs = sorted(distances.items(), key=lambda kv: kv[1])

  if not ranked_pairs:
    budget.lastReason = "no candidate pairs left"
    return best

  for (first, second), _distance in ranked_pairs[:maxPairAttempts]:
    if budget.exhausted():
      budget.lastReason = "budget exhausted"
      break

    taken = 0
    for merged in closePairVariants(work, first, second, originBmg, original_leaves,
                                     budget, set(visited), maxSteps, plateauBudget, branching):
      if taken >= mergeVariants:
        budget.lastReason = "mergeVariants exhausted"
        break
      if budget.exhausted():
        budget.lastReason = "budget exhausted"
        break
      taken += 1

      signature = _signature(merged)
      if signature in visited:
        continue
      visited.add(signature)

      found = _search(merged, originBmg, original_leaves, budget, visited,
                       maxPairAttempts, mergeVariants, maxSteps, plateauBudget, branching)
      if _quality(found) < _quality(best):
        best = found
      if _quality(best)[0] == 0:
        return best

  return best


def reduceViaBacktrackingTwinDistance(N: nx.DiGraph, originBmg: nx.DiGraph, target=None,
                                       maxMoveChecks: int = 20000, maxPairAttempts: int = 6,
                                       mergeVariants: int = 3, maxSteps: int = 24,
                                       plateauBudget: int = 2, branching: int = 5,
                                       budget: Budget = None) -> nx.DiGraph:
  """Entry point."""
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}
  current = N.copy()
  if budget is None:
    budget = Budget(maxMoveChecks)
  visited = {_signature(current)}

  result = _search(current, originBmg, original_leaves, budget, visited,
                    maxPairAttempts, mergeVariants, maxSteps, plateauBudget, branching)

  # Transform Tree to LRT
  flattenAll(result, originBmg, original_leaves)
  return result
