import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src import _try_move_edge
from src.heuristics.minimize_reticulation_number import allPullCombinedMoves, cleanup, score
from src.heuristics.greedy_twin_distance_v1 import flattenAll


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


def _quality(N: nx.DiGraph) -> tuple:
  return (score(N), N.number_of_nodes())


def _rankedCandidates(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set,
                       seen: set, r0: int, plateau: int, plateauBudget: int) -> list:
  """
  Every candidate pull move, cleaned up and scored, that does not make r(N) worse.
  Sideways (r == r0) moves are allowed up to plateauBudget in a row, mirroring backtracking_twin_distance.py's _descend."""
  ranked = []
  for u, v, new_u, new_v in allPullCombinedMoves(N):
    candidate = N.copy()
    if not _try_move_edge(candidate, u, v, new_u, new_v, originBmg, original_leaves):
      continue
    cleanup(candidate, originBmg, original_leaves)

    r = score(candidate)
    if r > r0:
      continue
    if r == r0 and plateau >= plateauBudget: # Plateau initial set to 2, see main
      continue

    signature = _signature(candidate)
    if signature in seen:
      continue
    ranked.append((r, signature, candidate))

  ranked.sort(key=lambda e: (e[0], str(e[1])))
  return ranked


def _search(work: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set,
            budget: Budget, seen: set, depth: int, plateau: int,
            maxDepth: int, plateauBudget: int, branching: int) -> nx.DiGraph:
  """
  Short explanations of what happens here:
  - the passed network is 1) initialized as best finding and r0 is the reticulation number of this network.
  - if r0 == 0, then the path is found that realizes a tree
  - Candidates being ranked with regards to their reduction of r(N). Plateaus (= No change in r(N)) are allowed "plateauBudget" times
  - When a path reaches r(N) = 0, exit and return
  """
  best = work
  r0 = score(work)
  if r0 == 0:
    return work
  if depth >= maxDepth:
    budget.lastReason = "maxDepth reached"
    return best
  if budget.exhausted():
    budget.lastReason = "budget exhausted"
    return best

  ranked = _rankedCandidates(work, originBmg, original_leaves, seen, r0, plateau, plateauBudget)

  if not ranked:
    budget.lastReason = "no improving move exists"
    return best

  explored = 0
  for r, signature, candidate in ranked:
    if explored >= branching:
      budget.lastReason = "branching exhausted"
      break
    if budget.exhausted():
      budget.lastReason = "budget exhausted"
      break
    explored += 1
    seen.add(signature)
    budget.spend()

    found = _search(candidate, originBmg, original_leaves, budget, seen,
                     depth + 1, plateau + 1 if r == r0 else 0,
                     maxDepth, plateauBudget, branching)
    if _quality(found) < _quality(best):
      best = found
    if score(best) == 0:
      return best

  return best


def reduceViaBacktrackingRNumber(N: nx.DiGraph, originBmg: nx.DiGraph, target=None,
                                  maxMoveChecks: int = 20000, maxDepth: int = 24,
                                  plateauBudget: int = 2, branching: int = 5,
                                  budget: Budget = None) -> nx.DiGraph:
  """Entry point."""
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}
  current = cleanup(N.copy(), originBmg, original_leaves)
  if budget is None:
    budget = Budget(maxMoveChecks)
  seen = {_signature(current)}

  result = _search(current, originBmg, original_leaves, budget, seen,
                    0, 0, maxDepth, plateauBudget, branching)

  flattenAll(result, originBmg, original_leaves)
  return result
