"""
The reticulation number represents the degree to which the graph G differs from a tree (degree-based perspective)
Here, the Pull-Only approaches are defined.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src import _try_move_edge, cleanUpDummyVertices, removingRedundantVertices
from samples.generate_bucketed_samples import loadBucket

def score(N: nx.DiGraph):
  return sum(max(0, N.in_degree(v) - 1) for v in N.nodes)

def cleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set) -> nx.DiGraph:

  while True:
    changed_contract = False
    while cleanUpDummyVertices(N, originBmg, original_leaves):
      changed_contract = True

    changed_merge = False
    while removingRedundantVertices(N, originBmg, original_leaves):
      changed_merge = True

    if not changed_contract and not changed_merge:
      break;

  return N


def allPullUpMoves(N: nx.DiGraph) -> list:
  """Defines all pull up moves."""
  moves = []
  for u, v in list(N.edges()):
    for parent_u in list(N.predecessors(u)):
      moves.append((u, v, parent_u, v))
    for parent_v in list(N.predecessors(v)):
      if parent_v != u:
        moves.append((u, v, u, parent_v))
  return moves


def bestPullUpCleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set):
  """Tries all pull-up moves + cleanup (contraction and removal) and return the network with the best move applied."""
  best_network = None
  best_r = score(N)

  for u, v, new_u, new_v in allPullUpMoves(N):
    candidate = N.copy()
    if not _try_move_edge(candidate, u, v, new_u, new_v, originBmg, original_leaves):
      continue

    cleanup(candidate, originBmg, original_leaves)
    r = score(candidate)
    if r < best_r:
      best_r = r
      best_network = candidate

  return best_network


def _signature(N: nx.DiGraph) -> frozenset:
  return frozenset(N.edges())


def anyPullUpCleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set, avoid: set):
  """Selects an arbitrary pull-up move that was not already tried; fallback mechanism."""
  for u, v, new_u, new_v in allPullUpMoves(N):
    candidate = N.copy()
    if not _try_move_edge(candidate, u, v, new_u, new_v, originBmg, original_leaves):
      continue

    cleanup(candidate, originBmg, original_leaves)
    if _signature(candidate) in avoid:
      continue
    return candidate
  return None


def reduceReticulationsPullUp(N: nx.DiGraph, originBmg: nx.DiGraph,
                               maxSteps: int = 200) -> nx.DiGraph:
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}

  current = cleanup(N.copy(), originBmg, original_leaves)
  visited = {_signature(current)}
  current.graph["hitMaxSteps"] = False

  for _ in range(maxSteps):
    improved = bestPullUpCleanup(current, originBmg, original_leaves)
    if improved is not None:
      current = improved
      visited.add(_signature(current))
      continue

    sideways = anyPullUpCleanup(current, originBmg, original_leaves, visited)
    if sideways is None:
      current.graph["hitMaxSteps"] = False
      return current  # Everything's been seen

    current = sideways
    visited.add(_signature(current))

  current.graph["hitMaxSteps"] = True  # ran out of steps, not a dead end
  return current


def allPullDownMoves(N: nx.DiGraph) -> list:
  """Defines all pull-down moves, similar to pull up."""
  moves = []
  for u, v in list(N.edges()):
    for child_u in list(N.successors(u)):
      if child_u != v:
        moves.append((u, v, child_u, v))
    if N.in_degree(v) > 1:
      for child_v in list(N.successors(v)):
        moves.append((u, v, u, child_v))
  return moves


def bestPullDownCleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set):
  """Equivalent to bestPullUpCleanup, over allPullDownMoves instead."""
  best_network = None
  best_r = score(N)

  for u, v, new_u, new_v in allPullDownMoves(N):
    candidate = N.copy()
    if not _try_move_edge(candidate, u, v, new_u, new_v, originBmg, original_leaves):
      continue

    cleanup(candidate, originBmg, original_leaves)
    r = score(candidate)
    if r < best_r:
      best_r = r
      best_network = candidate

  return best_network


def anyPullDownCleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set,
                        avoid: set):
  """Equivalent to anyPullUpCleanup, over allPullDownMoves instead."""
  for u, v, new_u, new_v in allPullDownMoves(N):
    candidate = N.copy()
    if not _try_move_edge(candidate, u, v, new_u, new_v, originBmg, original_leaves):
      continue

    cleanup(candidate, originBmg, original_leaves)
    if _signature(candidate) in avoid:
      continue
    return candidate
  return None


def reduceReticulationsPullDown(N: nx.DiGraph, originBmg: nx.DiGraph,
                                 maxSteps: int = 200) -> nx.DiGraph:
  """Equivalent to reduceReticulationsPullUp, driven by pull-down moves."""
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}

  current = cleanup(N.copy(), originBmg, original_leaves)
  visited = {_signature(current)}
  current.graph["hitMaxSteps"] = False

  for _ in range(maxSteps):
    improved = bestPullDownCleanup(current, originBmg, original_leaves)
    if improved is not None:
      current = improved
      visited.add(_signature(current))
      continue

    sideways = anyPullDownCleanup(current, originBmg, original_leaves, visited)
    if sideways is None:
      current.graph["hitMaxSteps"] = False
      return current

    current = sideways
    visited.add(_signature(current))

  current.graph["hitMaxSteps"] = True  # ran out of steps, not a dead end
  return current


def allPullCombinedMoves(N: nx.DiGraph) -> list: 
  """Returns both types of moves for greedy approach w/o exit"""
  return allPullUpMoves(N) + allPullDownMoves(N)


def bestPullCombinedCleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set):
  """Decides on both types on moves, equals the greedy approach without exit when no good solution was found."""
  best_network = None
  best_r = score(N)

  for u, v, new_u, new_v in allPullCombinedMoves(N):
    candidate = N.copy()
    if not _try_move_edge(candidate, u, v, new_u, new_v, originBmg, original_leaves):
      continue

    cleanup(candidate, originBmg, original_leaves)
    r = score(candidate)
    if r < best_r:
      best_r = r
      best_network = candidate

  return best_network


def anyPullCombinedCleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set, avoid: set):
  """Fallback for pullcombined, when no move reduced r."""
  for u, v, new_u, new_v in allPullCombinedMoves(N):
    candidate = N.copy()
    if not _try_move_edge(candidate, u, v, new_u, new_v, originBmg, original_leaves):
      continue

    cleanup(candidate, originBmg, original_leaves)
    if _signature(candidate) in avoid:
      continue
    return candidate
  return None


def reduceReticulationsPullCombined(N: nx.DiGraph, originBmg: nx.DiGraph, maxSteps: int = 200) -> nx.DiGraph:
  """Greedy approach without exit. Takes all possible pull up and down moves and selects the best. Falls back when no best candidate is found (i.e., when no move reduces R)"""
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}

  current = cleanup(N.copy(), originBmg, original_leaves)
  visited = {_signature(current)}
  current.graph["hitMaxSteps"] = False

  for _ in range(maxSteps):
    improved = bestPullCombinedCleanup(current, originBmg, original_leaves)
    if improved is not None:
      current = improved
      visited.add(_signature(current))
      continue

    sideways = anyPullCombinedCleanup(current, originBmg, original_leaves, visited)
    if sideways is None:
      current.graph["hitMaxSteps"] = False
      return current

    current = sideways
    visited.add(_signature(current))

  current.graph["hitMaxSteps"] = True
  return current


def reduceReticulationsPullAlternating(N: nx.DiGraph, originBmg: nx.DiGraph, maxSteps: int = 200) -> nx.DiGraph:
  """Alternates the pull-strategy (up/down)"""
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}

  current = cleanup(N.copy(), originBmg, original_leaves)
  visited = {_signature(current)}
  current.graph["hitMaxSteps"] = False

  directions = [
      (bestPullUpCleanup, anyPullUpCleanup),
      (bestPullDownCleanup, anyPullDownCleanup),
  ]

  for step in range(maxSteps):
    scheduled = step % 2
    order = (directions[scheduled], directions[1 - scheduled])

    moved = False
    for bestFn, anyFn in order:
      improved = bestFn(current, originBmg, original_leaves)
      if improved is not None:
        current = improved
        visited.add(_signature(current))
        moved = True
        break

      sideways = anyFn(current, originBmg, original_leaves, visited)
      if sideways is not None:
        current = sideways
        visited.add(_signature(current))
        moved = True
        break

    if not moved:
      current.graph["hitMaxSteps"] = False
      return current

  current.graph["hitMaxSteps"] = True
  return current


def main():
  maxSteps = 500

  for variant in ("variantA", "variantB"):
    samples = loadBucket(name=f"samples_R0_1_20_{variant}.pkl")

    for name, heuristic in (
      ("Pull-Up", reduceReticulationsPullUp),
      ("Pull-Down", reduceReticulationsPullDown),
      ("Pull-Combined", reduceReticulationsPullCombined),
      ("Pull-Alternating", reduceReticulationsPullAlternating)
      ):

      print(f"\n=== {variant} / {name} ===")
      improved = 0
      trees = 0
      maxStepsHit = 0
      reductionSum = 0.0

      for instance in samples:
        network = instance["network"]
        bmg = instance["bmg"]
        R0 = score(network) # R0 is the reticulation number of the initial network

        started = time.time()
        result = heuristic(network.copy(), bmg, maxSteps=maxSteps)
        elapsed = time.time() - started

        hitMaxSteps = result.graph.get("hitMaxSteps", False)

        R = score(result)
        if R0:
          reductionSum += 1 - R / R0
        if R < R0:
          improved += 1
        if R == 0:
          trees += 1
        if hitMaxSteps:
          maxStepsHit += 1

        note = "MAXSTEPS" if hitMaxSteps else ""
        print(f"number of species={instance['numSpecies']:<3} seed={instance['seed']:<3} "
              f"R0={R0:<5} R={R:<5} {elapsed:6.1f}s {note}")

      n = len(samples)
      print(f"\n{variant} / {name}: {improved}/{n} improved, {trees}/{n} reached a tree (R=0), "
            f"{maxStepsHit}/{n} exited due to maxsteps constraint, "
            f"mean R-reduction {reductionSum / n:.1%}")


if __name__ == "__main__":
  main()