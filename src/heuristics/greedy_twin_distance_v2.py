"""
Eager variant of reduceViaTwinDistance, i.e., whenever the distance of a pair was decreased, re-compute the set of distances
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src import removingRedundantVertices
from src.heuristics.greedy_twin_distance_v1 import (
    allTwinDistances,
    bestPairMove,
    flattenAll,
    twinDistance,
)


def onePairStep(N: nx.DiGraph, first, second, originBmg: nx.DiGraph, original_leaves: set) -> nx.DiGraph:
  current = N.copy()
  if twinDistance(current, first, second) == 0:
    removingRedundantVertices(current, originBmg, original_leaves)
    return current

  improved = bestPairMove(current, first, second, originBmg, original_leaves)
  return improved if improved is not None else current


def reduceViaTwinDistanceEager(N: nx.DiGraph, originBmg: nx.DiGraph, maxSteps: int = 200) -> nx.DiGraph:
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}
  current = N.copy()  # Currently best network
  visited = {frozenset(current.edges())}

  changed = True
  steps = 0
  while changed and steps < maxSteps:
    changed = False
    distances = allTwinDistances(current)
    for (first, second), distance in sorted(distances.items(), key=lambda kv: kv[1]):
      new_network = onePairStep(current, first, second, originBmg, original_leaves)
      signature = frozenset(new_network.edges())
      if signature in visited:
        continue  # Prevent hopping between two already seen states

      if new_network.number_of_nodes() < current.number_of_nodes() or \
         twinDistance(new_network, first, second) < distance:
        current = new_network
        visited.add(signature)
        changed = True
        steps += 1
        break  # After each pull, the distances are stale -> Recompute

  current.graph["hitMaxSteps"] = changed  # loop stopped on the steps cap, not a dead end

  # Flatten (Tree to LRT!) when no more changes happens
  flattenAll(current, originBmg, original_leaves)
  return current
