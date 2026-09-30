import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src import _try_move_edge
from src.heuristics.minimize_reticulation_number import allPullCombinedMoves, cleanup, score
from src.heuristics.greedy_twin_distance_v1 import flattenAll

# Follows the same logic as greedy_twin_distance.

def reduceViaRNumber(N: nx.DiGraph, originBmg: nx.DiGraph, maxSteps: int = 200) -> nx.DiGraph:
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}
  current = cleanup(N.copy(), originBmg, original_leaves)
  visited = {frozenset(current.edges())}

  changed = True
  while changed:
    changed = False
    r0 = score(current)

    ranked = []
    for u, v, new_u, new_v in allPullCombinedMoves(current):
      candidate = current.copy()
      if not _try_move_edge(candidate, u, v, new_u, new_v, originBmg, original_leaves):
        continue
      cleanup(candidate, originBmg, original_leaves)
      ranked.append((score(candidate), candidate))
    ranked.sort(key=lambda e: e[0])

    for r, candidate in ranked:
      signature = frozenset(candidate.edges())
      if signature in visited:
        continue
      if r < r0:
        current = candidate
        visited.add(signature)
        changed = True
        break

  flattenAll(current, originBmg, original_leaves)
  return current
