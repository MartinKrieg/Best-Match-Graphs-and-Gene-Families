import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src import _try_contract, _try_move_edge, removingRedundantVertices


def twinDistance(N: nx.DiGraph, first, second) -> int:
  parents = set(N.predecessors(first)) ^ set(N.predecessors(second))
  children = set(N.successors(first)) ^ set(N.successors(second))
  return len(parents) + len(children)


def internalVertices(N: nx.DiGraph) -> list:
  return [v for v in N.nodes() if N.in_degree(v) > 0 and N.out_degree(v) > 0]


def allTwinDistances(N: nx.DiGraph) -> dict:
  """
  Calculates the twin-distances between all pairs of vertices
  """
  vertices = internalVertices(N)
  distances = {}
  for i, first in enumerate(vertices):
    for second in vertices[i + 1:]:
      distances[(first, second)] = twinDistance(N, first, second)
  return distances


def pairFocus(N: nx.DiGraph, first, second) -> set:
  focus = {first, second}
  for v in (first, second):
    focus |= set(N.predecessors(v))
    focus |= set(N.successors(v))
  return focus


def pairMoves(N: nx.DiGraph, first, second) -> list:
  """
  The 4 pull variants (tail up, tail down, head up, head down) plus
  contraction, restricted to the pair's neighbourhood (focus = both vertices
  plus their immediate parents and children). Head-down is only proposed
  when the head has another parent left afterwards, otherwise it could strip
  a vertex down to in-degree 0 and split the network into two roots.
  """
  focus = pairFocus(N, first, second)
  edges = {edge for w in focus
           for edge in list(N.in_edges(w)) + list(N.out_edges(w))}

  moves = []
  for tail, head in edges:
    variants = []
    variants.extend((parent, head) for parent in N.predecessors(tail))
    variants.extend((child, head) for child in N.successors(tail) if child != head)
    variants.extend((tail, parent) for parent in N.predecessors(head) if parent != tail)
    if N.in_degree(head) > 1:
      variants.extend((tail, child) for child in N.successors(head))

    for newTail, newHead in variants:
      if newTail == newHead or N.has_edge(newTail, newHead):
        continue
      if first not in (tail, head, newTail, newHead) and \
         second not in (tail, head, newTail, newHead):
        continue
      moves.append(("pull", tail, head, newTail, newHead))

  for node in focus - {first, second}:
    if N.in_degree(node) == 1 and N.out_degree(node) == 1:
      moves.append(("contract", node))

  return moves


def applyMove(N: nx.DiGraph, move, originBmg: nx.DiGraph, original_leaves: set) -> bool:
  if move[0] == "pull":
    _, tail, head, newTail, newHead = move
    return _try_move_edge(N, tail, head, newTail, newHead, originBmg, original_leaves)
  _, node = move
  return _try_contract(N, node, originBmg, original_leaves)


def bestPairMove(N: nx.DiGraph, first, second, originBmg: nx.DiGraph, original_leaves: set):
  """
  Tries every local move for (first, second) on a copy of N and returns
  whichever result lowers twinDistance(first, second) the most - only if
  strictly lower than the current distance. None if nothing improves.
  """
  baseline = twinDistance(N, first, second)
  best_network = None
  best_distance = baseline

  for move in pairMoves(N, first, second):
    candidate = N.copy()
    if not applyMove(candidate, move, originBmg, original_leaves):
      continue

    distance = twinDistance(candidate, first, second)
    if distance < best_distance:
      best_distance = distance
      best_network = candidate

  return best_network


def closePair(N: nx.DiGraph, first, second, originBmg: nx.DiGraph, original_leaves: set,
              maxSteps: int = 200) -> nx.DiGraph:
  """
  Repeatedly applies bestPairMove for (first, second) until the distance
  reaches 0 - i.e. local optimization. Afterwards the removal is done.
  """
  current = N.copy()

  for _ in range(maxSteps):
    if twinDistance(current, first, second) == 0:
      removingRedundantVertices(current, originBmg, original_leaves)
      return current

    improved = bestPairMove(current, first, second, originBmg, original_leaves)
    if improved is None:
      return current
    current = improved

  return current


def _restore(N: nx.DiGraph, backup: nx.DiGraph) -> None:
  N.clear()
  N.add_nodes_from(backup.nodes(data=True))
  N.add_edges_from(backup.edges(data=True))


def tryFlatten(N: nx.DiGraph, node, originBmg: nx.DiGraph, original_leaves: set) -> bool:
  if N.in_degree(node) != 1 or N.out_degree(node) < 1:
    return False
  # out_degree == 1 is plain contraction: the loop below pulls zero children

  parent = next(N.predecessors(node))
  backup = N.copy()

  for child in list(N.successors(node))[:-1]:
    if not _try_move_edge(N, node, child, parent, child, originBmg, original_leaves):
      _restore(N, backup)
      return False

  if not _try_contract(N, node, originBmg, original_leaves):
    _restore(N, backup)
    return False

  return True


def flattenAll(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set) -> bool:
  """Flattens every eligible node to a fixpoint. Returns whether anything changed."""
  changed = False
  progress = True
  while progress:
    progress = False
    for node in list(N.nodes()):
      if tryFlatten(N, node, originBmg, original_leaves):
        changed = progress = True
  return changed


def reduceViaTwinDistance(N: nx.DiGraph, originBmg: nx.DiGraph, maxSteps: int = 200) -> nx.DiGraph:
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}
  current = N.copy() # Currently best network
  visited = {frozenset(current.edges())}

  changed = True
  while changed:
    changed = False
    distances = allTwinDistances(current)
    for (first, second), distance in sorted(distances.items(), key=lambda kv: kv[1]):
      new_network = closePair(current, first, second, originBmg, original_leaves, maxSteps)
      signature = frozenset(new_network.edges())
      if signature in visited:
        continue  # Prevent hopping between two already seen states

      if new_network.number_of_nodes() < current.number_of_nodes() or \
         twinDistance(new_network, first, second) < distance:
        current = new_network
        visited.add(signature)
        changed = True
        break # After each pull, the distances are stale -> Recompute

  # Flatten (Tree to LRT!) when no more changes happens
  flattenAll(current, originBmg, original_leaves)
  return current
