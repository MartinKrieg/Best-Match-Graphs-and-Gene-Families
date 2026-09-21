"""
The reticulation number represents the degree to which the graph G differs from a tree (degree-based perspective)
"""

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src import _try_move_edge, cleanUpDummyVertices, removingRedundantVertices
from samples.generate_samples import loadSamples

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
  moves = []
  for u, v in list(N.edges()):
    for parent_u in list(N.predecessors(u)):
      moves.append((u, v, parent_u, v))
    for parent_v in list(N.predecessors(v)):
      if parent_v != u:
        moves.append((u, v, u, parent_v))
  return moves


def bestPullUpCleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set):
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


def anyPullUpCleanup(N: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set,
                      avoid: set):
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
                               maxSteps: int = 2000) -> nx.DiGraph:
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}

  current = cleanup(N.copy(), originBmg, original_leaves)
  visited = {_signature(current)}

  for _ in range(maxSteps):
    improved = bestPullUpCleanup(current, originBmg, original_leaves)
    if improved is not None:
      current = improved
      visited.add(_signature(current))
      continue

    sideways = anyPullUpCleanup(current, originBmg, original_leaves, visited)
    if sideways is None:
      return current  # Everything's been seen

    current = sideways
    visited.add(_signature(current))

  return current


def allPullDownMoves(N: nx.DiGraph) -> list:
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
                                 maxSteps: int = 2000) -> nx.DiGraph:
  """Equivalent to reduceReticulationsPullUp, driven by pull-down moves."""
  original_leaves = {node for node in N.nodes() if N.out_degree(node) == 0}

  current = cleanup(N.copy(), originBmg, original_leaves)
  visited = {_signature(current)}

  for _ in range(maxSteps):
    improved = bestPullDownCleanup(current, originBmg, original_leaves)
    if improved is not None:
      current = improved
      visited.add(_signature(current))
      continue

    sideways = anyPullDownCleanup(current, originBmg, original_leaves, visited)
    if sideways is None:
      return current

    current = sideways
    visited.add(_signature(current))

  return current


class Timeout(Exception):
  pass


def _raiseTimeout(signum, frame):
  raise Timeout()


def runWithTimeout(heuristic, network: nx.DiGraph, bmg: nx.DiGraph, seconds: int = 30):

  signal.signal(signal.SIGALRM, _raiseTimeout)
  signal.alarm(seconds)
  try:
    result = heuristic(network.copy(), bmg)
    note = ""
  except Timeout:
    result, note = network, "TIMEOUT"
  finally:
    signal.alarm(0)
  return result, note


def main():
  samples = loadSamples()

  for name, heuristic in (("Pull-Up", reduceReticulationsPullUp),
                           ("Pull-Down", reduceReticulationsPullDown)):
    print(f"\n=== {name} ===")
    improved = 0
    timeouts = 0
    reductionSum = 0.0

    for instance in samples:
      network = instance["network"]
      bmg = instance["bmg"]
      R0 = score(network) # R0 is the reticulation number of the initial network

      started = time.time()
      result, note = runWithTimeout(heuristic, network, bmg, seconds=30)
      elapsed = time.time() - started

      R = score(result)
      if R0:
        reductionSum += 1 - R / R0
      if R < R0:
        improved += 1
      if note:
        timeouts += 1

      print(f"ns={instance['numSpecies']:<3} seed={instance['seed']:<3} "
            f"R0={R0:<5} R={R:<5} {elapsed:6.1f}s {note}")

    n = len(samples)
    print(f"\n{name}: {improved}/{n} verbessert, {timeouts}/{n} Timeout, "
          f"mittlere R-Reduktion {reductionSum / n:.1%}")


if __name__ == "__main__":
  main()