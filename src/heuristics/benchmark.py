"""
Shared benchmark for the task 2e heuristics.

Every heuristic is measured on the same cached instances and with the same
metrics, so that the results of different heuristics can be compared directly.
"""

import signal
import sys
import time
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import networkExplainsBmg  


# Metrics

def reticulationNumber(network: nx.DiGraph) -> int:
    """R = sum of the excess in-degrees, zero exactly for trees."""
    return sum(max(0, network.in_degree(v) - 1) for v in network.nodes())


def isTree(network: nx.DiGraph) -> bool:
    return reticulationNumber(network) == 0


def leafSet(network: nx.DiGraph) -> set:
    return {v for v in network.nodes() if network.out_degree(v) == 0}


def clusters(network: nx.DiGraph) -> set:
    """The leaf sets below the vertices, usable as a topology identity."""
    return {
        frozenset({d for d in nx.descendants(network, v) | {v}
                   if network.out_degree(d) == 0})
        for v in network.nodes()
    }


def preservesBmg(network: nx.DiGraph, bmg: nx.DiGraph) -> bool:
    return networkExplainsBmg(network, bmg)


def sameClusters(network: nx.DiGraph, target: nx.DiGraph) -> bool:
    """
    Whether the network has the cluster system of T*.
    """
    return (leafSet(network) == leafSet(target)
            and clusters(network) == clusters(target))


def isPhylogenetic(network: nx.DiGraph) -> bool:
    """
    Whether no vertex has exactly one child.
    """
    return all(network.out_degree(v) != 1 for v in network.nodes())


def hasSingleRoot(network: nx.DiGraph) -> bool:
    """
    Whether exactly one vertex has in-degree 0.
    """
    return sum(1 for v in network.nodes() if network.in_degree(v) == 0) == 1


def equalsTarget(network: nx.DiGraph, target: nx.DiGraph) -> bool:
    """Whether the network *is* T*: a phylogenetic tree with T*'s clusters."""
    return (isTree(network) and isPhylogenetic(network)
            and hasSingleRoot(network) and sameClusters(network, target))


def metrics(network: nx.DiGraph, bmg: nx.DiGraph, target: nx.DiGraph) -> dict:
    return {
        "R": reticulationNumber(network),
        "V": network.number_of_nodes(),
        "E": network.number_of_edges(),
        "isTree": isTree(network),
        "bmgOk": preservesBmg(network, bmg),
        "clustersOk": sameClusters(network, target),
        "phylo": isPhylogenetic(network),
        "root1": hasSingleRoot(network),
        "isTarget": equalsTarget(network, target),
    }


# Runner

class _Timeout(Exception):
    pass


def _raise(signum, frame):
    raise _Timeout()


def run(heuristic, name: str, instances: list, timeout: int = 120,
        verbose: bool = True) -> list:
    """
    Run a heuristic over the given instances and print one row per instance.

    'solved' means: the result is a tree, it still explains the BMG, and it is
    T*. Anything else is a partial result and the R column says how far it got.
    """
    rows = []

    if verbose:
        print(f"\n=== {name} ===")
        print(f"{'ns/seed':<10} {'|L|':<5} {'R0':<6} {'R':<6} {'|V|':<6} "
              f"{'tree':<6} {'bmg':<6} {'root1':<6} {'clust':<6} {'=T*':<6} {'sec':<7} note")

    for instance in instances:
        network = instance["network"].copy()
        bmg = instance["bmg"]
        target = instance["target"]
        startR = reticulationNumber(network)
        note = ""
        started = time.time()

        signal.signal(signal.SIGALRM, _raise)
        signal.alarm(timeout)
        try:
            result = heuristic(network, bmg, target)
        except _Timeout:
            result, note = network, "TIMEOUT"
        except Exception as error:
            result, note = network, f"ERROR {type(error).__name__}: {error}"
        finally:
            signal.alarm(0)

        elapsed = time.time() - started
        if result is None:
            result, note = network, note or "returned None"

        row = metrics(result, bmg, target)
        row.update({
            "instance": f"{instance['numSpecies']}/{instance['seed']}",
            "leaves": len(leafSet(instance["network"])),
            "R0": startR,
            "seconds": elapsed,
            "note": note,
            "solved": row["isTarget"] and row["bmgOk"],
            "valid": row["bmgOk"] and row["root1"],
        })
        rows.append(row)

        if verbose:
            print(f"{row['instance']:<10} {row['leaves']:<5} {row['R0']:<6} "
                  f"{row['R']:<6} {row['V']:<6} {str(row['isTree']):<6} "
                  f"{str(row['bmgOk']):<6} {str(row['root1']):<6} {str(row['clustersOk']):<6} "
                  f"{str(row['isTarget']):<6} {elapsed:<7.1f} {note}")

    if verbose:
        solved = sum(1 for row in rows if row["solved"])
        trees = sum(1 for row in rows if row["isTree"] and row["bmgOk"])
        broken = sum(1 for row in rows if not row["bmgOk"])
        split = sum(1 for row in rows if not row["root1"])
        reduction = [1 - row["R"] / row["R0"] for row in rows if row["R0"]]
        rightTopology = sum(1 for row in rows if row["clustersOk"] and row["bmgOk"])
        print(f"\nsolved (=T*): {solved}/{len(rows)}   "
              f"tree+bmgOk: {trees}/{len(rows)}   "
              f"clusters of T* reached: {rightTopology}/{len(rows)}   "
              f"BMG broken: {broken}   split into 2 roots: {split}   "
              f"mean R reduction: {sum(reduction) / len(reduction):.1%}")
    return rows
