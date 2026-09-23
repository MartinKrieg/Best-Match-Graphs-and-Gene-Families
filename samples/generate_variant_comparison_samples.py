"""
Builds two parallel sets of five R0-bucketed sample sets (100 instances
each), one per BIC-cherry+expansion variant, to compare whether the choice
of starting-network construction (task 2b's variant (a) vs (b)) affects
downstream editing-heuristic difficulty (tasks 2e/2f).
"""

import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from src import (
    buildBaseBigCherry,
    extendBicCherryNetwork,
    extendBicCherryNetworkEdgeRestricted,
    getLeaves,
    leastResolvedTreeFromBmg,
    networkExplainsBmg,
    toNetwork,
)
from utils import generateGeneTree, generateTreeBmg

HERE = Path(__file__).resolve().parent
TARGET_PER_BUCKET = 100
MAX_NODES = 400
SPECIES_RANGE = range(3, 12)

RANGES = [
    (1, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 100),
]

VARIANTS = {
    "variantA": extendBicCherryNetwork,
    "variantB": extendBicCherryNetworkEdgeRestricted,
}


def reticulationNumber(network: nx.DiGraph) -> int:
    return sum(max(0, network.in_degree(v) - 1) for v in network.nodes())


def bucketNameFor(R0: int, variantLabel: str):
    for lo, hi in RANGES:
        if lo <= R0 < hi:
            return f"samples_R0_{lo}_{hi}_{variantLabel}.pkl"
    return None


def _sharedDraw(numSpecies: int, seed: int):
    """The part that is identical for both variants: gene tree, BMG, target
    and the base BIC-cherry network. Built once per draw and reused."""
    random.seed(seed)
    np.random.seed(seed)

    _, geneTree, _, geneColors = generateGeneTree(numSpecies)
    bmg = generateTreeBmg(geneTree)

    palette = {}
    for key, value in geneColors.items():
        if hasattr(value, "tolist"):
            value = tuple(value.tolist())
        elif isinstance(value, list):
            value = tuple(value)
        palette[key] = value
    reconc = {leaf.label: leaf.reconc for leaf in geneTree.leaves()}

    target, _ = toNetwork(leastResolvedTreeFromBmg(bmg))
    base, _, parents = buildBaseBigCherry(bmg, palette)

    return geneTree, bmg, palette, reconc, target, base, parents


def _instanceForVariant(extend, geneTree, bmg, palette, reconc, target, base, parents,
                         numSpecies, seed):
    network = extend(base, parents, bmg, palette)
    targetCopy = target.copy()
    for graph in (network, targetCopy):
        for v in graph.nodes():
            if graph.out_degree(v) == 0:
                graph.nodes[v]["reconc"] = reconc[v]

    return {
        "startExplains": networkExplainsBmg(network, bmg),
        "numSpecies": numSpecies,
        "seed": seed,
        "geneTree": geneTree,
        "network": network,
        "bmg": bmg,
        "target": targetCopy,
        "colors": palette,
        "reconc": reconc,
        "R0": reticulationNumber(network),
    }


def buildBucketedSamples(perBucket: int = TARGET_PER_BUCKET) -> dict:
    """Returns {variantLabel: {bucketFileName: [instances]}}."""
    collected = {
        label: {f"samples_R0_{lo}_{hi}_{label}.pkl": [] for lo, hi in RANGES}
        for label in VARIANTS
    }

    def variantDone(label):
        return all(len(bucket) >= perBucket for bucket in collected[label].values())

    def allDone():
        return all(variantDone(label) for label in VARIANTS)

    seed = 0
    while not allDone():
        for numSpecies in SPECIES_RANGE:
            if allDone():
                break
            try:
                draw = _sharedDraw(numSpecies, seed)
            except Exception:
                continue
            geneTree, bmg, palette, reconc, target, base, parents = draw

            for label, extend in VARIANTS.items():
                if variantDone(label):
                    continue

                try:
                    instance = _instanceForVariant(
                        extend, geneTree, bmg, palette, reconc, target, base, parents,
                        numSpecies, seed,
                    )
                except Exception:
                    continue
                if not instance["startExplains"]:
                    continue
                if instance["network"].number_of_nodes() > MAX_NODES:
                    continue

                name = bucketNameFor(instance["R0"], label)
                buckets = collected[label]
                if name is None or len(buckets[name]) >= perBucket:
                    continue

                buckets[name].append(instance)
                print(f"  [{label} {name} {len(buckets[name])}/{perBucket}] "
                      f"ns={numSpecies} seed={seed}: |L|={len(getLeaves(instance['network']))} "
                      f"R0={instance['R0']} |V|={instance['network'].number_of_nodes()}")
        seed += 1

    return collected


def loadVariantBucket(bucketName: str) -> list:
    path = HERE / bucketName
    if not path.exists():
        raise FileNotFoundError(f"{path} not built yet. Run this script to create it.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _allBucketNames():
    return [
        f"samples_R0_{lo}_{hi}_{label}.pkl"
        for label in VARIANTS
        for lo, hi in RANGES
    ]


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    allNames = _allBucketNames()

    if rebuild or not all((HERE / name).exists() for name in allNames):
        print("building all ten buckets (5 R0-ranges x 2 variants)")
        collected = buildBucketedSamples()
        for buckets in collected.values():
            for name, instances in buckets.items():
                with (HERE / name).open("wb") as handle:
                    pickle.dump(instances, handle)
                print(f"saved {len(instances)} instances to {HERE / name}")
    else:
        print("all ten buckets already exist, pass --rebuild to regenerate")

    for name in allNames:
        instances = loadVariantBucket(name)
        if not instances:
            print(f"{name}: n=0")
            continue
        Rs = sorted(i["R0"] for i in instances)
        print(f"{name}: n={len(instances)} R0 min={Rs[0]} max={Rs[-1]}")
