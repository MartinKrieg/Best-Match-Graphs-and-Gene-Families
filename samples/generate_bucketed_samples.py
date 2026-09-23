"""
Builds five R0-bucketed sample sets of 100 instances each, for evaluating
heuristics separately by starting difficulty:

    samples_R0_1_20.pkl     1   <= R0 < 20   (R0 == 0 excluded: already a tree, no reduction work)
    samples_R0_20_40.pkl    20  <= R0 < 40
    samples_R0_40_60.pkl    40  <= R0 < 60
    samples_R0_60_80.pkl    60  <= R0 < 80
    samples_R0_80_100.pkl   80  <= R0 < 100
"""

import pickle
import random
import sys
import numpy as np
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from src import (
    buildBaseBigCherry,
    extendBicCherryNetwork,
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

BUCKETS = [
    (1, 20, "samples_R0_1_20.pkl"),
    (20, 40, "samples_R0_20_40.pkl"),
    (40, 60, "samples_R0_40_60.pkl"),
    (60, 80, "samples_R0_60_80.pkl"),
    (80, 100, "samples_R0_80_100.pkl"),
]


def reticulationNumber(network: nx.DiGraph) -> int:
    return sum(max(0, network.in_degree(v) - 1) for v in network.nodes())


def _buildInstance(numSpecies: int, seed: int) -> dict:
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
    network = extendBicCherryNetwork(base, parents, bmg, palette)
    for graph in (network, target):
        for v in graph.nodes():
            if graph.out_degree(v) == 0:
                graph.nodes[v]["reconc"] = reconc[v]

    return {
        "startExplains": networkExplainsBmg(network, bmg),
        "numSpecies": numSpecies,
        "seed": seed,
        "network": network,
        "bmg": bmg,
        "target": target,
        "colors": palette,
        "reconc": reconc,
    }


def bucketFor(R0: int):
    for lo, hi, name in BUCKETS:
        if lo <= R0 < hi:
            return name
    return None


def buildBucketedSamples(perBucket: int = TARGET_PER_BUCKET) -> dict:
    collected = {name: [] for _, _, name in BUCKETS}
    seed = 0
    while any(len(collected[name]) < perBucket for _, _, name in BUCKETS):
        for numSpecies in SPECIES_RANGE:
            try:
                instance = _buildInstance(numSpecies, seed)
            except Exception:
                continue
            if not instance["startExplains"]:
                continue
            if instance["network"].number_of_nodes() > MAX_NODES:
                continue

            R0 = reticulationNumber(instance["network"])
            name = bucketFor(R0)
            if name is None or len(collected[name]) >= perBucket:
                continue

            collected[name].append(instance)
            print(f"  [{name} {len(collected[name])}/{perBucket}] "
                  f"ns={numSpecies} seed={seed}: |L|={len(getLeaves(instance['network']))} "
                  f"R0={R0} |V|={instance['network'].number_of_nodes()}")
        seed += 1

    return collected


def loadBucket(name: str, rebuild: bool = False) -> list:
    path = HERE / name
    if path.exists() and not rebuild:
        with path.open("rb") as handle:
            return pickle.load(handle)
    raise FileNotFoundError(f"{path} not built yet -- run this script directly first")


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    if rebuild or not all((HERE / name).exists() for _, _, name in BUCKETS):
        print("building all five buckets")
        collected = buildBucketedSamples()
        for _, _, name in BUCKETS:
            with (HERE / name).open("wb") as handle:
                pickle.dump(collected[name], handle)
            print(f"saved {len(collected[name])} instances to {HERE / name}")
    else:
        print("all five buckets already exist, pass --rebuild to regenerate")

    for _, _, name in BUCKETS:
        instances = loadBucket(name)
        Rs = sorted(reticulationNumber(i["network"]) for i in instances)
        print(f"{name}: n={len(instances)} R0 min={Rs[0]} max={Rs[-1]}")
