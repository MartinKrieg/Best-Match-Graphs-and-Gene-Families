"""
Builds a fixed set of evaluation instances for task 2(e) heuristics and caches
them to samples/samples.pkl.
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.heuristics.benchmark import _buildInstance, leafSet, reticulationNumber  # noqa: E402

CACHE = Path(__file__).resolve().parent / "samples.pkl"
TARGET_COUNT = 100
MAX_NODES = 400
SPECIES_RANGE = range(3, 11)  # 3..10 species per seed


def buildSamples(count: int = TARGET_COUNT) -> list:
    instances = []
    seed = 0
    while len(instances) < count:
        for numSpecies in SPECIES_RANGE:
            try:
                instance = _buildInstance(numSpecies, seed)
            except Exception:
                continue
            if not instance["startExplains"]:
                continue
            if instance["network"].number_of_nodes() > MAX_NODES:
                continue

            instances.append(instance)
            print(f"  [{len(instances)}/{count}] ns={numSpecies} seed={seed}: "
                  f"|L|={len(leafSet(instance['network']))} "
                  f"R={reticulationNumber(instance['network'])} "
                  f"|V|={instance['network'].number_of_nodes()}")

            if len(instances) >= count:
                break
        seed += 1

    return instances


def loadSamples(rebuild: bool = False) -> list:
    if CACHE.exists() and not rebuild:
        with CACHE.open("rb") as handle:
            return pickle.load(handle)

    instances = buildSamples()
    with CACHE.open("wb") as handle:
        pickle.dump(instances, handle)
    return instances


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    print("building samples" if rebuild else "loading samples (cached if present)")
    result = loadSamples(rebuild=rebuild)
    print(f"\n{len(result)} samples ready in {CACHE}")
    bySpecies = {}
    for inst in result:
        bySpecies.setdefault(inst["numSpecies"], 0)
        bySpecies[inst["numSpecies"]] += 1
    print("by numSpecies:", dict(sorted(bySpecies.items())))
