### Study for task 1a

from pathlib import Path
import argparse
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src import generateHybridNetwork, bestMatchGraphs
from utils import generateGeneTree

CATEGORIES = ["common", "onlyBmg", "onlyWbmg"]
COLORS = {"common": "#2a78d6", "onlyBmg": "#eb6834", "onlyWbmg": "#1baf7a"}
LABELS = {
    "common": "best match & weak best match",
    "onlyBmg": "best match only",
    "onlyWbmg": "weak best match only",
}


def compareBestMatches(network) -> dict:
    """Edge-set comparison between the strict and the weak best match graph."""
    bmg, wbmg = bestMatchGraphs(network)
    bmgEdges, wbmgEdges = set(bmg.edges()), set(wbmg.edges())
    return {
        "common": len(bmgEdges & wbmgEdges),
        "onlyBmg": len(bmgEdges - wbmgEdges),
        "onlyWbmg": len(wbmgEdges - bmgEdges),
    }


def runComparison(perCombo: int, speciesRange: range, hybRange: range) -> list:
    results = []
    combos = [(numSpecies, numHyb) for numSpecies in speciesRange for numHyb in hybRange]
    for numSpecies, numHyb in combos:
        for i in range(perCombo):
            print(f"=== numSpecies={numSpecies} numHyb={numHyb}: network {i + 1}/{perCombo} ===")
            try:
                nxGeneTree, _, rootID, geneColors = generateGeneTree(numSpecies)
                network, _ = generateHybridNetwork(nxGeneTree, rootID, numHyb, geneColors)
                result = compareBestMatches(network)
                result["numSpecies"] = numSpecies
                result["numHybridizations"] = numHyb
                results.append(result)
            except Exception as error:
                print(f"  skipped: {error}")
            finally:
                plt.close("all")  # generateHybridNetwork() plots each network
    return results


def plotResults(results: list, outPath: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    numHybValues = sorted({r["numHybridizations"] for r in results})
    exactMatchRate = []
    for numHybValue in numHybValues:
        subset = [r for r in results if r["numHybridizations"] == numHybValue]
        exact = sum(1 for r in subset if r["onlyWbmg"] == 0)
        exactMatchRate.append(100 * exact / len(subset))

    ax.bar([str(numHybValue) for numHybValue in numHybValues], exactMatchRate, color=COLORS["common"])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of hybridizations")
    ax.set_ylabel("Networks where wBMG = BMG exactly (%)")
    ax.set_title(f"Exact w(BMG) matches comparison for increasing hybridization (n={len(results)})")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(exactMatchRate):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=150, bbox_inches="tight")
    print(f"-> Saved plot to {outPath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of networks per (species, hyb) combination")
    parser.add_argument("--species", type=int, default=6, help="Number of species")
    parser.add_argument("--hyb", type=int, default=10, help="Number of hybridizations")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate networks instead of using the cached results")
    args = parser.parse_args()

    speciesRange = range(2, args.species + 1)
    hybRange = range(1, args.hyb + 1)
    runTag = f"n{args.n}_species{args.species}_hyb{args.hyb}"

    cachePath = Path(f"./plots/task_1a/results_{runTag}.pkl")
    if cachePath.exists() and not args.rebuild:
        print(f"-> Loading cached results from {cachePath} (pass --rebuild to regenerate)")
        with cachePath.open("rb") as handle:
            results = pickle.load(handle)
    else:
        results = runComparison(args.n, speciesRange, hybRange)
        cachePath.parent.mkdir(parents=True, exist_ok=True)
        with cachePath.open("wb") as handle:
            pickle.dump(results, handle)

    total = args.n * len(speciesRange) * len(hybRange)
    violations = sum(1 for r in results if r["onlyBmg"] > 0)
    print(
        f"-> {len(results)}/{total} networks generated successfully; "
        f"{violations} of them have a best match that is not a weak best match"
    )

    plotResults(results, Path(f"./plots/task_1a/bmg_vs_wbmg_{runTag}.png"))
