### Study for task 1d
### For both variants restricted and standard BIC-cherry+expansion

from pathlib import Path
import argparse
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src import (
    generateHybridNetwork,
    weakBestMatchGraph,
    colorsFromGraph,
    buildBaseBigCherry,
    extendBicCherryNetwork,
    extendBicCherryNetworkEdgeRestricted,
    networkExplainsBmg,
)
from utils import generateGeneTree

COLOR = "#2a78d6"
VARIANTS = {
    "restricted": extendBicCherryNetworkEdgeRestricted,
    "standard": extendBicCherryNetwork,
}


def compareExplanation(network, extend) -> dict:
    """Does the expansion's weak BMG match the original?"""
    wbmg = weakBestMatchGraph(network)
    geneColors = colorsFromGraph(wbmg)
    base, _, parents = buildBaseBigCherry(wbmg, geneColors)
    explainingNetwork = extend(base, parents, wbmg, geneColors)
    return {"explains": networkExplainsBmg(explainingNetwork, wbmg, weak=True)}


def runComparison(perCombo: int, speciesRange: range, hybRange: range, extend) -> list:
    results = []
    combos = [(numSpecies, numHyb) for numSpecies in speciesRange for numHyb in hybRange]
    for numSpecies, numHyb in combos:
        for i in range(perCombo):
            print(f"=== numSpecies={numSpecies} numHyp={numHyb}: network {i + 1}/{perCombo} ===")
            try:
                nxGeneTree, _, rootID, geneColors = generateGeneTree(numSpecies)
                network, _ = generateHybridNetwork(nxGeneTree, rootID, numHyb, geneColors)
                result = compareExplanation(network, extend)
                result["numSpecies"] = numSpecies
                result["numHybridizations"] = numHyb
                results.append(result)
            except Exception as error:
                print(f"  skipped: {error}")
            finally:
                plt.close("all")  # generateHybridNetwork() plots each network
    return results


def plotResults(results: list, outPath: Path, variant: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    hybValues = sorted({r["numHybridizations"] for r in results})
    successRate = []
    for numHyb in hybValues:
        subset = [r for r in results if r["numHybridizations"] == numHyb]
        success = sum(1 for r in subset if r["explains"])
        successRate.append(100 * success / len(subset))

    ax.bar([str(numHyb) for numHyb in hybValues], successRate, color=COLOR)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Number of hybridizations")
    ax.set_ylabel("Networks where wBMG(N') = wBMG(N) (%)")
    ax.set_title(f"{variant.capitalize()} BIC-cherry+expansion explains wBMG (n={len(results)})")
    ax.spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(successRate):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=150, bbox_inches="tight")
    print(f"-> Saved plot to {outPath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of networks per (numSpecies, numHyb) combination")
    parser.add_argument("--species", type=int, default=6, help="Largest number of species")
    parser.add_argument("--hyb", type=int, default=10, help="Largest number of hybridizations")
    parser.add_argument("--variant", choices=list(VARIANTS), default="restricted", help="Which BIC-cherry+expansion variant to test")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate networks instead of using the cached results")
    args = parser.parse_args()

    speciesRange = range(2, args.species + 1)
    hybRange = range(1, args.hyb + 1)
    runTag = f"{args.variant}_n{args.n}_species{args.species}_hyb{args.hyb}"

    cachePath = Path(f"./plots/task_1d/results_{runTag}.pkl")
    if cachePath.exists() and not args.rebuild:
        print(f"-> Loading cached results from {cachePath} (pass --rebuild to regenerate)")
        with cachePath.open("rb") as handle:
            results = pickle.load(handle)
    else:
        results = runComparison(args.n, speciesRange, hybRange, VARIANTS[args.variant])
        cachePath.parent.mkdir(parents=True, exist_ok=True)
        with cachePath.open("wb") as handle:
            pickle.dump(results, handle)

    total = args.n * len(speciesRange) * len(hybRange)
    successes = sum(1 for r in results if r["explains"])
    print(
        f"-> {len(results)}/{total} networks generated successfully; "
        f"{successes}/{len(results)} explaining networks match wBMG(N) exactly"
    )

    plotResults(results, Path(f"./plots/task_1d/wbmg_explained_{runTag}.png"), args.variant)
