### Study for task 1a

from pathlib import Path
import argparse
import csv
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src import generateHybridNetwork, bestMatchGraphs
from utils import generateGeneTree

CSV_FIELDS = ["numSpecies", "numHybridizations", "onlyBmg", "onlyWbmg"]
SPECIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
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


def appendResults(rows: list, csvPath: Path):
    """Append the rows to the CSV, writing the header only when the file is new."""
    csvPath.parent.mkdir(parents=True, exist_ok=True)
    isNew = not csvPath.exists()
    with csvPath.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if isNew:
            writer.writeheader()
        writer.writerows(rows)


def loadResults(csvPath: Path) -> list:
    with csvPath.open(newline="") as handle:
        return [{key: int(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def runComparison(perCombo: int, speciesRange: range, hybRange: range, csvPath: Path):
    for numSpecies in speciesRange:
        rows = []
        for numHyb in hybRange:
            for i in range(perCombo):
                print(f"=== numSpecies={numSpecies} numHyb={numHyb}: network {i + 1}/{perCombo} ===")
                try:
                    nxGeneTree, _, rootID, geneColors = generateGeneTree(numSpecies)
                    network, _ = generateHybridNetwork(nxGeneTree, rootID, numHyb, geneColors)
                    result = compareBestMatches(network)
                    rows.append({
                        "numSpecies": numSpecies,
                        "numHybridizations": numHyb,
                        "onlyBmg": result["onlyBmg"],
                        "onlyWbmg": result["onlyWbmg"],
                    })
                except Exception as error:
                    print(f"  skipped: {error}")
                finally:
                    plt.close("all")  # generateHybridNetwork() plots each network, close them
        appendResults(rows, csvPath)  # only what the plot needs


def plotResults(results: list, outPath: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for numSpecies in sorted({r["numSpecies"] for r in results}):
        speciesRows = [r for r in results if r["numSpecies"] == numSpecies]
        numHybValues = sorted({r["numHybridizations"] for r in speciesRows})
        exactMatchRate = []
        for numHybValue in numHybValues:
            subset = [r for r in speciesRows if r["numHybridizations"] == numHybValue]
            exact = sum(1 for r in subset if r["onlyWbmg"] == 0)
            exactMatchRate.append(100 * exact / len(subset))

        ax.plot(
            numHybValues, exactMatchRate, marker="o", markersize=6, linewidth=2,
            color=SPECIES_COLORS[(numSpecies - 2) % len(SPECIES_COLORS)],
            label=f"{numSpecies} species (n={len(speciesRows)})",
        )

    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Number of hybridizations")
    ax.set_ylabel("Networks where wBMG = BMG exactly (%)")
    ax.set_title("Exact w(BMG) matches for increasing hybridization, by number of species")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, title="Species")

    fig.tight_layout()
    outPath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outPath, dpi=150, bbox_inches="tight")
    print(f"-> Saved plot to {outPath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of networks per (species, hyb) combination")
    parser.add_argument("--species", type=int, default=5, help="Number of species")
    parser.add_argument("--hyb", type=int, default=25, help="Number of hybridizations")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate networks instead of using the cached results")
    args = parser.parse_args()

    speciesRange = range(2, args.species + 1)
    hybRange = range(1, args.hyb + 1)
    runTag = f"n{args.n}_species{args.species}_hyb{args.hyb}"

    csvPath = Path(f"./plots/task_1a/results_{runTag}.csv")
    if args.rebuild or not csvPath.exists():
        csvPath.unlink(missing_ok=True)
        runComparison(args.n, speciesRange, hybRange, csvPath)
    else:
        print(f"-> Loading cached results from {csvPath} (pass --rebuild to regenerate)")
    results = loadResults(csvPath)

    total = args.n * len(speciesRange) * len(hybRange)
    violations = sum(1 for r in results if r["onlyBmg"] > 0)

    plotResults(results, Path(f"./plots/task_1a/bmg_vs_wbmg_{runTag}.png"))
