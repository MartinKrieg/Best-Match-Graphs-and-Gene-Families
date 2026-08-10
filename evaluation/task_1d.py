### TASK:
### 1st: Generate a network
### 2nd: Compute the weak best match graph
### 3rd: Produce a explaining network using method 1b
### 4th: Compute the weak best match graph for the explaining network
### 5th: Compare

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import generateHybridNetwork, buildBaseBigCherry, extendBicCherryNetworkEdgeRestricted, weakBestMatchGraph
from utils import generateGeneTree, generateBmg, mapLeafColorsToNxTree
import networkx as nx

# NOTE: BIC-Cherry produces explaining networks
# NOTE: BMG's are created in another way

SPECIES_VARIATION = [10, 15, 20, 50]
HYBRIDIZATION_VARIATION = [3, 4, 5]
for num_species in SPECIES_VARIATION:
    for num_hybridizations in HYBRIDIZATION_VARIATION:
        print(f"Generating network with {num_species} species and {num_hybridizations} hybridizations...")

        # 1st step: Generate a network
        nxGeneTree, geneTree, rootID, geneColors = generateGeneTree(num_species)
        hybrid_network, _ = generateHybridNetwork(nxGeneTree, rootID, num_hybridizations, geneColors)
        hybrid_colors = mapLeafColorsToNxTree(hybrid_network, geneColors)

        # 2nd step: Compute the weak best match graph for the generated network
        wBMG = weakBestMatchGraph(hybrid_network, hybrid_colors)

        # 3rd step: Produce an explaining network using method 1b (restricted cases) (using normal bmg for test cases now)
        base_bic_cherry, root, parents = buildBaseBigCherry(wBMG, gene_colors=geneColors)
        explaining_network = extendBicCherryNetworkEdgeRestricted(base_bic_cherry, parents, wBMG, gene_colors=geneColors)
        explaining_colors = mapLeafColorsToNxTree(explaining_network, geneColors)

        # 4th step: Compute the weak best match graph for the explaining network
        wBMG_explaining = weakBestMatchGraph(explaining_network, explaining_colors)

        # 5th step: Compare the two best match graphs
        if nx.is_isomorphic(wBMG, wBMG_explaining):
            print(f"Success: The WMGs are isomorphic for {num_species} species and {num_hybridizations} hybridizations.")
        else:
            print(f"Failure: The WMGs are NOT isomorphic for {num_species} species and {num_hybridizations} hybridizations.")