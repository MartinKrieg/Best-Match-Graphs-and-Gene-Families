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

SPECIES_VARIATION = range(2, 5)
HYBRIDIZATION_VARIATION = range(2, 30)
for num_species in SPECIES_VARIATION:
    for num_hybridizations in HYBRIDIZATION_VARIATION:
        #print(f"Generating network with {num_species} species and {num_hybridizations} hybridizations...")

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
        # TODO Question: What could change here that causes the 4th step to fail (i.e. making the explaining network not explain the original network)? Is it the fact that we are using a restricted version of the BIC-Cherry algorithm? Or is it something else?

        # 4th step: Compute the weak best match graph for the explaining network
        wBMG_explaining = weakBestMatchGraph(explaining_network, explaining_colors)

        # 5th step: Compare the two best match graphs
        def has_same_best_matches(graph_a, graph_b):
            if set(graph_a.nodes()) != set(graph_b.nodes()):
                return False

            for node in graph_a.nodes():
                if graph_a.nodes[node].get("color") != graph_b.nodes[node].get("color"):
                    return False

            return set(graph_a.edges()) == set(graph_b.edges())

        isIsomorphic = nx.is_isomorphic(wBMG, wBMG_explaining)
        hasSameBestMatches = has_same_best_matches(wBMG, wBMG_explaining)

        if not isIsomorphic or not hasSameBestMatches:
            print(f"Mismatch found for {num_species} species and {num_hybridizations} hybridizations.")
            print(f"Isomorphic: {isIsomorphic}, Same Best Matches: {hasSameBestMatches}")
            # Optionally, you can save the graphs for further inspection
            nx.write_gml(wBMG, f"plots/mismatch/wBMG_{num_species}_{num_hybridizations}.gml")
            nx.write_gml(wBMG_explaining, f"plots/mismatch/wBMG_explaining_{num_species}_{num_hybridizations}.gml")
 