### TASK:
### 1st: Generate a network
### 2nd: Compute the weak best match graph
### 3rd: Produce a explaining network using method 1b
### 4th: Compute the weak best match graph for the explaining network
### 5th: Compare

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import generateHybridNetwork, buildBaseBigCherry, extendBicCherryNetworkEdgeRestricted
from utils import generateGeneTree, generateBmg
import networkx as nx

# NOTE: BIC-Cherry produces explaining networks
# NOTE: BMG's are created in another way

SPECIES_VARIATION = [5, 10, 15, 20, 50]
HYBRIDIZATION_VARIATION = [1, 2, 3, 4, 5]

for num_species in SPECIES_VARIATION:
    for num_hybridizations in HYBRIDIZATION_VARIATION:
        print(f"Generating network with {num_species} species and {num_hybridizations} hybridizations...")

        # 1st step: Generate a network
        nxGeneTree, geneTree, rootID, geneColors = generateGeneTree(num_species)
        hybrid_network = generateHybridNetwork(nxGeneTree, rootID, num_hybridizations, geneColors)

        # 2nd step: Compute the weak best match graph for the generated network
        bmg, geneColors = generateBmg(geneTree, geneColors)

        # 3rd step: Produce an explaining network using method 1b (restricted cases) (using normal bmg for test cases now)
        base_bic_cherry, base_root, base_parents = buildBaseBigCherry(bmg, geneColors)
        restricted_bic_cherry = extendBicCherryNetworkEdgeRestricted(base_bic_cherry, base_parents, bmg, geneColors)

        # 4th step: Compute the weak best match graph for the explaining network
        bmg_explaining, geneColors_explaining = generateBmg(restricted_bic_cherry, geneColors)

        # 5th step: Compare the two best match graphs
        if nx.is_isomorphic(bmg, bmg_explaining):
            print(f"Success: The BMGs are isomorphic for {num_species} species and {num_hybridizations} hybridizations.")
        else:
            print(f"Failure: The BMGs are NOT isomorphic for {num_species} species and {num_hybridizations} hybridizations.")