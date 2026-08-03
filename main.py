from utils import generateGeneTree, generateBmg, visualizeHierarchicalNetwork
from src import generateHybridNetwork, buildBaseBigCherry, extendBicCherryNetwork, extendBicCherryNetworkEdgeRestricted
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl", type=int, help="Number of leaves in gene tree")
    parser.add_argument("--nh", type=int, help="Number of hybridizations")
    args = parser.parse_args()

    numberGeneTreeLeaves = args.nl
    numberHybridizations = args.nh

    # Hybridization
    nxGeneTree, geneTree, rootID, geneColors = generateGeneTree(numberGeneTreeLeaves)
    generateHybridNetwork(nxGeneTree, rootID, numberHybridizations, geneColors)
    
    # BigCherry
    bmg, geneColors = generateBmg(geneTree, geneColors)
    base_network, root, base_parents = buildBaseBigCherry(bmg, geneColors)
    visualizeHierarchicalNetwork(base_network, "Unmodified_Base_BIC_Cherry_Network", root, geneColors)
    
    final_network = extendBicCherryNetwork(base_network, base_parents, bmg, geneColors)
    visualizeHierarchicalNetwork(final_network, "Expanded_Phylogenetic_Network", root, geneColors)
    
    network = extendBicCherryNetworkEdgeRestricted(base_network, base_parents,bmg,geneColors)
    visualizeHierarchicalNetwork(network, "Expanded_Phylogenetic_Network_Restricted", root, geneColors)
    
