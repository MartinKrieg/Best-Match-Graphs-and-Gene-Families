from utils import generateGeneTree, generateTreeBmg, visualizeBmg, visualizeHierarchicalNetwork
from src import generateHybridNetwork, buildBaseBigCherry, extendBicCherryNetwork, extendBicCherryNetworkEdgeRestricted
from src import leastResolvedTree, explainsBmg, toNetwork
from src import missingArcs, networkExplainsBmg
from src import surveyEditMoves, formatMoveSurvey
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, help="Number of species")
    parser.add_argument("--nh", type=int, help="Number of hybridizations")
    args = parser.parse_args()

    numberSpecies = args.ns
    numberHybridizations = args.nh

    # Hybridization
    nxGeneTree, geneTree, rootID, geneColors = generateGeneTree(numberSpecies)
    generateHybridNetwork(nxGeneTree, rootID, numberHybridizations, geneColors)
    
    # BMG Generation
    bmg = generateTreeBmg(geneTree)
    bmg, geneColors = visualizeBmg(bmg, geneTree, geneColors)

    # Least resolved tree
    lrt = leastResolvedTree(geneTree)
    lrtNetwork, lrtRoot = toNetwork(lrt)
    print(f"-> Least resolved tree explains the BMG: {explainsBmg(lrt, bmg)}")
    visualizeHierarchicalNetwork(lrtNetwork, "Least_Resolved_Tree", lrtRoot, geneColors, "./plots/least_resolved_tree")

    base_network, root, base_parents = buildBaseBigCherry(bmg, geneColors)
    visualizeHierarchicalNetwork(base_network, "Unmodified_Base_BIC_Cherry_Network", root, geneColors, "./plots/big_cherry")
    
    print(f"-> Missing arcs, i.e. expansions to perform: {len(missingArcs(bmg))}")

    final_network = extendBicCherryNetwork(base_network, base_parents, bmg, geneColors)
    print(f"-> Expansion explains the BMG: strict={networkExplainsBmg(final_network, bmg)} weak={networkExplainsBmg(final_network, bmg, weak=True)}")
    visualizeHierarchicalNetwork(final_network, "Expanded_Phylogenetic_Network", root, geneColors, "./plots/big_cherry")
    
    network = extendBicCherryNetworkEdgeRestricted(base_network, base_parents,bmg,geneColors)
    print(f"-> Edge-restricted expansion explains the BMG: strict={networkExplainsBmg(network, bmg)} weak={networkExplainsBmg(network, bmg, weak=True)}")
    visualizeHierarchicalNetwork(network, "Expanded_Phylogenetic_Network_Restricted", root, geneColors, "./plots/big_cherry")

    survey = surveyEditMoves(network)
    print(f"-> Task 2d edit-move BMG check:\n{formatMoveSurvey(survey)}")
    