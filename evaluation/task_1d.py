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
from utils import generateGeneTree, mapLeafColorsToNxTree
import networkx as nx
import numpy as np


def sanitize_for_gml(G: nx.DiGraph) -> nx.DiGraph:
    """Cast numpy scalars in node/edge attributes to native Python types."""
    def cast(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    for _, data in G.nodes(data=True):
        for key, value in data.items():
            data[key] = cast(value)
    for _, _, data in G.edges(data=True):
        for key, value in data.items():
            data[key] = cast(value)
    return G

def relabel_leaves_by_gene_id(G: nx.DiGraph) -> nx.DiGraph:
    """Relabel leaf nodes to their gene-ID 'label' attribute."""
    mapping = {
        node: data["label"]
        for node, data in G.nodes(data=True)
        if G.out_degree(node) == 0 and "label" in data
    }
    return nx.relabel_nodes(G, mapping, copy=True)

def annotate_leaf_colors_for_export(G: nx.DiGraph, leaf_colors: dict, attr_name: str = "gene_color") -> nx.DiGraph:
    """Return a copy of G where leaf color labels are stored as node attributes."""
    H = G.copy()
    attrs = {}
    for node, color in leaf_colors.items():
        if node in H:
            # as str -> robust for GML roundtrip
            attrs[node] = {attr_name: str(color)}
    nx.set_node_attributes(H, attrs)
    return sanitize_for_gml(H)

def isEqual(G1, G2):

    # Vertex set equal?
    if set(G1.nodes()) != set(G2.nodes()):
        return False

    # Edge set equal?
    if set(G1.edges()) != set(G2.edges()):
        return False

    if not nx.is_isomorphic(G1, G2):
        return False

    return True

SPECIES_VARIATION = [2] # for iterative testing, set range(X,Y)
HYBRIDIZATION_VARIATION = [1]
found = False
while not found:
    for num_species in SPECIES_VARIATION:
        for num_hybridizations in HYBRIDIZATION_VARIATION:
            #print(f"Generating network with {num_species} species and {num_hybridizations} hybridizations...")

            # 1st step: Generate a network 
            nxGeneTree, geneTree, rootID, geneColors = generateGeneTree(num_species)
            nxGeneTree = relabel_leaves_by_gene_id(nxGeneTree)
            hybrid_network, _ = generateHybridNetwork(nxGeneTree, rootID, num_hybridizations, geneColors)
            hybrid_colors = mapLeafColorsToNxTree(hybrid_network, geneColors)

            # 2nd step: Compute the weak best match graph for the generated network
            wBMG = weakBestMatchGraph(hybrid_network, hybrid_colors)

            # 3rd step: Produce an explaining network using method 1b (restricted cases)
            base_bic_cherry, _, parents = buildBaseBigCherry(wBMG, gene_colors=geneColors)
            explaining_network = extendBicCherryNetworkEdgeRestricted(base_bic_cherry, parents, wBMG, gene_colors=geneColors)
            explaining_colors = mapLeafColorsToNxTree(explaining_network, geneColors)

            # 4th step: Compute the weak best match graph for the explaining network
            wBMG_explaining = weakBestMatchGraph(explaining_network, explaining_colors)

            isEqualGraphs = isEqual(wBMG, wBMG_explaining)

            if not isEqualGraphs:
                found = True
                print(f"Mismatch found for {num_species} species and {num_hybridizations} hybridizations.")
                out_dir = Path(f"plots/mismatch/{num_species}_{num_hybridizations}")
                out_dir.mkdir(parents=True, exist_ok=True)

                # Store graphs as GML files
                nx.write_gml(
                    annotate_leaf_colors_for_export(nxGeneTree, geneColors),
                    out_dir / f"geneTree_{num_species}_{num_hybridizations}.gml"
                )
                nx.write_gml(
                    annotate_leaf_colors_for_export(hybrid_network, hybrid_colors),
                    out_dir / f"hybrid_network_{num_species}_{num_hybridizations}.gml"
                )
                nx.write_gml(
                    annotate_leaf_colors_for_export(explaining_network, explaining_colors),
                    out_dir / f"explaining_network_{num_species}_{num_hybridizations}.gml"
                )
                nx.write_gml(
                    annotate_leaf_colors_for_export(wBMG, geneColors),
                    out_dir / f"wBMG_{num_species}_{num_hybridizations}.gml"
                )
                nx.write_gml(
                    annotate_leaf_colors_for_export(wBMG_explaining, geneColors),
                    out_dir / f"wBMG_explaining_{num_species}_{num_hybridizations}.gml"
                )

                print(f"Found mismatch for {num_species} species and {num_hybridizations} hybridizations. Graphs saved in {out_dir}/")