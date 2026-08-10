import asymmetree.treeevolve as te
from datetime import datetime
from pathlib import Path
import networkx as nx
from asymmetree.visualization.tree_vis import visualize, assign_colors
from tralda.datastructures import Tree


def generateGeneTree(numSpecies: int) -> tuple[nx.DiGraph, Tree, int, dict]:
    print(f"-> Generating species tree with {numSpecies} species .")
    current_date = datetime.now().strftime("%Y-%m-%d")
    save_path = Path(f"./plots/base_tree/genetree_{current_date}.png")

    speciesTree = te.species_tree_n_age(
        n=numSpecies,
        age=1.0,
        #model="BDP",
        #innovation=True,
        #birth_rate=1.0,
        #death_rate=0.5,
        #contraction_probability=0.2,
    )
    geneTree = te.dated_gene_tree(
        speciesTree,
        dupl_rate=0.7,
        #loss_rate=0.7,
        #hgt_rate=0.7,
        #gc_rate=0.7,
        #dupl_polytomy=0.5,
        #replace_prob=0.5,
        transfer_distance_bias="inverse",
    )

    # Best matches are only defined for the observable genes. Loss leaves carry an edge of the species tree as their 'reconc',
    # so leaving them in would add spurious genes and spurious species to the BMG
    geneTree = te.prune_losses(geneTree)

    _, gene_colors = assign_colors(speciesTree, geneTree)
    visualize(geneTree, color_dict=gene_colors, save_as=str(save_path))

    nxGeneTree, root_id = geneTree.to_nx()
    print(f"Number of vertices: {nxGeneTree.number_of_nodes()}, Number of edges: {nxGeneTree.number_of_edges()}")

    # TODO: Consider returning a small dataclass here if more tree variants or
    # color mappings need to travel together in the future.
    return nxGeneTree, geneTree, root_id, gene_colors


def mapLeafColorsToNxTree(nxTree: nx.DiGraph, geneColors: dict) -> dict:
    """Map a leaf-color dictionary onto the NetworkX node IDs of a tree."""
    mapped_colors = {}
    for node in nxTree.nodes():
        if nxTree.out_degree(node) != 0:
            continue

        if node in geneColors:
            mapped_colors[node] = geneColors[node]
            continue

        label = nxTree.nodes[node].get("label")
        if label in geneColors:
            mapped_colors[node] = geneColors[label]
            continue

        reconc = nxTree.nodes[node].get("reconc")
        if reconc in geneColors:
            mapped_colors[node] = geneColors[reconc]
            continue

        raise ValueError(f"no color assigned to the leaf {node}")

    return mapped_colors
