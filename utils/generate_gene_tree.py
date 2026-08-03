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

    _, gene_colors = assign_colors(speciesTree, geneTree)
    visualize(geneTree, color_dict=gene_colors, save_as=str(save_path))

    nxGeneTree, root_id = geneTree.to_nx()

    return nxGeneTree, geneTree, root_id, gene_colors
