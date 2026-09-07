import asymmetree.treeevolve as te
from datetime import datetime
from pathlib import Path
import networkx as nx
from asymmetree.visualization.tree_vis import visualize, assign_colors
from tralda.datastructures import Tree
from asymmetree.analysis import best_matches


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

    return nxGeneTree, geneTree, root_id, gene_colors

def generateTreeBmg(geneTree: Tree) -> nx.DiGraph:
    """
    The best match graph explained by a leaf-colored gene tree.

    The nodes are the leaf labels and carry their species as the 'color'
    attribute.
    """
    return best_matches.bmg_from_tree(geneTree)

def generateNetworkBmg(network: nx.DiGraph) -> nx.DiGraph:
    """
    Computes the Best Match Graph (BMG) explained by a leaf-colored phylogenetic network.
    
    The nodes are the leaf labels and carry their species as the 'color' attribute.
    An edge (x, y) exists if y is a best match of x among all leaves of y's color.
    """
    bmg = nx.DiGraph()
    leaves = [n for n in network.nodes() if network.out_degree(n) == 0]
    
    colors = {}
    for leaf in leaves:
        color_val = network.nodes[leaf].get('reconc')
        if color_val is None:
            raise ValueError(f"Leaf {leaf} is missing the 'color' attribute.")
        colors[leaf] = color_val
        bmg.add_node(leaf, color=color_val)
        
    # 2. Precompute ancestors and descendants for efficiency
    descendants = {n: nx.descendants(network, n) for n in network.nodes()}
    ancestors = {n: nx.ancestors(network, n) for n in network.nodes()}
    for n in network.nodes():
        ancestors[n].add(n) # Include the node itself for LCA intersection
        
    # 3. Compute LCAs for all pairs of leaves
    lca = {}
    for x in leaves:
        lca[x] = {}
        for y in leaves:
            if x == y:
                lca[x][y] = {x}
                continue
            
  
            commonAncestor = ancestors[x].intersection(ancestors[y])
            
            # An ancestor u is an LCA if NONE of its descendants are also in CA
            lca[x][y] = {u for u in commonAncestor if not any(v in commonAncestor for v in descendants[u])}
            
    # 4. Determine Best Matches based on network LCA partial ordering
    for x in leaves:
        # Group all other leaves by their color
        color_to_leaves = {}
        for y in leaves:
            if x != y:
                c = colors[y]
                color_to_leaves.setdefault(c, []).append(y)
                
        # Evaluate best match condition per color group
        for c, targets in color_to_leaves.items():
            for y in targets:
                is_best_match = False
                
                # y is a best match of x if AT LEAST ONE of their LCAs (u) is not "beaten"
                for u in lca[x][y]:
                    u_is_beaten = False
                    
                    # Check against all other targets y' of the same color
                    for y_prime in targets:
                        for v in lca[x][y_prime]:
                            if v in descendants[u]: # v is a strict descendant of u (closer to leaves)
                                u_is_beaten = True
                                break
                        if u_is_beaten:
                            break
                            
                    # If we found an LCA for (x,y) that no (x,y') improves upon, it's a best match!
                    if not u_is_beaten:
                        is_best_match = True
                        break 
                        
                if is_best_match:
                    bmg.add_edge(x, y)
                    
    return bmg

def generateLeastResolvedTree(geneTree: Tree) -> Tree:
    """T*, obtained by contracting the redundant edges of the gene tree."""
    return best_matches.lrt_from_tree(geneTree)