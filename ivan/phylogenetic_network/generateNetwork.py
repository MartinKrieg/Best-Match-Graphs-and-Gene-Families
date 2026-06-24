import asymmetree.treeevolve as te
from asymmetree.visualize.TreeVis import assign_gene_colors, visualize
import random
import networkx as nx
import matplotlib.pyplot as plt

def visualize_tree(tree, save_as='tree.pdf'):
    gene_colors = assign_gene_colors(tree)          # leitet Farben aus reconc (σ) ab
    visualize(tree, color_dict=gene_colors, save_as=save_as)

def visualize_network(N, save_as='network.pdf'):
    plt.figure()
    # Layout nach Tiefe (Wurzel oben)
    for v in nx.topological_sort(N):
        preds = list(N.predecessors(v))
        N.nodes[v]['layer'] = 0 if not preds else 1 + max(N.nodes[p]['layer'] for p in preds)
    pos = nx.multipartite_layout(N, subset_key='layer', align='horizontal')
    pos = {v: (x, -y) for v, (x, y) in pos.items()}
    # Hybridisierungskanten (in Knoten mit Eingangsgrad >= 2) rot, Rest schwarz
    colors = ['red' if N.in_degree(v) >= 2 else 'black' for _, v in N.edges()]
    nx.draw(N, pos, edge_color=colors, node_size=200, with_labels=False)
    plt.savefig(save_as, dpi=300) if save_as else plt.show()
    plt.close()

def generate_tree(n_species, age, dupl_rate, loss_rate, hgt_rate, prune):
    """
    - generate a leaf-colored gene tree with AsymmeTree
    - leaves carry the attribute 'reconc' = the species sigma(leaf)
    """

    S = te.species_tree_n_age(n_species, age)
    T = te.dated_gene_tree(S, dupl_rate=dupl_rate, loss_rate=loss_rate, hgt_rate=hgt_rate)
    return te.prune_losses(T) if prune else T

def tree_to_network(tree):
    """Convert a tralda/AsymmeTree tree into a NetworkX DiGraph network"""
    G, root = tree.to_nx()
    N = nx.DiGraph(G)
    N.graph['root'] = root
    return N

def leaves(N):
    return [v for v in N if N.out_degree(v) == 0]

def sigma(N):
    """Leaf coloring sigma: leaf -> species id ('reconc' attribute)."""
    return {v: N.nodes[v].get('reconc') for v in leaves(N)}

def _fresh_id(N):
    nid = N.graph.get('_next_id')
    if nid is None:
        nid = max(N.nodes) + 1
    N.graph['_next_id'] = nid + 1
    return nid

def insert_hybridization(N):
    """
    -   Insert one hybridization vertex by subdividing two random edges
        and connecting the two new vertices, keeping the network acyclic.
    -   Returns the new hybridization vertex (in-degree 2, out-degree 1).
    """
    # sample 2 random edges
    # u... parent, v... children
    (u1, v1), (u2, v2) = random.sample(list(N.edges()), 2)

    a, b = _fresh_id(N), _fresh_id(N)

    # insert vertices a, b into edges (u1, v1), (u2, v2)
    N.remove_edge(u1, v1)
    N.add_edge(u1, a)
    N.add_edge(a, v1)
    N.remove_edge(u2, v2)
    N.add_edge(u2, b)
    N.add_edge(b, v2)

    # orient the hybridization edge so we never create a cycle:
    # if b can already reach a, a must become the hybridization vertex, else b.
    if nx.has_path(N, b, a):
        source, hybridization_vertex = b, a
    else:
        source, hybridization_vertex = a, b

    N.add_edge(source, hybridization_vertex)
    N.nodes[hybridization_vertex]['hybridization'] = True
    
    # final security check for acyclicity
    assert nx.is_directed_acyclic_graph(N), "insertion broke acyclicity"

    return hybridization_vertex

def generate_network(n_species, n_hybridizations, age, dupl_rate, loss_rate, hgt_rate, prune):
    """Generate a tree and turn it into a network with k hybridizations."""

    tree = generate_tree(n_species, age, dupl_rate, loss_rate, hgt_rate, prune)

    visualize_tree(tree)

    N = tree_to_network(tree)
    for _ in range(n_hybridizations):
        insert_hybridization(N)
    return N

if __name__ == '__main__':
    SEED = 420
    random = random.Random(SEED)

    N = generate_network(
        n_species=3,
        n_hybridizations=1,
        age=1.0,
        dupl_rate=0.5,
        loss_rate=0.5,
        hgt_rate=0.5,
        prune=True,
    )
    hybridization_vertices = [v for v in N if N.in_degree(v) == 2]
    root = [v for v in N if N.in_degree(v) == 0]

    print(f'nodes={N.number_of_nodes()} edges={N.number_of_edges()}')
    print(f'leaves={len(leaves(N))} hybridizations={len(hybridization_vertices)} root={root}')
    print(f'is_DAG={nx.is_directed_acyclic_graph(N)}')
    # coloring of the leafs
    print('sigma (leaf: species) -->', sigma(N))

    visualize_network(N)