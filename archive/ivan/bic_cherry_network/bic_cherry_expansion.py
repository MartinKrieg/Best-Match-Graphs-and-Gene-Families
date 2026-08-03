"""
Implement the BIC-cherry+expansion algorithm for given graphs
(G, sigma) as in the tech report from Patricia Ebert and Marc Hellmuth

Input : a colored digraph (G, sigma) that is properly colored and has the
        sicor-in-hub property (i.e. a candidate best match graph)
        - G     : networkx.DiGraph whose nodes are the leaves X
        - sigma : dict mapping each leaf to its color, sigma[x]
Output: a leaf-colored network N (networkx.DiGraph) explaining (G, sigma).
        N.graph['root'] holds the root vertex
"""

import itertools
import networkx as nx


def _p(x, y):
    """Canonical id of the cherry vertex p_xy (unordered in {x, y})"""
    return ('p', frozenset((x, y)))

def bic_cherry_network(X, sigma):
    """Build the BIC-cherry network on leaf set X (Definition 2.3)"""
    X = list(X)
    N = nx.DiGraph()
    N.add_nodes_from(X)

    if len(X) == 2:
        # p_xy takes the role of the root
        x, y = X[0], X[1]
        root = _p(x, y)
        N.add_edges_from([(root, x), (root, y)])
    else:
        root = 'rho'
        N.add_node(root)
        for x, y in itertools.combinations(X, 2):
            if sigma[x] != sigma[y]:
                p = _p(x, y)
                N.add_edges_from([(root, p), (p, x), (p, y)])

    N.graph['root'] = root
    return N

def bic_cherry_expansion(G, sigma):
    """Algorithm 1: construct a network N explaining the BMG (G, sigma)"""
    X = list(G.nodes())
    N = bic_cherry_network(X, sigma)

    # group leaves by color so we can pick a partner y' quickly
    color_classes = {}
    for x in X:
        color_classes.setdefault(sigma[x], []).append(x)

    q_id = 0
    for x in X:
        for y in X:
            # only differently-colored non-arcs trigger an expansion
            if sigma[x] == sigma[y] or G.has_edge(x, y):
                continue
            # (x, y) not in E and sicor-in-hub => y is not a sicor,
            # hence a same-colored partner y' != y exists
            y_prime = next(yp for yp in color_classes[sigma[y]] if yp != y)
            # extension [xy : xy']: new vertex q under p_xy, children x and y'
            q = ('q', q_id)
            q_id += 1
            N.add_edges_from([(_p(x, y), q), (q, x), (q, y_prime)])

    return N

if __name__ == '__main__':
    # Example from Figure 2 of the tech report:
    # X = {x, y, z}, colors sigma(x)=A, sigma(y)=sigma(z)=B.
    # Target BMG = all differently-colored arcs EXCEPT (x, y).
    sigma = {'x': 'A', 'y': 'B', 'z': 'B'}
    G = nx.DiGraph()
    G.add_nodes_from(sigma)
    G.add_edges_from([('x', 'z'), ('y', 'x'), ('z', 'x')])  # (x, y) is missing

    N = bic_cherry_expansion(G, sigma)

    leaves = [v for v in N if N.out_degree(v) == 0]
    roots = [v for v in N if N.in_degree(v) == 0]
    
    print('nodes:', N.number_of_nodes(), 'edges:', N.number_of_edges())
    print('leaves:', sorted(leaves))
    print('root:', N.graph['root'], '| roots found:', roots)
    print('is_DAG:', nx.is_directed_acyclic_graph(N))
    print('edges:', sorted(N.edges(), key=str))
