from networkx import DiGraph, lowest_common_ancestor, all_simple_paths
from itertools import combinations
from utils import extract_sigma_from_graph

### Difference between best matches and weak best matches

def construct_weak_best_match_graph(G: DiGraph):
    G_bmg = DiGraph()
    sigma = extract_sigma_from_graph(G) # todo
    for x, y in combinations(G.nodes(), 2):
        if isWeakBestMatch(G=G, sigma=sigma, x=x, y=y):
            G_bmg.add_edge(x,y)
    return G_bmg

# Check whether y is a best match for x or not
def isWeakBestMatch(G: DiGraph, sigma, x, y):
    if sigma[x] == sigma[y]:
        return False

    y_primes = [node for node in G.nodes() if sigma[node] == sigma[y]]

    lca_xy = lowest_common_ancestor(G, x, y)

    for y_prime in y_primes:
        lca_xy_prime = lowest_common_ancestor(G, x, y_prime)
        # lca(x, y_prime) must be an ancestor for lca(x, y)
        if isAncestor(G, lca_xy_prime, lca_xy): # Check is Pre-/Successor lca(x,y) <= lca(x,y_prime)
            return False

    return True

# Returns whether or not x is higher than y, thus fulfilling 
def isAncestor(G, ancestor, sibling):
    paths = all_simple_paths(G, G.root, sibling)
    for path in paths:
        if path.__contains__(ancestor):
            return True

    return False