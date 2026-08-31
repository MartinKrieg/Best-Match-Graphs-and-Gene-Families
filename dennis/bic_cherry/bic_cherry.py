import itertools

import networkx as nx
from networkx import DiGraph
from itertools import combinations

def construct_bic_cherry_plus_expansion(G: DiGraph) -> DiGraph:
    """
    Constructs the bic cherry plus expansion from a given directed graph G.
    """

    # Check if G satisfies the sicor-inhub property

    if not is_sicor_inhub(G): # This is a prior step, separate construction from preparation/verification of the input graph.
        raise ValueError("The input graph does not satisfy the sicor-inhub property. Thus, the graphs cannot be a BMG.")

    N = construct_base_bic_cherry(
        list(G.nodes()),
        {v: G.nodes[v]["color"] for v in G.nodes()}
    )


def construct_base_bic_cherry(X: list, sigma) -> DiGraph:
  """Build the BIC-cherry network on leaf set X (Definition 2.3)"""
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

def _p(x, y):
    """Canonical id of the cherry vertex p_xy (unordered in {x, y})"""
    return ('p', frozenset((x, y)))

def is_sicor_inhub(G: DiGraph) -> bool:
    """
    Checks if the directed graph G satisfies the sicor-inhub property.
    """

    for v1, v2 in combinations(G.nodes(), 2):
        if G.nodes[v1]["color"] != G.nodes[v2]["color"]:
            if not (G.has_edge(v1, v2) or G.has_edge(v2, v1)):
                return False
    return True