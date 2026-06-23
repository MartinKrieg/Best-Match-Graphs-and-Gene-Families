from networkx import DiGraph
from itertools import combinations

def construct_base_bic_cherry(G: DiGraph) -> DiGraph:
    """
    Constructs the base bic cherry from a given directed graph G.
    """

    # Check if G satisfies the sicor-inhub property

    if not check_sicor_inhub_property(G): # This is a prior step, separate construction from preparation/verification of the input graph.
        raise ValueError("The input graph does not satisfy the sicor-inhub property.")

    N = DiGraph()
    N.add_nodes_from(G.nodes(data=True))

    root_node = "rho"
    N.add_node(root_node)
    
    color_diff_tuples = collect_color_diff_tuples(G)
    
    # Construct the base bic cherry network N

    for v1, v2 in color_diff_tuples:
        min_leaf, max_leaf = min(v1, v2), max(v1, v2)
        new_vertex = f"p{min_leaf}{max_leaf}"
        N.add_node(new_vertex)
        N.add_edge(root_node, new_vertex)
        N.add_edge(new_vertex, v1)
        N.add_edge(new_vertex, v2)

    return N

def constuct_bic_cherry_expanded(N: DiGraph, G: DiGraph) -> DiGraph:
    """
    Constructs the expansion of the base bic cherry network
    """

    color_diff_tuples = collect_color_diff_tuples(G)

    for v1, v2 in color_diff_tuples:
        if not G.has_edge(v1, v2):
            N = extend_graph(v1, v2, N, G)
        if not G.has_edge(v2, v1):
            N = extend_graph(v2, v1, N, G)
    
    return N

def extend_graph(x, y, N: DiGraph, G: DiGraph) -> DiGraph:
    
    # Find alternative vertex y' with same color as y
    y_color = G.nodes[y]["color"]
    y_prime = None
    for v in G.nodes():
        if v != y and G.nodes[v]["color"] == y_color:
            y_prime = v
            break

    # y_prime should never be None, because of the sicor-inhub property

    # Add new vertex q_xy'

    new_vertex = f"q{x}{y_prime}" # = q_xy'
    N.add_node(new_vertex) # = q_xy'

    # Define p_xy as lexikographically sorted tuple of x and y, so that we can easily find the corresponding p_xy for the given x and y.
    p_xy = f"p{min(x, y)}{max(x, y)}"

    # Add edges
    ## Add edge p_xy -> q_xy'
    N.add_edge(p_xy, new_vertex)
    ## Add edge q_xy' -> x
    N.add_edge(new_vertex, x)
    ## Add edge q_xy' -> y'
    N.add_edge(new_vertex, y_prime)

    return N

def constuct_bic_cherry_expanded_edge_restricted(N: DiGraph, G: DiGraph) -> DiGraph:
    
    color_diff_tuples = collect_color_diff_tuples(G)

    for v1, v2 in color_diff_tuples:
        if not G.has_edge(v1, v2):
            N = extend_graph_edge_restricted(v1, v2, N, G)
        if not G.has_edge(v2, v1):
            N = extend_graph_edge_restricted(v2, v1, N, G)
    
    return N

def extend_graph_edge_restricted(x, y, N: DiGraph, G: DiGraph) -> DiGraph:
    
    # Find alternative vertex y' with same color as y
    y_color = G.nodes[y]["color"]
    y_prime = None
    for v in G.nodes():
        if v != y and G.nodes[v]["color"] == y_color:
            if G.has_edge(x, v):
                y_prime = v
                break
            else:
                continue
    if y_prime is None:
        raise ValueError(f"No alternative vertex with same color as {y} and edge from {x} to that vertex exists.")

    # y_prime should never be None, because of the sicor-inhub property

    # Define p_xy as lexikographically sorted tuple of x and y, so that we can easily find the corresponding p_xy for the given x and y.
    p_xy = f"p{min(x, y)}{max(x, y)}"

   # Define p_xz (in our variables p_x_y') as lexikographically sorted tuple of x and y_prime
    p_xz = f"p{min(x, y_prime)}{max(x, y_prime)}"

    N.add_edge(p_xy, p_xz)

    return N

def collect_color_diff_tuples(G: DiGraph) -> list:
    """
    Collects unique 2-tuples of vertices with different colors in the graph G.
    """
    color_diff_tuples = []
    
    # combinations(G.nodes(), 2) automatically gives you unique pairs like (1, 2) 
    # and completely skips the reversed (2, 1) pair.
    for v1, v2 in combinations(G.nodes(), 2):
        if G.nodes[v1]["color"] != G.nodes[v2]["color"]:
            color_diff_tuples.append((v1, v2))
            
    return color_diff_tuples

def is_graph_properly_colored(G: DiGraph) -> bool:
    """
    Checks if the given directed graph G is properly colored, i.e., no two adjacent vertices share the same color.
    """
    for v1, v2 in G.edges():
        if G.nodes[v1]["color"] == G.nodes[v2]["color"]:
            return False
    return True

def check_sicor_inhub_property(G: DiGraph) -> bool:
    """
    Checks if the given directed graph G satisfies the sicor-inhub (single colored-inhub) property.
    This property applies iff every sicor is a inhub.
    """

    # Iterating through all vertices and check sicor property first and if that's satisfied, then check inhub property.

    V = G.nodes()

    for vertex in V:
        if is_sicor(vertex, G):
          if is_inhub(vertex, G):
              return True
          
    return False # No vertex exists that satisfies both properties, hence G does not satisfy the sicor-inhub property.

def is_inhub(vertex: str, G: DiGraph) -> bool:
    
    vertex_color = G.nodes[vertex]["color"]

    for v in G.nodes():
        v_color = G.nodes[v]["color"]
      
        if v != vertex and v_color != vertex_color: # For all vertices with different color the edge v -> vertex must exist
            if G.has_edge(v, vertex):
                continue
            else:
                return False
    return True

def is_sicor(vertex: str, G: DiGraph) -> bool:
    """
    Checks if the given directed graph G satisfies the sicor (single colored) property.
    """
    vertex_color = G.nodes[vertex]["color"]

    for v in G.nodes():
        v_color = G.nodes[v]["color"]
        if v != vertex and v_color == vertex_color:
            return False
    return True




