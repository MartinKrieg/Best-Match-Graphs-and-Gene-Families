import networkx as nx
from utils import generateNetworkBmg

def preserveNetworkLeaves(original_leaves: set, network: nx.DiGraph) -> bool:
    '''
    Checks if the edit operation preserves the exact leaf set of the network.
    '''
    current_leaves = {node for node in network.nodes() if network.out_degree(node) == 0}
    return original_leaves == current_leaves

def checkBmgRelations(originBmg: nx.DiGraph, editBmg: nx.DiGraph) -> bool:
    '''
    Checks if the best match relations are exactly maintained.
    Assumes BMG nodes and edges must perfectly match.
    '''
    return (set(originBmg.nodes()) == set(editBmg.nodes()) and set(originBmg.edges()) == set(editBmg.edges()))

def _try_move_edge(network: nx.DiGraph, u, v, new_u, new_v, originBmg, original_leaves) -> bool:
    """
    Helper function to safely test an edge move.
    Returns True if the move is successful and kept, False if reverted.
    """
    if new_u == new_v or network.has_edge(new_u, new_v):
        return False
        
    network.remove_edge(u, v)
    network.add_edge(new_u, new_v) # Trivially would remove the edge just to add it back

    # 1. Cycle check
    if not nx.is_directed_acyclic_graph(network):
        network.remove_edge(new_u, new_v)
        network.add_edge(u, v)
        return False
    
    # 2. Leaves check
    if not preserveNetworkLeaves(original_leaves, network):
        network.remove_edge(new_u, new_v)
        network.add_edge(u, v)
        return False
        
    # 3. BMG check
    newBmg = generateNetworkBmg(network)
    if checkBmgRelations(originBmg, newBmg):
        return True
    
    network.remove_edge(new_u, new_v)
    network.add_edge(u, v)
    return False

def _try_contract(network: nx.DiGraph, node, originBmg, original_leaves) -> bool:
    '''
    Try to contract a 1-in/1-out node; revert if BMG or leaves break.
    '''
    if network.in_degree(node) != 1 or network.out_degree(node) != 1:
        return False
    parent = next(network.predecessors(node))
    child = next(network.successors(node))
    if network.has_edge(parent, child):
        return False

    network.add_edge(parent, child)
    network.remove_node(node)

    if (nx.is_directed_acyclic_graph(network)
            and preserveNetworkLeaves(original_leaves, network)
            and checkBmgRelations(originBmg, generateNetworkBmg(network))):
        return True

    network.remove_edge(parent, child)
    network.add_edge(parent, node)
    network.add_edge(node, child)
    return False


def _network_signature(network: nx.DiGraph) -> frozenset:
    '''Canonical state hash: the edge set (nodes are implied by edges plus leaves).'''
    return frozenset(network.edges())

def pullingUpEditing(network: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set) -> bool:
    '''
    Attempts to pull the source or target of an edge UP to a parent node.
    Returns True if at least one edit was made.
    '''
    edges = list(network.edges())
    for u, v in edges:
        # Try pulling tail (u) UP to parents of u
        for parent_u in list(network.predecessors(u)):
            if _try_move_edge(network, u, v, parent_u, v, originBmg, original_leaves):
                return True
                
        # Try pulling head (v) UP to parents of v (excluding u to prevent trivial loops)
        for parent_v in list(network.predecessors(v)):
            if parent_v != u:
                if _try_move_edge(network, u, v, u, parent_v, originBmg, original_leaves):
                    return True
    return False

def pullingDownEditing(network: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set) -> bool:
    '''
    Attempts to pull the source or target of an edge DOWN to a child node.
    Returns True if at least one edit was made.
    '''
    edges = list(network.edges())
    for u, v in edges:
        # Try pulling tail (u) DOWN to children of u (excluding v)
        for child_u in list(network.successors(u)):
            if child_u != v:
                if _try_move_edge(network, u, v, child_u, v, originBmg, original_leaves):
                    return True
                    
        # Try pulling head (v) DOWN to children of v
        for child_v in list(network.successors(v)):
            if _try_move_edge(network, u, v, u, child_v, originBmg, original_leaves):
                return True
    return False

def removingRedundantVertices(network: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set) -> bool:
    '''
    Removes ONE vertex sharing the exact same parents and children as another,
    if the BMG and leaf set are preserved. Returns True if a removal was made.
    '''
    internal_nodes = [n for n in network.nodes() 
                      if network.in_degree(n) > 0 and network.out_degree(n) > 0]

    signatures = {}
    for node in internal_nodes:
        sig = (frozenset(network.predecessors(node)),
               frozenset(network.successors(node)))
        signatures.setdefault(sig, []).append(node)

    for sig, nodes in signatures.items():
        if len(nodes) <= 1:
            continue
        rn = nodes[1]  # attempt only the first duplicate candidate; recompute next call
        backup = network.copy()
        network.remove_node(rn)

        # If nodes[1] fails, nodes[n] will fail, too and vice versa if it works.
        if not (preserveNetworkLeaves(original_leaves, network)
                and checkBmgRelations(originBmg, generateNetworkBmg(network))):
            network.clear()  # revert, in-place
            network.add_nodes_from(backup.nodes(data=True))
            network.add_edges_from(backup.edges(data=True))
        else:
            return True  # one removal committed per call
    return False


def cleanUpDummyVertices(network: nx.DiGraph,originBmg: nx.DiGraph, original_leaves ) -> bool:
    '''
    Doing edge contraction.
    Helper to bypass and remove vertices with in_degree == 1 and out_degree == 1,
    which are often left behind by pulling operations.
    '''
    changed = False
    for node in list(network.nodes()):
        if _try_contract(network, node, originBmg, original_leaves):
            changed = True
    return changed

def editingNetwork(network: nx.DiGraph, originBmg: nx.DiGraph) -> nx.DiGraph:
    '''
    Main function: iteratively applies simplification operations until stable.
    Operations are applied one at a time. A change producing an already-visited
    state is reverted, and other operations are still attempted.
    '''
    editedNetwork = network.copy()
    original_leaves = {node for node in editedNetwork.nodes()
                       if editedNetwork.out_degree(node) == 0}

    visited = {_network_signature(editedNetwork)}

    operations = [
            removingRedundantVertices,
            cleanUpDummyVertices,
            pullingUpEditing,
            pullingDownEditing,
        ]

    stable = False
    while not stable:
        stable = True

        for op in operations:
            backup = editedNetwork.copy()
            changed = op(editedNetwork, originBmg, original_leaves)

            if not changed:
                continue

            sig = _network_signature(editedNetwork)
            if sig in visited:
                editedNetwork.clear()
                editedNetwork.add_nodes_from(backup.nodes(data=True))
                editedNetwork.add_edges_from(backup.edges(data=True))
                continue

            visited.add(sig)
            stable = False
            break

    return editedNetwork
