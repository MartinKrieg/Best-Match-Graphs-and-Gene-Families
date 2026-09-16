import networkx as nx
from utils import generateNetworkBmg

def preserveNetworkLeaves(original_leaves: set, network: nx.DiGraph) -> bool:
    '''
    Checks if the edit operation preserves the exact leaf set of the network.
    '''
    current_leaves = {node for node in network.nodes() if network.out_degree(node) == 0}
    return original_leaves == current_leaves

def preservesPhylogeneticNetwork(original_leaves: set, network: nx.DiGraph) -> bool:
    '''
    Checks the three properties an edit has to keep for the result to still be
    a phylogenetic network: acyclic, same leaf set, and a single root.

    Without the root condition an edit can detach a subnetwork and leave a
    second source behind. src.network_best_matches then refuses to compute best
    matches at all, so "the edited network explains the same BMG" would be a
    claim about a graph the definition does not apply to.
    '''
    if not nx.is_directed_acyclic_graph(network):
        return False
    if not preserveNetworkLeaves(original_leaves, network):
        return False
    roots = [node for node in network.nodes() if network.in_degree(node) == 0]
    return len(roots) == 1

def checkBmgRelations(originBmg: nx.DiGraph, editBmg: nx.DiGraph) -> bool:
    '''
    Checks if the best match relations are exactly maintained.
    Assumes BMG nodes and edges must perfectly match.
    '''
    return (set(originBmg.nodes()) == set(editBmg.nodes()) and set(originBmg.edges()) == set(editBmg.edges()))

def _tryMoveEdge(network: nx.DiGraph, u, v, new_u, new_v, originBmg, original_leaves, checkBMG: bool = True) -> bool:
    """
    Helper function to safely test an edge move.
    Returns True if the move is successful and kept, False if reverted.

    checkBMG=False skips the BMG comparison, which makes originBmg unused and
    lets a caller enumerate the moves that merely leave a phylogenetic network,
    without filtering them. That is what task 2d needs in order to measure how
    many of them happen to preserve the BMG.
    """
    if new_u == new_v or network.has_edge(new_u, new_v):
        return False
        
    network.remove_edge(u, v)
    network.add_edge(new_u, new_v)

    # 1. Geometry check: acyclic, same leaves, single root
    if not preservesPhylogeneticNetwork(original_leaves, network):
        network.remove_edge(new_u, new_v)
        network.add_edge(u, v)
        return False

    # 2. BMG check
    if not checkBMG or checkBmgRelations(originBmg, generateNetworkBmg(network)):
        return True

    network.remove_edge(new_u, new_v)
    network.add_edge(u, v)
    return False

def _tryContract(network: nx.DiGraph, node, originBmg, original_leaves, checkBMG: bool = True) -> bool:
    '''
    Try to contract a 1-in/1-out node; revert if BMG or geometry break.
    '''
    if network.in_degree(node) != 1 or network.out_degree(node) != 1:
        return False
    parent = next(network.predecessors(node))
    child = next(network.successors(node))
    if network.has_edge(parent, child):
        return False

    attributes = dict(network.nodes[node])
    network.add_edge(parent, child)
    network.remove_node(node)

    if preservesPhylogeneticNetwork(original_leaves, network):
        if not checkBMG or checkBmgRelations(originBmg, generateNetworkBmg(network)):
            return True

    network.remove_edge(parent, child)
    network.add_node(node, **attributes)
    network.add_edge(parent, node)
    network.add_edge(node, child)
    return False


def _networkSignature(network: nx.DiGraph) -> frozenset:
    '''Canonical state hash: the edge set (nodes are implied by edges plus leaves).'''
    return frozenset(network.edges())

def _restoreInPlace(network: nx.DiGraph, backup: nx.DiGraph) -> None:
    '''Rolls network back to backup, keeping the object identity of network.'''
    network.clear()
    network.add_nodes_from(backup.nodes(data=True))
    network.add_edges_from(backup.edges(data=True))

def pullingUpEditing(network: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set, checkBMG: bool = True) -> bool:
    '''
    Attempts to pull the source or target of an edge UP to a parent node.
    Returns True if at least one edit was made.
    '''
    edges = list(network.edges())
    for u, v in edges:
        # Try pulling tail (u) UP to parents of u
        for parent_u in list(network.predecessors(u)):
            if _tryMoveEdge(network, u, v, parent_u, v, originBmg, original_leaves, checkBMG):
                return True
                
        # Try pulling head (v) UP to parents of v (excluding u to prevent trivial loops)
        for parent_v in list(network.predecessors(v)):
            if parent_v != u:
                if _tryMoveEdge(network, u, v, u, parent_v, originBmg, original_leaves, checkBMG):
                    return True
    return False

def pullingDownEditing(network: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set, checkBMG: bool = True) -> bool:
    '''
    Attempts to pull the source or target of an edge DOWN to a child node.
    Returns True if at least one edit was made.
    '''
    edges = list(network.edges())
    for u, v in edges:
        # Try pulling tail (u) DOWN to children of u (excluding v)
        for child_u in list(network.successors(u)):
            if child_u != v:
                if _tryMoveEdge(network, u, v, child_u, v, originBmg, original_leaves, checkBMG):
                    return True
                    
        # Try pulling head (v) DOWN to children of v
        for child_v in list(network.successors(v)):
            if _tryMoveEdge(network, u, v, u, child_v, originBmg, original_leaves, checkBMG):
                return True
    return False

def redundantVertexGroups(network: nx.DiGraph) -> list:
    '''
    Groups the inner vertices that share both their parents and their children.

    Every group of two or more is a set of interchangeable vertices, so all but
    one of them are candidates for removal.
    '''
    signatures = {}
    for node in network.nodes():
        if network.in_degree(node) == 0 or network.out_degree(node) == 0:
            continue
        sig = (frozenset(network.predecessors(node)),
               frozenset(network.successors(node)))
        signatures.setdefault(sig, []).append(node)

    return [nodes for nodes in signatures.values() if len(nodes) > 1]

def _tryRemoveRedundant(network: nx.DiGraph, node, originBmg, original_leaves, checkBMG: bool = True) -> bool:
    '''
    Try to drop a vertex that another vertex duplicates; revert if BMG or
    geometry break. The caller decides that the vertex really is redundant.
    '''
    backup = network.copy()
    network.remove_node(node)

    if preservesPhylogeneticNetwork(original_leaves, network) \
       and (not checkBMG or checkBmgRelations(originBmg, generateNetworkBmg(network))):
        return True

    _restoreInPlace(network, backup)
    return False

def removingRedundantVertices(network: nx.DiGraph, originBmg: nx.DiGraph, original_leaves: set, checkBMG: bool = True) -> bool:
    '''
    Removes ONE vertex sharing the exact same parents and children as another,
    if the BMG and leaf set are preserved. Returns True if a removal was made.
    '''
    for nodes in redundantVertexGroups(network):
        # attempt only the first duplicate candidate; recompute next call
        if _tryRemoveRedundant(network, nodes[1], originBmg, original_leaves, checkBMG):
            return True

    return False


def cleanUpDummyVertices(network: nx.DiGraph,originBmg: nx.DiGraph, original_leaves, checkBMG: bool = True) -> bool:
    '''
    Doing edge contraction.
    Helper to bypass and remove vertices with in_degree == 1 and out_degree == 1,
    which are often left behind by pulling operations.
    '''
    for node in list(network.nodes()):
        if _tryContract(network, node, originBmg, original_leaves, checkBMG):
            return True
    return False

EDIT_OPERATIONS = (
    removingRedundantVertices,
    cleanUpDummyVertices,
    pullingUpEditing,
    pullingDownEditing,
)

def iterEditingSteps(network: nx.DiGraph, originBmg: nx.DiGraph, checkBMG: bool = True):
    '''
    Yields the edited network after every committed operation, in order.

    The first item is the network after one move, the k-th after k moves, so a
    caller can take exactly as many steps as it wants to inspect. A state that
    was already visited is reverted instead of yielded, which keeps the walk
    from cycling.

    Note that this is one greedy path: every operation commits the first
    candidate it accepts. It is not an enumeration of the moves reachable in k
    steps, which is what src.edit_move_checks provides for task 2d.
    '''
    editedNetwork = network.copy()
    original_leaves = {node for node in editedNetwork.nodes()
                       if editedNetwork.out_degree(node) == 0}

    visited = {_networkSignature(editedNetwork)}

    stable = False
    while not stable:
        stable = True

        for op in EDIT_OPERATIONS:
            backup = editedNetwork.copy()

            if not op(editedNetwork, originBmg, original_leaves, checkBMG):
                continue

            sig = _networkSignature(editedNetwork)
            if sig in visited:
                _restoreInPlace(editedNetwork, backup)
                continue

            visited.add(sig)
            yield editedNetwork.copy()
            stable = False
            break

def editingNetwork(network: nx.DiGraph, originBmg: nx.DiGraph, checkBMG: bool = True, numberMoves: int | None = None) -> nx.DiGraph:
    '''
    Applies simplification operations to the network.

    checkBMG=True  : moves are only committed if the BMG is preserved.
    checkBMG=False : moves are applied blindly (only the geometry is checked).

    numberMoves    : max number of committed operations.
                     None (default) -> run until stable (old behaviour).
    '''
    editedNetwork = network.copy()
    if numberMoves is not None and numberMoves <= 0:
        return editedNetwork

    for movesDone, step in enumerate(iterEditingSteps(network, originBmg, checkBMG), start=1):
        editedNetwork = step
        if numberMoves is not None and movesDone >= numberMoves:
            break

    return editedNetwork

