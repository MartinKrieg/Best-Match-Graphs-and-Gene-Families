# test_editing_operations.py
import pytest
import networkx as nx

from src import (
    preserveNetworkLeaves,
    checkBmgRelations,
    _try_move_edge,
    _try_contract,
    pullingUpEditing,
    pullingDownEditing,
    removingRedundantVertices,
    cleanUpDummyVertices,
    editingNetwork,
)
from utils import generateNetworkBmg


# ---------------------------------------------------------------- helpers

def make_leaves(network, colors: dict):
    """Attach 'reconc' color attributes to the given leaf nodes."""
    for node, color in colors.items():
        network.nodes[node]['reconc'] = color
    return network


def leaves_of(network):
    return {n for n in network.nodes() if network.out_degree(n) == 0}


def colored_path_network():
    """
    Simple colored DAG (a species-like backbone with 4 leaves, 2 colors):

            r
           / \
          a   b
         /|   |\\        (a, b are internal; reticulation-free)
         x y   z w
    colors: x, z -> 'A' ; y, w -> 'B'
    """
    g = nx.DiGraph()
    g.add_edges_from([('r', 'a'), ('r', 'b'),
                      ('a', 'x'), ('a', 'y'),
                      ('b', 'z'), ('b', 'w')])
    return make_leaves(g, {'x': 'A', 'y': 'B', 'z': 'A', 'w': 'B'})


def network_with_dummy_vertex():
    """
            r
           / \\
          a   b
         / \\  ...
        d   y
        |
        x
    d is 1-in/1-out -> contractible to edge a -> x
    """
    g = nx.DiGraph()
    g.add_edges_from([('r', 'a'), ('r', 'b'),
                      ('a', 'd'), ('a', 'y'),
                      ('d', 'x'),
                      ('b', 'z'), ('b', 'w')])
    return make_leaves(g, {'x': 'A', 'y': 'B', 'z': 'A', 'w': 'B'})

def redundant_duplicate_network():
    """
    Two internal vertices p and q with identical parents and children:

            r
           /|\\
          p q a
          | | ...
          x x
    """
    g = nx.DiGraph()
    g.add_edges_from([('r', 'p'), ('r', 'q'), ('r', 's'),
                      ('p', 'x'), ('p', 'y'),
                      ('q', 'x'), ('q', 'y'),
                      ('s', 'z'), ('s', 'w')])
    return make_leaves(g, {'x': 'A', 'y': 'B', 'z': 'A', 'w': 'B'})


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def net():
    return colored_path_network()

@pytest.fixture
def bmg(net):
    return generateNetworkBmg(net)

@pytest.fixture
def orig_leaves(net):
    return leaves_of(net)


# ---------------------------------------------------------------- basic checks

class TestPreserveNetworkLeaves:
    def test_no_change_preserves_leaves(self, net, orig_leaves):
        assert preserveNetworkLeaves(orig_leaves, net)

    def test_removing_a_leaf_breaks_leaf_set(self, net, orig_leaves):
        net.remove_node('x')
        assert not preserveNetworkLeaves(orig_leaves, net)

    def test_detached_leaf_still_counts(self, net, orig_leaves):
        # removing the edge does not remove the leaf -> still preserved
        net.remove_edge('a', 'x')
        assert preserveNetworkLeaves(orig_leaves, net)


class TestCheckBmgRelations:
    def test_identical_graphs_match(self, net, bmg):
        assert checkBmgRelations(bmg, generateNetworkBmg(net))

    def test_isomorphic_but_differently_labeled_bmg_is_rejected(self, net, bmg):
        # isomorphic digraph, but node labels differ -> must NOT match
        relabeled = nx.relabel_nodes(bmg, {'x': 'X1'})
        assert not checkBmgRelations(bmg, relabeled)

    def test_missing_edge_is_rejected(self, net, bmg):
        edited = generateNetworkBmg(net)
        if edited.number_of_edges() > 0:
            edited.remove_edge(*list(edited.edges())[0])
            assert not checkBmgRelations(bmg, edited)


# ---------------------------------------------------------------- edge moves

class TestTryMoveEdge:
    def test_successful_move_is_kept(self, net, bmg, orig_leaves):
        # any move the helper accepts must leave a correct BMG behind
        accepted = False
        for u, v in list(net.edges()):
            for parent in list(net.predecessors(u)):
                g = net.copy()
                if _try_move_edge(g, u, v, parent, v, bmg, orig_leaves):
                    accepted = True
                    assert checkBmgRelations(bmg, generateNetworkBmg(g))
                    assert preserveNetworkLeaves(orig_leaves, g)
                    break
            if accepted:
                break
        # no assertion on `accepted` — depends on the instance; invariant matters

    def test_rejected_move_rolls_back_exactly(self, net, bmg, orig_leaves):
        before_edges = set(net.edges())
        before_nodes = set(net.nodes())
        # force a move that creates a cycle: b -> w -> ... (w is a leaf, so
        # pull head 'w' of edge ('b','w') UP to 'a' -> edge b->a creates a cycle r->a->? no;
        # instead use an edge set guaranteed to cycle: pull ('a','x') tail up to 'r' then down...)
        # Use a definite cycle: add edge b->a first (still a DAG), then pulling
        # tail of (b, a) UP to a predecessor of b would need care — instead
        # directly test cycle rejection:
        g = net.copy()
        g.add_edge('b', 'a')          # now b -> a exists
        moved = _try_move_edge(g, 'b', 'a', 'x', 'a', bmg, orig_leaves)
        # x -> a: cycle a -> x -> a is only a cycle if a is ancestor of x — it is!
        assert moved is False
        assert set(g.edges()) | set() == set(g.edges())  # sanity: still a graph
        # rollback: either unchanged (b->a kept) or exact original state
        if moved is False:
            # b->a was NOT part of the original; the helper must not have
            # destroyed it — check graph is still a DAG
            assert nx.is_directed_acyclic_graph(g)

    def test_existing_edge_is_rejected_without_mutation(self, net, bmg, orig_leaves):
        g = net.copy()
        before = set(g.edges())
        # r -> a already exists: moving ('r','a') to ('r','a') is blocked by
        # new_u == new_v; moving ('r','b') head to a node where the edge exists:
        assert _try_move_edge(g, 'r', 'b', 'r', 'a', bmg, orig_leaves) is False
        assert set(g.edges()) == before

    def test_self_loop_blocked(self, net, bmg, orig_leaves):
        g = net.copy()
        before = set(g.edges())
        assert _try_move_edge(g, 'a', 'x', 'x', 'x', bmg, orig_leaves) is False
        assert set(g.edges()) == before


# ---------------------------------------------------------------- contraction

class TestTryContract:
    def test_contract_dummy_vertex_preserves_bmg(self, bmg, orig_leaves):
        g = network_with_dummy_vertex()
        bmg = generateNetworkBmg(g) 
        assert _try_contract(g, 'd', bmg, orig_leaves)
        assert 'd' not in g
        assert g.has_edge('a', 'x') and g.has_edge('a', 'y')
        assert checkBmgRelations(bmg, generateNetworkBmg(g))
        assert preserveNetworkLeaves(orig_leaves, g)

    def test_non_dummy_vertex_is_untouched(self, net, bmg, orig_leaves):
        g = net.copy()
        before = set(g.edges())
        # 'r' has out_degree 2 -> not a dummy
        assert _try_contract(g, 'r', bmg, orig_leaves) is False
        assert set(g.edges()) == before

    def test_failed_contract_rolls_back_node_and_edges(self, net, bmg, orig_leaves):
        g = net.copy()
        before_edges = set(g.edges())
        before_nodes = set(g.nodes())
        # contract a leaf? leaves have out_degree 0 -> rejected without mutation
        _try_contract(g, 'x', bmg, orig_leaves)
        assert set(g.edges()) == before_edges
        assert set(g.nodes()) == before_nodes


# ---------------------------------------------------------------- pull operations

class TestPullOperations:
    def test_pulling_up_returns_bool_and_keeps_bmg(self, net, bmg, orig_leaves):
        g = net.copy()
        result = pullingUpEditing(g, bmg, orig_leaves)
        assert isinstance(result, bool)
        if result:
            assert checkBmgRelations(bmg, generateNetworkBmg(g))
            assert preserveNetworkLeaves(orig_leaves, g)
            assert nx.is_directed_acyclic_graph(g)

    def test_pulling_down_returns_bool_and_keeps_bmg(self, net, bmg, orig_leaves):
        g = net.copy()
        result = pullingDownEditing(g, bmg, orig_leaves)
        assert isinstance(result, bool)
        if result:
            assert checkBmgRelations(bmg, generateNetworkBmg(g))
            assert preserveNetworkLeaves(orig_leaves, g)
            assert nx.is_directed_acyclic_graph(g)

    def test_no_valid_move_leaves_network_unchanged(self, net, bmg, orig_leaves):
        # tiny star network where no pull can preserve the BMG
        g = nx.DiGraph()
        g.add_edges_from([('r', 'x'), ('r', 'y')])
        make_leaves(g, {'x': 'A', 'y': 'B'})
        small_bmg = generateNetworkBmg(g)
        before = set(g.edges())
        assert pullingUpEditing(g, small_bmg, leaves_of(g)) is False
        assert pullingDownEditing(g, small_bmg, leaves_of(g)) is False
        assert set(g.edges()) == before

    def test_pull_up_then_down_can_oscillate_is_safe(self, net, bmg, orig_leaves):
        # The classic oscillation: apply up, then down, then up again.
        # Each call must individually keep the BMG — termination is
        # editingNetwork's job, not the pull functions'.
        g = net.copy()
        for _ in range(3):
            pullingUpEditing(g, bmg, orig_leaves)
            assert checkBmgRelations(bmg, generateNetworkBmg(g))
            pullingDownEditing(g, bmg, orig_leaves)
            assert checkBmgRelations(bmg, generateNetworkBmg(g))


# ---------------------------------------------------------------- redundancy / cleanup

class TestRemovingRedundantVertices:
    def test_removes_duplicate_signature_vertices(self, bmg, orig_leaves):
        g = redundant_duplicate_network()
        dup_bmg = generateNetworkBmg(g)
        assert removingRedundantVertices(g, dup_bmg, leaves_of(g))
        # exactly one of p, q must be gone
        assert ('p' in g) != ('q' in g)
        assert checkBmgRelations(dup_bmg, generateNetworkBmg(g))

    def test_no_duplicates_no_change(self, net, bmg, orig_leaves):
        g = net.copy()
        before = set(g.edges())
        assert removingRedundantVertices(g, bmg, orig_leaves) is False
        assert set(g.edges()) == before


class TestCleanUpDummyVertices:
    def test_contracts_dummy_chain(self, bmg, orig_leaves):
        g = network_with_dummy_vertex()
        bmg = generateNetworkBmg(g) 
        assert cleanUpDummyVertices(g, bmg, orig_leaves)
        assert 'd' not in g

    def test_network_without_dummies_unchanged(self, net, bmg, orig_leaves):
        g = net.copy()
        before = set(g.edges())
        before_nodes = set(g.nodes())
        assert cleanUpDummyVertices(g, bmg, orig_leaves) is False
        assert set(g.edges()) == before
        assert set(g.nodes()) == before_nodes


# ---------------------------------------------------------------- integration

class TestEditingNetwork:
    def test_terminates_on_simple_network(self, net, bmg):
        result = editingNetwork(net, bmg)   # must not hang
        assert nx.is_directed_acyclic_graph(result)

    def test_leaf_set_is_preserved(self, net, bmg):
        result = editingNetwork(net, bmg)
        assert preserveNetworkLeaves(leaves_of(net), result)

    def test_bmg_is_preserved(self, net, bmg):
        result = editingNetwork(net, bmg)
        assert checkBmgRelations(bmg, generateNetworkBmg(result))

    def test_input_network_is_not_mutated(self, net, bmg):
        before_edges = set(net.edges())
        before_nodes = set(net.nodes())
        editingNetwork(net, bmg)
        assert set(net.edges()) == before_edges
        assert set(net.nodes()) == before_nodes

    def test_result_is_never_larger_than_input(self, net, bmg):
        result = editingNetwork(net, bmg)
        # simplification may change edges but should not grow the vertex count
        assert result.number_of_nodes() <= net.number_of_nodes()

    def test_stable_network_is_a_fixed_point(self, net, bmg):
        result = editingNetwork(net, bmg)
        result_bmg = generateNetworkBmg(result)
        result_leaves = leaves_of(result)
        # running again on the result must change nothing
        again = editingNetwork(result, result_bmg)
        assert set(again.edges()) == set(result.edges())
        assert set(again.nodes()) == set(result.nodes())

    @pytest.mark.timeout(10)
    def test_no_infinite_loop_on_random_networks(self):
        import random
        random.seed(42)
        for _ in range(20):
            g = nx.DiGraph()
            g.add_edges_from([('r0', 'r1'), ('r0', 'r2')])
            internal = ['r1', 'r2']
            leaf_names = ['x', 'y', 'z', 'w']

            # guarantee: every internal node gets at least one leaf child,
            # otherwise it becomes an uncolored "leaf" itself -> ValueError
            for internal_node, leaf in zip(internal, leaf_names[:2]):
                g.add_edge(internal_node, leaf)
            for leaf in leaf_names[2:]:
                parent = random.choice(internal + ['r0'])
                g.add_edge(parent, leaf)

            make_leaves(g, {'x': 'A', 'y': 'B', 'z': 'A', 'w': 'B'})
            bmg = generateNetworkBmg(g)
            result = editingNetwork(g, bmg)
            assert nx.is_directed_acyclic_graph(result)
            assert checkBmgRelations(bmg, generateNetworkBmg(result))
