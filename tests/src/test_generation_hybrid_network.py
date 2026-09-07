import pytest
import networkx as nx
from src.generate_hybrid_network import generateHybridNetwork
from utils import generateGeneTree
from hypothesis import given, settings, strategies as st


@pytest.fixture
def simple_network():
    """Return a minimal sample network representation for tests."""
    G = nx.DiGraph()
    G.add_nodes_from(["1", "2", "3"])
    G.add_edges_from([("1", "2"), ("1", "3")])
    return G

@pytest.fixture
def network_too_few_edges():
    """Return a network with fewer than two edges for testing."""
    G = nx.DiGraph()
    G.add_nodes_from(["1", "2"])
    G.add_edges_from([("1", "2")])  # Only one edge
    return G


class TestGenerationHybridNetwork:
    """Tests for generation of hybrid networks."""

    def test_simple_network_fixture(self, simple_network):
        network, root_id = generateHybridNetwork(
            simple_network,
            rootID="1",
            num_hybridizations=1,
            geneColors={"1": "red", "2": "blue", "3": "green"},
        )

        assert len(network.edges()) == len(simple_network.edges()) + 3

    def test_less_than_two_edges_network(self, network_too_few_edges):
        # Assert ValueError
        with pytest.raises(ValueError):
            generateHybridNetwork(
                network_too_few_edges,
                rootID="1",
                num_hybridizations=1,
                geneColors={"1": "red", "2": "blue"},
            )

    @settings(deadline=None)
    @given(
        num_hybridizations=st.integers(min_value=1, max_value=10),
        nodes=st.integers(min_value=3, max_value=100)
    )
    def test_hybridization_increases_edges_invariant(self, num_hybridizations, nodes):
        gene_tree = generateGeneTree(numSpecies=nodes)
        nx_gene_tree, _, root_id, gene_colors = gene_tree
        edges_before = len(nx_gene_tree.edges())
        hybridized_network, _ = generateHybridNetwork(
            nx_gene_tree,
            rootID=root_id,
            num_hybridizations=num_hybridizations,
            geneColors=gene_colors
        )
        edges_after = len(hybridized_network.edges())
        assert edges_after == edges_before + 3 * num_hybridizations

    @settings(deadline=None)
    @given(
        num_hybridizations=st.integers(min_value=1, max_value=10),
        nodes=st.integers(min_value=3, max_value=100)
    )
    def test_hybridization_increases_vertices_invariant(self, num_hybridizations, nodes):
        gene_tree = generateGeneTree(numSpecies=nodes)
        nx_gene_tree, _, root_id, gene_colors = gene_tree
        vertices_before = len(nx_gene_tree.nodes())
        hybridized_network, _ = generateHybridNetwork(
            nx_gene_tree,
            rootID=root_id,
            num_hybridizations=num_hybridizations,
            geneColors=gene_colors
        )
        vertices_after = len(hybridized_network.nodes())
        assert vertices_after == vertices_before + 2 * num_hybridizations