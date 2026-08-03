import pytest
import networkx as nx
from src.generate_hybrid_network import generateHybridNetwork


@pytest.fixture
def simple_network():
    """Return a minimal sample network representation for tests."""
    G = nx.DiGraph()
    G.add_nodes_from(["1", "2", "3"])
    G.add_edges_from([("1", "2"), ("1", "3")])
    return G


@pytest.fixture
def medium_network():
	"""Return a minimal sample network representation for tests."""
	G = nx.DiGraph()
	G.add_nodes_from(["1", "2", "3", "4", "5", "6", "7"])
	G.add_edges_from([("1", "2"), ("1", "3"), ("2", "4"), ("2", "5"), ("3", "6"), ("3", "7")])
	


class TestBicCherryNetwork:
    """Tests for bic cherry network functionality."""

    def test_simple_network_fixture(self, simple_network):
        network, root_id = generateHybridNetwork(
            simple_network,
            rootID="1",
            num_hybridizations=1,
            geneColors=None,
        )
        assert root_id == "1"
        assert len(network.edges()) >= len(simple_network.edges())

