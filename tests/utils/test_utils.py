import pytest
import networkx as nx
from utils.generate_bmg import generateBmg
from utils.generate_gene_tree import generateGeneTree
from hypothesis import given, settings, strategies as st

@pytest.fixture
def valid_gene_tree():
    None

@settings(deadline=None)
@given(
    num_leaves=st.integers(min_value=1, max_value=10)
)
def test_generate_valid_gene_tree(num_leaves):
    gene_tree, _, root_id, gene_colors = generateGeneTree(num_leaves)
    assert isinstance(gene_tree, nx.DiGraph)
    assert len(gene_tree.nodes()) == num_leaves + 1