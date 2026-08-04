import networkx as nx
from utils import extract_sigma_from_graph

import unittest

class TestUtils(unittest.TestCase):

  def init(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def testExtractSigmaFromGraph(self):
    G = nx.DiGraph()
    G.add_node("1", color='red')
    G.add_node("2", color='blue')
    G.add_node("3", color='red')

    G.add_edge("1", "2")
    G.add_edge("2", "3")

    sigma = extract_sigma_from_graph(G)

    self.assertEqual({
      "1": "red",
      "2": "blue",
      "3": "red"
    }, sigma)

    