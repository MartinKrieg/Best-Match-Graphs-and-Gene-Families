from .bic_cherry import check_sicor_inhub_property, construct_base_bic_cherry, constuct_bic_cherry_expanded, constuct_bic_cherry_expanded_edge_restricted, is_graph_properly_colored, is_inhub, is_sicor
from networkx import DiGraph
import unittest

class TestBicCherryConstruction(unittest.TestCase):

  def init(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def test_is_graph_properly_colored(self):
    G = DiGraph()
    G.add_node("1", color='red')
    G.add_node("2", color='blue')
    G.add_node("3", color='red')

    G.add_edge("1", "2")
    G.add_edge("2", "3")

    self.assertTrue(is_graph_properly_colored(G))

    G.add_edge("1", "3") # Add edge between two vertices of the same color

    self.assertFalse(is_graph_properly_colored(G))

  def test_is_sicor(self):
    G = DiGraph()

    G.add_node("1", color='red')
    G.add_node("2", color='blue')
    G.add_node("3", color='red')

    self.assertFalse(is_sicor("1", G))
    self.assertTrue(is_sicor("2", G))
    self.assertFalse(is_sicor("3", G))

  def test_is_inhub(self):
    G = DiGraph()
    G.add_node("1", color='red')
    G.add_node("2", color='blue')
    G.add_node("3", color='red')

    G.add_edge("1", "2")
    G.add_edge("1", "3")
    G.add_edge("3", "2")

    self.assertFalse(is_inhub("1", G))
    self.assertTrue(is_inhub("2", G))
    self.assertFalse(is_inhub("3", G))

  def test_check_sicor_inhub_property_false(self):
    G = DiGraph()
    G.add_node("1", color='red')
    G.add_node("2", color='blue')
    G.add_node("3", color='red')

    G.add_edge("1", "2")
    G.add_edge("1", "3")

    self.assertTrue(is_sicor("2", G))
    self.assertFalse(is_inhub("2", G))
    self.assertFalse(check_sicor_inhub_property(G))

  def test_check_sicor_inhub_property_true(self):
    G = DiGraph()
    G.add_node("1", color='red')
    G.add_node("2", color='blue')
    G.add_node("3", color='red')

    G.add_edge("1", "2")
    G.add_edge("3", "2")

    self.assertTrue(is_sicor("2", G))
    self.assertTrue(is_inhub("2", G))
    self.assertTrue(check_sicor_inhub_property(G))

  def test_construct_base_bic_cherry(self):
      G = DiGraph()
      G.add_node("1", color='red')
      G.add_node("2", color='blue')
      G.add_node("3", color='red')

      G.add_edge("1", "2")
      G.add_edge("3", "2")
      G.add_edge("1", "3")

      N = construct_base_bic_cherry(G)

      # p_12 
      self.assertIn("p12", N.nodes())
      self.assertIn(("rho", "p12"), N.edges())
      self.assertIn(("p12", "1"), N.edges())
      self.assertIn(("p12", "2"), N.edges())

      # p_23
      self.assertIn("p23", N.nodes())
      self.assertIn(("rho", "p23"), N.edges())
      self.assertIn(("p23", "2"), N.edges())
      self.assertIn(("p23", "3"), N.edges())

      self.assertEqual(len(N.nodes()), 6) # root + 2 inner vertices + 3 leaves (original graph)
      self.assertEqual(len(N.edges()), 6)

  def test_constuct_bic_cherry_expanded(self):
      G = DiGraph()
      G.add_node("1", color='red')
      G.add_node("2", color='blue')
      G.add_node("3", color='red')

      G.add_edge("1", "2")
      G.add_edge("3", "2")
      G.add_edge("1", "3")

      N = construct_base_bic_cherry(G)
      N = constuct_bic_cherry_expanded(N, G)

      # q_12'
      self.assertIn("q21", N.nodes())
      self.assertIn(("p23", "q21"), N.edges())
      self.assertIn(("q21", "1"), N.edges())
      self.assertIn(("q21", "2"), N.edges())

      # q_23'
      self.assertIn("q23", N.nodes())
      self.assertIn(("p12", "q23"), N.edges())
      self.assertIn(("q23", "2"), N.edges())
      self.assertIn(("q23", "3"), N.edges())

  def test_constuct_bic_cherry_expanded_edge_restricted(self):
      G = DiGraph()
      G.add_node("1", color='red')
      G.add_node("2", color='blue')
      G.add_node("3", color='red')

      G.add_edge("1", "2")
      G.add_edge("3", "2")
      G.add_edge("2", "3")

      N = construct_base_bic_cherry(G)
      N = constuct_bic_cherry_expanded_edge_restricted(N, G)

      self.assertIn("p12", N.nodes())
      self.assertIn("p23", N.nodes())
      self.assertEqual(len(N.nodes()), 6)
      self.assertEqual(len(N.edges()), 7)
      self.assertNotIn("q21", N.nodes())
      self.assertNotIn("q23", N.nodes())
      self.assertIn(("p12", "p23"), N.edges())

if __name__ == "__main__":
    unittest.main()


# TODO: Extend test cases to dynamic graphs or something else