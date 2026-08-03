import matplotlib.pyplot as plt
import networkx as nx
from utils import extract_sigma_from_graph, generate_random_colored_digraph
from best_match_construction import construct_weak_best_match_graph

import unittest

class TestBestMatchConstruction(unittest.TestCase):

  def init(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def testConstructWeakBestMatch(self):
    G = generate_random_colored_digraph(n_nodes=100)

    G_bmg = construct_weak_best_match_graph(G=G)
    
    nx.draw(G, with_labels=True)
    plt.show()

