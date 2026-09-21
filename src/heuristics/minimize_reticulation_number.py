"""
The reticulation number represents the degree to which the graph G differs from a tree (degree-based perspective)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src import (
    _try_contract,
    bestMatchGraphs,
    bicCherryExplanation,
    buildBaseBigCherry,
    generateHybridNetwork,
    pullingUpEditing,
    removingRedundantVertices,
)
from utils import generateGeneTree

def score(N: nx.DiGraph):
  r = sum(max(0, N.in_degree(v) - 1) for v in N.nodes)

  return r

def cleanup(N: nx.DiGraph):
  # Apply contraction as often as possible
  # Apply removal as often as possible

  while True:
    changed_contract = False
    while _try_contract(N):
      changed_contract = True

    changed_merge = False
    while removingRedundantVertices(N):
      changed_merge = True

    if not changed_contract and not changed_merge:
      break;

  return N

def contract(N: nx.DiGraph):
  for v in N.nodes:
    result = _try_contract(N, v, )
    if result:
      return True
  return False

def main():
  # Create several networks
  for num_species in range(1,10):
    for num_hybrid in range(1,10):
      nxGeneTree, geneTree, root_id, gene_colors = generateGeneTree(species=num_species)
      hybrid_network = generateHybridNetwork(nxGeneTree, root_id, num_hybrid, gene_colors)
      network_bmg = bestMatchGraphs(hybrid_network)
      explaining_network = bicCherryExplanation(network_bmg)

  return False

if __name__.equals(main):
  main()