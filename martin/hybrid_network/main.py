from generate_hybrid_network import generate_hybrid_network
import networkx as nx

if __name__ == "__main__":
    net, root = generate_hybrid_network(num_leaves=3, num_hybridizations=1)

    print(f"Is valid DAG Network: {nx.is_directed_acyclic_graph(net)}")
    print(f"Total nodes in network: {net.number_of_nodes()}")
