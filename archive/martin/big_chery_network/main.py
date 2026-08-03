import big_cherry

if __name__ == "__main__":
    bmg, geneColors = big_cherry.generate_bmg(num_leaves=3)
    print("Generating Base BIC-Cherry Network...")
    
    base_network, root, base_parents = big_cherry.build_base_bic_cherry(bmg, geneColors)
    big_cherry.visualize_hierarchical_network(
        base_network, "Unmodified_Base_BIC_Cherry_Network", root, geneColors
    )
    print("-> Executing Function 2: Extending Network for Non-Arcs...")
    final_network = big_cherry.extend_bic_cherry_network(base_network, root, base_parents, bmg, geneColors)
    big_cherry.visualize_hierarchical_network(final_network, "Expanded_Phylogenetic_Network", root, geneColors)

    print("-> Done! Both steps visualized successfully.")