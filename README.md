# Best Match Graphs and Gene Families

Project for the course *Advanced Methods in Bioinformatics*

Best match graphs are read off simulated gene families and networks are built that explain the same graphs. The project then measures which edit operations make an explaining network more tree-like while leaving the best match graph unchanged.

Group members: Martin Krieg, Dennis Schiese, Ivan Bondarenko


## Scope

`main` implements tasks 0 through 2d.


| Task    | What it does                                                                           |
| ------- | -------------------------------------------------------------------------------------- |
| 0       | Simulate a gene tree and insert hybridizations                                         |
| 1a / 1b | BIC-cherry network and both expansions                                                 |
| 1c      | Tests of the constructions                                                             |
| 1d      | Variant (b) against weak BMGs of hybrid networks; counterexamples in `plots/mismatch/` |
| 2a      | Tree-BMG and the least resolved tree T^{*}                                             |
| 2b      | The same construction applied to the tree-BMGs from 2a                                 |
| 2c      | Pull-up, pull-down, and removal of redundant vertices                                  |
| 2d      | Single moves and short sequences, checked against the strict and the weak BMG          |
| 2e / 2f | Search a path from variant (b) to T^{*}                                                |


## Installation

From the project directory:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Pipeline

Both arguments are required. `--ns` is the number of species, `--nh` the number of hybridizations.

```bash
python main.py --ns 4 --nh 2
```

One run produces, in order:

1. a gene tree (`plots/base_tree/`) and a hybrid network (`plots/hybrid_network/`),
2. the tree-BMG and the least resolved tree,
3. the BIC-cherry base network and both expansions (`plots/big_cherry/`, `plots/least_resolved_tree/`),
4. a sample of the task-2d edit moves on the edge-restricted expansion
5. tests different paths to T^{*}

The console lines report whether T^{*} explains the BMG, how many arcs are missing and whether each expansion explains the graph strictly and weakly. On tree-BMGs, variant (b) explains the graph. Variant (a) does so only when the chosen expansion target happens to be a real arc.

Example output:

```text
-> Least resolved tree explains the BMG: True
-> Missing arcs, i.e. expansions to perform: 6
-> Expansion explains the BMG: strict=False weak=False
-> Edge-restricted expansion explains the BMG: strict=True weak=True
```

Without duplications every species has exactly one gene. No arcs are then missing, both expansions do nothing, and the run is trivial. The simulation therefore uses `dupl_rate=0.7`.

## Tests

```bash
python -m pytest tests
```

238 tests: Among other things they check that both routes to T^{*} agree, that variant (b) explains the tree-BMGs and that a geometrically valid move does not automatically preserve the BMG.

`evaluation/task_1d.py` searches for cases in which variant (b) fails to explain the weak BMG of a hybrid network and writes them to `plots/mismatch/`.

## Layout


| Path                             | Task       | Contents                                                      |
| -------------------------------- | ---------- | ------------------------------------------------------------- |
| `utils/generate_tree.py`         | 0, 2a      | Species tree and gene tree, tree-BMG, weak network BMG        |
| `src/generate_hybrid_network.py` | 0          | Hybridization; the result stays a DAG                         |
| `src/big_cherry_network.py`      | 1a, 1b     | BIC-cherry base and both expansions                           |
| `src/network_best_matches.py`    | 1d, 2b, 2d | Strict and weak BMG, `networkExplainsBmg`                     |
| `src/least_resolved_tree.py`     | 2a         | T^{*}, `explainsBmg`                                          |
| `src/bic_cherry_explanations.py` | 2b         | Tree-BMG, T^{*}, and both variants in one step                |
| `src/editing_operations.py`      | 2c         | Pull-up, pull-down, redundancy, greedy path                   |
| `src/edit_move_checks.py`        | 2d         | The same moves without the BMG filter; strict and weak counts |
| `evaluation/task_1d.py`          | 1d         | Search for counterexamples                                    |
| `plots/mismatch/`                | 1d         | Archived cases in which variant (b) misses the weak BMG       |
| `data/`                          | —          | The two papers and the project description                    |
| `main.py`                        | Demo       | Pipeline from task 0 through the 2d survey on variant (b)     |


The color of a gene is its species. In the code it is stored as the node attribute `color` or as `reconc` on gene trees from AsymmeTree.

## Results in brief

A tree or a network *explains* a colored digraph (G,\sigma) when the best match graph read off it is exactly (G,\sigma).

- On trees, the least resolved tree T^{*} is the unique explanation with the fewest inner edges. It is the target of the later editing tasks.
- Variant (a) expands a missing arc (x,y) towards an arbitrary y' of the same color. That makes y' a best match of x, even when (x,y') is not an arc of G. On tree-BMGs it therefore fails regularly.
- Variant (b) expands only along existing arcs and explains the tree-BMGs from task 2a, both strictly and weakly. It is the starting point for tasks 2c and 2d.
- The same variant (b) does not explain every weak BMG of a hybrid network. The counterexamples are in `plots/mismatch/`.
- A move that leaves the network acyclic and keeps the leaves does not by itself preserve the BMG. Task 2c keeps only steps whose weak BMG is unchanged; task 2d counts how often that fails.



## References

The papers are in `data/`.

1. M. Geiß et al. *Best Match Graphs.* J. Math. Biol. 78:2015–2057, 2019.
2. P. A. Ebert, M. Hellmuth. *Best Matches in the Context of Phylogenetic Networks.* Unpublished manuscript, 2026.

Simulation and the tree-BMG routines come from [AsymmeTree](https://github.com/david-schaller/AsymmeTree). Graphs are stored with [NetworkX](https://networkx.org/).
