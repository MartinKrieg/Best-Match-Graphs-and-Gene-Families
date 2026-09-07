"""
So far: The goal is to find a way to convert a network into a tree, i.e., a graph with each vertex having in-degree = 1 (except for the root).

The only edit operation that decreases the in-degree of a node is "removing redundant vertices". All others, namely "pulling up/down" (decreases out-degree or changes the 
vertices that are connected to a node) and "contracting" (removes vertices with in-degree = 1 and out-degree = 1) do not decrease the in-degree of a node.
Thus, we must use the latter operations to change the network in a way that allows us to remove redundant vertices.

As soon as we have a tree, removing redundant vertices will convert it into a LRT, by definition.

---

What can Pulling up/down do? How do they restructure the network? Can we derive a strategy to use them to prepare the network for removing redundant vertices?

For an edge (u,v) (check for one example, needs to be checked if it can be generalized):

Pulling up (tail, u):
    - In-degree of u is unchanged
    - Out-degree of u is decreased by 1, but the out-degree of p (parent of u) is increased by 1.
    - In-degree of v is unchanged, but the edges differ: (u,v) is replaced by (p,v), whereas p is a parent of u.
    - Out-degree of v is unchanged
    => Contraction of u is maybe possible

Pulling up (head, v):
    - In-degree of u is unchanged
    - Out-degree of u is unchanged, but the edges differ: (u,v) is replaced by (u,p), whereas p is a parent of v.
    - In-degree of v is decreased by 1, but the in-degree of p is increased by 1.
    - Out-degree of v is unchanged
    => Contraction of v is maybe possible
    
Pulling down (tail, u):
    - In-degree of u is unchanged
    - Out-degree of u is decreased by 1, but the out-degree of c (child of u) is increased by 1.
    - In-degree of v is unchanged, but the edges differ: (u,v) is replaced by (c,v), whereas c is a child of u.
    - Out-degree of v is unchanged
    => Contraction of u is maybe possible

Pulling down (head, v):
    - In-degree of u is unchanged
    - Out-degree of u is unchanged, but the edges differ: (u,v) is replaced by (u,c), whereas c is a child of v.
    - In-degree of v is decreased by 1, but the in-degree of c is increased by 1.
    - Out-degree of v is unchanged
    => Contraction of v is maybe possible

---

Ideas:
- Look for vertices with in-degree > 1
"""



from src.editing_operations import removingRedundantVertices

# A lot to think about ;-)