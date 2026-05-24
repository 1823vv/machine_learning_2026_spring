# NetworkX & Graph Theory

---

## 1. Graph (Undirected)

### Concept

A graph is a pair:

$$
G = (V, E)
$$

* $V$: set of nodes
* $E$: set of edges
* Undirected: $(u, v) = (v, u)$

---

### Mathematical Form

$$
E \subseteq { {u, v} \mid u, v \in V }
$$

---

### Minimal Code

```python
G = nx.Graph()
G.add_edge("A", "B")
```

---

## 2. Directed Graph

### Concept

Edges have direction.

---

### Mathematical Form

$$
E \subseteq V \times V
$$

$$
(u, v) \neq (v, u)
$$

---

### Minimal Code

```python
G = nx.DiGraph()
G.add_edge("A", "B")  # A → B
```

---

## 3. Weighted Graph

### Concept

Each edge has a numerical value.

---

### Mathematical Form

$$
w: E \rightarrow \mathbb{R}
$$

$$
G = (V, E, w)
$$

---

### Minimal Code

```python
G = nx.Graph()
G.add_edge("A", "B", weight=3)
```

---

## 4. Adjacency Matrix

### Concept

Matrix representation of a graph.

---

### Mathematical Form

$$
A_{ij} =
\begin{cases}
w(i,j), & \text{if edge exists} \\
0, & \text{otherwise}
\end{cases}
$$

---

### Minimal Code

```python
A = nx.to_numpy_array(G)
```

---

## 5. Degree

### Concept

Number of connections of a node.

---

### Mathematical Form

$$
\deg(v) = |\{u \in V : (u,v) \in E\}|
$$

---

### Minimal Code

```python
G.degree("A")
```

---

## 6. In-Degree / Out-Degree

### Concept

For directed graphs.

---

### Mathematical Form

$$
\deg_{in}(v) = |{u : (u,v) \in E}|
$$

$$
\deg_{out}(v) = |{u : (v,u) \in E}|
$$

---

### Minimal Code

```python
G.in_degree("A")
G.out_degree("A")
```

---

## 7. Path

### Concept

Sequence of connected nodes.

---

### Mathematical Form

$$
v_1 \rightarrow v_2 \rightarrow \cdots \rightarrow v_k
$$

---

### Minimal Code

```python
nx.shortest_path(G, "A", "B")
```

---

## 8. Distance (Shortest Path Length)

### Concept

Length of shortest path.

---

### Mathematical Form

$$
d(u, v) = \min \text{path length}
$$

---

### Minimal Code

```python
nx.shortest_path_length(G, "A", "B")
```

---

## 9. Density

### Concept

How full the graph is.

---

### Mathematical Form (Undirected)

$$
\text{Density} = \frac{2|E|}{|V|(|V|-1)}
$$

---

### Minimal Code

```python
nx.density(G)
```

---

## 10. Clustering Coefficient

### Concept

How connected neighbors are.

---

### Mathematical Form

$$
C(v) = \frac{2 \cdot |{(u,w)}|}{k_v (k_v - 1)}
$$

---

### Minimal Code

```python
nx.clustering(G, "A")
```

---

## 11. Degree Centrality

### Concept

Node importance by number of connections.

---

### Mathematical Form

$$
C_D(v) = \frac{\deg(v)}{|V| - 1}
$$

---

### Minimal Code

```python
nx.degree_centrality(G)
```

---

## 12. Betweenness Centrality

### Concept

How often a node lies on shortest paths.

---

### Mathematical Form

$$
C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}
$$

---

### Minimal Code

```python
nx.betweenness_centrality(G)
```

---

## 13. Connected Components

### Concept

Groups where every node is reachable.

---

### Mathematical Form

$$
\forall u,v \in C,\ \exists \text{ path between } u,v
$$

---

### Minimal Code

```python
nx.connected_components(G)
```

---

## 14. Complete Graph

### Concept

All nodes are connected.

---

### Mathematical Form

$$
|E| = \frac{n(n-1)}{2}
$$

---

### Minimal Code

```python
nx.complete_graph(5)
```

---

## 15. Random Graph (Erdős–Rényi)

### Concept

Edges appear randomly.

---

### Mathematical Form

$$
P((u,v) \in E) = p
$$

---

### Minimal Code

```python
nx.erdos_renyi_graph(10, 0.2)
```

---

## 16. Graph Laplacian (Important for GNN)

### Concept

Captures graph structure.

---

### Mathematical Form

$$
L = D - A
$$

* $D$: degree matrix
* $A$: adjacency matrix

---

### Minimal Code

```python
L = nx.laplacian_matrix(G).toarray()
```

---

## 17. Neighbors

### Concept

Nodes directly connected to a node.

---

### Mathematical Form

$$
\mathcal{N}(v) = \{u \mid (u,v) \in E\}
$$

---

### Minimal Code

```python
list(G.neighbors("A"))
```
