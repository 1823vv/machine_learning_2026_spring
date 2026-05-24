# Attention as a Graph

---

## 1. Motivation

In the previous lecture, we saw that standard self-attention is **too flexible**:

* Every token can attend to every other token
* This leads to invalid information flow (future leakage, padding noise)

To understand masking more deeply, we need a better mental model.

> Instead of thinking of attention as a matrix computation, we can think of it as a **graph**.

This perspective will make all masking strategies intuitive.

---

## 2. Tokens as Nodes, Attention as Edges

Consider a sequence of tokens:

$$
x_1, x_2, \dots, x_T
$$

We can represent this sequence as a graph:

* Each token $x_i$ is a **node**
* Each attention connection from $i$ to $j$ is a **directed edge**

---

### Graph Interpretation

For a query position $i$, attention computes:

$$
\alpha_{ij} = \text{softmax}_j \left( \frac{q_i \cdot k_j}{\sqrt{d_k}} \right)
$$

Then the output is:

$$
o_i = \sum_{j=1}^{T} \alpha_{ij} v_j
$$

Where:

* $q_i \in \mathbb{R}^{d_k}$ = query vector at position $i$
* $k_j \in \mathbb{R}^{d_k}$ = key vector at position $j$
* $v_j \in \mathbb{R}^{d_v}$ = value vector at position $j$

This means:

> Node $i$ aggregates information from all nodes $j$ through weighted edges $\alpha_{ij}$.

---

## 3. Fully Connected Attention Graph

In standard self-attention:

* Every node connects to every other node
* The graph is **fully connected**

---

### Adjacency View

We can think of the attention logits:

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

as a **weighted adjacency matrix**:

* $S_{ij}$ = strength of edge from node $i$ to node $j$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $S \in \mathbb{R}^{T \times T}$ = attention score matrix

After softmax:

* $\alpha_{ij}$ = normalized edge weights

---

### Key Insight

> Self-attention builds a **dense directed graph** over the sequence.

Every node receives information from all others.

---

## 4. Why Fully Connected Graphs Are Problematic

A fully connected graph assumes:

* All nodes are equally valid sources of information
* All connections are allowed

But real tasks impose constraints:

---

### 4.1 Temporal Constraint

In autoregressive generation:

* Node $i$ should not receive information from nodes $j > i$

But the graph includes edges:

$$
i \rightarrow j \quad \text{for all } j
$$

This introduces **invalid future edges**.

---

### 4.2 Validity Constraint

With padding:

* Some nodes do not represent real tokens

Yet the graph still includes edges:

$$
i \rightarrow \text{PAD}
$$

These edges introduce **noise into the computation**.

---

## 5. Masking as Edge Removal

Masking can now be understood as a simple operation:

> **Remove invalid edges from the attention graph**

---

### Masked Attention

We modify the attention computation:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $V \in \mathbb{R}^{T \times d_v}$ = value matrix
* $M_{ij} = 0$ → keep edge $i \rightarrow j$
* $M_{ij} = -\infty$ → remove edge $i \rightarrow j$

---

### Effect of $-\infty$

After softmax:

$$
\alpha_{ij} = 0
$$

So the edge effectively disappears.

---

## 6. Example: Causal Mask as Graph Pruning

Without masking:

```
1 → 1, 2, 3, 4
2 → 1, 2, 3, 4
3 → 1, 2, 3, 4
4 → 1, 2, 3, 4
```

With causal masking:

```
1 → 1
2 → 1, 2
3 → 1, 2, 3
4 → 1, 2, 3, 4
```

We have removed all edges that point to the future.

---

## 7. Example: Padding Mask as Graph Filtering

Suppose position 4 is padding:

Without masking:

```
All nodes → 4 (PAD)
```

With padding mask:

```
No node → 4
```

The padding node is effectively **disconnected** from the graph.

---

## 8. A Unified Interpretation

All masking strategies can be viewed as:

> **Editing the adjacency matrix of the attention graph**

| Mask Type            | Graph Operation                           |
| -------------------- | ----------------------------------------- |
| Causal mask          | Remove future edges                       |
| Padding mask         | Remove edges to invalid nodes             |
| Cross-attention mask | Remove edges across sequences selectively |

---

## 9. Connection to Transformer Architectures

This graph view applies to all Transformer variants:

* **Decoder-only models** (e.g., GPT):

  * Use causal masks → enforce temporal structure

* **Encoder-only models** (e.g., BERT):

  * Use padding masks → no temporal restriction

* **Encoder-decoder models**:

  * Combine multiple masks → structured information flow
