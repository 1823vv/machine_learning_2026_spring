# Padding Mask and Variable-Length Sequences

---

## 1. Motivation

In real-world applications, sequences rarely have the same length.

Examples:

* "Hello." → very short
* "The quick brown fox jumps over the lazy dog." → longer
* Documents or paragraphs → much longer

However, neural networks require **fixed-size tensors** for efficient computation.

---

### The Practical Solution

We pad shorter sequences to match the longest one in the batch.

Example:

```
Sequence 1: [The] [cat]  [sat]              → length 3
Sequence 2: [I]   [love] [Shanghai] [today] → length 4

After padding:

Sequence 1: [The] [cat]  [sat]        [PAD]
Sequence 2: [I]   [love] [Shanghai]   [today]
```

---

### The Problem

Padding introduces tokens that:

* Do not carry meaning
* Should not influence the model

---

## 2. Attention with Padding (The Issue)

Recall attention:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $V \in \mathbb{R}^{T \times d_v}$ = value matrix

For a query at position $i$:

$$
o_i = \sum_{j=1}^{T} \alpha_{ij} v_j
$$

If position $j$ is `[PAD]`, then:

* $v_j$ is meaningless
* $\alpha_{ij}$ may still be non-zero

---

### Consequences

Without masking:

* The model attends to `[PAD]` tokens
* Outputs become contaminated by noise
* Gradients flow through invalid positions

---

## 3. Padding Mask: Defining Valid Tokens

We introduce a **padding mask** to indicate which tokens are valid.

---

### Definition

For a sequence of length $T$, define:

$$
m_j =
\begin{cases}
1 & \text{if position } j \text{ is a real token} \\
0 & \text{if position } j \text{ is padding}
\end{cases}
$$

---

### Example

```
Tokens:        [The] [cat] [sat] [PAD]
Padding mask:    1     1     1     0
```

---

## 4. Applying the Padding Mask

We convert the binary mask into a matrix $M$:

$$
M_{ij} =
\begin{cases}
0 & m_j = 1 \\
-\infty & m_j = 0
\end{cases}
$$

Notice:

* The mask depends only on **key position $j$**
* All queries share the same masking over padding

---

### Masked Attention

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $V \in \mathbb{R}^{T \times d_v}$ = value matrix

This ensures:

$$
\alpha_{ij} = 0 \quad \text{if } j \text{ is padding}
$$

---

## 5. Graph Interpretation

From the graph perspective:

* Nodes = tokens
* Edges = attention connections

---

### Without Padding Mask

```
All nodes → PAD
```

The padding node is connected and influences the graph.

---

### With Padding Mask

```
No node → PAD
```

> All edges pointing to padding nodes are removed.

---

## 6. Important Detail: Masking Keys, Not Queries

Padding masks are applied to **keys (and values)**, not queries.

Why?

* We want to prevent attention **to** padding
* Queries at padding positions may still exist during computation

---

## 7. Batch Computation and Efficiency

Padding enables:

* Parallel computation across sequences
* Efficient GPU utilization

Without padding:

* Each sequence would require separate processing
* Computation would be slow and inefficient

---

### Trade-off

Padding introduces:

* Extra tokens
* Extra computation

Padding masks ensure:

> These extra tokens do not affect learning.

---

## 8. Where It Is Used

Padding masks are used in:

* **Encoder-only models** (e.g., BERT)

  * No causal constraint, only padding mask

* **Decoder-only models** (e.g., GPT)

  * Combined with causal mask

* **Encoder-decoder models**

  * Applied to encoder outputs
  * Used in both self-attention and cross-attention
