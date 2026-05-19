# Multi-Head Attention in the Original Transformer (2017)


---

## (1) Global Configuration

The original Transformer uses:

$$
d_{\text{model}} = 512, \quad h = 8
$$

This immediately determines the per-head dimension:

$$
d_{\text{head}} = \frac{d_{\text{model}}}{h} = \frac{512}{8} = 64
$$

Thus:

$$
d_k = d_v = 64
$$

---

## (2) Per-Head Linear Projections

Each head has its own projection matrices:

$$
W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d_{model} \times d_{head}} = \mathbb{R}^{512 \times 64}
$$

So for one head:

* input: $X \in \mathbb{R}^{n \times 512}$
* output:

$$
Q^{(i)}, K^{(i)}, V^{(i)} \in \mathbb{R}^{n \times 64}
$$

Across all heads:

* total number of projection matrices: $8 \times 3 = 24$
* each matrix: $512 \times 64$

---

## (3) Attention Computation per Head

For each head:

$$
Q^{(i)} (K^{(i)})^T \in \mathbb{R}^{n \times n}
$$

Scaling factor:

$$
\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{64}} = \frac{1}{8}
$$

Attention output:

$$
\text{head}^{(i)} \in \mathbb{R}^{n \times 64}
$$

---

## (4) Concatenation Across Heads

After computing all 8 heads:

$$
\text{Concat}\left(\text{head}^{(1)}, \dots, \text{head}^{(8)}\right)
\in \mathbb{R}^{n \times (8 \cdot 64)} = \mathbb{R}^{n \times 512}
$$

Key observation:

$$
8 \times 64 = 512 = d_{\text{model}}
$$

So concatenation restores the original model dimension exactly.

---

## (5) Output Projection $W_O$

Final projection:

$$
W_O \in \mathbb{R}^{(h \cdot d_v) \times d_{model}} = \mathbb{R}^{512 \times 512}
$$

Thus:

$$
\text{MultiHead}(X) \in \mathbb{R}^{n \times 512}
$$

Important point:

* MHA is dimension-preserving
* input and output both live in $\mathbb{R}^{512}$

---

## (6) Parameter Count

Per head:

* $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$: each has $512 \times 64 = 32768$ parameters
* total per head:

$$
3 \times 32768 = 98304
$$

Across 8 heads:

$$
8 \times 98304 = 786432
$$

Output projection:

$$
W_O: 512 \times 512 = 262144
$$

Total parameters in MHA:

$$
786432 + 262144 = 1{,}048{,}576
$$

---

## (7) Why These Numbers Matter

### (a) Constant Model Width

$$
h \cdot d_{head} = 8 \times 64 = 512 = d_{model}
$$

* splitting into heads does not change total dimensionality
* only redistributes it into subspaces

---

### (b) Smaller Attention Spaces

Each head operates in:

$$
\mathbb{R}^{64}
$$

instead of:

$$
\mathbb{R}^{512}
$$

This:

* reduces computation per head
* stabilizes dot products via scaling

---

### (c) Parallel Subspaces

Instead of one large attention:

* 1 head with $512$-dim similarity

we get:

* 8 heads with $64$-dim similarity

This enables:

* multiple independent attention patterns
* richer representation decomposition

---

## (8) Summary

The original MHA is numerically defined by:

$$
d_{\text{model}} = 512, \quad h = 8, \quad d_k = d_v = 64
$$

and implemented as:

$$
\text{MultiHead}(X)=
\text{Concat}\left(\text{head}^{(1)}, \dots, \text{head}^{(h)}\right) W_O
$$

with:

* 8 parallel attention heads
* each head: $64$-dimensional
* total parameters: $1{,}048{,}576$
* output dimension preserved at $512$

