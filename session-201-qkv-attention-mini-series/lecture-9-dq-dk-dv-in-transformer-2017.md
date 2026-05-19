
# 1. Original Transformer (2017) — Q/K/V Projections

In **Attention Is All You Need (2017)**, the Query-Key-Value (Q/K/V) framework is defined as follows.

---

## (1) Definitions — Single Token

For a single token represented as a **row vector** $x \in \mathbb{R}^{1 \times d_{\text{model}}}$:

$$
q = x W_Q \in \mathbb{R}^{1 \times d_k}, \quad
k = x W_K \in \mathbb{R}^{1 \times d_k}, \quad
v = x W_V \in \mathbb{R}^{1 \times d_v}
$$

with weight matrices:

$$
W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

**Notes**:

* $d_q = d_k$ → required for dot product similarity
* $d_v$ → free parameter, controls output dimension

---

## (2) Definitions — Entire Sequence

For a sequence of $n$ tokens, stacked as $X \in \mathbb{R}^{n \times d_{\text{model}}}$:

$$
Q = X W_Q \in \mathbb{R}^{n \times d_k}, \quad
K = X W_K \in \mathbb{R}^{n \times d_k}, \quad
V = X W_V \in \mathbb{R}^{n \times d_v}
$$

* Attention scores are computed as:

$$
\boxed{\text{Attention}(Q, K, V)=
\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V}
$$


* or, for self attention:

$$
\boxed{\text{Attention}(X)=
\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V}
$$


* The attention matrix $QK^T \in \mathbb{R}^{n \times n}$ is **square**, because queries and keys come from the same sequence.

---

## (3) Multi-Head Design

Original Transformer uses:

* $d_{\text{model}} = 512$, $h = 8$ heads

Per head:

$$
d_k = d_q = 64, \quad d_v = 64
$$

Total dimensions satisfy:

$$
d_{\text{model}} = h \cdot d_k = h \cdot d_v
$$

**Explanation**:

* $d_k$ → similarity space, controls **how tokens attend to each other**
* $d_v$ → output space, controls **what information flows forward**
* Multi-head allows parallel subspaces for richer representation

---

## (4) Key Constraints and Freedom

* $d_q = d_k$ → **mathematically required** for dot-product attention
* $d_v$ → **free parameter**
* Original Transformer convenience: $d_v = d_k$

---

## (5) Modern Interpretation (GPT-style)

Most modern Transformers adopt:

$$
d_q = d_k = d_v = d_{\text{head}}, \quad
d_{\text{head}} = \frac{d_{\text{model}}}{h}
$$

* Treat Q, K, V **symmetrically**
* Simplifies implementation and standardizes design
