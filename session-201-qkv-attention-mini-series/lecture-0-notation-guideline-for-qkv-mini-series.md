# Notation Guideline for the QKV Attention Mini-Series (Exhaustive Edition)

This document is the canonical notation reference for all lectures and exercises in
`session-201-qkv-attention-mini-series`.

It is intentionally exhaustive so every symbol has a single, stable meaning.

---

## 1) Core Conventions

1. **Token = row** in all sequence matrices.
2. **Batch dimension is omitted by default** in this mini-series unless explicitly introduced.
3. **Index roles are fixed**:
   - $i$: query-token position (row index)
   - $j$: key/value-token position (column index in score/weight matrices)
   - $h$: attention-head index
   - $f$: generic feature-channel index when needed
4. **Row-wise softmax** is the default in attention: normalize over $j$ for fixed $i$.

---

## 2) Scalar Dimensions and Index Domains

| Symbol | Meaning | Domain |
|---|---|---|
| $n$ | sequence length | integer, $n\ge 1$ |
| $d_{\text{model}}$ | embedding/model width | integer |
| $d_k$ | query/key per-token feature width | integer |
| $d_v$ | value per-token feature width | integer |
| $H$ | number of heads (MHA) | integer |
| $d_h$ | per-head width in MHA (often $d_{\text{model}}/H$) | integer |

Index domains (single sequence):

$$
i,j\in\{0,1,\dots,n-1\}
$$

---

## 3) Sequence Representation and Shapes

| Symbol | Meaning | Shape |
|---|---|---|
| $x_i$ | token embedding row at position $i$ | $\mathbb{R}^{1\times d_{\text{model}}}$ |
| $X$ | row-stacked token matrix | $\mathbb{R}^{n\times d_{\text{model}}}$ |

Row-stacked convention:

$$
X=
\begin{bmatrix}
x_0\\x_1\\\vdots\\x_{n-1}
\end{bmatrix}
\in\mathbb{R}^{n\times d_{\text{model}}}
$$

---

## 4) Q/K/V Projection Parameters and Outputs

### 4.1 Projection matrices

| Symbol | Meaning | Shape |
|---|---|---|
| $W_Q$ | query projection | $\mathbb{R}^{d_{\text{model}}\times d_k}$ |
| $W_K$ | key projection | $\mathbb{R}^{d_{\text{model}}\times d_k}$ |
| $W_V$ | value projection | $\mathbb{R}^{d_{\text{model}}\times d_v}$ |
| $b_Q,b_K,b_V$ | optional bias vectors | $\mathbb{R}^{1\times d_k},\mathbb{R}^{1\times d_k},\mathbb{R}^{1\times d_v}$ |

### 4.2 Projected matrices

Without biases:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

With biases (broadcast row-wise):

$$
Q=XW_Q+\mathbf{1}b_Q,\quad K=XW_K+\mathbf{1}b_K,\quad V=XW_V+\mathbf{1}b_V
$$

where $\mathbf{1}\in\mathbb{R}^{n\times 1}$.

Shapes:

$$
Q,K\in\mathbb{R}^{n\times d_k},\qquad V\in\mathbb{R}^{n\times d_v}
$$

Per-token rows:

$$
q_i=x_iW_Q,\quad k_i=x_iW_K,\quad v_i=x_iW_V
$$

with $q_i,k_i\in\mathbb{R}^{1\times d_k}$ and $v_i\in\mathbb{R}^{1\times d_v}$.

---

## 5) Score Matrix, Scaling, Softmax Axis

### 5.1 Raw/scaled scores

| Symbol | Meaning | Shape |
|---|---|---|
| $s_{ij}$ | scalar compatibility score | scalar |
| $S$ | full score matrix | $\mathbb{R}^{n\times n}$ |

$$
S=\frac{QK^T}{\sqrt{d_k}},\qquad s_{ij}=\frac{q_ik_j^T}{\sqrt{d_k}}
$$

### 5.2 Why scaling appears

- Dot products grow in magnitude with feature width.
- Dividing by $\sqrt{d_k}$ stabilizes logit scale before softmax.
- This helps avoid overly sharp/flat distributions during training.

### 5.3 Softmax convention

| Symbol | Meaning | Shape |
|---|---|---|
| $a_{ij}$ | attention weight from query $i$ to key $j$ | scalar |
| $A$ | attention weight matrix | $\mathbb{R}^{n\times n}$ |

$$
A=\operatorname{softmax}(S)\quad\text{(row-wise)}
$$

Equivalent element form:

$$
a_{ij}=\frac{\exp(s_{ij})}{\sum_{j'=0}^{n-1}\exp(s_{ij'})}
$$

Row normalization property:

$$
\sum_{j=0}^{n-1}a_{ij}=1\quad\text{for each fixed }i
$$

---

## 6) Attention Output

| Symbol | Meaning | Shape |
|---|---|---|
| $z_i$ | output vector at position $i$ | $\mathbb{R}^{1\times d_v}$ |
| $Z$ | output matrix | $\mathbb{R}^{n\times d_v}$ |

Matrix form:

$$
Z=AV=\operatorname{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Token form:

$$
z_i=\sum_{j=0}^{n-1}a_{ij}v_j
$$

---

## 7) Self-Attention vs Cross-Attention Shapes

### 7.1 Self-attention

Single source matrix $X$:

$$
Q=XW_Q,\ K=XW_K,\ V=XW_V
$$

### 7.2 Cross-attention

Separate query-side and memory-side inputs:

- Query-side states: $X^{(q)}\in\mathbb{R}^{n_q\times d_{\text{model}}}$
- Memory-side states: $X^{(m)}\in\mathbb{R}^{n_m\times d_{\text{model}}}$

Then:

$$
Q=X^{(q)}W_Q\in\mathbb{R}^{n_q\times d_k}
$$
$$
K=X^{(m)}W_K\in\mathbb{R}^{n_m\times d_k}
$$
$$
V=X^{(m)}W_V\in\mathbb{R}^{n_m\times d_v}
$$

Scores/weights/output:

$$
S\in\mathbb{R}^{n_q\times n_m},\quad A\in\mathbb{R}^{n_q\times n_m},\quad Z\in\mathbb{R}^{n_q\times d_v}
$$

---

## 8) Multi-Head Attention (MHA) Notation

Per head $h\in\{1,\dots,H\}$:

$$
Q^{(h)}=XW_Q^{(h)},\quad K^{(h)}=XW_K^{(h)},\quad V^{(h)}=XW_V^{(h)}
$$

Head output:

$$
Z^{(h)}=\operatorname{softmax}\!\left(\frac{Q^{(h)}K^{(h)T}}{\sqrt{d_h}}\right)V^{(h)}
$$

Concatenate and project:

$$
Z_{\text{MHA}}=\operatorname{Concat}(Z^{(1)},\dots,Z^{(H)})W_O
$$

with

$$
W_O\in\mathbb{R}^{(H\cdot d_h)\times d_{\text{model}}}
$$

---

## 9) Masks in Attention Logits (Interface Convention)

Although detailed mask definitions are in Session-203, Session-201 adopts:

$$
A=\operatorname{softmax}(S+M)
$$

with mask $M$ broadcast-compatible with score shape, and
- allowed entries add $0$,
- blocked entries add $-\infty$ (or implementation large negative constant).

---

## 10) Batch-Aware Extension (Optional Reference)

If batch size $B$ is explicit:

- $X\in\mathbb{R}^{B\times n\times d_{\text{model}}}$,
- $Q,K\in\mathbb{R}^{B\times n\times d_k}$,
- $V\in\mathbb{R}^{B\times n\times d_v}$,
- $S,A\in\mathbb{R}^{B\times n\times n}$.

Softmax remains over key axis for each batch/query row.

---

## 11) Canonical Symbol Contract for This Series

- $i$: query-token position index.
- $j$: key/value-token position index.
- $f$: generic feature-channel index when required.
- plain $k$: **not** used as token index in this series; kept free to stay compatible with PE channel indexing convention.
- $d_k$: query/key feature width symbol.

---

## 12) Consistency with Sessions 200/202/203/204

This file is fully aligned with the unified 20X guideline and compatible with:

- `session-200-welcome-to-attention-transformer/lecture-0-notation-guideline-for-transformer-attention-20X-series.md`
- `session-202-positional-encoding-mini-series/lecture-0-notation-guideline-for-pe-mini-series.md`
- `session-203-masking-mini-series/lecture-0-notation-guideline-for-masking-mini-series.md`
- `session-204-loss-function-mini-series/lecture-0-notation-guideline-for-loss-function-mini-series.md`

Shared global rules retained:
- token=row,
- row-wise attention softmax,
- $i$ token position,
- $j$ key position,
- plain $k$ reserved for PE channel-pair indexing,
- $c$ vocabulary-class index in loss formulas (outside Session-201 attention equations).

---

## 13) What to Avoid

| Avoid | Why |
|---|---|
| Using column-vector token convention by default | conflicts with all 20X matrix equations |
| Reusing $i$ for channels/classes | collides with token-position meaning |
| Reusing $j$ for vocabulary class in attention chapters | collides with key-position meaning |
| Omitting $1/\sqrt{d_k}$ in scaled dot-product attention | changes logit scale behavior |
| Applying softmax over the wrong axis | breaks attention normalization semantics |
| Mixing self-attention and cross-attention shapes without $n_q,n_m$ distinction | creates silent shape ambiguity |
| Reusing a local symbol like $c$ for a one-off constant inside a short derivation | **Allowed** if scope is explicit and it does not redefine the global class-index role in loss notation |


---

### Scoped Local-Symbol Flexibility (Important)

Short-lived local symbols used inside a clearly scoped derivation are **allowed** even if they overlap with global symbol letters, provided all of the following hold:

1. The scope is explicit (only within a small local derivation/block).
2. The local meaning is stated immediately (e.g., "for any constant $c$").
3. The symbol is not reused to redefine the mini-series canonical role outside that local block.
4. Reader understanding is not impacted (no ambiguity about global notation contract).

Examples that are acceptable:
- using $c$ as an additive constant in a one-off log-sum-exp derivation,
- using temporary $i,j$ as summation counters in a short standalone algebra step when their scope is obvious.


### Transpose notation policy

- Use only `^T` for matrix transpose in this series (e.g., `QK^T`, `q_i k_j^T`).
- Never use `^\top` in lecture content, equations, or examples.
