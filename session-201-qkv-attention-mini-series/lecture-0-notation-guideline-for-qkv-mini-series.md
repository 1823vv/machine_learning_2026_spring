# Notation Guideline for the QKV Attention Mini-Series (Exhaustive Edition)

This document is the canonical notation reference for all lectures and exercises in
`session-201-qkv-attention-mini-series`.

It is intentionally exhaustive so every symbol has a single, stable meaning.

---

## 1) Core Conventions

1. **Token = row** in all sequence matrices.
2. **Batch dimension is omitted by default** in this mini-series unless explicitly introduced.
3. **Index roles are fixed**:
   - $i$: query-token position (row index); also head index in MHA superscripts $(i)$
   - $j$: key/value-token position (column index in score/weight matrices)
   - $h$: number of attention heads
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
| $h$ | number of heads (MHA) | integer |
| $d_{\text{head}}$ | per-head width in MHA (often $d_{\text{model}}/h$) | integer |

Index domains (single sequence):

$$
i,j\in\{0,1,\dots,n-1\}
$$

In multi-head attention, the head index is also denoted by $i$ (or by explicit numerals $1,\dots,h$) in superscript form $(i)$; it never appears in the same formula as the token-position $i$, so no ambiguity arises.

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
| $s_{ij}$ | dot-product score (raw, or unscaled) | scalar |
| $S$ | full score matrix (raw, or scaled) | $\mathbb{R}^{n\times n}$ |

Raw dot product:

$$
s_{ij}=q_i\cdot k_j
$$

Scaled dot product:

$$
s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_k}}
$$


Raw score matrix:

$$
S=QK^T
$$


Scaled score matrix:

$$
S=\frac{QK^T}{\sqrt{d_k}}
$$

The softmax operates on the scaled scores.

### 5.2 Why scaling appears

- Dot products grow in magnitude with feature width.
- Dividing by $\sqrt{d_k}$ stabilizes logit scale before softmax.
- This helps avoid overly sharp/flat distributions during training.

### 5.3 Softmax convention

| Symbol | Meaning | Shape |
|---|---|---|
| $\alpha_{ij}$ | attention weight from query $i$ to key $j$ | scalar |
| $A$ | attention weight matrix | $\mathbb{R}^{n\times n}$ |

$$
A=\operatorname{softmax}(S)\quad\text{(row-wise)}
$$

or 

$$
A=\operatorname{softmax}(S/\sqrt{d_k})\quad\text{(row-wise)}
$$

Equivalent element form (softmax on scaled scores):

$$
\alpha_{ij}=\frac{\exp(s_{ij}/\sqrt{d_k})}{\sum_{j'=0}^{n-1}\exp(s_{ij'}/\sqrt{d_k})}
$$

Row normalization property:

$$
\sum_{j=0}^{n-1}\alpha_{ij}=1\quad\text{for each fixed }i
$$

**Policy:** Scalar attention weights are always written as $\alpha_{ij}$; the full attention weight matrix is $A$, with entries $A_{ij}=\alpha_{ij}$. Never use $a_{ij}$ for attention weights in this series.

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
z_i=\sum_{j=0}^{n-1}\alpha_{ij}v_j
$$

---

## 7) Self-Attention vs Cross-Attention Shapes

### 7.1 Self-attention

Single source matrix $X$:

$$
Q=XW_Q,\ K=XW_K,\ V=XW_V
$$

### 7.2 Cross-attention

Separate query-side (decoder) and memory-side (encoder) inputs:

- Decoder states: $H_{\text{dec}}\in\mathbb{R}^{n_{\text{dec}}\times d_{\text{model}}}$
- Encoder memory: $H_{\text{enc}}\in\mathbb{R}^{n_{\text{enc}}\times d_{\text{model}}}$

Then:

$$
Q=H_{\text{dec}}W_Q\in\mathbb{R}^{n_{\text{dec}}\times d_k}
$$
$$
K=H_{\text{enc}}W_K\in\mathbb{R}^{n_{\text{enc}}\times d_k}
$$
$$
V=H_{\text{enc}}W_V\in\mathbb{R}^{n_{\text{enc}}\times d_v}
$$

Scores/weights/output:

$$
S\in\mathbb{R}^{n_{\text{dec}}\times n_{\text{enc}}},\quad A\in\mathbb{R}^{n_{\text{dec}}\times n_{\text{enc}}},\quad Z\in\mathbb{R}^{n_{\text{dec}}\times d_v}
$$

---

## 8) Multi-Head Attention (MHA) Notation

Per head $i\in\{1,\dots,h\}$:

$$
Q^{(i)}=XW_Q^{(i)},\quad K^{(i)}=XW_K^{(i)},\quad V^{(i)}=XW_V^{(i)}
$$

Head output:

$$
Z^{(i)}=\operatorname{softmax}\!\left(\frac{Q^{(i)}K^{(i)T}}{\sqrt{d_{\text{head}}}}\right)V^{(i)}
$$

Concatenate and project:

$$
Z_{\text{MHA}}=\operatorname{Concat}(Z^{(1)},\dots,Z^{(h)})W_O
$$

with

$$
W_O\in\mathbb{R}^{(h\cdot d_{\text{head}})\times d_{\text{model}}}
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

- $i$: query-token position index (or head index in MHA superscripts $(i)$; the two uses never collide in the same formula).
- $j$: key/value-token position index.
- $f$: generic feature-channel index when required.
- plain $k$: **not** used as token index in this series; kept free to stay compatible with PE channel indexing convention.
- $s_{ij}$: raw (unscaled) dot-product score $q_i \cdot k_j$.
- $S$: scaled score matrix $QK^T / \sqrt{d_k}$.
- $\alpha_{ij}$: scalar attention weight from query $i$ to key $j$.
- $A$: attention weight matrix (row-wise softmax of $S$).
- $d_k$: query/key feature width symbol.
- $h$: number of attention heads.
- $d_{\text{head}}$: per-head feature width in MHA.

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
| Mixing self-attention and cross-attention shapes without $n_{\text{dec}},n_{\text{enc}}$ distinction | creates silent shape ambiguity |
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
