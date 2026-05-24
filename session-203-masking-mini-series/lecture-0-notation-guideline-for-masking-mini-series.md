# Notation Guideline for the Masking Mini-Series

This document fixes notation for all lectures in `session-203-masking-mini-series` and aligns it with Session-201 (QKV) and Session-202 (PE).

---

## 1. Shared Base Symbols (Aligned with Session-201/202)

| Symbol | Meaning | Shape |
|---|---|---|
| $n$ | sequence length | integer |
| $d_{\text{model}}$ | embedding width | integer |
| $X$ | token embedding matrix (token=row) | $\mathbb{R}^{n\times d_{\text{model}}}$ |
| $x_i$ | token row at position $i$ | $\mathbb{R}^{1\times d_{\text{model}}}$ |
| $Q,K,V$ | projected query/key/value matrices | $Q,K\in\mathbb{R}^{n\times d_k},\ V\in\mathbb{R}^{n\times d_v}$ |
| $S$ | raw attention score matrix | $\mathbb{R}^{n\times n}$ |
| $A$ | normalized attention weight matrix | $\mathbb{R}^{n\times n}$ |
| $Z$ | attention output | $\mathbb{R}^{n\times d_v}$ |

Core equations:

$$
S=\frac{QK^T}{\sqrt{d_k}},\qquad A=\operatorname{softmax}(S),\qquad Z=AV
$$

Softmax is row-wise (fixed query row $i$, across key index $j$).

---

## 2. Mask Symbols

| Symbol | Meaning | Shape / Domain |
|---|---|---|
| $M$ | additive attention mask applied to logits | $\mathbb{R}^{n\times n}$ |
| $m_{ij}$ | mask entry for query $i$, key $j$ | scalar |
| $\mathcal{A}_i$ | allowed key index set for query $i$ | subset of $\{0,\dots,n-1\}$ |
| $M^{\text{causal}}$ | causal mask | $\mathbb{R}^{n\times n}$ |
| $M^{\text{pad}}$ | padding mask (broadcasted to pairwise form) | compatible with $n\times n$ logits |

Masked attention:

$$
A=\operatorname{softmax}(S+M),\qquad Z=AV
$$

Conventions for $m_{ij}$:
- $m_{ij}=0$ means **allowed**.
- $m_{ij}=-\infty$ (or very negative constant in implementation) means **blocked**.

Equivalent set form:

$$
a_{ij}=0\quad\text{if }j\notin\mathcal{A}_i
$$

---

## 3. Canonical Masks in This Mini-Series

### 3.1 Causal mask

$$
M^{\text{causal}}_{ij}=\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

### 3.2 Padding mask

If token $j$ is padding, it must not be attended to:

$$
M^{\text{pad}}_{ij}=\begin{cases}
-\infty, & j\in\mathcal{P}\\
0, & j\notin\mathcal{P}
\end{cases}
$$

where $\mathcal{P}$ is the set of padding positions.

### 3.3 Combined mask

$$
M^{\text{combined}} = M^{\text{causal}} + M^{\text{pad}}
$$

Any blocked source remains blocked.

---

## 4. Index Rules (Must Stay Stable)

- $i$: query token position (row index).
- $j$: key/value token position (column index in $S,A,M$).
- Do **not** reuse $i,j$ for feature channels.

---

## 5. Consistency Check Across Four Guideline Files

This masking guideline is consistent with:
- `session-201-qkv-attention-mini-series/lecture-0-notation-guideline-for-qkv-mini-series.md`
- `session-202-positional-encoding-mini-series/lecture-0-notation-guideline-for-pe-mini-series.md`
- `session-204-loss-function-mini-series/lecture-0-notation-guideline-for-loss-function-mini-series.md`

### Canonical Rule Set

- Keep $i$ as token position index and $j$ as key-position index in attention/masking.
- Reserve plain $k$ for PE channel/frequency index.
- Keep $d_k$ as query/key dimension symbol.
- Use $c$ as class index in loss formulas.



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
