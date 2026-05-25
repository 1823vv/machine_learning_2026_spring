# Notation Guideline for the Positional Encoding Mini-Series

This document fixes the notation used across all lectures and exercises in the `session-202-positional-encoding-mini-series` folder.

---

## 1. Sequence and Tokens

| Symbol | Meaning | Domain / Shape |
|---|---|---|
| $n$ | sequence length (number of tokens) | integer |
| $d_{\text{model}}$ | model embedding dimension | integer |
| $X$ | input token embedding matrix | $\mathbb{R}^{n \times d_{\text{model}}}$ |
| $x_i$ | embedding vector (row) of the token at position $i$ | $\mathbb{R}^{1 \times d_{\text{model}}}$ |

The matrix $X$ is constructed by stacking token embeddings row-wise:

$$
X =
\begin{bmatrix}
x_0 \\
x_1 \\
\vdots \\
x_{n-1}
\end{bmatrix}
\in \mathbb{R}^{n \times d_{\text{model}}}
$$

Each row corresponds to one token position.

---

## 2. Positional Encoding

| Symbol | Meaning | Domain / Shape |
|---|---|---|
| $i$ | **token position index** (0-based) | $i \in \{0,1,\dots,n-1\}$ |
| $p_i$ | positional encoding vector (row) for position $i$ | $\mathbb{R}^{1 \times d_{\text{model}}}$ |
| $PE$ | full positional encoding matrix | $\mathbb{R}^{n \times d_{\text{model}}}$ |
| $k$ | **channel-pair index** | $k \in \{0,1,\dots,\frac{d_{\text{model}}}{2}-1\}$ |
| $\omega_k$ | angular frequency of the $k$-th pair | scalar |

The full positional encoding matrix $PE$ is constructed by stacking $p_i$ row-wise, exactly like $X$:

$$
PE =
\begin{bmatrix}
p_0 \\
p_1 \\
\vdots \\
p_{n-1}
\end{bmatrix}
\in \mathbb{R}^{n \times d_{\text{model}}}
$$

Each $p_i$ is a **row vector**:

$$
p_i \in \mathbb{R}^{1 \times d_{\text{model}}}
$$

---

## 3. Combined Representation

The content-location joint representation is written with a tilde:

$$
\tilde{x}_i = x_i + p_i
$$

In matrix form:

$$
\tilde{X} = X + PE, \qquad PE \in \mathbb{R}^{n \times d_{\text{model}}}
$$

---

## 4. Standard Sinusoidal Formula

For position index $i$ and channel-pair index $k$:

$$
PE_{i,\,2k} = \sin\!\left(\frac{i}{10000^{\,2k/d_{\text{model}}}}\right)
$$

$$
PE_{i,\,2k+1} = \cos\!\left(\frac{i}{10000^{\,2k/d_{\text{model}}}}\right)
$$

> **Rule of thumb:**
> - $i$ always refers to a **position in the sequence**.
> - $k$ always refers to a **frequency channel pair**.

Explicit row-vector form of $p_i$ (using $m = d_{\text{model}}/2$ channel pairs):

$$
p_i =
\begin{bmatrix}
\sin(\omega_0 i) &
\cos(\omega_0 i) &
\sin(\omega_1 i) &
\cos(\omega_1 i) &
\cdots &
\sin(\omega_{m-1} i) &
\cos(\omega_{m-1} i)
\end{bmatrix}
\in \mathbb{R}^{1 \times d_{\text{model}}}
$$

---

## 5. Why Token = Row Is Natural in Transformer

Because attention computation is

$$
Q = XW_Q
$$

with

$$
X \in \mathbb{R}^{n \times d_{\text{model}}}, \quad W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}
$$

we obtain

$$
Q \in \mathbb{R}^{n \times d_k}
$$

Thus every row is one token’s query, and the token–token similarity matrix $QK^{T}$ is naturally row-oriented. The Transformer community almost universally adopts the convention:

> **token = row**

---

## 6. Attention Pipeline (for Reference)

Projections use combined representations:

$$
Q = \tilde{X}W_Q, \quad K = \tilde{X}W_K, \quad V = \tilde{X}W_V
$$

Scaled dot-product attention:

$$
\operatorname{Attn}(\tilde{X}) = \operatorname{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

---

## 7. What Not to Do

| Avoid | Reason |
|---|---|
| Using $pos$ instead of $i$ for position | breaks consistency with earlier lectures |
| Using $i$ for channel-pair index | collides with the standard use of $i$ for position |
| Parenthesized subscripts like $PE_{(pos,2i)}$ | the comma subscript $PE_{i,2k}$ is the convention here |
| Writing $x_i \in \mathbb{R}^{d_{\text{model}}}$ without specifying row vs. column | $\mathbb{R}^{1 \times d_{\text{model}}}$ removes ambiguity |
| Writing $p_i$ as a comma-separated list without row-vector brackets | use $\begin{bmatrix}\dots\end{bmatrix}$ to make the row shape explicit |
| Using $P$ for the positional encoding matrix | $P$ is easily confused with permutation / probability / projection matrices |



---

## 8. Unified 20X Rule Alignment

- Keep $i$ as token position index across all mini-series.
- Keep plain $k$ for PE channel-pair index only.
- Keep $d_k$ as attention feature dimension symbol in QKV/masking contexts.
- Use $c$ for vocabulary class index in loss formulas.


## 9. Tolerated Differences (Explicitly OK)

- Using plain $k$ as PE channel-pair index is **OK** and canonical in this series.
- Reusing attention symbols (e.g., $Q,K,d_k$) in reference sections is **OK** as contextual reminders; they do not redefine PE index semantics.


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
