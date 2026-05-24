# Notation Guideline for the Positional Encoding Mini-Series

This document fixes the notation used across all lectures and exercises in the `session-202-positional-encoding-mini-series` folder.

---

## 1. Sequence and Tokens

| Symbol | Meaning | Domain / Shape |
|---|---|---|
| $n$ | sequence length (number of tokens) | integer |
| $d_{\text{model}}$ | model embedding dimension | integer |
| $X$ | input token embedding matrix | $\mathbb{R}^{n \times d_{\text{model}}}$ |
| $x_i$ | embedding vector of the token at position $i$ | $\mathbb{R}^{d_{\text{model}}}$ |

The matrix $X$ is stacked row-wise:

$$
X = [x_0;\, x_1;\, \dots;\, x_{n-1}]
$$

---

## 2. Positional Encoding

| Symbol | Meaning | Domain / Shape |
|---|---|---|
| $i$ | **token position index** (0-based) | $i \in \{0,1,\dots,n-1\}$ |
| $p_i$ | positional encoding vector for position $i$ | $\mathbb{R}^{d_{\text{model}}}$ |
| $PE$ | full positional encoding matrix | $\mathbb{R}^{n \times d_{\text{model}}}$ |
| $k$ | **channel-pair index** | $k \in \{0,1,\dots,\frac{d_{\text{model}}}{2}-1\}$ |
| $\omega_k$ | angular frequency of the $k$-th pair | scalar |

---

## 3. Combined Representation

The content-location joint representation is written with a tilde:

$$
\tilde{x}_i = x_i + p_i
$$

In matrix form:

$$
\tilde{X} = X + P, \qquad P \in \mathbb{R}^{n \times d_{\text{model}}}
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

---

## 5. Attention Pipeline (for Reference)

Projections use combined representations:

$$
Q = \tilde{X}W_Q, \quad K = \tilde{X}W_K, \quad V = \tilde{X}W_V
$$

Scaled dot-product attention:

$$
\operatorname{Attn}(\tilde{X}) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

---

## 6. What Not to Do

| Avoid | Reason |
|---|---|
| Using $pos$ instead of $i$ for position | breaks consistency with earlier lectures |
| Using $i$ for channel-pair index | collides with the standard use of $i$ for position |
| Parenthesized subscripts like $PE_{(pos,2i)}$ | the comma subscript $PE_{i,2k}$ is the convention here |
