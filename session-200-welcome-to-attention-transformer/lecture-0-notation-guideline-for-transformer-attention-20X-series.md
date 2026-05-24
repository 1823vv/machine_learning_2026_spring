# Unified Notation Guideline for Transformer Attention 20X Mini-Series

This is the canonical notation specification for:
- `session-201-qkv-attention-mini-series`
- `session-202-positional-encoding-mini-series`
- `session-203-masking-mini-series`
- `session-204-loss-function-mini-series`

It defines the shared symbol contract so all four mini-series stay mutually consistent.

---

## 1) Universal Conventions

1. **Token = row** in sequence matrices.
2. Mathematical formulas default to **single-sequence** form (batch suppressed unless stated).
3. Attention softmax is **row-wise over key axis**.
4. Symbols have fixed roles by domain (position, key, channel, class, width).

---

## 2) Global Indices and Dimension Symbols

### 2.1 Indices

| Symbol | Canonical meaning |
|---|---|
| $i$ | token position / query row index |
| $j$ | key/value token position index (attention score column index) |
| $k$ | PE channel-pair/frequency index (Session-202 formulas) |
| $c$ | vocabulary class index in loss formulas |
| $h$ | attention head index |

### 2.2 Dimensions

| Symbol | Meaning |
|---|---|
| $n$ | sequence length |
| $d_{\text{model}}$ | model embedding width |
| $d_k$ | query/key feature width |
| $d_v$ | value feature width |
| $H$ | number of attention heads |
| $|\mathcal{V}|$ | vocabulary size |

---

## 3) Shared Sequence Representation

$$
X=\begin{bmatrix}x_0\\x_1\\\vdots\\x_{n-1}\end{bmatrix}\in\mathbb{R}^{n\times d_{\text{model}}},\quad
x_i\in\mathbb{R}^{1\times d_{\text{model}}}
$$

This row-stacked representation is assumed unless explicitly overridden.

---

## 4) QKV Attention Core (Sessions 201/203)

Projection parameters:

$$
W_Q,W_K\in\mathbb{R}^{d_{\text{model}}\times d_k},\qquad W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}
$$

Projected tensors:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

with

$$
Q,K\in\mathbb{R}^{n\times d_k},\qquad V\in\mathbb{R}^{n\times d_v}
$$

Scaled scores, weights, output:

$$
S=\frac{QK^T}{\sqrt{d_k}},\qquad A=\operatorname{softmax}(S),\qquad Z=AV
$$

Element view:

$$
a_{ij}=\frac{\exp(s_{ij})}{\sum_{j'=0}^{n-1}\exp(s_{ij'})},\qquad z_i=\sum_{j=0}^{n-1}a_{ij}v_j
$$

---

## 5) Positional Encoding Core (Session 202)

Combined representation:

$$
\tilde{x}_i=x_i+p_i,\qquad \tilde{X}=X+PE
$$

where $p_i\in\mathbb{R}^{1\times d_{\text{model}}}$ and $PE\in\mathbb{R}^{n\times d_{\text{model}}}$.

Sinusoidal form (with plain $k$ as channel-pair index):

$$
PE_{i,2k}=\sin\left(\frac{i}{10000^{2k/d_{\text{model}}}}\right),\qquad
PE_{i,2k+1}=\cos\left(\frac{i}{10000^{2k/d_{\text{model}}}}\right)
$$

---

## 6) Masking Core (Session 203)

Additive-logit mask convention:

$$
A=\operatorname{softmax}(S+M)
$$

where mask entries follow:
- $0$ for allowed,
- $-\infty$ (or implementation large negative constant) for blocked.

Canonical causal mask:

$$
M^{\text{causal}}_{ij}=\begin{cases}
0,& j\le i\\
-\infty,& j>i
\end{cases}
$$

---

## 7) Loss/Objective Core (Session 204)

Per token-position $i$, classifier outputs:

$$
o_i,p_i,y_i\in\mathbb{R}^{1\times |\mathcal{V}|},\qquad p_i=\operatorname{softmax}(o_i)
$$

Class index is $c$:

$$
\ell_i=-\sum_{c=0}^{|\mathcal{V}|-1}y_{ic}\log p_{ic},\qquad
\ell_i=-\log p_{i,t_i}\;\text{(one-hot target)}
$$

Masked sequence average:

$$
\mathcal{L}=\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}}\ell_i
$$

---

## 8) Multi-Head Attention Cross-Series Convention

Per head $h$:

$$
Z^{(h)}=\operatorname{softmax}\!\left(\frac{Q^{(h)}K^{(h)T}}{\sqrt{d_h}}\right)V^{(h)}
$$

Concatenate and project:

$$
Z_{\text{MHA}}=\operatorname{Concat}(Z^{(1)},\dots,Z^{(H)})W_O
$$

This convention is used when MHA is introduced in Session-201 and referenced elsewhere.

---

## 9) Canonical Index Rules (Normative)

1. Use $i$ for token position index.
2. Use $j$ for key/value position index in attention.
3. Use plain $k$ for PE channel-pair index.
4. Use $d_k$ only as query/key feature width.
5. Use $c$ for vocabulary class index in loss formulas.
6. Keep attention softmax row-wise over key axis.

---

## 10) Compatible Variations (Still Canonical)

The following stylistic choices are acceptable and remain compatible:

- Matrix-only vs token-wise equation presentations.
- Including or omitting bias terms in projection equations.
- Suppressing batch dimension in core derivations (recommended default), while adding it in implementation notes.

These do not change symbol meanings.

- Temporary local symbols inside a short derivation (e.g., using $c$ for an additive constant in log-sum-exp) are **OK** when they are clearly scoped and do not redefine global index roles.


---

## 11) Mini-Series Compliance Checklist

A lecture/exercise is compliant if it preserves all of the following:

- token=row,
- $i$ as token position,
- $j$ as key index in attention,
- plain $k$ only in PE channel-pair formulas,
- $c$ for vocabulary class index in loss formulas,
- row-wise softmax in attention.

If any item is violated, update notation to this guideline.


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
---

## 12) Session-by-Session Mirrored Detailed Specifications (Superset Guarantee)

This unified file is intentionally a **superset** of all other `lecture-0-notation-guideline-*.md` files.

The detailed per-series specifications are mirrored below for completeness.

### 12.1 Session-201 QKV Detailed Spec (Mirror)

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


---

### 12.2 Session-202 PE Detailed Spec (Mirror)

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


---

### 12.3 Session-203 Masking Detailed Spec (Mirror)

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


---

### 12.4 Session-204 Loss Detailed Spec (Mirror)

# Notation Guideline for the Loss-Function Mini-Series

This document fixes notation for all lectures in `session-204-loss-function-mini-series` and aligns it with Session-201/202/203.

---

## 1. Shared Transformer Backbone Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $n$ | sequence length | integer |
| $d_{\text{model}}$ | model width | integer |
| $X$ | token embedding matrix (token=row) | $\mathbb{R}^{n\times d_{\text{model}}}$ |
| $Q,K,V$ | attention projections | $Q,K\in\mathbb{R}^{n\times d_k},\ V\in\mathbb{R}^{n\times d_v}$ |
| $Z$ | contextual output representation | $\mathbb{R}^{n\times d_{\text{model}}}$ (after block projections) |

Index meaning is inherited unchanged:
- $i$: token position (row index).
- $c$: vocabulary class index in output vocabulary distributions.

---

## 2. Vocabulary Prediction Symbols

| Symbol | Meaning | Shape |
|---|---|---|
| $|\mathcal{V}|$ | vocabulary size | integer |
| $o_i$ | output logits for token position $i$ | $\mathbb{R}^{1\times |\mathcal{V}|}$ |
| $p_i$ | predicted probability vector | $\mathbb{R}^{1\times |\mathcal{V}|}$ |
| $y_i$ | target distribution (one-hot or smoothed) | $\mathbb{R}^{1\times |\mathcal{V}|}$ |
| $t_i$ | gold token id at position $i$ | integer in $[0,|\mathcal{V}|-1]$ |

$$
p_i=\operatorname{softmax}(o_i)
$$

---

## 3. Cross-Entropy and NLL

Per-token cross-entropy:

$$
\ell_i = -\sum_{c=0}^{|\mathcal{V}|-1} y_{ic}\log p_{ic}
$$

If $y_i$ is one-hot with class $t_i$:

$$
\ell_i = -\log p_{i,t_i}
$$

Sequence loss with valid-index set $\mathcal{I}$ (after masking ignored positions):

$$
\mathcal{L}=\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}}\ell_i
$$

---

## 4. Objective-Specific Masks

| Symbol | Meaning |
|---|---|
| $r_i$ | training eligibility indicator for token $i$ ($1$=counted, $0$=ignored) |
| $\mathcal{I}=\{i\mid r_i=1\}$ | set of positions included in loss |

Examples:
- CLM/GPT: $r_i=1$ for next-token-prediction positions.
- MLM/BERT: $r_i=1$ only on masked-token positions.
- Seq2Seq: $r_i=1$ on decoder target positions excluding padding.

Equivalent weighted form:

$$
\mathcal{L}=\frac{\sum_{i=0}^{n-1}r_i\ell_i}{\sum_{i=0}^{n-1}r_i}
$$

---

## 5. Label Smoothing Convention

With smoothing parameter $\varepsilon\in[0,1)$:

$$
y^{(\text{LS})}_{ic}=(1-\varepsilon)\,\mathbf{1}[c=t_i] + \frac{\varepsilon}{|\mathcal{V}|}
$$

and

$$
\ell_i^{(\text{LS})}=-\sum_c y^{(\text{LS})}_{ic}\log p_{ic}
$$

---

## 6. Consistency Check Across Four Guideline Files

This file is aligned with:
- `session-201-qkv-attention-mini-series/lecture-0-notation-guideline-for-qkv-mini-series.md`
- `session-202-positional-encoding-mini-series/lecture-0-notation-guideline-for-pe-mini-series.md`
- `session-203-masking-mini-series/lecture-0-notation-guideline-for-masking-mini-series.md`

### Canonical Rule Set

- Use $j$ only for key-position index in attention contexts.
- Use $c$ for vocabulary class index in loss formulas.
- Reserve plain $k$ for PE channel-pair indexing and $d_k$ for attention feature dimension.


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
