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
