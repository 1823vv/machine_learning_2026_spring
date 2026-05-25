# Positional Encoding: Why Addition Works

![](./img/ppppp2.jpg)

---
## 1. Design Choice

Transformer uses

$$
\tilde{X}=X+PE
$$

instead of concatenation or separate pipelines.

---
## 2. Why Not Concatenation

Concatenation increases dimensionality and delays content-position interaction until an extra projection mixes them.

---
## 3. Expansion Inside Attention

With

$$
Q=(X+PE)W_Q, \quad K=(X+PE)W_K
$$

we obtain

$$
QK^T=
XW_Q(XW_K)^T+
XW_Q(PEW_K)^T+
PEW_Q(XW_K)^T+
PEW_Q(PEW_K)^T
$$

---
## 4. Interpretation of the Four Terms

### Content-content interaction

$$
XW_Q(XW_K)^T
$$

### Content-position interaction

$$
XW_Q(PEW_K)^T
$$

### Position-content interaction

$$
PEW_Q(XW_K)^T
$$

### Position-position interaction

$$
PEW_Q(PEW_K)^T
$$

> [!INFO]
> A single additive step enables rich interactions without modifying the core attention architecture.

---
## 5. Practical Merits

- same dimensionality across the block
- immediate mixing in first projection
- parameter-efficient and implementation-simple

---
## 6. Summary

Addition is minimal but expressive: it injects order while preserving Transformer structure.
