# Positional Encoding: Why Addition Works

![](./img/ppppp2.jpg)

---
## 1. Design Choice

Transformer uses

$$
\tilde{X}=X+P
$$

instead of concatenation or separate pipelines.

---
## 2. Why Not Concatenation

Concatenation increases dimensionality and delays content-position interaction until an extra projection mixes them.

---
## 3. Expansion Inside Attention

With

$$
Q=(X+P)W_Q, \quad K=(X+P)W_K
$$

we obtain

$$
QK^\top=
XW_Q(XW_K)^\top+
XW_Q(PW_K)^\top+
PW_Q(XW_K)^\top+
PW_Q(PW_K)^\top
$$

---
## 4. Interpretation of the Four Terms

### Content-content interaction

$$
XW_Q(XW_K)^\top
$$

### Content-position interaction

$$
XW_Q(PW_K)^\top
$$

### Position-content interaction

$$
PW_Q(XW_K)^\top
$$

### Position-position interaction

$$
PW_Q(PW_K)^\top
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
