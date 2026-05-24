# Combining Masks in Attention

---

## 1. Motivation

In previous lectures, we introduced two fundamental masks:

* **Causal mask** → enforces temporal order (no future access)
* **Padding mask** → removes invalid tokens (no attention to `[PAD]`)

In real Transformer models, we often need **both constraints at the same time**.

> The key question: how do we combine them into a single attention computation?

---

## 2. Two Constraints, One Goal

Let us restate the two constraints clearly.

---

### 2.1 Causal Constraint

For autoregressive modeling:

$$
i \text{ can attend to } j \quad \text{only if } j \le i
$$

---

### 2.2 Padding Constraint

For variable-length sequences:

$$
i \text{ can attend to } j \quad \text{only if } j \text{ is not padding}
$$

---

### Combined Constraint

We want:

$$
i \text{ can attend to } j
\quad \text{only if}
\quad j \le i \;\; \text{and} \;\; j \text{ is valid}
$$

---

## 3. Mask Combination Principle

Both masks are implemented by adding a matrix before softmax:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $V \in \mathbb{R}^{T \times d_v}$ = value matrix

So we simply combine masks:

$$
M = M_{\text{causal}} + M_{\text{padding}}
$$

---

### Why Addition Works

Because:

* Allowed positions → $0 + 0 = 0$
* Blocked positions → $-\infty + 0 = -\infty$
* Blocked by either → still $-\infty$

> If any constraint blocks an edge, the edge is removed.

---

## 4. Graph Interpretation

From the graph perspective:

* Causal mask removes **future edges**
* Padding mask removes **invalid nodes**

---

### Step-by-Step

Without masking:

```
Fully connected graph
```

Apply causal mask:

```
Remove edges to future nodes
```

Apply padding mask:

```
Remove edges to PAD nodes
```

Final graph:

```
Only past + valid nodes remain
```

---

## 5. Matrix Example

Consider a sequence:

```
Tokens: [A] [B] [C] [PAD]
```

---

### Causal Mask

```
      A    B    C   PAD
A     0  -inf -inf -inf
B     0    0  -inf -inf
C     0    0    0  -inf
PAD   0    0    0    0
```

---

### Padding Mask

```
      A    B    C   PAD
A     0    0    0  -inf
B     0    0    0  -inf
C     0    0    0  -inf
PAD -inf -inf -inf -inf
```

---

### Combined Mask

```
      A    B    C   PAD
A     0  -inf -inf -inf
B     0    0  -inf -inf
C     0    0    0  -inf
PAD -inf -inf -inf -inf
```

---

## 6. Where This Is Used

Combined masks appear in:

* **Decoder-only models (GPT)**

  * causal + padding

* **Decoder in encoder-decoder models**

  * causal + padding (self-attention)

* **Encoder-only models (BERT)**

  * only padding mask (no causal constraint)
