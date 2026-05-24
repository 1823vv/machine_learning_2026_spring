# Encoder-Only Models — Masked Language Modeling (MLM)

---

## 1. Setting: Full-Context Encoding

Given a sequence:

$$
x_1, x_2, \dots, x_T
$$

An encoder (no causal mask) produces:

$$
h_t = \text{Encoder}(x_1, \dots, x_T)
$$

---

### Property

* each position sees **all tokens**
* bidirectional context

---

## 2. Why Direct Prediction Fails

If we train:

$$
P(x_t \mid x_{\neq t})
$$

the model has access to $x_t$ itself.

---

### Result

> trivial identity mapping (no useful learning)

---

## 3. Masked Language Modeling

Introduce a mask set $\mathcal{M}$:

* replace $x_t$ with a mask token for $t \in \mathcal{M}$

Define corrupted input:

$$
\tilde{x}_1, \dots, \tilde{x}_T
$$

---

### Objective

For each masked position:

$$
\mathcal{L}_t = -\log P(x_t \mid \tilde{x})
$$

Total loss:

$$
\mathcal{L} = \sum_{t \in \mathcal{M}} \mathcal{L}_t
$$

---

### Key Insight

> prediction is only required at masked positions

---

## 4. Learning Signal

For each $t \in \mathcal{M}$:

* $x_t$ is hidden
* model must infer it from context

---

### Effect

* prevents copying
* forces use of global context

---

## 5. What Is Learned

MLM optimizes:

$$
P(x_t \mid \text{context})
$$

with **full bidirectional context**.

---

### Result

* contextual embeddings
* semantic understanding
* non-causal representations

---

### Representative Model

* BERT
