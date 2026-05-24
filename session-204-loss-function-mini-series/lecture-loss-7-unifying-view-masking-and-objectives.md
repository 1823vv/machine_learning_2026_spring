# Unifying View — Masking and Objectives

---

## 1. A Single Perspective

All Transformer variants share the same core pipeline:

$$
h_t \rightarrow z_t \rightarrow P_t \rightarrow \mathcal{L}
$$

They differ only in:

* **what the model can see**
* **what the model is asked to predict**

---

## 2. Two Design Axes

### (1) Masking → Information Flow

Defines the visible context:

* full visibility
* partial visibility
* causal (left-to-right)

---

### (2) Objective → Prediction Target

Defines the learning goal:

* predict masked tokens
* predict next token
* predict target sequence conditioned on input

---

### Key Insight

> Masking controls **input**
> Objective controls **output**

---

## 3. Three Model Families

---

### Encoder-Only (MLM)

* visibility: full sequence with random masking
* objective:

$$
\mathcal{L} = \sum_{t \in \mathcal{M}} -\log P(x_t \mid \tilde{x})
$$

---

### Decoder-Only (CLM)

* visibility: causal (no future tokens)
* objective:

$$
\mathcal{L} = \sum_{t=1}^{T} -\log P(x_t \mid x_{<t})
$$

---

### Encoder–Decoder (Seq2Seq)

* encoder: full visibility over $x$
* decoder: causal over $y$ + cross-attention to $x$
* objective:

$$
\mathcal{L} = \sum_{t=1}^{T_d} -\log P(y_t \mid y_{<t}, x)
$$

---

## 4. Unified Table

| Model           | Masking                       | Objective      |
| --------------- | ----------------------------- | -------------- |
| Encoder-only    | random masking                | MLM            |
| Decoder-only    | causal masking                | CLM            |
| Encoder-decoder | encoder full + decoder causal | conditional LM |

---

## 5. What Actually Changes

Across all variants:

* architecture is similar (attention + FFN)
* training loss is cross-entropy

---

### The only differences are:

* visibility constraints (masking)
* prediction targets (objective)

---

## 6. Interpretation

We can rewrite all objectives as:

$$
\mathcal{L} = \sum_t \mathcal{L}_t = \sum_t -\log P(\text{target}_t \mid \text{visible context})
$$

---

### Examples

* MLM: target = masked tokens
* CLM: target = next token
* Seq2Seq: target = output sequence

---

## 7. Final Insight

All Transformer models are instances of the same principle:

> **learn a probability distribution under a visibility constraint**

* Masking defines **what is known**
* Objective defines **what must be predicted**


> Different models = different choices of masking + objective

