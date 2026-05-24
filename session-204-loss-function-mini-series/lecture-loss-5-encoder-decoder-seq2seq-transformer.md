# Encoder–Decoder Models — Seq2Seq Transformer

---

## 1. Setting: Conditional Sequence Modeling

We model a mapping:

$$
x_1, x_2, \dots, x_{T_e} \rightarrow y_1, y_2, \dots, y_{T_d}
$$

The goal is to learn:

$$
P(y_1, y_2, \dots, y_{T_d} \mid x_1, x_2, \dots, x_{T_e})
$$

---

## 2. Factorization (Autoregressive Decoder)

We decompose:

$$
P(y_1, y_2, \dots, y_{T_d} \mid x) = \prod_{t=1}^{T_d} P(y_t \mid y_{<t}, x)
$$

---

### Position-Level Loss

$$
\mathcal{L}_t = -\log P(y_t \mid y_{<t}, x)
$$

### Training Objective

$$
\mathcal{L} = \sum_{t=1}^{T_d} \mathcal{L}_t
$$

---

## 3. Encoder: Full-Context Representation

The encoder processes the input:

$$
h_1, h_2, \dots, h_{T_e} = \text{Encoder}(x_1, x_2, \dots, x_{T_e})
$$

---

### Property

* bidirectional attention
* full visibility over $x$

---

### Role

> build a contextual representation of the source sequence

---

## 4. Decoder: Conditional Generation

The decoder predicts targets:

$$
h_t = \text{Decoder}(y_{<t}, h_1, h_2, \dots, h_{T_e})
$$

---

### Mechanisms

* causal mask on $y$ (no future targets)
* cross-attention to encoder outputs (access to source)

---

### Output

$$
P(y_t \mid y_{<t}, x)
$$

---

## 5. Shift and Alignment

We construct decoder inputs and labels:

```text
decoder input:  <BOS>  y_1   y_2   ... y_{T_d-1}
labels:         y_1    y_2   y_3   ... y_{T_d}
```

---

### Interpretation

At position $t$:

* input provides $y_{<t}$
* target is $y_t$

---

## 6. Teacher Forcing and Parallel Training

During training:

* use ground-truth $y_{<t}$
* full sequence is available

---

### Consequence

All steps are computed in parallel:

$$
\mathcal{L}_1, \dots, \mathcal{L}_{T_d}
$$

---

### Key Insight

> parallelism comes from teacher forcing + masking

---

## 7. What Is Learned

The model learns a **conditional language model**:

$$
P(y \mid x)
$$

---

### Result

* input understanding (encoder)
* output generation (decoder)
* alignment via cross-attention

---

### Representative Model

* Transformer 2017, T5

