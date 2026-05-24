# The Goal of Prediction — From Vectors to Words

---

## 1. From Representation to Prediction

In Transformer models:

* attention builds contextual representations
* positional encoding injects order
* masking controls information flow

At each position $t$, we get a vector:

$$
h_t \in \mathbb{R}^{d}
$$

This is not a word, but a **continuous representation**.

---

### Key Question

> How do we turn $h_t$ into a token?

---

## 2. Projection to Vocabulary

We map hidden states to vocabulary space:

$$
z_t = h_t W_o
$$

where:

* $W_o \in \mathbb{R}^{d \times V}$
* $V$ is vocabulary size
* $z_t \in \mathbb{R}^{V}$ are **logits**

---

### Interpretation

Each dimension of $z_t$ scores one token:

> the model scores all words, not selects one

---

## 3. From Scores to Probabilities

We apply softmax:

$$
P_c = \frac{\exp(z_c)}{\sum_{c^\prime=1}^{V} \exp(z_{c^\prime})}
$$

This gives:

$$
P \in \mathbb{R}^{V}, \quad \sum_c P_c = 1
$$

---

### Interpretation

At each position $t$, the model outputs:

> a categorical distribution over the vocabulary

---

## 4. Ground Truth as One-Hot

If the correct token is:

```text
city
```

we represent it as:

$$
y_c =
\begin{cases}
1 & \text{if } c = \text{city} \\
0 & \text{otherwise}
\end{cases}
$$

---

### Interpretation

* prediction: $P$ (soft distribution)
* target: $y$ (one-hot distribution)

---

## 5. The Learning Goal

We want:

> predicted distribution ≈ ground truth distribution

This means:

* increase $P_{\text{correct}}$
* decrease others

---

### Intuition

* correct model: $P_{\text{correct}} \rightarrow 1$
* wrong model: $P_{\text{correct}} \ll 1$

