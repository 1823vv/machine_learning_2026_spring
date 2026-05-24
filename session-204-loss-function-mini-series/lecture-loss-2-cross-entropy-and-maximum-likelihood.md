# Cross-Entropy and Maximum Likelihood

---

## 1. Cross-Entropy Loss (Token-Level)

For a single position, the cross-entropy loss is defined as:

$$
\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{V} y_c \log P_c
$$

where:

* $V$ is the vocabulary size
* $y_c$ is the target distribution
* $P_c$ is the predicted probability

---

### One-Hot Simplification

In language modeling, $y$ is usually one-hot.

So the loss becomes:

$$
\mathcal{L}_{\text{CE}} = -\log P_{\text{correct}}
$$

---

### Interpretation

> The loss only depends on the probability assigned to the correct token

* If $P_{\text{correct}} = 1$, loss is 0
* If $P_{\text{correct}}$ is small, loss is large

---

## 2. Example

Consider the context:

```text
Shanghai has a very beautiful
```

Suppose the correct next word is:

```text
skyline
```

The model predicts:

| Token   | Probability |
| ------- | ----------- |
| skyline | 0.60        |
| river   | 0.20        |
| park    | 0.10        |
| food    | 0.05        |

---

### Loss Computation

$$
\mathcal{L} = -\log 0.60
$$

If instead the model predicted:

$$
P(\text{skyline}) = 0.10
$$

then:

$$
\mathcal{L} = -\log 0.10
$$

---

### Key Insight

> The model is punished for being uncertain about the correct answer

---

## 3. Sequence-Level Loss

Language modeling involves sequences:

$$
x_1, x_2, \dots, x_T
$$

We compute loss at every position:

$$
\mathcal{L}_t = -\log P(x_t \mid x_{<t})
$$

Total loss:

$$
\mathcal{L} = \sum_{t=1}^{T} -\log P(x_t \mid x_{<t})
$$

---

## 4. Maximum Likelihood Estimation (MLE)

We now connect loss to probability.

The probability of a sequence is:

$$
P(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

Taking the logarithm:

$$
\log P(x_1, \dots, x_T) = \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

---

### Training Objective

We want to maximize this probability:

$$
\max_\theta \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)
$$

---

### Equivalent Formulation

Minimizing cross-entropy is equivalent to:

$$
\min_\theta \mathcal{L}=
\max_\theta \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)
$$

---

### Key Insight

> Training a language model = maximizing the probability of the training data

---

## 5. Cross-Entropy Gradient

Let $z_c$ be logits and $P_c = \text{softmax}(z_c)$.

The gradient is:

$$
\frac{\partial \mathcal{L}}{\partial z_c} = P_c - y_c
$$

---

### Interpretation

* $P_c$ = model prediction
* $y_c$ = target

So:

> gradient = prediction − target

---

### Consequence

* If the model overestimates a token → push it down
* If it underestimates the correct token → push it up

---

## 6. Putting It All Together

At each step:

$$
h_t \rightarrow z_t \rightarrow P_t \rightarrow \mathcal{L}_t
$$

Across a sequence:

$$
\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t
$$
