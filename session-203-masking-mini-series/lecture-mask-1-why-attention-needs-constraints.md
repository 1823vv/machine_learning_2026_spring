# Why Attention Needs Constraints

---

## 1. Motivation

Self-attention is one of the most powerful ideas in modern deep learning. It allows every token in a sequence to interact with every other token.

This flexibility is the source of its strength—but also its biggest problem.

Many real-world tasks impose **structural constraints**:

* Language generation is **causal** (future tokens should not be visible)
* Sequences have **variable length** (padding tokens should be ignored)
* Encoder-decoder models have **asymmetric roles** (different sources of information)

Standard self-attention ignores all of these constraints.

---

## 2. Standard Self-Attention is Fully Connected

The attention mechanism is defined as:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $V \in \mathbb{R}^{T \times d_v}$ = value matrix

The matrix $QK^T$ computes **similarity between every pair of positions**.

This means:

> Every token can attend to every other token.

In other words, attention creates a **fully connected interaction pattern**.

---

## 3. The Problem of Too Much Freedom

At first glance, this seems ideal. Why restrict information?

But unrestricted attention introduces serious problems.

---

### 3.1 Future Information Leakage

In autoregressive tasks (e.g., language modeling), the model should learn:

$$
P(x_t \mid x_{<t})
$$

However, without constraints, attention allows:

$$
P(x_t \mid x_1, \dots, x_T)
$$

This means:

* The model can “peek into the future”
* Training becomes artificially easy
* Inference becomes inconsistent

This violates the fundamental **causal structure** of the problem.

---

### 3.2 Noise from Padding Tokens

In batched training, sequences are padded:

```
[The] [cat]  [sat] [on]  [the] [mat]
[I]   [love] [AI]  [PAD] [PAD] [PAD]
```

Without constraints:

* Attention treats `[PAD]` like a real token
* The model assigns weights to meaningless positions
* Outputs become corrupted by noise

---

### 3.3 Mismatch Between Training and Inference

If the model uses information that will not be available at inference time:

* It learns dependencies that cannot be used
* Performance degrades during generation
* The model becomes unreliable

This is a form of **distribution mismatch**.

---

## 4. Attention Needs Structure

The key insight is:

> Not all information should be accessible.

We need a way to control:

* **Which tokens can interact**
* **Which connections are allowed**
* **Which information flows are valid**

This leads to the concept of **masking**.

---

## 5. Masking: The Core Idea

Masking modifies attention by adding a matrix $M$:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where:

* $Q \in \mathbb{R}^{T \times d_k}$ = query matrix
* $K \in \mathbb{R}^{T \times d_k}$ = key matrix
* $V \in \mathbb{R}^{T \times d_v}$ = value matrix
* $M_{ij} = 0$ → interaction allowed
* $M_{ij} = -\infty$ → interaction blocked

After softmax:

* allowed positions keep probability mass
* blocked positions become exactly zero

---

## 6. What Masking Solves

Masking allows us to enforce:

* **Causality** → prevent future leakage
* **Validity** → ignore padding tokens
* **Structure** → control how different parts of the model interact

Instead of removing attention entirely, we **selectively disable connections**.
