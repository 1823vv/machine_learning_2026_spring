# Batch Normalization

---

## 1. Motivation: Internal Covariate Shift

During training of deep neural networks, the distribution of activations in intermediate layers keeps changing as parameters update.

Under the row-vector convention, one linear layer computes:

$$
z = x W + b
$$

where $x \in \mathbb{R}^{1 \times d_{in}}$, $W \in \mathbb{R}^{d_{in} \times d_{out}}$, $b \in \mathbb{R}^{1 \times d_{out}}$, and $z \in \mathbb{R}^{1 \times d_{out}}$.

As training progresses:

* input distributions to later layers shift
* later layers must continuously adapt
* optimization can become unstable

This phenomenon motivates Batch Normalization.

---

## 2. Core Idea

Batch Normalization standardizes intermediate activations within a mini-batch, feature by feature.

For a batch $Z \in \mathbb{R}^{m \times d_{out}}$, where rows are samples and columns are activation features:

$$
Z = X W + \mathbf{1}b
$$

Compute batch statistics for each feature $j$:

$$
\mu_{B,j} = \frac{1}{m} \sum_{i=1}^{m} Z_{i,j}
$$

$$
\sigma_{B,j}^2 = \frac{1}{m} \sum_{i=1}^{m} (Z_{i,j} - \mu_{B,j})^2
$$

Normalize each activation:

$$
\hat{Z}_{i,j} = \frac{Z_{i,j} - \mu_{B,j}}{\sqrt{\sigma_{B,j}^2 + \epsilon}}
$$

Equivalently, $\mu_B, \sigma_B^2, \gamma, \beta \in \mathbb{R}^{1 \times d_{out}}$ are row vectors that broadcast across the $m$ samples.

---

## 3. Learnable Transformation

After normalization, we restore representation flexibility using:

* scale parameter $\gamma$
* shift parameter $\beta$

Final output:

$$
Y_{i,j} = \gamma_j \hat{Z}_{i,j} + \beta_j
$$

This ensures the network can recover any necessary distribution if normalization is not optimal.

---

## 4. Intuition

Batch Normalization does two things:

* stabilizes activation distribution
* reduces sensitivity to parameter initialization

Effectively:

* each layer receives inputs with controlled mean and variance
* optimization becomes easier and more predictable

---

## 5. Placement in Neural Networks

A standard structure is:

$$
\text{Linear} \rightarrow \text{BatchNorm} \rightarrow \text{Activation}
$$

Example:

$$
x W + b \rightarrow \text{BN} \rightarrow \text{ReLU}
$$

This means the affine pre-activation is normalized before the nonlinearity.

---

## 6. Training Behavior

During training:

* statistics are computed from the current mini-batch
* each batch introduces slight randomness
* $\gamma$ and $\beta$ are learned by gradient descent like other parameters

Running estimates are also maintained for inference:

$$
\mu_{running} \leftarrow (1 - \alpha)\mu_{running} + \alpha \mu_B
$$

$$
\sigma^2_{running} \leftarrow (1 - \alpha)\sigma^2_{running} + \alpha \sigma_B^2
$$

These estimates approximate global dataset statistics.

---

## 7. Inference Behavior

During inference:

* mini-batch statistics may be noisy or unavailable
* running estimates are used instead
* dropout-like randomness is not introduced

Normalization becomes:

$$
\hat{z} = \frac{z - \mu_{running}}{\sqrt{\sigma^2_{running} + \epsilon}}
$$

This ensures deterministic behavior.

---

## 8. Effects on Optimization

Batch Normalization improves training in several ways.

### 8.1 Faster Convergence

* allows higher learning rates
* reduces sensitivity to initialization

---

### 8.2 Gradient Stability

* reduces exploding gradients
* mitigates vanishing gradients

---

### 8.3 Implicit Regularization

Because batch statistics vary:

* noise is injected during training
* model becomes more robust
* overfitting can be reduced

---

## 9. Relation to Standardization

Standardization applies to input features:

$$
x \rightarrow \frac{x - \mu}{\sigma}
$$

Batch Normalization applies the same idea internally:

* inside hidden layers
* during training dynamics
* separately for each activation feature

Thus it can be viewed as:

> standardization applied repeatedly inside a deep network

---

## 10. Summary

Batch Normalization:

* normalizes each activation feature across the mini-batch
* learns $\gamma$ and $\beta$ to preserve representation flexibility
* uses mini-batch statistics during training and running estimates during inference
* stabilizes intermediate activations and improves optimization speed

It is a core technique for training deep neural networks efficiently.
