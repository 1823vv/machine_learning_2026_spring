# Softmax Cross-Entropy with Logits: Full Batch Gradient Descent

---

## Overview

The `softmax_crossentropy_with_logits` function in `code-my_nn.py` implements the **combined** softmax + cross-entropy computation and its gradient in a numerically stable way. This combined approach is preferred over computing softmax and cross-entropy separately because it yields a mathematically elegant gradient.

---

## 1. Problem Setup: Full Batch Setting

### 1.1 Dataset Notation

We have a dataset with $n$ samples:

- **Input matrix**: $X \in \mathbb{R}^{n \times d}$ — each row $x_i \in \mathbb{R}^{1 \times d}$ is one input sample
- **Label vector**: $y \in \mathbb{R}^{n}$ — each $y_i \in \{0, 1, \ldots, K-1\}$ is a class index
- **One-hot labels**: $Y \in \mathbb{R}^{n \times K}$ — each row $Y_i$ is a one-hot vector with $Y_{i, y_i} = 1$

### 1.2 Network Output

For a network with $L$ layers, the forward pass produces:

$$
Z^{(L)} = A^{(L-1)} W^{(L)} + b^{(L)}, \quad \hat{Y} = \text{softmax}(Z^{(L)})
$$

Where:

- $Z^{(L)} \in \mathbb{R}^{n \times K}$ — raw logits (scores) for all $n$ samples
- $\hat{Y} \in \mathbb{R}^{n \times K}$ — predicted probabilities, each row $\hat{Y}_i$ sums to 1

### 1.3 Loss Function

In full batch gradient descent, we use the **mean** cross-entropy loss over all $n$ samples:

$$
\boxed{\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} Y_{i,k} \log \hat{Y}_{i,k}}
$$

This is equivalent to:

$$
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} Y_{i,k} \log \hat{Y}_{i,k} = -\frac{1}{n} \sum_{i=1}^{n} \left( Y_i \odot \log \hat{Y}_i \right) \mathbf{1}^\top
$$

Where $\odot$ denotes element-wise multiplication and $\mathbf{1}$ is a column vector of ones.

> **Why mean instead of sum?** Using the mean ensures the loss scale is independent of dataset size $n$, making learning rates and gradient magnitudes more interpretable across different dataset sizes.

---

## 2. Softmax Function

### 2.1 Definition

The softmax function converts raw logits $z_k$ into probabilities that sum to 1:

$$
\hat{y}_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

For the full batch, applying softmax row-wise to $Z^{(L)}$:

$$
\hat{Y}_{i,k} = \frac{e^{Z^{(L)}_{i,k}}}{\sum_{j=1}^{K} e^{Z^{(L)}_{i,j}}}
$$

### 2.2 Numerical Stability

Directly computing $e^{z_k}$ can cause overflow when $z_k$ is large. We use the **log-sum-exp trick**:

$$
\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_j e^{z_j}} = \frac{e^{z_k - M}}{\sum_j e^{z_j - M}}
$$

Where $M = \max_i z_i$ is the maximum logit in each row. This shifts all logits so the largest is 0, preventing overflow.

In code:

```python
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

### 2.3 Jacobian of Softmax (Optional Deep Dive)

The Jacobian matrix of softmax with respect to its input is:

$$
\frac{\partial \text{softmax}(z)}{\partial z} = \text{diag}(\hat{y}) - \hat{y}\hat{y}^\top
$$

This is a **negative semi-definite** matrix, which means softmax is a **log-concave** function — an important property for optimization.

---

## 3. Cross-Entropy Loss

### 3.1 Definition

For a single sample with one-hot label $y$ and prediction $\hat{y}$:

$$
\ell(y, \hat{y}) = -\sum_{k=1}^{K} y_k \log \hat{y}_k
$$

This is the **categorical cross-entropy** (also called **negative log-likelihood**).

### 3.2 Intuition

When $y$ is one-hot (say $y_k = 1$ for the true class $k$):

$$
\ell = -\log \hat{y}_k
$$

- If $\hat{y}_k \to 1$: loss $\to 0$ (perfect prediction)
- If $\hat{y}_k \to 0$: loss $\to \infty$ (confident wrong prediction is heavily penalized)

The cross-entropy measures the "surprise" of the prediction relative to the true label.

---

## 4. The Magic: Softmax + Cross-Entropy Gradient

### 4.1 The Key Result

This is the most elegant result in multiclass classification:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial Z^{(L)}} = \frac{1}{n}(\hat{Y} - Y)}
$$

**The gradient is simply the difference between predicted probabilities and true labels!**

### 4.2 Why Does This Happen?

Let's derive this step by step for a single sample, then extend to the full batch.

#### Step 1: Cross-Entropy Gradient w.r.t. Softmax Outputs

For a single sample:

$$
\ell = -\sum_{k=1}^{K} y_k \log \hat{y}_k
$$

Taking the derivative w.r.t. $\hat{y}_j$:

$$
\frac{\partial \ell}{\partial \hat{y}_j} = -\frac{y_j}{\hat{y}_j}
$$

#### Step 2: Softmax Gradient w.r.t. Logits

We need $\frac{\partial \hat{y}_j}{\partial z_i}$. Using the quotient rule:

$$
\hat{y}_j = \frac{e^{z_j}}{\sum_k e^{z_k}}
$$

- **If $i = j$ (diagonal elements)**:
  $$
  \frac{\partial \hat{y}_j}{\partial z_j} = \frac{e^{z_j} \cdot \sum_k e^{z_k} - e^{z_j} \cdot e^{z_j}}{(\sum_k e^{z_k})^2} = \hat{y}_j(1 - \hat{y}_j)
  $$

- **If $i \neq j$ (off-diagonal elements)**:
  $$
  \frac{\partial \hat{y}_j}{\partial z_i} = \frac{-e^{z_j} \cdot e^{z_i}}{(\sum_k e^{z_k})^2} = -\hat{y}_j \hat{y}_i
  $$

This can be written compactly as:
$$
\frac{\partial \hat{y}_j}{\partial z_i} = \hat{y}_j (\delta_{ij} - \hat{y}_i)
$$
Where $\delta_{ij}$ is the Kronecker delta.

#### Step 3: Chain Rule — Putting It Together

Using the chain rule:

$$
\frac{\partial \ell}{\partial z_i} = \sum_{j=1}^{K} \frac{\partial \ell}{\partial \hat{y}_j} \cdot \frac{\partial \hat{y}_j}{\partial z_i}
$$

Substituting:

$$
\frac{\partial \ell}{\partial z_i} = \sum_{j=1}^{K} \left(-\frac{y_j}{\hat{y}_j}\right) \cdot \hat{y}_j(\delta_{ij} - \hat{y}_i)
= -\sum_{j=1}^{K} y_j(\delta_{ij} - \hat{y}_i)
$$

Breaking this sum:
- When $j = i$: $-y_i(1 - \hat{y}_i) = -y_i + y_i\hat{y}_i$
- When $j \neq i$: $-y_j(-\hat{y}_i) = y_j\hat{y}_i$

So:
$$
\frac{\partial \ell}{\partial z_i} = -y_i + y_i\hat{y}_i + \sum_{j \neq i} y_j\hat{y}_i
= -y_i + \hat{y}_i \sum_{j=1}^{K} y_j
$$

Since $y$ is one-hot, $\sum_j y_j = 1$:

$$
\boxed{\frac{\partial \ell}{\partial z_i} = \hat{y}_i - y_i}
$$

#### Step 4: Full Batch Extension

For the full batch with mean loss:

$$
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n} \ell_i
$$

The gradient w.r.t. each row of $Z^{(L)}$ is:

$$
\frac{\partial \mathcal{L}}{\partial Z^{(L)}_{i,:}} = \frac{1}{n}(\hat{Y}_i - Y_i)
$$

In matrix form:

$$
\boxed{\Delta^{(L)} = \frac{\partial \mathcal{L}}{\partial Z^{(L)}} = \frac{1}{n}(\hat{Y} - Y)}
$$

### 4.3 Interpretation

This gradient has a beautiful intuitive meaning:

- For the **correct class** $k$: $\Delta^{(L)}_{i,k} = \frac{1}{n}(\hat{Y}_{i,k} - 1)$
  - If prediction is confident and correct ($\hat{Y}_{i,k} \approx 1$): gradient $\approx 0$
  - If prediction is wrong ($\hat{Y}_{i,k} \approx 0$): gradient $\approx -\frac{1}{n}$ (negative signal)

- For **incorrect classes** $j \neq k$: $\Delta^{(L)}_{i,j} = \frac{1}{n}\hat{Y}_{i,j}$
  - Positive signal pushing logits up for classes the model assigns probability to

---

## 5. Implementation in Code

### 5.1 The Function

```python
def softmax_crossentropy_with_logits(logits, labels):
    # Create one-hot vectors from labels
    batch_size = logits.shape[0]
    one_hot_labels = np.zeros_like(logits)
    one_hot_labels[np.arange(batch_size), labels] = 1

    # Compute softmax (with numeric stability)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy loss
    loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size

    # Gradient of cross-entropy loss w.r.t. logits
    grad = (softmax_probs - one_hot_labels) / batch_size

    return loss, grad
```

### 5.2 Step-by-Step Breakdown

| Step | Code | Purpose |
|:----:|:-----|:--------|
| 1 | `batch_size = logits.shape[0]` | Get number of samples ($n$) |
| 2 | `one_hot_labels[...] = 1` | Convert integer labels to one-hot matrix $Y$ |
| 3 | `exp_logits = np.exp(logits - np.max(...))` | Numerically stable softmax |
| 4 | `softmax_probs = ... / np.sum(...)` | Compute $\hat{Y} = \text{softmax}(Z^{(L)})$ |
| 5 | `loss = -np.sum(one_hot_labels * np.log(...)) / batch_size` | Compute $\mathcal{L} = -\frac{1}{n}Y \cdot \log \hat{Y}$ |
| 6 | `grad = (softmax_probs - one_hot_labels) / batch_size` | Compute $\Delta^{(L)} = \frac{1}{n}(\hat{Y} - Y)$ |

### 5.3 Key Implementation Details

#### Numerical Stability: Log Probabilities

```python
np.log(softmax_probs + 1e-9)
```

Adding `1e-9` prevents `log(0)`, which would produce `-inf`. The value `1e-9` is small enough not to affect the loss meaningfully.

#### Why Divide by `batch_size` in Both Loss and Gradient?

The division by `batch_size` serves two purposes:

1. **Loss**: Gives the *mean* loss across samples, making it comparable across different dataset sizes
2. **Gradient**: Scales the gradient so that its magnitude is independent of dataset size

This means:
- **Loss**: $\mathcal{L} \in \mathbb{R}$ (scalar, the mean negative log-likelihood)
- **Gradient**: $\Delta^{(L)} \in \mathbb{R}^{n \times K}$ (matrix where each row is the mean gradient for that sample)

---

## 6. Integration with Backpropagation

### 6.1 Role in the Training Loop

In the `train` function:

```python
def train(network, X, y):
    # Forward pass
    activations = forward(network, X)
    logits = activations[-1]

    # Compute loss and initial gradient
    loss, grad_logits = softmax_crossentropy_with_logits(logits, y)

    # Backward pass (backpropagation)
    grad_output = grad_logits
    for i in range(len(network))[::-1]:  # Reversed order
        layer = network[i]
        grad_output = layer.backward(grad_output)

    return loss
```

The `grad_logits` returned is exactly $\Delta^{(L)} = \frac{1}{n}(\hat{Y} - Y)$.

### 6.2 Backpropagation Flow

Following the framework from `lecture-5-backprop-in-neural-networks.md`:

1. **Output Layer Error** (already computed by `softmax_crossentropy_with_logits`):
   $$
   \Delta^{(L)} = \frac{1}{n}(\hat{Y} - Y)
   $$

2. **Output Layer Parameter Gradients** (in `Dense.backward`):
   $$
   \frac{\partial \mathcal{L}}{\partial W^{(L)}} = (A^{(L-1)})^T \Delta^{(L)}
   $$
   $$
   \frac{\partial \mathcal{L}}{\partial b^{(L)}} = \sum_{i=1}^{n} \Delta^{(L)}_{i,:}
   $$

3. **Hidden Layer Error Propagation**:
   $$
   \Delta^{(l)} = \Delta^{(l+1)} (W^{(l+1)})^T \odot (f^{(l)})'(Z^{(l)})
   $$

4. **Hidden Layer Parameter Gradients**:
   $$
   \frac{\partial \mathcal{L}}{\partial W^{(l)}} = (A^{(l-1)})^T \Delta^{(l)}
   $$
   $$
   \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \sum_{i=1}^{n} \Delta^{(l)}_{i,:}
   $$

### 6.3 The Dense Layer Implementation

```python
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        # He initialization
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        self.biases = np.zeros(output_units)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, grad_output):
        # Gradient of loss w.r.t. weights: input^T · grad_output
        grad_weights = np.dot(self.input.T, grad_output)

        # Gradient of loss w.r.t. biases: sum grad_output over batch dimension
        grad_biases = np.sum(grad_output, axis=0)

        # Gradient of loss w.r.t. input: grad_output · weights^T
        grad_input = np.dot(grad_output, self.weights.T)

        # Update parameters using gradient descent
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
```

Note: The $\frac{1}{n}$ factor is **not** divided again in `backward` because it's already incorporated in `grad_output = Δ^{(L)}` from `softmax_crossentropy_with_logits`.
