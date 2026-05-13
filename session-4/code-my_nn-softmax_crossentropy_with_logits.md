# Softmax Cross-Entropy with Logits

## Overview

The `softmax_crossentropy_with_logits` function in `code-my_nn.py` implements the combined softmax activation and cross-entropy loss function for multiclass classification. This function is crucial for training neural networks on classification tasks like MNIST digit recognition.

## Function Implementation

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

## Theoretical Foundation

### 1. Softmax Function

From the theory, the softmax function converts raw scores (logits) into probabilities:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

Where:
- $z_i$ are the logits (raw scores from the network)
- $\hat{y}_i$ are the predicted probabilities
- $K$ is the number of classes

### 2. Cross-Entropy Loss

The cross-entropy loss measures the difference between predicted probabilities and true labels:

$$
\mathcal{L} = -\sum_{k=1}^{K} y_k \log \hat{y}_k
$$

Where $y_k$ is 1 for the true class and 0 for all others (one-hot encoding).

### 3. Gradient Derivation

The gradient of cross-entropy loss with respect to logits simplifies to:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i
$$

This beautiful simplification occurs because:
$$
\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} = -\frac{y_i}{\hat{y}_i} \cdot \hat{y}_i(1 - \hat{y}_i) + \sum_{j \neq i} \frac{y_j}{\hat{y}_j} \cdot \hat{y}_i \hat{y}_j = \hat{y}_i - y_i
$$

## Implementation Details

### 1. One-Hot Encoding

```python
one_hot_labels = np.zeros_like(logits)
one_hot_labels[np.arange(batch_size), labels] = 1
```

Converts integer labels (0-9 for MNIST) into one-hot vectors where the true class has value 1 and others have value 0.

### 2. Numerical Stability

```python
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
```

Subtracts the maximum logit before exponentiation to prevent numerical overflow when logits are large.

### 3. Loss Computation

```python
loss = -np.sum(one_hot_labels * np.log(softmax_probs + 1e-9)) / batch_size
```

- Uses `np.log(softmax_probs + 1e-9)` to avoid `log(0)` which would be undefined
- Averages over the batch size for consistent training dynamics

### 4. Gradient Computation

```python
grad = (softmax_probs - one_hot_labels) / batch_size
```

This implements the theoretical gradient $\hat{y} - y$ with a crucial addition: **averaging over the batch**.

#### Why Divide by Batch Size?

The theoretical gradient $\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$ gives the gradient for a **single sample**. However, in full-batch training we use all $n$ samples simultaneously. Here's why we divide by `batch_size` (which equals $n$ in full-batch mode):

**1. Gradient Averaging**
- Without division: $\text{grad} = \sum_{i=1}^{n} (\hat{y}_i - y_i)$
- With division: $\text{grad} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$
- This gives us the **mean gradient** across all $n$ samples in the dataset

**2. Training Stability**
- Larger datasets would produce larger gradient magnitudes without averaging
- Division ensures gradient scale is **independent of dataset size**
- Learning rate $\eta$ becomes more interpretable and consistent

**3. Connection to Theory**
From `lecture-5-backprop-in-neural-networks.md`, full-batch gradients are:
$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (A^{(l-1)})^T \Delta^{(l)}
$$

$$
\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \sum_{i=1}^{n} \Delta^{(l)}_{i,:}
$$

The $\frac{1}{n}$ factor is absorbed into $\Delta^{(l)}$ at the output layer, so subsequent parameter gradients do **not** divide by $n$ again.

**4. Practical Example**
If dataset size $n = 1000$ and we have:
- Sample 1: $\hat{y}_1 - y_1 = [0.1, -0.1, 0.0]$
- Sample 2: $\hat{y}_2 - y_2 = [0.2, -0.1, -0.1]$
- ... (continuing for all 1000 samples)

Without averaging: gradients would scale with dataset size
With averaging: $\text{grad} = \frac{1}{1000} \sum_{i=1}^{1000} (\hat{y}_i - y_i)$ gives stable, representative gradient direction

**Note:** This represents full-batch gradient descent using the entire dataset. Mini-batch training (using $m < n$ samples) will be covered in later sessions.

## Connection to Backpropagation Theory

This function provides the **output layer error signal** (mean over full batch):

$$
\Delta^{(L)} = \frac{1}{n}(\hat{Y} - Y)
$$

This error signal is then propagated backward through the network using:

$$
\Delta^{(l)} = \Delta^{(l+1)} (W^{(l+1)})^T \odot (f^{(l)})'(Z^{(l)})
$$

And parameter gradients are computed as (no additional $\frac{1}{n}$ factor needed):

$$
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = (A^{(l-1)})^T \Delta^{(l)}
$$

$$
\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \sum_{i=1}^{n} \Delta^{(l)}_{i,:}
$$

> **Remark:** Because the loss is defined as the mean over all $n$ samples, the factor $\frac{1}{n}$ is introduced during output-layer gradient computation. Therefore, subsequent weight and bias gradients should not divide by $n$ again.

## Usage in Training

In the `train` function, this gradient is used to start backpropagation:

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

The `grad_logits` returned by `softmax_crossentropy_with_logits` becomes the initial error signal that gets propagated through each layer's `backward` method.
