## Question: Linear Regression as a One-Layer Neural Network

Treat linear regression with one output as a one-layer neural network with identity activation. For one training example, use the row-vector convention

$$
z = xW + b,
$$

$$
\hat y = z,
$$

where $x \in \mathbb{R}^{1 \times d}$, $W \in \mathbb{R}^{d \times 1}$, $b \in \mathbb{R}^{1 \times 1}$, and $y \in \mathbb{R}^{1 \times 1}$. The per-example squared-error loss is

$$
\ell = (\hat y - y)^2.
$$

1. Draw the computation graph from $x$ to $z$ to $\hat y$ to $\ell$. Write $\frac{\partial \ell}{\partial \hat y}$.
2. Write $\frac{\partial \hat y}{\partial z}$. Use the chain rule to write $\delta = \frac{\partial \ell}{\partial z}$.
3. Write $\frac{\partial \ell}{\partial W}$ in terms of $x^T$ and $\delta$. Write $\frac{\partial \ell}{\partial b}$.
4. Write the gradient descent updates for $W$ and $b$ using learning rate $\eta$ and the notation $\leftarrow$. Explain why this is backpropagation even though the model has only one layer.
