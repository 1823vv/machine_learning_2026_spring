## Question: Linear Regression as a Computation Graph

Treat linear regression with one output as a one-layer neural network with identity activation. For one example, use the row-vector convention

$$
z=xW+b,
$$

$$
\hat y=z,
$$

where $x \in \mathbb{R}^{1 \times d}$, $W \in \mathbb{R}^{d \times 1}$, $b \in \mathbb{R}^{1 \times 1}$, and $y \in \mathbb{R}^{1 \times 1}$. The per-example squared-error loss is

$$
\ell=(\hat y-y)^2.
$$

1. Draw the computation graph from $x$ to $z$ to $\hat y$ to $\ell$ and label the identity activation.
2. Compute $\frac{\partial \ell}{\partial \hat y}$ and $\frac{\partial \hat y}{\partial z}$.
3. Use the chain rule to compute $\delta=\frac{\partial \ell}{\partial z}$.
4. Write $\frac{\partial \ell}{\partial W}$ in terms of $x^T$ and $\delta$, and write $\frac{\partial \ell}{\partial b}$.
