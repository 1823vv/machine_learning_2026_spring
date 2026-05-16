## Question: Logistic Regression as a Sigmoid Neuron

Treat binary logistic regression as a one-layer neural network with sigmoid activation. For one example,

$$
z = xW + b,
$$

$$
\hat y = \sigma(z) = \frac{1}{1+e^{-z}},
$$

where $x \in \mathbb{R}^{1 \times d}$, $W \in \mathbb{R}^{d \times 1}$, $b \in \mathbb{R}^{1 \times 1}$, and $y \in \{0,1\}$. The binary cross-entropy loss is

$$
\ell = -\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right].
$$

1. Draw the computation graph from $x$ to $z$ to $\hat y$ to $\ell$.
2. Write the derivative $\frac{d\sigma}{dz}$ in terms of $\hat y$. Write $\frac{\partial \ell}{\partial \hat y}$.
3. Use the chain rule to show that $\frac{\partial \ell}{\partial z}=\hat y-y$. Write $\frac{\partial \ell}{\partial W}$ in terms of $x^T$ and $\hat y-y$.
4. Write $\frac{\partial \ell}{\partial b}$. Explain why the error signal $\hat y-y$ is central to logistic-regression training.
