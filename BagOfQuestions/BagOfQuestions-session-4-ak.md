## Question: Logistic Regression as a Sigmoid Computation Graph

Treat binary logistic regression as a one-layer neural network with sigmoid activation. For one example,

$$
z=xW+b,
$$

$$
\hat y=\sigma(z)=\frac{1}{1+e^{-z}},
$$

where $x \in \mathbb{R}^{1 \times d}$, $W \in \mathbb{R}^{d \times 1}$, $b \in \mathbb{R}^{1 \times 1}$, and $y \in \{0,1\}$. The binary cross-entropy loss is

$$
\ell=-\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right].
$$

1. Draw the computation graph from $x$ to $z$ to $\hat y$ to $\ell$.
2. Write $\frac{\partial \ell}{\partial \hat y}$ and $\frac{\partial \hat y}{\partial z}$.
3. Multiply these two derivatives and simplify to get $\delta=\frac{\partial \ell}{\partial z}=\hat y-y$.
4. Write $\frac{\partial \ell}{\partial W}=x^T\delta$ and $\frac{\partial \ell}{\partial b}=\delta$. Explain why this error signal is simpler than the two separate derivatives look at first.
