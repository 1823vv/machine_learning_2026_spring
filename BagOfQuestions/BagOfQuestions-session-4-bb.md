## Question: Why Sigmoid Plus Binary Cross-Entropy Simplifies

For a sigmoid neuron,

$$
\hat y = \sigma(z)=\frac{1}{1+e^{-z}}.
$$

The binary cross-entropy loss for one example is

$$
\ell = -\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right].
$$

1. Compute $\frac{\partial \ell}{\partial \hat y}$.
2. Compute $\frac{\partial \hat y}{\partial z}$. Multiply the two derivatives and simplify.
3. Show the final result $\frac{\partial \ell}{\partial z}=\hat y-y$. Explain why this simplification is useful for implementation.
4. Explain what happens to $\hat y-y$ when the prediction is too high for an example with $y=0$. Explain what happens to $\hat y-y$ when the prediction is too low for an example with $y=1$.
