## Question: Compare Error Signals in Linear and Logistic Regression

Linear regression and binary logistic regression can both be viewed as one-layer neural networks. They differ in the output activation and loss.

For linear regression:

$$
z=xW+b, \quad \hat y=z, \quad \ell=(\hat y-y)^2.
$$

For binary logistic regression:

$$
z=xW+b, \quad \hat y=\sigma(z), \quad \ell=-\left[y\log(\hat y)+(1-y)\log(1-\hat y)\right].
$$

1. For linear regression, compute $\delta=\frac{\partial \ell}{\partial z}$.
2. For binary logistic regression with sigmoid plus binary cross-entropy, compute $\delta=\frac{\partial \ell}{\partial z}$. Explain why the linear-regression error signal includes a factor of 2 under this squared-error convention.
3. Explain why the logistic-regression error signal simplifies to $\hat y-y$. For both models, write the common formula $\frac{\partial \ell}{\partial W}=x^T\delta$.
4. Draw two side-by-side computation graphs and highlight where the two derivations differ.
