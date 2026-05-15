## Question: Batch Gradient Formula for Linear Regression

For multiple linear regression under the row-vector convention:

$$
\hat{Y}=XW+b
$$

and MSE loss:

$$
\mathcal{L}(W,b)=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}^{(i)}-y^{(i)})^2
$$

1. Write the batch gradient formula for $\frac{\partial \mathcal{L}}{\partial W}$.
2. Write the batch gradient formula for $\frac{\partial \mathcal{L}}{\partial b}$.
3. Explain what the vector $\hat{Y}-Y$ represents.
4. If $X\in\mathbb{R}^{n\times d}$ and $\hat{Y}-Y\in\mathbb{R}^{n\times 1}$, what is the shape of $X^{\mathsf{T}}(\hat{Y}-Y)$?
5. Why should this shape match the shape of $W$?
6. Write the gradient descent updates for $W$ and $b$ using leftarrow notation.
7. Draw a computation graph from $X,W,b$ to $\hat{Y}$ to loss, and then show gradients flowing backward to $W$ and $b$.

## Question: Error Signal

1. In linear regression, why is $\hat{Y}-Y$ called an error signal?
2. What happens to the gradient if predictions are already very close to targets?
