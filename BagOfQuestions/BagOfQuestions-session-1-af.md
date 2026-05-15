## Question: Debug a Linear Regression Implementation

A student writes this code in the `fit` function:

```python
y_predicted = X * self.weights + self.bias

dw = (2 / n_samples) * X.T * (y_predicted - y)
db = (2 / n_samples) * np.sum(y_predicted - y)

self.weights = self.weights + self.lr * dw
self.bias = self.bias + self.lr * db
```

1. What is wrong with using `X * self.weights` for multiple linear regression?
2. What should the prediction line be?
3. What is wrong with using `X.T * (y_predicted - y)` for `dw`?
4. What should the correct `dw` line be?
5. What is wrong with updating with `+ self.lr * dw`?
6. Rewrite the corrected code block.
7. Explain why these bugs are shape bugs, optimization-direction bugs, or both.

## Question: Shape Debugging

Assume $X\in\mathbb{R}^{50\times 4}$ and `self.weights.shape == (4,)`.

1. What shape should `y_predicted` have?
2. What shape should `dw` have?
3. Why is `np.dot(X.T, (y_predicted - y))` the right shape?
