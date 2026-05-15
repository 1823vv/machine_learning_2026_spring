## Question: Debug Linear Regression Implementation

TODO: improve the whole this file.

TODO: make the background of code from scratch a bit more in detail. 
We are working on `class MyOwnLinearRegression`. A student writes this code in the `fit` function:

```python
y_predicted = X * self.weights + self.bias

dw = (2 / n_samples) * X.T * (y_predicted - y)
db = (2 / n_samples) * np.sum(y_predicted - y)

self.weights = self.weights + self.lr * dw
self.bias = self.bias + self.lr * db
```

1. What is wrong with using `X * self.weights` for multiple linear regression? What should the prediction line be after correction?
3. What is wrong with using `X.T * (y_predicted - y)` for `dw`?
4. What should the correct `dw` line be?
5. What is wrong with updating with `+ self.lr * dw`? how to fix?

Assume $X\in\mathbb{R}^{50\times 4}$ and `self.weights.shape == (4,)`.

6. What shape should `y_predicted` have?
7. What shape should `dw` have?
