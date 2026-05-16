## Question: Debug Linear Regression from Scratch

We are implementing linear regression `class MyOwnLinearRegression` from scratch with NumPy. The feature matrix has shape `X.shape == (50, 4)`, the target vector has shape `y.shape == (50,)`, and `self.weights.shape == (4,)`. A correct multiple linear regression prediction should represent

$$
\hat y_i = w_1x_{i1} + w_2x_{i2} + w_3x_{i3} + w_4x_{i4} + b.
$$

A student writes the following incorrect code inside the `fit` function:

```python
y_predicted = X * self.weights + self.bias

dw = (2 / n_samples) * X.T * (y_predicted - y)
db = (2 / n_samples) * np.sum(y_predicted - y)

self.weights = self.weights + self.lr * dw
self.bias = self.bias + self.lr * db
```

1. Explain what is wrong with `X * self.weights` for multiple linear regression, and write the corrected prediction line using `np.dot`.
2. State the corrected shape of `y_predicted`, explain what is wrong with `X.T * (y_predicted - y)` for `dw`, and write the corrected `dw` line using `np.dot`.
3. State the corrected shape of `dw` and explain why checking array shapes is useful when debugging linear regression from scratch.
4. Explain why the update direction is wrong in `self.weights = self.weights + self.lr * dw`, then write the corrected weight and bias update lines.
