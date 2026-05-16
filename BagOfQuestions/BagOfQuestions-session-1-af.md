## Question: Debug Linear Regression from Scratch

We are implementing linear regression `class MyOwnLinearRegression` from scratch with NumPy. The model receives a matrix `X` with shape `(50, 4)`, so there are 50 training examples and 4 input features. The target vector `y` has shape `(50,)`, and `self.weights.shape == (4,)`.

A correct multiple linear regression prediction should represent

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

Answer the questions below.

1. What is wrong with `X * self.weights` for multiple linear regression?
2. Write the corrected prediction line using `np.dot`.
3. What shape should the corrected `y_predicted` have?
4. What is wrong with `X.T * (y_predicted - y)` for computing `dw`?
5. Write the corrected `dw` line using `np.dot`.
6. What shape should the corrected `dw` have?
7. Why is the update direction wrong in `self.weights = self.weights + self.lr * dw`?
8. Write the corrected weight update line.
9. Write the corrected bias update line.
10. In one sentence, explain why checking array shapes is a useful debugging habit when implementing linear regression from scratch.
